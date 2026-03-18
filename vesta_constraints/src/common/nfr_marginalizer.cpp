#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/nfr_marginalizer.h>
#include <vesta_constraints/common/qr_marginalizer.h>
#include <vesta_core/eigen.h>
#include <vesta_core/graph.h>

#include <glog/logging.h>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vesta_constraints
{

namespace
{

// Eigenvalue threshold relative to the maximum eigenvalue
constexpr double EIGENVALUE_REL_THRESHOLD = 1e-10;
constexpr double EIGENVALUE_ABS_THRESHOLD = 1e-14;

double computeEigenvalueThreshold(double max_eigenvalue)
{
  double threshold = max_eigenvalue * EIGENVALUE_REL_THRESHOLD;
  return threshold < EIGENVALUE_ABS_THRESHOLD ? EIGENVALUE_ABS_THRESHOLD : threshold;
}

/// Represents a selected edge in the sparsity pattern
struct SparsityEdge
{
  int var_a;
  int var_b;
  double mutual_information;
};

/// Union-Find data structure for Kruskal's algorithm
class UnionFind
{
public:
  explicit UnionFind(int n) : parent_(n), rank_(n, 0)
  {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  int find(int x)
  {
    while (parent_[x] != x)
    {
      parent_[x] = parent_[parent_[x]];
      x = parent_[x];
    }
    return x;
  }

  bool unite(int x, int y)
  {
    int rx = find(x);
    int ry = find(y);
    if (rx == ry)
    {
      return false;
    }
    if (rank_[rx] < rank_[ry])
    {
      std::swap(rx, ry);
    }
    parent_[ry] = rx;
    if (rank_[rx] == rank_[ry])
    {
      ++rank_[rx];
    }
    return true;
  }

private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

/**
 * @brief Compute MI between two variable blocks using only the needed sigma subblocks.
 *
 * MI(xi, xj) = 0.5 * log(det(Sigma_ii) * det(Sigma_jj) / det(Sigma_ij_block))
 * All blocks are extracted as lightweight Eigen::Block views — no copies.
 */
double computeMutualInformation(const vesta_core::MatrixXd& sigma, int off_i, int dim_i, int off_j, int dim_j)
{
  // Use Eigen::Block views to avoid copies
  auto sigma_ii = sigma.block(off_i, off_i, dim_i, dim_i);
  auto sigma_jj = sigma.block(off_j, off_j, dim_j, dim_j);

  double det_ii = sigma_ii.determinant();
  double det_jj = sigma_jj.determinant();

  if (det_ii <= 0.0 || det_jj <= 0.0)
  {
    return 0.0;
  }

  // Build the joint 2x2 block matrix and compute its determinant
  // det(joint) = det(S_ii) * det(S_jj - S_ji * S_ii^{-1} * S_ij) [Schur complement]
  // This avoids allocating the full joint matrix
  auto sigma_ij = sigma.block(off_i, off_j, dim_i, dim_j);

  // For small blocks (typical dim 3-7), the Schur complement approach with solve is efficient
  vesta_core::MatrixXd sigma_ii_copy = sigma_ii;
  Eigen::LLT<vesta_core::MatrixXd> llt_ii(sigma_ii_copy);
  if (llt_ii.info() != Eigen::Success)
  {
    return 0.0;
  }

  // Schur complement: S_jj - S_ji * S_ii^{-1} * S_ij
  vesta_core::MatrixXd sigma_ij_copy = sigma_ij;
  vesta_core::MatrixXd schur = vesta_core::MatrixXd(sigma_jj) -
                                vesta_core::MatrixXd(sigma.block(off_j, off_i, dim_j, dim_i)) *
                                    llt_ii.solve(sigma_ij_copy);

  double det_schur = schur.determinant();
  if (det_schur <= 0.0)
  {
    return 0.0;
  }

  // det(joint) = det(S_ii) * det(schur)
  double det_joint = det_ii * det_schur;

  double mi = 0.5 * std::log(det_ii * det_jj / det_joint);
  return std::max(mi, 0.0);
}

/**
 * @brief Compute X_e = (A_e Sigma A_e^T)^{-1} directly from sigma subblocks.
 *
 * For A_e = [-I_a, I_b], the product A_e * Sigma * A_e^T = S_aa - S_ab - S_ba + S_bb.
 * This avoids constructing the full sparse A_e matrix.
 */
vesta_core::MatrixXd computeEdgeInformation(const vesta_core::MatrixXd& sigma, int off_a, int dim_a, int off_b,
                                             int dim_b, int meas_dim)
{
  // A_e * Sigma * A_e^T = S_aa - S_ab - S_ba + S_bb (for the meas_dim x meas_dim block)
  vesta_core::MatrixXd asa(meas_dim, meas_dim);
  for (int r = 0; r < meas_dim; ++r)
  {
    for (int c = 0; c < meas_dim; ++c)
    {
      double val = 0.0;
      if (r < dim_a && c < dim_a)
      {
        val += sigma(off_a + r, off_a + c);
      }
      if (r < dim_b && c < dim_b)
      {
        val += sigma(off_b + r, off_b + c);
      }
      if (r < dim_a && c < dim_b)
      {
        val -= sigma(off_a + r, off_b + c);
      }
      if (r < dim_b && c < dim_a)
      {
        val -= sigma(off_b + r, off_a + c);
      }
      asa(r, c) = val;
    }
  }
  asa = (asa + asa.transpose()) * 0.5;

  // Invert: X_e = asa^{-1}
  Eigen::LLT<vesta_core::MatrixXd> llt(asa);
  if (llt.info() == Eigen::Success)
  {
    return llt.solve(vesta_core::MatrixXd::Identity(meas_dim, meas_dim));
  }

  // Fallback: pseudoinverse
  Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig(asa);
  vesta_core::MatrixXd result = vesta_core::MatrixXd::Zero(meas_dim, meas_dim);
  if (eig.info() == Eigen::Success)
  {
    double thresh = computeEigenvalueThreshold(eig.eigenvalues().maxCoeff());
    for (int k = 0; k < eig.eigenvalues().size(); ++k)
    {
      if (eig.eigenvalues()(k) > thresh)
      {
        result +=
            (1.0 / eig.eigenvalues()(k)) * eig.eigenvectors().col(k) * eig.eigenvectors().col(k).transpose();
      }
    }
  }
  return result;
}

/**
 * @brief Compute sqrt-info matrix Lt such that Lt^T * Lt = X (via Cholesky when possible).
 */
vesta_core::MatrixXd computeSqrtInfo(const vesta_core::MatrixXd& X)
{
  // Try Cholesky first — faster than eigendecomposition for small PSD matrices
  Eigen::LLT<vesta_core::MatrixXd> llt(X);
  if (llt.info() == Eigen::Success)
  {
    // L * L^T = X, return L^T as the sqrt-info
    return llt.matrixL().transpose();
  }

  // Fallback: eigendecomposition
  Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig(X);
  const auto& vals = eig.eigenvalues();
  const auto& vecs = eig.eigenvectors();

  vesta_core::MatrixXd sqrt_d = vesta_core::MatrixXd::Zero(X.rows(), X.cols());
  for (int k = 0; k < vals.size(); ++k)
  {
    if (vals(k) > 0.0)
    {
      sqrt_d(k, k) = std::sqrt(vals(k));
    }
  }

  // L = V * sqrt(D), Lt = sqrt(D) * V^T
  return (sqrt_d * vecs.transpose());
}

/**
 * @brief Project a symmetric matrix onto the PSD cone via Cholesky check, falling back to eigendecomposition.
 */
vesta_core::MatrixXd projectPSD(const vesta_core::MatrixXd& X)
{
  // Fast path: if already PSD, Cholesky succeeds and we skip the projection
  Eigen::LLT<vesta_core::MatrixXd> llt(X);
  if (llt.info() == Eigen::Success)
  {
    return X;
  }

  // Slow path: eigendecomposition and clamp
  Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig(X);
  return eig.eigenvectors() * eig.eigenvalues().cwiseMax(0.0).asDiagonal() * eig.eigenvectors().transpose();
}

}  // namespace

NfrMarginalizer::NfrMarginalizer(bool use_fej) : use_fej_(use_fej)
{
}

vesta_core::Transaction NfrMarginalizer::marginalize(const std::string& source,
                                                     const std::vector<vesta_core::UUID>& marginalized_variables,
                                                     const vesta_core::Graph& graph)
{
  // 1. Classify variables
  auto classified = detail::classifyVariables(marginalized_variables, graph);

  // 2. If no non-stamped, delegate to QR
  if (classified.non_stamped.empty())
  {
    return QRMarginalizer(use_fej_).marginalize(source, marginalized_variables, graph);
  }

  // 3. Setup
  const size_t num_non_stamped = classified.non_stamped.size();
  const size_t num_marginalized = marginalized_variables.size();
  auto variable_order = detail::buildSchurEliminationOrder(classified.non_stamped, classified.stamped);

  // 4. Linearize and bucket
  auto lin_result = detail::linearizeAndBucket(source, marginalized_variables, num_marginalized, graph,
                                               std::move(variable_order), use_fej_);

  // 5. Collect Schur terms
  auto schur_terms = detail::collectSchurTerms(lin_result.linear_terms, num_non_stamped);

  // 6. Compute Schur complement
  if (!schur_terms.empty())
  {
    auto schur_result = detail::computeSchurComplement(schur_terms, num_non_stamped, graph, lin_result.variable_order);

    if (schur_result)
    {
      auto& H_oo = schur_result->H_oo;
      auto& eta_o = schur_result->eta_o;
      const auto& other_indices = schur_result->other_indices;
      const auto& other_offsets = schur_result->other_offsets;
      const auto& var_tangent_sizes = schur_result->var_tangent_sizes;
      int total_other_dim = schur_result->total_other_dim;

      int num_remaining = static_cast<int>(other_indices.size());

      // Helper: emit block-diagonal priors for a set of variable indices
      auto emitBlockDiagonalPriors = [&](const std::vector<unsigned int>& indices) {
        for (auto idx : indices)
        {
          int off = other_offsets.at(idx);
          int dim = var_tangent_sizes.at(idx);
          vesta_core::MatrixXd H_ii = H_oo.block(off, off, dim, dim);
          vesta_core::VectorXd eta_i = eta_o.segment(off, dim);

          auto single_term = detail::createSingleVariableTerm(idx, H_ii, eta_i);
          if (!single_term.variables.empty())
          {
            if (idx >= num_marginalized)
            {
              auto mc = detail::createMarginalConstraint(source, single_term, graph, lin_result.variable_order);
              lin_result.transaction.addConstraint(std::move(mc));
            }
            else
            {
              lin_result.linear_terms[idx].push_back(std::move(single_term));
            }
          }
        }
      };

      if (num_remaining <= 1)
      {
        emitBlockDiagonalPriors(other_indices);
      }
      else
      {
        // Compute covariance directly from eigendecomposition of H_oo.
        // This is needed for MI computation and edge information recovery.
        // We reuse the eigendecomposition for both rank checking and covariance.
        Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig_hoo(H_oo);
        if (eig_hoo.info() != Eigen::Success)
        {
          LOG(ERROR) << "NfrMarginalizer: eigendecomposition of H_oo failed. "
                     << "Falling back to block-diagonal priors.";
          emitBlockDiagonalPriors(other_indices);
        }
        else
        {
          const auto& eigenvalues = eig_hoo.eigenvalues();
          const auto& eigenvectors = eig_hoo.eigenvectors();
          double threshold = computeEigenvalueThreshold(eigenvalues.maxCoeff());

          int rank = 0;
          for (int k = 0; k < eigenvalues.size(); ++k)
          {
            if (eigenvalues(k) > threshold)
            {
              ++rank;
            }
          }

          if (rank == 0)
          {
            LOG(WARNING) << "NfrMarginalizer: H_oo has zero rank. No marginal constraints generated.";
          }
          else
          {
            // Compute covariance from eigendecomposition (no redundant Cholesky):
            // Sigma = V * D^{-1} * V^T (using only positive eigenvalues)
            vesta_core::MatrixXd sigma(total_other_dim, total_other_dim);
            sigma.setZero();
            for (int k = 0; k < eigenvalues.size(); ++k)
            {
              if (eigenvalues(k) > threshold)
              {
                sigma.noalias() +=
                    (1.0 / eigenvalues(k)) * (eigenvectors.col(k) * eigenvectors.col(k).transpose());
              }
            }

            // ---- NFR Algorithm ----

            // Step 1: Compute MI for all variable pairs and select Chow-Liu tree
            std::vector<SparsityEdge> all_edges;
            all_edges.reserve(static_cast<size_t>(num_remaining * (num_remaining - 1) / 2));

            for (int a = 0; a < num_remaining; ++a)
            {
              int off_a = other_offsets.at(other_indices[a]);
              int dim_a = var_tangent_sizes.at(other_indices[a]);
              for (int b = a + 1; b < num_remaining; ++b)
              {
                int off_b = other_offsets.at(other_indices[b]);
                int dim_b = var_tangent_sizes.at(other_indices[b]);

                double mi = computeMutualInformation(sigma, off_a, dim_a, off_b, dim_b);
                if (mi > 0.0)
                {
                  all_edges.push_back({ a, b, mi });
                }
              }
            }

            // Use partial_sort to only sort the top (num_remaining - 1) edges
            // needed for the MST, instead of fully sorting all O(n²) edges.
            int num_tree_needed = num_remaining - 1;
            if (static_cast<int>(all_edges.size()) > num_tree_needed * 2)
            {
              std::partial_sort(all_edges.begin(),
                                all_edges.begin() + std::min(static_cast<int>(all_edges.size()), num_tree_needed * 3),
                                all_edges.end(), [](const SparsityEdge& a, const SparsityEdge& b) {
                                  return a.mutual_information > b.mutual_information;
                                });
            }
            else
            {
              std::sort(all_edges.begin(), all_edges.end(), [](const SparsityEdge& a, const SparsityEdge& b) {
                return a.mutual_information > b.mutual_information;
              });
            }

            // Kruskal's MST (inline to avoid re-sorting in selectChowLiuTree)
            UnionFind uf(num_remaining);
            std::vector<SparsityEdge> selected_edges;
            selected_edges.reserve(static_cast<size_t>(num_tree_needed));
            for (const auto& edge : all_edges)
            {
              if (static_cast<int>(selected_edges.size()) >= num_tree_needed)
              {
                break;
              }
              if (uf.unite(edge.var_a, edge.var_b))
              {
                selected_edges.push_back(edge);
              }
            }

            // Step 2: Recover pairwise edge information matrices using direct block extraction
            int num_selected = static_cast<int>(selected_edges.size());
            std::vector<vesta_core::MatrixXd> x_blocks(num_selected);

            // Accumulate H_edges diagonal blocks per variable (only diag blocks needed for NFR)
            // Instead of building full H_edges, track per-variable diagonal contribution
            std::vector<vesta_core::MatrixXd> h_edges_diag(num_remaining);
            for (int v = 0; v < num_remaining; ++v)
            {
              int dim = var_tangent_sizes.at(other_indices[v]);
              h_edges_diag[v] = vesta_core::MatrixXd::Zero(dim, dim);
            }

            for (int e = 0; e < num_selected; ++e)
            {
              const auto& edge = selected_edges[e];
              int off_a = other_offsets.at(other_indices[edge.var_a]);
              int dim_a = var_tangent_sizes.at(other_indices[edge.var_a]);
              int off_b = other_offsets.at(other_indices[edge.var_b]);
              int dim_b = var_tangent_sizes.at(other_indices[edge.var_b]);
              int meas_dim = std::min(dim_a, dim_b);

              // Compute X_e directly from sigma subblocks (no full A_e matrix)
              x_blocks[e] = computeEdgeInformation(sigma, off_a, dim_a, off_b, dim_b, meas_dim);

              // Accumulate diagonal blocks: A_e^T X_e A_e has:
              //   block(a,a) += X_e[0:dim_a, 0:dim_a]
              //   block(b,b) += X_e[0:dim_b, 0:dim_b]
              for (int r = 0; r < meas_dim && r < dim_a; ++r)
              {
                for (int c = 0; c < meas_dim && c < dim_a; ++c)
                {
                  h_edges_diag[edge.var_a](r, c) += x_blocks[e](r, c);
                }
              }
              for (int r = 0; r < meas_dim && r < dim_b; ++r)
              {
                for (int c = 0; c < meas_dim && c < dim_b; ++c)
                {
                  h_edges_diag[edge.var_b](r, c) += x_blocks[e](r, c);
                }
              }
            }

            // Step 3: Emit pairwise relative factors
            for (int e = 0; e < num_selected; ++e)
            {
              const auto& edge = selected_edges[e];
              unsigned int idx_a = other_indices[edge.var_a];
              unsigned int idx_b = other_indices[edge.var_b];
              int dim_a = var_tangent_sizes.at(idx_a);
              int dim_b = var_tangent_sizes.at(idx_b);
              int meas_dim = std::min(dim_a, dim_b);

              vesta_core::MatrixXd Lt = computeSqrtInfo(x_blocks[e]);

              detail::LinearTerm rel_term;
              rel_term.variables = { idx_a, idx_b };

              vesta_core::MatrixXd A_a = vesta_core::MatrixXd::Zero(meas_dim, dim_a);
              vesta_core::MatrixXd A_b = vesta_core::MatrixXd::Zero(meas_dim, dim_b);
              int copy_dim = std::min({ meas_dim, dim_a, dim_b });
              A_a.block(0, 0, meas_dim, copy_dim) = -Lt.block(0, 0, meas_dim, copy_dim);
              A_b.block(0, 0, meas_dim, copy_dim) = Lt.block(0, 0, meas_dim, copy_dim);

              rel_term.A = { A_a, A_b };
              rel_term.b = vesta_core::VectorXd::Zero(meas_dim);

              if (idx_a >= num_marginalized && idx_b >= num_marginalized)
              {
                auto mc = detail::createMarginalConstraint(source, rel_term, graph, lin_result.variable_order);
                lin_result.transaction.addConstraint(std::move(mc));
              }
              else
              {
                auto min_var = std::min(idx_a, idx_b);
                lin_result.linear_terms[min_var].push_back(std::move(rel_term));
              }
            }

            // Step 4: Emit per-variable absolute prior factors from residual information
            for (int v = 0; v < num_remaining; ++v)
            {
              unsigned int idx = other_indices[v];
              int off = other_offsets.at(idx);
              int dim = var_tangent_sizes.at(idx);

              // Residual: H_oo_ii - H_edges_ii (using tracked diagonal, not full H_edges)
              vesta_core::MatrixXd H_res = H_oo.block(off, off, dim, dim) - h_edges_diag[v];
              H_res = (H_res + H_res.transpose()) * 0.5;
              H_res = projectPSD(H_res);

              vesta_core::VectorXd eta_res = eta_o.segment(off, dim);

              auto single_term = detail::createSingleVariableTerm(idx, H_res, eta_res);
              if (!single_term.variables.empty())
              {
                if (idx >= num_marginalized)
                {
                  auto mc = detail::createMarginalConstraint(source, single_term, graph, lin_result.variable_order);
                  lin_result.transaction.addConstraint(std::move(mc));
                }
                else
                {
                  lin_result.linear_terms[idx].push_back(std::move(single_term));
                }
              }
            }
          }
        }
      }
    }
  }

  // 7. QR phase for stamped variables
  detail::marginalizeStampedVariables(lin_result.linear_terms, num_non_stamped, num_marginalized);

  // 8. Emit remaining constraints
  detail::emitRemainingConstraints(source, lin_result.linear_terms, num_marginalized, graph, lin_result.variable_order,
                                   lin_result.transaction);

  return std::move(lin_result.transaction);
}

}  // namespace vesta_constraints
