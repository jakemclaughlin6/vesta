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

// Tikhonov regularization constant for covariance computation
constexpr double TIKHONOV_EPS = 1e-6;

// Eigenvalue threshold relative to the maximum eigenvalue
constexpr double EIGENVALUE_REL_THRESHOLD = 1e-10;
constexpr double EIGENVALUE_ABS_THRESHOLD = 1e-14;

/**
 * @brief Compute eigenvalue threshold for numerical rank determination
 */
double computeEigenvalueThreshold(double max_eigenvalue)
{
  double threshold = max_eigenvalue * EIGENVALUE_REL_THRESHOLD;
  if (threshold < EIGENVALUE_ABS_THRESHOLD)
  {
    threshold = EIGENVALUE_ABS_THRESHOLD;
  }
  return threshold;
}

/**
 * @brief Compute the mutual information between two variable blocks given the joint covariance.
 */
double computeMutualInformation(const vesta_core::MatrixXd& sigma, int off_i, int dim_i, int off_j, int dim_j)
{
  vesta_core::MatrixXd sigma_ii = sigma.block(off_i, off_i, dim_i, dim_i);
  vesta_core::MatrixXd sigma_jj = sigma.block(off_j, off_j, dim_j, dim_j);

  int joint_dim = dim_i + dim_j;
  vesta_core::MatrixXd sigma_joint(joint_dim, joint_dim);
  sigma_joint.block(0, 0, dim_i, dim_i) = sigma_ii;
  sigma_joint.block(0, dim_i, dim_i, dim_j) = sigma.block(off_i, off_j, dim_i, dim_j);
  sigma_joint.block(dim_i, 0, dim_j, dim_i) = sigma.block(off_j, off_i, dim_j, dim_i);
  sigma_joint.block(dim_i, dim_i, dim_j, dim_j) = sigma_jj;

  double det_ii = sigma_ii.determinant();
  double det_jj = sigma_jj.determinant();
  double det_joint = sigma_joint.determinant();

  if (det_joint <= 0.0 || det_ii <= 0.0 || det_jj <= 0.0)
  {
    return 0.0;
  }

  double mi = 0.5 * std::log(det_ii * det_jj / det_joint);
  return std::max(mi, 0.0);
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
 * @brief Select edges for the Chow-Liu maximum spanning tree using Kruskal's algorithm.
 */
std::vector<SparsityEdge> selectChowLiuTree(std::vector<SparsityEdge>& edges, int num_vars)
{
  std::sort(edges.begin(), edges.end(),
            [](const SparsityEdge& a, const SparsityEdge& b) { return a.mutual_information > b.mutual_information; });

  UnionFind uf(num_vars);
  std::vector<SparsityEdge> tree_edges;
  tree_edges.reserve(static_cast<size_t>(num_vars - 1));

  for (const auto& edge : edges)
  {
    if (static_cast<int>(tree_edges.size()) >= num_vars - 1)
    {
      break;
    }
    if (uf.unite(edge.var_a, edge.var_b))
    {
      tree_edges.push_back(edge);
    }
  }

  return tree_edges;
}

/**
 * @brief Compute the matrix square root L such that L * L^T = X for a PSD matrix X.
 */
vesta_core::MatrixXd matrixSqrt(const vesta_core::MatrixXd& X)
{
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

  return vecs * sqrt_d;
}

/**
 * @brief Project a symmetric matrix onto the PSD cone by clamping negative eigenvalues to zero.
 */
vesta_core::MatrixXd projectPSD(const vesta_core::MatrixXd& X)
{
  Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig(X);
  const auto& vals = eig.eigenvalues();
  const auto& vecs = eig.eigenvectors();

  vesta_core::VectorXd clamped_vals = vals.cwiseMax(0.0);
  return vecs * clamped_vals.asDiagonal() * vecs.transpose();
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
        // Eigendecomposition of H_oo for rank handling and covariance computation
        Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig_hoo(H_oo);
        if (eig_hoo.info() != Eigen::Success)
        {
          LOG(ERROR) << "NfrMarginalizer: eigendecomposition of H_oo failed. "
                     << "Falling back to block-diagonal priors.";
          emitBlockDiagonalPriors(other_indices);
        }
        else
        {
          const auto& hoo_eigenvalues = eig_hoo.eigenvalues();
          const auto& hoo_eigenvectors = eig_hoo.eigenvectors();
          double hoo_threshold = computeEigenvalueThreshold(hoo_eigenvalues.maxCoeff());

          int rank = 0;
          for (int k = 0; k < hoo_eigenvalues.size(); ++k)
          {
            if (hoo_eigenvalues(k) > hoo_threshold)
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
            bool rank_deficient = (rank < total_other_dim);

            // Compute covariance Sigma_o = H_oo^{-1} (with rank handling)
            vesta_core::MatrixXd sigma;
            if (rank_deficient)
            {
              sigma = vesta_core::MatrixXd::Zero(total_other_dim, total_other_dim);
              for (int k = 0; k < hoo_eigenvalues.size(); ++k)
              {
                if (hoo_eigenvalues(k) > hoo_threshold)
                {
                  sigma += (1.0 / hoo_eigenvalues(k)) * hoo_eigenvectors.col(k) * hoo_eigenvectors.col(k).transpose();
                }
              }
            }
            else
            {
              vesta_core::MatrixXd H_reg =
                  H_oo + TIKHONOV_EPS * vesta_core::MatrixXd::Identity(total_other_dim, total_other_dim);
              Eigen::LLT<vesta_core::MatrixXd> llt_sigma(H_reg);
              if (llt_sigma.info() != Eigen::Success)
              {
                sigma = vesta_core::MatrixXd::Zero(total_other_dim, total_other_dim);
                for (int k = 0; k < hoo_eigenvalues.size(); ++k)
                {
                  double val = hoo_eigenvalues(k) + TIKHONOV_EPS;
                  if (val > 0.0)
                  {
                    sigma += (1.0 / val) * hoo_eigenvectors.col(k) * hoo_eigenvectors.col(k).transpose();
                  }
                }
              }
              else
              {
                sigma = llt_sigma.solve(vesta_core::MatrixXd::Identity(total_other_dim, total_other_dim));
              }
            }

            // ---- NFR Algorithm ----
            // 1. Select Chow-Liu tree edges (same as CLT)
            // 2. Recover pairwise relative factor information matrices via closed-form
            // 3. Compute the reconstructed information from edges: H_edges = sum A_e^T X_e A_e
            // 4. Compute residual diagonal information for absolute priors: H_res_i = H_oo_ii - H_edges_ii
            // 5. Emit both pairwise and absolute prior factors

            // Step 1: Compute mutual information for all variable pairs
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

            // Step 2: Select Chow-Liu tree
            std::vector<SparsityEdge> selected_edges = selectChowLiuTree(all_edges, num_remaining);

            // Step 3: Recover pairwise edge information matrices (CLT closed-form)
            // For each edge, the Jacobian A_e = [-I_a, I_b], and X_e = (A_e Sigma A_e^T)^{-1}
            int num_selected = static_cast<int>(selected_edges.size());
            std::vector<vesta_core::MatrixXd> x_blocks(num_selected);

            // Accumulate reconstructed information matrix from edges
            vesta_core::MatrixXd H_edges = vesta_core::MatrixXd::Zero(total_other_dim, total_other_dim);

            for (int e = 0; e < num_selected; ++e)
            {
              const auto& edge = selected_edges[e];
              int off_a = other_offsets.at(other_indices[edge.var_a]);
              int dim_a = var_tangent_sizes.at(other_indices[edge.var_a]);
              int off_b = other_offsets.at(other_indices[edge.var_b]);
              int dim_b = var_tangent_sizes.at(other_indices[edge.var_b]);
              int meas_dim = std::min(dim_a, dim_b);

              // Build A_e row: [-I, I] acting on the full state vector
              vesta_core::MatrixXd A_e = vesta_core::MatrixXd::Zero(meas_dim, total_other_dim);
              for (int r = 0; r < meas_dim; ++r)
              {
                if (r < dim_a)
                {
                  A_e(r, off_a + r) = -1.0;
                }
                if (r < dim_b)
                {
                  A_e(r, off_b + r) = 1.0;
                }
              }

              // Compute X_e = (A_e Sigma A_e^T)^{-1}
              vesta_core::MatrixXd asa = A_e * sigma * A_e.transpose();
              asa = (asa + asa.transpose()) * 0.5;

              Eigen::LLT<vesta_core::MatrixXd> llt_asa(asa);
              if (llt_asa.info() == Eigen::Success)
              {
                x_blocks[e] = llt_asa.solve(vesta_core::MatrixXd::Identity(meas_dim, meas_dim));
              }
              else
              {
                Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig_asa(asa);
                x_blocks[e] = vesta_core::MatrixXd::Zero(meas_dim, meas_dim);
                if (eig_asa.info() == Eigen::Success)
                {
                  double thresh = computeEigenvalueThreshold(eig_asa.eigenvalues().maxCoeff());
                  for (int k = 0; k < eig_asa.eigenvalues().size(); ++k)
                  {
                    if (eig_asa.eigenvalues()(k) > thresh)
                    {
                      x_blocks[e] += (1.0 / eig_asa.eigenvalues()(k)) * eig_asa.eigenvectors().col(k) *
                                     eig_asa.eigenvectors().col(k).transpose();
                    }
                  }
                }
              }

              x_blocks[e] = projectPSD(x_blocks[e]);

              // Accumulate into H_edges: H_edges += A_e^T X_e A_e
              H_edges += A_e.transpose() * x_blocks[e] * A_e;
            }

            // Step 4: Emit pairwise relative factors
            for (int e = 0; e < num_selected; ++e)
            {
              const auto& edge = selected_edges[e];
              unsigned int idx_a = other_indices[edge.var_a];
              unsigned int idx_b = other_indices[edge.var_b];
              int dim_a = var_tangent_sizes.at(idx_a);
              int dim_b = var_tangent_sizes.at(idx_b);
              int meas_dim = std::min(dim_a, dim_b);

              vesta_core::MatrixXd L = matrixSqrt(x_blocks[e]);
              vesta_core::MatrixXd Lt = L.transpose();

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

            // Step 5: Emit per-variable absolute prior factors from residual information
            // The residual captures information that the tree edges cannot represent
            // (diagonal information not fully covered by the off-diagonal edges).
            for (int v = 0; v < num_remaining; ++v)
            {
              unsigned int idx = other_indices[v];
              int off = other_offsets.at(idx);
              int dim = var_tangent_sizes.at(idx);

              // Residual information for this variable: H_oo_ii - H_edges_ii
              vesta_core::MatrixXd H_res = H_oo.block(off, off, dim, dim) - H_edges.block(off, off, dim, dim);
              H_res = (H_res + H_res.transpose()) * 0.5;

              // Project onto PSD cone (the residual can have negative eigenvalues
              // if the tree edges over-estimate diagonal information)
              H_res = projectPSD(H_res);

              // Residual information vector: eta_res_i = eta_o_i - (H_edges * x_bar contribution)
              // At the linearization point, x - x_bar = 0, so the information vector
              // from edges is zero. Thus eta_res = eta_o for the prior.
              vesta_core::VectorXd eta_res = eta_o.segment(off, dim);

              // But we also need to subtract the portion of eta accounted for by the edges.
              // Since the edge factors have b=0 (at linearization point), they contribute
              // nothing to eta. So the full eta goes to the priors.

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
