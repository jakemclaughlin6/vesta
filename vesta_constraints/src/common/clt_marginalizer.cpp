#include <vesta_constraints/common/clt_marginalizer.h>
#include <vesta_constraints/common/marginalize_variables.h>
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

// L-BFGS memory size
constexpr int LBFGS_MEMORY = 10;

// Line search parameters (Armijo backtracking)
constexpr double ARMIJO_C = 1e-4;
constexpr double ARMIJO_RHO = 0.5;
constexpr int MAX_LINE_SEARCH_ITERS = 20;

/// Represents a selected edge in the sparsity pattern
struct SparsityEdge
{
  int var_a;  ///< Index into other_indices
  int var_b;  ///< Index into other_indices
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
      parent_[x] = parent_[parent_[x]];  // path compression
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
 *
 * MI(xi, xj) = 0.5 * log(det(Sigma_ii) * det(Sigma_jj) / det(Sigma_ij_block))
 * where Sigma_ij_block is the 2x2 block submatrix [[Sigma_ii, Sigma_ij], [Sigma_ji, Sigma_jj]]
 */
double computeMutualInformation(const vesta_core::MatrixXd& sigma, int off_i, int dim_i, int off_j, int dim_j)
{
  vesta_core::MatrixXd sigma_ii = sigma.block(off_i, off_i, dim_i, dim_i);
  vesta_core::MatrixXd sigma_jj = sigma.block(off_j, off_j, dim_j, dim_j);

  // Build the joint block
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

/**
 * @brief Select edges for the Chow-Liu maximum spanning tree using Kruskal's algorithm.
 *
 * @param edges All candidate edges with MI weights
 * @param num_vars Number of variables (nodes)
 * @return Vector of selected tree edges (N-1 edges for N nodes)
 */
std::vector<SparsityEdge> selectChowLiuTree(std::vector<SparsityEdge>& edges, int num_vars)
{
  // Sort edges by MI descending
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
 * @brief Select edges for the Chow-Liu tree plus additional chords.
 *
 * @param edges All candidate edges with MI weights (will be sorted)
 * @param num_vars Number of variables (nodes)
 * @param chord_ratio Ratio of additional chords to tree edges
 * @return Vector of selected edges (tree + chords)
 */
std::vector<SparsityEdge> selectSubgraph(std::vector<SparsityEdge>& edges, int num_vars, double chord_ratio)
{
  // Sort edges by MI descending
  std::sort(edges.begin(), edges.end(),
            [](const SparsityEdge& a, const SparsityEdge& b) { return a.mutual_information > b.mutual_information; });

  UnionFind uf(num_vars);
  std::vector<SparsityEdge> selected_edges;
  std::vector<SparsityEdge> chord_candidates;

  // First pass: build tree
  for (const auto& edge : edges)
  {
    if (uf.unite(edge.var_a, edge.var_b))
    {
      selected_edges.push_back(edge);
    }
    else
    {
      chord_candidates.push_back(edge);
    }
  }

  // Second pass: add chords (already sorted by MI descending via chord_candidates order)
  int num_tree_edges = static_cast<int>(selected_edges.size());
  int num_chords = std::max(1, static_cast<int>(std::round(chord_ratio * num_tree_edges)));
  num_chords = std::min(num_chords, static_cast<int>(chord_candidates.size()));

  for (int c = 0; c < num_chords; ++c)
  {
    selected_edges.push_back(chord_candidates[c]);
  }

  return selected_edges;
}

/**
 * @brief Compute the matrix square root L such that L * L^T = X for a PSD matrix X.
 * Uses eigendecomposition: X = V D V^T => L = V sqrt(D).
 * Returns L^T (so that L_transpose * L_transpose^T = ... no, we want L L^T = X).
 * Actually returns L such that L L^T = X, where L = V * sqrt(D).
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

  return vecs * sqrt_d;  // L such that L * L^T = X
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

/**
 * @brief Solve the nonlinear graph sparsification using Projected Quasi-Newton (PQN).
 *
 * Minimizes DKL(p||q) = <A^T X A, Sigma> - log det(A^T X A)
 * where X = blkdiag(X_1, ..., X_m) and each X_i must be PSD.
 *
 * @param A_rows Jacobian block rows for each edge (each is measurement_dim x total_var_dim)
 * @param block_sizes Size of each measurement block
 * @param sigma Covariance matrix
 * @param x_init Initial X blocks (from closed-form tree solution or similar)
 * @param max_iterations Maximum PQN iterations
 * @param gradient_tolerance Convergence tolerance
 * @return Optimized X blocks
 */
std::vector<vesta_core::MatrixXd> solvePQN(const std::vector<vesta_core::MatrixXd>& A_rows,
                                           const std::vector<int>& block_sizes, const vesta_core::MatrixXd& sigma,
                                           const std::vector<vesta_core::MatrixXd>& x_init, int max_iterations,
                                           double gradient_tolerance)
{
  int num_edges = static_cast<int>(A_rows.size());
  int total_var_dim = sigma.rows();

  // Working copy of X blocks
  std::vector<vesta_core::MatrixXd> x_blocks = x_init;

  // Compute total measurement dimension
  int total_meas_dim = 0;
  std::vector<int> block_offsets(num_edges);
  for (int e = 0; e < num_edges; ++e)
  {
    block_offsets[e] = total_meas_dim;
    total_meas_dim += block_sizes[e];
  }

  // Build the full A matrix (total_meas_dim x total_var_dim)
  vesta_core::MatrixXd A_full(total_meas_dim, total_var_dim);
  for (int e = 0; e < num_edges; ++e)
  {
    A_full.block(block_offsets[e], 0, block_sizes[e], total_var_dim) = A_rows[e];
  }

  // Helper: build X_diag from blocks
  auto buildXDiag = [&](const std::vector<vesta_core::MatrixXd>& blocks) -> vesta_core::MatrixXd {
    vesta_core::MatrixXd X_diag = vesta_core::MatrixXd::Zero(total_meas_dim, total_meas_dim);
    for (int e = 0; e < num_edges; ++e)
    {
      X_diag.block(block_offsets[e], block_offsets[e], block_sizes[e], block_sizes[e]) = blocks[e];
    }
    return X_diag;
  };

  // Helper: compute objective = tr(A^T X A * Sigma) - log det(A^T X A)
  auto computeObjective = [&](const std::vector<vesta_core::MatrixXd>& blocks) -> double {
    vesta_core::MatrixXd X_diag = buildXDiag(blocks);
    vesta_core::MatrixXd Q = A_full.transpose() * X_diag * A_full;

    // Make Q symmetric
    Q = (Q + Q.transpose()) * 0.5;

    Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig(Q);
    if (eig.info() != Eigen::Success)
    {
      return std::numeric_limits<double>::infinity();
    }

    double log_det = 0.0;
    double trace_val = (Q * sigma).trace();

    for (int k = 0; k < eig.eigenvalues().size(); ++k)
    {
      if (eig.eigenvalues()(k) <= 0.0)
      {
        return std::numeric_limits<double>::infinity();
      }
      log_det += std::log(eig.eigenvalues()(k));
    }

    return trace_val - log_det;
  };

  // Helper: compute gradient blocks
  // grad_i = {A [Sigma - Q^{-1}] A^T}_i  where Q = A^T X A
  auto computeGradient =
      [&](const std::vector<vesta_core::MatrixXd>& blocks) -> std::vector<vesta_core::MatrixXd> {
    vesta_core::MatrixXd X_diag = buildXDiag(blocks);
    vesta_core::MatrixXd Q = A_full.transpose() * X_diag * A_full;
    Q = (Q + Q.transpose()) * 0.5;

    // Q^{-1} via Cholesky
    Eigen::LLT<vesta_core::MatrixXd> llt(Q);
    if (llt.info() != Eigen::Success)
    {
      // Fallback: eigendecomposition pseudoinverse
      Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig(Q);
      vesta_core::MatrixXd Q_inv = vesta_core::MatrixXd::Zero(total_var_dim, total_var_dim);
      double thresh = computeEigenvalueThreshold(eig.eigenvalues().maxCoeff());
      for (int k = 0; k < eig.eigenvalues().size(); ++k)
      {
        if (eig.eigenvalues()(k) > thresh)
        {
          Q_inv += (1.0 / eig.eigenvalues()(k)) * eig.eigenvectors().col(k) * eig.eigenvectors().col(k).transpose();
        }
      }
      vesta_core::MatrixXd diff = sigma - Q_inv;
      vesta_core::MatrixXd G_full = A_full * diff * A_full.transpose();

      std::vector<vesta_core::MatrixXd> grads(num_edges);
      for (int e = 0; e < num_edges; ++e)
      {
        grads[e] = G_full.block(block_offsets[e], block_offsets[e], block_sizes[e], block_sizes[e]);
        grads[e] = (grads[e] + grads[e].transpose()) * 0.5;
      }
      return grads;
    }

    vesta_core::MatrixXd Q_inv = llt.solve(vesta_core::MatrixXd::Identity(total_var_dim, total_var_dim));
    vesta_core::MatrixXd diff = sigma - Q_inv;
    vesta_core::MatrixXd G_full = A_full * diff * A_full.transpose();

    std::vector<vesta_core::MatrixXd> grads(num_edges);
    for (int e = 0; e < num_edges; ++e)
    {
      grads[e] = G_full.block(block_offsets[e], block_offsets[e], block_sizes[e], block_sizes[e]);
      // Symmetrize
      grads[e] = (grads[e] + grads[e].transpose()) * 0.5;
    }
    return grads;
  };

  // Helper: vectorize/devectorize gradient blocks
  auto vectorize = [&](const std::vector<vesta_core::MatrixXd>& blocks) -> vesta_core::VectorXd {
    int total_params = 0;
    for (int e = 0; e < num_edges; ++e)
    {
      total_params += block_sizes[e] * block_sizes[e];
    }
    vesta_core::VectorXd vec(total_params);
    int offset = 0;
    for (int e = 0; e < num_edges; ++e)
    {
      int sz = block_sizes[e];
      // Use column-major vectorization for compatibility with Eigen's Map
      for (int c = 0; c < sz; ++c)
      {
        for (int r = 0; r < sz; ++r)
        {
          vec(offset++) = blocks[e](r, c);
        }
      }
    }
    return vec;
  };

  auto devectorize = [&](const vesta_core::VectorXd& vec) -> std::vector<vesta_core::MatrixXd> {
    std::vector<vesta_core::MatrixXd> blocks(num_edges);
    int offset = 0;
    for (int e = 0; e < num_edges; ++e)
    {
      int sz = block_sizes[e];
      blocks[e].resize(sz, sz);
      for (int c = 0; c < sz; ++c)
      {
        for (int r = 0; r < sz; ++r)
        {
          blocks[e](r, c) = vec(offset++);
        }
      }
    }
    return blocks;
  };

  // Helper: project all blocks onto PSD cone
  auto projectAllPSD = [&](std::vector<vesta_core::MatrixXd>& blocks) {
    for (int e = 0; e < num_edges; ++e)
    {
      blocks[e] = projectPSD(blocks[e]);
    }
  };

  // L-BFGS memory
  std::vector<vesta_core::VectorXd> sk_history;
  std::vector<vesta_core::VectorXd> y_history;
  std::vector<double> rho_history;
  sk_history.reserve(LBFGS_MEMORY);
  y_history.reserve(LBFGS_MEMORY);
  rho_history.reserve(LBFGS_MEMORY);

  vesta_core::VectorXd prev_x_vec = vectorize(x_blocks);
  vesta_core::VectorXd prev_grad_vec = vectorize(computeGradient(x_blocks));

  double prev_obj = computeObjective(x_blocks);

  for (int iter = 0; iter < max_iterations; ++iter)
  {
    // Check convergence
    double grad_norm = prev_grad_vec.norm();
    if (grad_norm < gradient_tolerance)
    {
      break;
    }

    // Compute L-BFGS direction
    vesta_core::VectorXd q = prev_grad_vec;
    int m = static_cast<int>(sk_history.size());
    std::vector<double> alpha(m);

    for (int i = m - 1; i >= 0; --i)
    {
      alpha[i] = rho_history[i] * sk_history[i].dot(q);
      q -= alpha[i] * y_history[i];
    }

    // Initial Hessian approximation: H_0 = gamma * I
    double gamma = 1.0;
    if (m > 0)
    {
      double sy = sk_history.back().dot(y_history.back());
      double yy = y_history.back().dot(y_history.back());
      if (yy > 0.0)
      {
        gamma = sy / yy;
      }
    }
    vesta_core::VectorXd r = gamma * q;

    for (int i = 0; i < m; ++i)
    {
      double beta = rho_history[i] * y_history[i].dot(r);
      r += (alpha[i] - beta) * sk_history[i];
    }

    // Search direction is -H * g
    vesta_core::VectorXd direction = -r;

    // Armijo backtracking line search with PSD projection
    double step_size = 1.0;
    double dir_deriv = prev_grad_vec.dot(direction);

    // If not a descent direction, use negative gradient
    if (dir_deriv >= 0.0)
    {
      direction = -prev_grad_vec;
      dir_deriv = -prev_grad_vec.squaredNorm();
      // Clear L-BFGS memory on direction reset
      sk_history.clear();
      y_history.clear();
      rho_history.clear();
    }

    bool line_search_success = false;
    std::vector<vesta_core::MatrixXd> new_blocks;
    double new_obj = std::numeric_limits<double>::infinity();

    for (int ls = 0; ls < MAX_LINE_SEARCH_ITERS; ++ls)
    {
      vesta_core::VectorXd candidate_vec = prev_x_vec + step_size * direction;
      new_blocks = devectorize(candidate_vec);
      projectAllPSD(new_blocks);

      new_obj = computeObjective(new_blocks);
      if (std::isfinite(new_obj) && new_obj <= prev_obj + ARMIJO_C * step_size * dir_deriv)
      {
        line_search_success = true;
        break;
      }
      step_size *= ARMIJO_RHO;
    }

    if (!line_search_success)
    {
      // Line search failed; try a small gradient step
      vesta_core::VectorXd candidate_vec = prev_x_vec - 1e-6 * prev_grad_vec;
      new_blocks = devectorize(candidate_vec);
      projectAllPSD(new_blocks);
      new_obj = computeObjective(new_blocks);
      if (!std::isfinite(new_obj) || new_obj >= prev_obj)
      {
        LOG(WARNING) << "CltMarginalizer: PQN line search failed at iteration " << iter
                     << ". Stopping early.";
        break;
      }
    }

    // Update L-BFGS memory
    vesta_core::VectorXd new_x_vec = vectorize(new_blocks);
    vesta_core::VectorXd new_grad_vec = vectorize(computeGradient(new_blocks));

    vesta_core::VectorXd s = new_x_vec - prev_x_vec;
    vesta_core::VectorXd y = new_grad_vec - prev_grad_vec;
    double sy = s.dot(y);

    if (sy > 1e-16)
    {
      if (static_cast<int>(sk_history.size()) >= LBFGS_MEMORY)
      {
        sk_history.erase(sk_history.begin());
        y_history.erase(y_history.begin());
        rho_history.erase(rho_history.begin());
      }
      sk_history.push_back(s);
      y_history.push_back(y);
      rho_history.push_back(1.0 / sy);
    }

    x_blocks = new_blocks;
    prev_x_vec = new_x_vec;
    prev_grad_vec = new_grad_vec;
    prev_obj = new_obj;
  }

  return x_blocks;
}

}  // namespace

CltMarginalizer::CltMarginalizer(SparsityMode mode, bool use_fej, double chord_ratio,
                                                       int max_iterations, double gradient_tolerance)
  : mode_(mode), use_fej_(use_fej), chord_ratio_(chord_ratio), max_iterations_(max_iterations)
  , gradient_tolerance_(gradient_tolerance)
{
}

vesta_core::Transaction CltMarginalizer::marginalize(
    const std::string& source, const std::vector<vesta_core::UUID>& marginalized_variables,
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
    auto schur_result =
        detail::computeSchurComplement(schur_terms, num_non_stamped, graph, lin_result.variable_order);

    if (schur_result)
    {
      auto& H_oo = schur_result->H_oo;
      auto& eta_o = schur_result->eta_o;
      const auto& other_indices = schur_result->other_indices;
      const auto& other_offsets = schur_result->other_offsets;
      const auto& var_tangent_sizes = schur_result->var_tangent_sizes;
      int total_other_dim = schur_result->total_other_dim;

      // ---- Nonlinear graph sparsification begins here ----

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

      // Fallback: if 0 or 1 remaining variables, use block-diagonal priors (no pairwise edges possible)
      if (num_remaining <= 1)
      {
        emitBlockDiagonalPriors(other_indices);
      }
      else
      {
        // Step 4: Handle rank deficiency via eigendecomposition
        Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig_hoo(H_oo);
        if (eig_hoo.info() != Eigen::Success)
        {
          LOG(ERROR) << "CltMarginalizer: eigendecomposition of H_oo failed. "
                     << "Falling back to block-diagonal priors.";
          emitBlockDiagonalPriors(other_indices);
        }
        else
        {
          const auto& hoo_eigenvalues = eig_hoo.eigenvalues();
          const auto& hoo_eigenvectors = eig_hoo.eigenvectors();
          double hoo_threshold = computeEigenvalueThreshold(hoo_eigenvalues.maxCoeff());

          // Count positive eigenvalues to determine rank
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
            LOG(WARNING) << "CltMarginalizer: H_oo has zero rank. No marginal constraints generated.";
          }
          else
          {
            // Build projection matrix Pi and reduced-space covariance
            // Pi = eigenvectors of nonzero eigenvalues (total_other_dim x rank)
            // In reduced space: H_reduced = Lambda_+ (diagonal of positive eigenvalues)
            // Sigma_reduced = Lambda_+^{-1}

            bool rank_deficient = (rank < total_other_dim);
            vesta_core::MatrixXd Pi;  // total_other_dim x rank
            vesta_core::MatrixXd sigma;  // covariance in working space

            if (rank_deficient)
            {
              Pi.resize(total_other_dim, rank);
              sigma.resize(rank, rank);
              sigma.setZero();
              int col = 0;
              for (int k = 0; k < hoo_eigenvalues.size(); ++k)
              {
                if (hoo_eigenvalues(k) > hoo_threshold)
                {
                  Pi.col(col) = hoo_eigenvectors.col(k);
                  sigma(col, col) = 1.0 / hoo_eigenvalues(k);
                  ++col;
                }
              }
            }
            else
            {
              // Full rank: compute covariance with Tikhonov regularization
              vesta_core::MatrixXd H_reg = H_oo + TIKHONOV_EPS * vesta_core::MatrixXd::Identity(
                                                                      total_other_dim, total_other_dim);
              Eigen::LLT<vesta_core::MatrixXd> llt_sigma(H_reg);
              if (llt_sigma.info() != Eigen::Success)
              {
                // Fallback to eigendecomp-based pseudoinverse
                sigma.resize(total_other_dim, total_other_dim);
                sigma.setZero();
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

            int working_dim = rank_deficient ? rank : total_other_dim;

            // For MI computation, we need sigma in the original variable space.
            // In the rank-deficient case, the covariance on original variables is Pi * sigma * Pi^T.
            vesta_core::MatrixXd sigma_orig;
            if (rank_deficient)
            {
              sigma_orig = Pi * sigma * Pi.transpose();
            }
            else
            {
              sigma_orig = sigma;
            }

            // Step 6: Compute mutual information for all variable pairs
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

                double mi = computeMutualInformation(sigma_orig, off_a, dim_a, off_b, dim_b);
                // Edges with zero MI are excluded. This may produce a disconnected CLT if some variable pairs are
                // completely uncorrelated. Isolated nodes will fall back to independent diagonal priors.
                if (mi > 0.0)
                {
                  all_edges.push_back({ a, b, mi });
                }
              }
            }

            // Step 7: Select sparsity pattern
            std::vector<SparsityEdge> selected_edges;
            if (mode_ == SparsityMode::Tree)
            {
              selected_edges = selectChowLiuTree(all_edges, num_remaining);
            }
            else
            {
              selected_edges = selectSubgraph(all_edges, num_remaining, chord_ratio_);
            }

            // Track which variables are connected by at least one selected edge
            std::unordered_set<int> connected_vars;
            for (const auto& edge : selected_edges)
            {
              connected_vars.insert(edge.var_a);
              connected_vars.insert(edge.var_b);
            }

            if (selected_edges.empty())
            {
              // No edges selected: fall back to block-diagonal priors for all variables
              emitBlockDiagonalPriors(other_indices);
            }
            else
            {
              // Step 8: Build Jacobian A for selected virtual measurements
              // Each edge (a, b) contributes a block row [-I, I] in the variable tangent space.
              // If rank-deficient, we project into the reduced space via Pi.

              int num_selected = static_cast<int>(selected_edges.size());

              // Build A rows and compute block sizes
              std::vector<vesta_core::MatrixXd> A_rows(num_selected);
              std::vector<int> block_sizes(num_selected);

              for (int e = 0; e < num_selected; ++e)
              {
                const auto& edge = selected_edges[e];
                int idx_a = edge.var_a;
                int idx_b = edge.var_b;
                int off_a = other_offsets.at(other_indices[idx_a]);
                int dim_a = var_tangent_sizes.at(other_indices[idx_a]);
                int off_b = other_offsets.at(other_indices[idx_b]);
                int dim_b = var_tangent_sizes.at(other_indices[idx_b]);

                int meas_dim = std::min(dim_a, dim_b);
                if (dim_a != dim_b)
                {
                  LOG(WARNING) << "CltMarginalizer: pairwise edge between variables of different tangent "
                                  "dimensions ("
                               << dim_a << " vs " << dim_b << "). Using min dimension " << meas_dim << ".";
                }
                block_sizes[e] = meas_dim;

                A_rows[e] = vesta_core::MatrixXd::Zero(meas_dim, rank_deficient ? working_dim : total_other_dim);

                if (rank_deficient)
                {
                  // A_row in reduced space: -Pi_a^T + Pi_b^T per measurement row
                  for (int r = 0; r < meas_dim; ++r)
                  {
                    A_rows[e].row(r) = -Pi.row(off_a + r) + Pi.row(off_b + r);
                  }
                }
                else
                {
                  // A_row = [-I_a, 0, ..., I_b, 0, ...]
                  for (int r = 0; r < meas_dim; ++r)
                  {
                    A_rows[e](r, off_a + r) = -1.0;
                    A_rows[e](r, off_b + r) = 1.0;
                  }
                }
              }

              // Step 9: Solve for X
              std::vector<vesta_core::MatrixXd> x_blocks(num_selected);

              // Closed-form solution: X_e = ({A Sigma A^T}_e)^{-1}
              for (int e = 0; e < num_selected; ++e)
              {
                vesta_core::MatrixXd asa = A_rows[e] * sigma * A_rows[e].transpose();
                asa = (asa + asa.transpose()) * 0.5;  // symmetrize

                Eigen::LLT<vesta_core::MatrixXd> llt_asa(asa);
                if (llt_asa.info() == Eigen::Success)
                {
                  x_blocks[e] = llt_asa.solve(
                      vesta_core::MatrixXd::Identity(block_sizes[e], block_sizes[e]));
                }
                else
                {
                  // Fallback: pseudoinverse via eigendecomposition
                  Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig_asa(asa);
                  x_blocks[e] = vesta_core::MatrixXd::Zero(block_sizes[e], block_sizes[e]);
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

                // Ensure PSD
                x_blocks[e] = projectPSD(x_blocks[e]);
              }

              // For SUBGRAPH mode, refine via PQN if there are chord edges beyond the tree
              if (mode_ == SparsityMode::Subgraph && num_selected > num_remaining - 1)
              {
                x_blocks = solvePQN(A_rows, block_sizes, sigma, x_blocks, max_iterations_, gradient_tolerance_);
              }

              // Step 10: Construct MarginalConstraints from the solved X blocks
              for (int e = 0; e < num_selected; ++e)
              {
                const auto& edge = selected_edges[e];
                unsigned int idx_a = other_indices[edge.var_a];
                unsigned int idx_b = other_indices[edge.var_b];
                int meas_dim = block_sizes[e];

                // Compute L such that L L^T = X_e, then use L^T as sqrt-info
                vesta_core::MatrixXd L = matrixSqrt(x_blocks[e]);
                vesta_core::MatrixXd Lt = L.transpose();  // meas_dim x meas_dim

                // Build the LinearTerm: cost = ||(-L^T)(x_i - x_i_bar) + L^T(x_j - x_j_bar)||^2
                //                            = ||L^T((x_j - x_j_bar) - (x_i - x_i_bar))||^2
                //                            = (delta_j - delta_i)^T X_e (delta_j - delta_i)
                detail::LinearTerm pairwise_term;
                pairwise_term.variables = { idx_a, idx_b };

                int dim_a = var_tangent_sizes.at(idx_a);
                int dim_b = var_tangent_sizes.at(idx_b);

                // Build A blocks, padding with zeros if tangent dims differ from meas_dim
                vesta_core::MatrixXd A_a = vesta_core::MatrixXd::Zero(meas_dim, dim_a);
                vesta_core::MatrixXd A_b = vesta_core::MatrixXd::Zero(meas_dim, dim_b);
                int copy_dim = std::min({ meas_dim, dim_a, dim_b });
                A_a.block(0, 0, meas_dim, copy_dim) = -Lt.block(0, 0, meas_dim, copy_dim);
                A_b.block(0, 0, meas_dim, copy_dim) = Lt.block(0, 0, meas_dim, copy_dim);

                pairwise_term.A = { A_a, A_b };
                pairwise_term.b = vesta_core::VectorXd::Zero(meas_dim);

                // Only emit to transaction if both variables are remaining (not marginalized stamped)
                if (idx_a >= num_marginalized && idx_b >= num_marginalized)
                {
                  auto mc = detail::createMarginalConstraint(source, pairwise_term, graph,
                                                             lin_result.variable_order);
                  lin_result.transaction.addConstraint(std::move(mc));
                }
                else
                {
                  // One or both variables are stamped marginalized; place in linear_terms for QR phase
                  auto min_var = std::min(idx_a, idx_b);
                  lin_result.linear_terms[min_var].push_back(std::move(pairwise_term));
                }
              }

              // Add independent per-variable priors for isolated nodes (not connected by any edge)
              for (int v = 0; v < num_remaining; ++v)
              {
                if (connected_vars.find(v) == connected_vars.end())
                {
                  unsigned int idx = other_indices[v];
                  int off = other_offsets.at(idx);
                  int dim = var_tangent_sizes.at(idx);
                  vesta_core::MatrixXd H_ii = H_oo.block(off, off, dim, dim);
                  vesta_core::VectorXd eta_i = eta_o.segment(off, dim);

                  auto single_term = detail::createSingleVariableTerm(idx, H_ii, eta_i);
                  if (!single_term.variables.empty())
                  {
                    if (idx >= num_marginalized)
                    {
                      auto mc = detail::createMarginalConstraint(source, single_term, graph,
                                                                 lin_result.variable_order);
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
    }
  }

  // 7. QR phase for stamped variables
  detail::marginalizeStampedVariables(lin_result.linear_terms, num_non_stamped, num_marginalized);

  // 8. Emit remaining constraints
  detail::emitRemainingConstraints(source, lin_result.linear_terms, num_marginalized, graph,
                                   lin_result.variable_order, lin_result.transaction);

  return std::move(lin_result.transaction);
}

}  // namespace vesta_constraints
