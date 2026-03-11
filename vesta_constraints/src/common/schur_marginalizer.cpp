#include <vesta_constraints/common/marginal_constraint.h>
#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/qr_marginalizer.h>
#include <vesta_constraints/common/schur_marginalizer.h>
#include <vesta_constraints/common/uuid_ordering.h>
#include <vesta_core/eigen.h>
#include <vesta_core/graph.h>
#include <vesta_core/uuid.h>
#include <vesta_core/variable.h>
#include <vesta_variables/common/stamped.h>

#include <glog/logging.h>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vesta_constraints
{

SchurMarginalizer::SchurMarginalizer(bool use_fej) : use_fej_(use_fej)
{
}

vesta_core::Transaction SchurMarginalizer::marginalize(const std::string& source,
                                                       const std::vector<vesta_core::UUID>& marginalized_variables,
                                                       const vesta_core::Graph& graph)
{
  // Classify marginalized variables as non-stamped vs stamped
  std::vector<vesta_core::UUID> non_stamped_vars;
  std::vector<vesta_core::UUID> stamped_vars;
  for (const auto& uuid : marginalized_variables)
  {
    const auto& variable = graph.getVariable(uuid);
    if (dynamic_cast<const vesta_variables::Stamped*>(&variable) != nullptr)
    {
      stamped_vars.push_back(uuid);
    }
    else
    {
      non_stamped_vars.push_back(uuid);
    }
  }

  // If no non-stamped variables, delegate entirely to QR
  if (non_stamped_vars.empty())
  {
    return QRMarginalizer(use_fej_).marginalize(source, marginalized_variables, graph);
  }

  const size_t num_non_stamped = non_stamped_vars.size();
  const size_t num_stamped = stamped_vars.size();
  const size_t num_marginalized = num_non_stamped + num_stamped;

  // Build elimination order: non-stamped first (0..K-1), then stamped (K..M-1)
  auto variable_order = UuidOrdering();
  for (const auto& uuid : non_stamped_vars)
  {
    variable_order.push_back(uuid);
  }
  for (const auto& uuid : stamped_vars)
  {
    variable_order.push_back(uuid);
  }

  vesta_core::Transaction transaction;

  // Mark all marginalized variables for removal
  for (const auto& uuid : marginalized_variables)
  {
    transaction.removeVariable(uuid);
  }

  // Linearize all involved constraints and bucket by lowest-ordered variable
  auto used_constraints = std::unordered_set<vesta_core::UUID, vesta_core::uuid::hash>();
  std::vector<std::vector<detail::LinearTerm>> linear_terms(variable_order.size());

  for (size_t i = 0ul; i < num_marginalized; ++i)
  {
    const auto constraints = graph.getConnectedConstraints(variable_order[i]);
    for (const auto& constraint : constraints)
    {
      if (used_constraints.find(constraint.uuid()) == used_constraints.end())
      {
        used_constraints.insert(constraint.uuid());
        for (const auto& variable_uuid : constraint.variables())
        {
          variable_order.push_back(variable_uuid);
        }
        auto lt = detail::linearize(constraint, graph, variable_order, use_fej_);
        // Place in bucket of lowest-ordered variable
        auto min_var = *std::min_element(lt.variables.begin(), lt.variables.end());
        linear_terms[min_var].push_back(std::move(lt));
        transaction.removeConstraint(constraint.uuid());
      }
    }
  }

  // Expand to include connected variables
  linear_terms.resize(variable_order.size());

  // Collect all LinearTerms from non-stamped variable buckets (indices 0..K-1)
  // These terms involve at least one non-stamped variable
  std::vector<detail::LinearTerm> schur_terms;
  for (size_t i = 0; i < num_non_stamped; ++i)
  {
    for (auto& lt : linear_terms[i])
    {
      schur_terms.push_back(std::move(lt));
    }
    linear_terms[i].clear();
  }

  if (!schur_terms.empty())
  {
    // Determine tangent sizes for all variables appearing in Schur terms
    std::unordered_map<unsigned int, int> var_tangent_sizes;
    for (const auto& lt : schur_terms)
    {
      for (size_t j = 0; j < lt.variables.size(); ++j)
      {
        auto var_idx = lt.variables[j];
        if (var_tangent_sizes.find(var_idx) == var_tangent_sizes.end())
        {
          var_tangent_sizes[var_idx] = lt.A[j].cols();
        }
      }
    }

    // Compute offsets for the "other" (non-eliminated) variables in the info matrix
    // Non-stamped vars (indices 0..K-1) will be eliminated via Schur
    // All other variables (stamped marginalized + remaining) form H_oo

    // Separate non-stamped and other variable indices
    // Skip holdConstant variables (e.g. PinholeCameraFixed) — they have zero
    // Jacobians and would add zero rows/columns to the information matrix
    std::vector<unsigned int> nonstamped_indices;
    std::vector<unsigned int> other_indices;
    for (const auto& [var_idx, _] : var_tangent_sizes)
    {
      if (var_idx < num_non_stamped)
      {
        nonstamped_indices.push_back(var_idx);
      }
      else if (!graph.getVariable(variable_order[var_idx]).holdConstant())
      {
        other_indices.push_back(var_idx);
      }
    }
    std::sort(nonstamped_indices.begin(), nonstamped_indices.end());
    std::sort(other_indices.begin(), other_indices.end());

    // Compute offsets for non-stamped variables (for H_nn blocks)
    std::unordered_map<unsigned int, int> ns_offsets;
    std::unordered_map<unsigned int, int> ns_sizes;
    for (auto idx : nonstamped_indices)
    {
      ns_offsets[idx] = 0;  // Each is handled independently (block-diagonal)
      ns_sizes[idx] = var_tangent_sizes[idx];
    }

    // Compute offsets for other variables (for H_oo)
    std::unordered_map<unsigned int, int> other_offsets;
    int total_other_dim = 0;
    for (auto idx : other_indices)
    {
      other_offsets[idx] = total_other_dim;
      total_other_dim += var_tangent_sizes[idx];
    }

    if (total_other_dim > 0)
    {
      // Build H_oo and eta_o
      vesta_core::MatrixXd H_oo = vesta_core::MatrixXd::Zero(total_other_dim, total_other_dim);
      vesta_core::VectorXd eta_o = vesta_core::VectorXd::Zero(total_other_dim);

      // For each non-stamped variable, accumulate H_kk, H_no_k, eta_k
      // then apply Schur complement
      // H_kk is block for non-stamped var k
      // H_no_k contains cross-blocks between k and other vars
      // We process each non-stamped var independently (block-diagonal assumption)

      // First, accumulate H_oo and eta_o from terms that don't touch non-stamped vars
      // (these are terms where all variables are "other")
      // Also accumulate per non-stamped-var data

      // Per non-stamped variable: H_kk, H_ko (cross with others), eta_k
      struct NonStampedData
      {
        vesta_core::MatrixXd h_kk;
        vesta_core::MatrixXd h_ko;  // dim_k x total_other_dim
        vesta_core::VectorXd eta_k;
      };
      std::unordered_map<unsigned int, NonStampedData> ns_data;
      for (auto idx : nonstamped_indices)
      {
        int dk = ns_sizes[idx];
        ns_data[idx] = { vesta_core::MatrixXd::Zero(dk, dk), vesta_core::MatrixXd::Zero(dk, total_other_dim),
                         vesta_core::VectorXd::Zero(dk) };
      }

      // Accumulate info matrix contributions from each linear term
      for (const auto& lt : schur_terms)
      {
        // For each pair (i, j) in the linear term, add A_i^T * A_j to appropriate block
        for (size_t i = 0; i < lt.variables.size(); ++i)
        {
          auto vi = lt.variables[i];
          bool vi_ns = (vi < num_non_stamped);

          // eta contribution: A_i^T * b
          vesta_core::VectorXd atb = lt.A[i].transpose() * lt.b;
          if (vi_ns)
          {
            ns_data[vi].eta_k += atb;
          }
          else
          {
            int off_i = other_offsets[vi];
            eta_o.segment(off_i, atb.size()) += atb;
          }

          for (size_t j = 0; j < lt.variables.size(); ++j)
          {
            auto vj = lt.variables[j];
            bool vj_ns = (vj < num_non_stamped);

            vesta_core::MatrixXd block = lt.A[i].transpose() * lt.A[j];

            if (vi_ns && vj_ns)
            {
              // Both non-stamped: only accumulate H_kk (block-diagonal, so vi == vj)
              if (vi == vj)
              {
                ns_data[vi].h_kk += block;
              }
              // Cross between different non-stamped vars: should not happen
              // (no constraint connects two non-stamped vars), skip
            }
            else if (vi_ns && !vj_ns)
            {
              // Non-stamped row, other column: H_ko
              int off_j = other_offsets[vj];
              ns_data[vi].h_ko.block(0, off_j, block.rows(), block.cols()) += block;
            }
            else if (!vi_ns && vj_ns)
            {
              // Other row, non-stamped column: H_ok = H_ko^T (handled via transpose below)
              // Skip here — we handle the full Schur complement using H_ko
            }
            else
            {
              // Both other: H_oo
              int off_i = other_offsets[vi];
              int off_j = other_offsets[vj];
              H_oo.block(off_i, off_j, block.rows(), block.cols()) += block;
            }
          }
        }
      }

      // Apply block Schur complement for each non-stamped variable
      for (auto idx : nonstamped_indices)
      {
        auto& data = ns_data[idx];
        auto llt = data.h_kk.llt();
        if (llt.info() != Eigen::Success)
        {
          LOG(WARNING) << "SchurMarginalizer: Cholesky decomposition of H_kk failed for non-stamped variable "
                       << variable_order[idx] << " (index " << idx << ", size " << data.h_kk.rows() << "x"
                       << data.h_kk.cols() << "). Skipping this variable in Schur elimination.";
          continue;
        }
        // H_oo -= H_ko^T * H_kk^{-1} * H_ko
        vesta_core::MatrixXd Z = llt.solve(data.h_ko);
        H_oo -= data.h_ko.transpose() * Z;
        // eta_o -= H_ko^T * H_kk^{-1} * eta_k
        eta_o -= data.h_ko.transpose() * llt.solve(data.eta_k);
      }

      // Recover square-root form via eigendecomposition
      // H_oo after Schur complement is PSD (not necessarily PD), so we use
      // eigendecomposition to extract the square root, keeping only components
      // with positive eigenvalues
      Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig(H_oo);
      if (eig.info() == Eigen::Success)
      {
        const auto& eigenvalues = eig.eigenvalues();
        const auto& eigenvectors = eig.eigenvectors();

        // Threshold for numerical zero (relative to largest eigenvalue)
        double max_eigenvalue = eigenvalues.maxCoeff();
        double threshold = max_eigenvalue * 1e-10;
        if (threshold < 1e-14)
        {
          threshold = 1e-14;
        }

        // Count positive eigenvalues to determine rank
        int rank = 0;
        for (int k = 0; k < eigenvalues.size(); ++k)
        {
          if (eigenvalues(k) > threshold)
          {
            ++rank;
          }
        }

        if (rank > 0)
        {
          // Build J = sqrt(D_pos) * V_pos^T (rank x total_other_dim)
          vesta_core::MatrixXd J(rank, total_other_dim);
          int row = 0;
          for (int k = 0; k < eigenvalues.size(); ++k)
          {
            if (eigenvalues(k) > threshold)
            {
              J.row(row) = std::sqrt(eigenvalues(k)) * eigenvectors.col(k).transpose();
              ++row;
            }
          }

          // Recover b: J^T b = eta_o  =>  b = (J^T)^{-1} eta_o = (J J^T)^{-1} J eta_o
          // Since J = sqrt(D) V^T, J^T = V sqrt(D), b = D^{-1/2} V^T eta_o
          vesta_core::VectorXd b(rank);
          row = 0;
          for (int k = 0; k < eigenvalues.size(); ++k)
          {
            if (eigenvalues(k) > threshold)
            {
              b(row) = eigenvectors.col(k).dot(eta_o) / std::sqrt(eigenvalues(k));
              ++row;
            }
          }

          // Create a LinearTerm from J and b, split by variable
          detail::LinearTerm schur_result;
          schur_result.b = b;
          schur_result.variables.reserve(other_indices.size());
          schur_result.A.reserve(other_indices.size());
          for (auto idx : other_indices)
          {
            int off = other_offsets[idx];
            int cols = var_tangent_sizes[idx];
            schur_result.variables.push_back(idx);
            schur_result.A.push_back(J.block(0, off, rank, cols));
          }

          // Place the Schur result into the appropriate bucket
          if (!schur_result.variables.empty())
          {
            auto min_var = *std::min_element(schur_result.variables.begin(), schur_result.variables.end());
            linear_terms[min_var].push_back(std::move(schur_result));
          }
        }
      }
      else
      {
        LOG(ERROR) << "SchurMarginalizer: eigendecomposition of H_oo (" << H_oo.rows() << "x" << H_oo.cols()
                   << ") failed after Schur complement. Falling back to QR marginalization for " << num_non_stamped
                   << " non-stamped variables.";
        for (auto& lt : schur_terms)
        {
          auto min_var = *std::min_element(lt.variables.begin(), lt.variables.end());
          linear_terms[min_var].push_back(std::move(lt));
        }

        // Run QR for non-stamped variables
        for (size_t i = 0; i < num_non_stamped; ++i)
        {
          auto linear_marginal = detail::marginalizeNext(linear_terms[i]);
          if (!linear_marginal.variables.empty())
          {
            auto lowest = linear_marginal.variables.front();
            linear_terms[lowest].push_back(std::move(linear_marginal));
          }
        }
      }
    }
    else
    {
      // All marginalized are non-stamped and there are no other connected variables
      // Nothing to do — the information is purely on marginalized variables
    }
  }

  // QR phase for stamped variables (indices K..M-1)
  for (size_t i = num_non_stamped; i < num_marginalized; ++i)
  {
    auto linear_marginal = detail::marginalizeNext(linear_terms[i]);
    if (!linear_marginal.variables.empty())
    {
      auto lowest = linear_marginal.variables.front();
      linear_terms[lowest].push_back(std::move(linear_marginal));
    }
  }

  // Create MarginalConstraints from remaining buckets
  for (size_t i = num_marginalized; i < linear_terms.size(); ++i)
  {
    for (const auto& linear_term : linear_terms[i])
    {
      auto marginal_constraint = detail::createMarginalConstraint(source, linear_term, graph, variable_order);
      transaction.addConstraint(std::move(marginal_constraint));
    }
  }

  return transaction;
}

}  // namespace vesta_constraints
