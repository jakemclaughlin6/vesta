#include <vesta_constraints/common/block_diagonal_marginalizer.h>
#include <vesta_constraints/common/marginal_constraint.h>
#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/qr_marginalizer.h>
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

BlockDiagonalMarginalizer::BlockDiagonalMarginalizer(bool use_fej) : use_fej_(use_fej)
{
}

namespace
{

/**
 * @brief Extract a single-variable LinearTerm from an information matrix diagonal block
 *
 * Given H_ii (diagonal block of info matrix for variable i) and eta_i (info vector segment),
 * eigendecompose H_ii and produce a LinearTerm with J and b such that J^T J = H_ii and J^T b = eta_i.
 */
detail::LinearTerm createSingleVariableTerm(unsigned int var_idx, const vesta_core::MatrixXd& H_ii,
                                            const vesta_core::VectorXd& eta_i)
{
  Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig(H_ii);
  if (eig.info() != Eigen::Success)
  {
    return {};
  }

  const auto& eigenvalues = eig.eigenvalues();
  const auto& eigenvectors = eig.eigenvectors();

  double max_eigenvalue = eigenvalues.maxCoeff();
  double threshold = max_eigenvalue * 1e-10;
  if (threshold < 1e-14)
  {
    threshold = 1e-14;
  }

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
    return {};
  }

  // Build J = sqrt(D_pos) * V_pos^T (rank x dim)
  vesta_core::MatrixXd J(rank, H_ii.cols());
  int row = 0;
  for (int k = 0; k < eigenvalues.size(); ++k)
  {
    if (eigenvalues(k) > threshold)
    {
      J.row(row) = std::sqrt(eigenvalues(k)) * eigenvectors.col(k).transpose();
      ++row;
    }
  }

  // Recover b: b = D^{-1/2} V^T eta_i
  vesta_core::VectorXd b(rank);
  row = 0;
  for (int k = 0; k < eigenvalues.size(); ++k)
  {
    if (eigenvalues(k) > threshold)
    {
      b(row) = eigenvectors.col(k).dot(eta_i) / std::sqrt(eigenvalues(k));
      ++row;
    }
  }

  detail::LinearTerm term;
  term.variables = { var_idx };
  term.A = { J };
  term.b = b;
  return term;
}

}  // namespace

vesta_core::Transaction BlockDiagonalMarginalizer::marginalize(
    const std::string& source, const std::vector<vesta_core::UUID>& marginalized_variables,
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

  // If no non-stamped variables, use QR for all stamped variables,
  // then split multi-variable results into per-variable terms
  if (non_stamped_vars.empty())
  {
    // Build elimination order for QR
    auto elimination_order = computeEliminationOrder(marginalized_variables, graph);

    vesta_core::Transaction transaction;
    for (const auto& uuid : marginalized_variables)
    {
      transaction.removeVariable(uuid);
    }

    auto variable_order = elimination_order;
    auto used_constraints = std::unordered_set<vesta_core::UUID, vesta_core::uuid::hash>();
    std::vector<std::vector<detail::LinearTerm>> linear_terms(variable_order.size());

    for (size_t i = 0ul; i < marginalized_variables.size(); ++i)
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
          linear_terms[i].push_back(detail::linearize(constraint, graph, variable_order, use_fej_));
          transaction.removeConstraint(constraint.uuid());
        }
      }
    }

    linear_terms.resize(variable_order.size());

    // Marginalize each variable in order via QR
    for (size_t i = 0ul; i < marginalized_variables.size(); ++i)
    {
      auto linear_marginal = detail::marginalizeNext(linear_terms[i]);
      if (!linear_marginal.variables.empty())
      {
        auto lowest = linear_marginal.variables.front();
        linear_terms[lowest].push_back(std::move(linear_marginal));
      }
    }

    // Split remaining multi-variable LinearTerms into per-variable terms
    for (size_t i = marginalized_variables.size(); i < linear_terms.size(); ++i)
    {
      for (const auto& linear_term : linear_terms[i])
      {
        if (linear_term.variables.size() == 1)
        {
          // Already single-variable, create constraint directly
          auto marginal_constraint = detail::createMarginalConstraint(source, linear_term, graph, variable_order);
          transaction.addConstraint(std::move(marginal_constraint));
        }
        else
        {
          // Multi-variable term: compute H = A^T A and eta = A^T b per variable
          for (size_t v = 0; v < linear_term.variables.size(); ++v)
          {
            auto var_idx = linear_term.variables[v];
            const auto& A_v = linear_term.A[v];
            vesta_core::MatrixXd H_vv = A_v.transpose() * A_v;
            vesta_core::VectorXd eta_v = A_v.transpose() * linear_term.b;

            auto single_term = createSingleVariableTerm(var_idx, H_vv, eta_v);
            if (!single_term.variables.empty())
            {
              auto mc = detail::createMarginalConstraint(source, single_term, graph, variable_order);
              transaction.addConstraint(std::move(mc));
            }
          }
        }
      }
    }

    return transaction;
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
        auto min_var = *std::min_element(lt.variables.begin(), lt.variables.end());
        linear_terms[min_var].push_back(std::move(lt));
        transaction.removeConstraint(constraint.uuid());
      }
    }
  }

  linear_terms.resize(variable_order.size());

  // Collect all LinearTerms from non-stamped variable buckets (indices 0..K-1)
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

    // Separate non-stamped and other variable indices
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

    // Compute offsets for non-stamped variables
    std::unordered_map<unsigned int, int> ns_sizes;
    for (auto idx : nonstamped_indices)
    {
      ns_sizes[idx] = var_tangent_sizes[idx];
    }

    // Compute offsets for other variables
    std::unordered_map<unsigned int, int> other_offsets;
    int total_other_dim = 0;
    for (auto idx : other_indices)
    {
      other_offsets[idx] = total_other_dim;
      total_other_dim += var_tangent_sizes[idx];
    }

    if (total_other_dim > 0)
    {
      // Build H_oo and eta_o (same as SchurMarginalizer)
      vesta_core::MatrixXd H_oo = vesta_core::MatrixXd::Zero(total_other_dim, total_other_dim);
      vesta_core::VectorXd eta_o = vesta_core::VectorXd::Zero(total_other_dim);

      struct NonStampedData
      {
        vesta_core::MatrixXd h_kk;
        vesta_core::MatrixXd h_ko;
        vesta_core::VectorXd eta_k;
      };
      std::unordered_map<unsigned int, NonStampedData> ns_data;
      for (auto idx : nonstamped_indices)
      {
        int dk = ns_sizes[idx];
        ns_data[idx] = { vesta_core::MatrixXd::Zero(dk, dk), vesta_core::MatrixXd::Zero(dk, total_other_dim),
                         vesta_core::VectorXd::Zero(dk) };
      }

      // Accumulate info matrix contributions
      for (const auto& lt : schur_terms)
      {
        for (size_t i = 0; i < lt.variables.size(); ++i)
        {
          auto vi = lt.variables[i];
          bool vi_ns = (vi < num_non_stamped);

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
              if (vi == vj)
              {
                ns_data[vi].h_kk += block;
              }
            }
            else if (vi_ns && !vj_ns)
            {
              int off_j = other_offsets[vj];
              ns_data[vi].h_ko.block(0, off_j, block.rows(), block.cols()) += block;
            }
            else if (!vi_ns && !vj_ns)
            {
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
          LOG(WARNING) << "BlockDiagonalMarginalizer: Cholesky of H_kk failed for variable " << variable_order[idx]
                       << ". Skipping.";
          continue;
        }
        vesta_core::MatrixXd Z = llt.solve(data.h_ko);
        H_oo -= data.h_ko.transpose() * Z;
        eta_o -= data.h_ko.transpose() * llt.solve(data.eta_k);
      }

      // KEY DIFFERENCE: Instead of one eigendecomposition on full H_oo,
      // extract each variable's diagonal block and eigendecompose independently
      for (auto idx : other_indices)
      {
        int off = other_offsets[idx];
        int dim = var_tangent_sizes[idx];
        vesta_core::MatrixXd H_ii = H_oo.block(off, off, dim, dim);
        vesta_core::VectorXd eta_i = eta_o.segment(off, dim);

        auto single_term = createSingleVariableTerm(idx, H_ii, eta_i);
        if (!single_term.variables.empty())
        {
          // Place into the appropriate bucket for later constraint creation
          // If this variable is a stamped marginalized variable, it goes into
          // its QR bucket; otherwise create constraint directly
          if (idx >= num_marginalized)
          {
            auto mc = detail::createMarginalConstraint(source, single_term, graph, variable_order);
            transaction.addConstraint(std::move(mc));
          }
          else
          {
            linear_terms[idx].push_back(std::move(single_term));
          }
        }
      }
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

  // Create MarginalConstraints from remaining buckets, splitting multi-variable terms
  for (size_t i = num_marginalized; i < linear_terms.size(); ++i)
  {
    for (const auto& linear_term : linear_terms[i])
    {
      if (linear_term.variables.size() == 1)
      {
        auto mc = detail::createMarginalConstraint(source, linear_term, graph, variable_order);
        transaction.addConstraint(std::move(mc));
      }
      else
      {
        // Multi-variable term from QR: split into per-variable terms
        for (size_t v = 0; v < linear_term.variables.size(); ++v)
        {
          auto var_idx = linear_term.variables[v];
          const auto& A_v = linear_term.A[v];
          vesta_core::MatrixXd H_vv = A_v.transpose() * A_v;
          vesta_core::VectorXd eta_v = A_v.transpose() * linear_term.b;

          auto single_term = createSingleVariableTerm(var_idx, H_vv, eta_v);
          if (!single_term.variables.empty())
          {
            auto mc = detail::createMarginalConstraint(source, single_term, graph, variable_order);
            transaction.addConstraint(std::move(mc));
          }
        }
      }
    }
  }

  return transaction;
}

}  // namespace vesta_constraints
