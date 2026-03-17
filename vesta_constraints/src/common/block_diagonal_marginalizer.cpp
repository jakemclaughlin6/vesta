#include <vesta_constraints/common/block_diagonal_marginalizer.h>
#include <vesta_constraints/common/marginal_constraint.h>
#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/uuid_ordering.h>
#include <vesta_core/eigen.h>
#include <vesta_core/graph.h>
#include <vesta_core/uuid.h>
#include <vesta_core/variable.h>

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vesta_constraints
{

BlockDiagonalMarginalizer::BlockDiagonalMarginalizer(bool use_fej) : use_fej_(use_fej)
{
}

vesta_core::Transaction BlockDiagonalMarginalizer::marginalize(
    const std::string& source, const std::vector<vesta_core::UUID>& marginalized_variables,
    const vesta_core::Graph& graph)
{
  // 1. Classify marginalized variables as non-stamped vs stamped
  auto classified = detail::classifyVariables(marginalized_variables, graph);

  // 2. If no non-stamped variables, use QR for all stamped variables,
  //    then split multi-variable results into per-variable terms
  if (classified.non_stamped.empty())
  {
    // Build elimination order for QR (CCOLAMD-based)
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

    // BD unique logic: split remaining multi-variable LinearTerms into per-variable terms
    for (size_t i = marginalized_variables.size(); i < linear_terms.size(); ++i)
    {
      for (const auto& linear_term : linear_terms[i])
      {
        if (linear_term.variables.size() == 1)
        {
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

            auto single_term = detail::createSingleVariableTerm(var_idx, H_vv, eta_v);
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

  // 3. Setup
  const size_t num_non_stamped = classified.non_stamped.size();
  const size_t num_marginalized = marginalized_variables.size();
  auto variable_order = detail::buildSchurEliminationOrder(classified.non_stamped, classified.stamped);

  // 4. Linearize and bucket
  auto lin_result =
      detail::linearizeAndBucket(source, marginalized_variables, num_marginalized, graph,
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
      // BD unique logic: instead of producing a dense multi-variable factor,
      // extract each variable's diagonal block and create independent single-variable terms
      for (auto idx : schur_result->other_indices)
      {
        int off = schur_result->other_offsets[idx];
        int dim = schur_result->var_tangent_sizes[idx];
        vesta_core::MatrixXd H_ii = schur_result->H_oo.block(off, off, dim, dim);
        vesta_core::VectorXd eta_i = schur_result->eta_o.segment(off, dim);

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
    }
  }

  // 7. QR phase for stamped variables
  detail::marginalizeStampedVariables(lin_result.linear_terms, num_non_stamped, num_marginalized);

  // 8. Emit remaining constraints, splitting multi-variable terms into per-variable terms
  for (size_t i = num_marginalized; i < lin_result.linear_terms.size(); ++i)
  {
    for (const auto& linear_term : lin_result.linear_terms[i])
    {
      if (linear_term.variables.size() == 1)
      {
        auto mc = detail::createMarginalConstraint(source, linear_term, graph, lin_result.variable_order);
        lin_result.transaction.addConstraint(std::move(mc));
      }
      else
      {
        // BD unique logic: split multi-variable terms from QR into per-variable terms
        for (size_t v = 0; v < linear_term.variables.size(); ++v)
        {
          auto var_idx = linear_term.variables[v];
          const auto& A_v = linear_term.A[v];
          vesta_core::MatrixXd H_vv = A_v.transpose() * A_v;
          vesta_core::VectorXd eta_v = A_v.transpose() * linear_term.b;

          auto single_term = detail::createSingleVariableTerm(var_idx, H_vv, eta_v);
          if (!single_term.variables.empty())
          {
            auto mc = detail::createMarginalConstraint(source, single_term, graph, lin_result.variable_order);
            lin_result.transaction.addConstraint(std::move(mc));
          }
        }
      }
    }
  }

  return std::move(lin_result.transaction);
}

}  // namespace vesta_constraints
