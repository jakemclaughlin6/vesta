#include <vesta_constraints/common/marginal_constraint.h>
#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/qr_marginalizer.h>
#include <vesta_constraints/common/uuid_ordering.h>
#include <vesta_core/graph.h>
#include <vesta_core/uuid.h>

#include <algorithm>
#include <cassert>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vesta_constraints
{

QRMarginalizer::QRMarginalizer(bool use_fej) : use_fej_(use_fej)
{
}

vesta_core::Transaction QRMarginalizer::marginalize(const std::string& source,
                                                    const std::vector<vesta_core::UUID>& marginalized_variables,
                                                    const vesta_core::Graph& graph)
{
  return marginalize(source, marginalized_variables, graph, computeEliminationOrder(marginalized_variables, graph));
}

vesta_core::Transaction QRMarginalizer::marginalize(const std::string& source,
                                                    const std::vector<vesta_core::UUID>& marginalized_variables,
                                                    const vesta_core::Graph& graph,
                                                    const UuidOrdering& elimination_order)
{
  assert(std::all_of(marginalized_variables.begin(), marginalized_variables.end(),
                     [&elimination_order, &marginalized_variables](const vesta_core::UUID& variable_uuid) {
                       return elimination_order.exists(variable_uuid) &&
                              elimination_order.at(variable_uuid) < marginalized_variables.size();
                     }));

  vesta_core::Transaction transaction;

  // Mark all marginalized variables for removal
  for (const auto& variable_uuid : marginalized_variables)
  {
    transaction.removeVariable(variable_uuid);
  }

  // Copy the elimination order so we can add additional variables if needed
  auto variable_order = elimination_order;

  // Linearize all involved constraints
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

  // Expand to include connected variables
  linear_terms.resize(variable_order.size());

  // Marginalize each variable in order
  for (size_t i = 0ul; i < marginalized_variables.size(); ++i)
  {
    auto linear_marginal = detail::marginalizeNext(linear_terms[i]);
    if (!linear_marginal.variables.empty())
    {
      auto lowest_ordered_variable = linear_marginal.variables.front();
      linear_terms[lowest_ordered_variable].push_back(std::move(linear_marginal));
    }
  }

  // Convert remaining linear marginals into marginal constraints
  for (size_t i = marginalized_variables.size(); i < linear_terms.size(); ++i)
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
