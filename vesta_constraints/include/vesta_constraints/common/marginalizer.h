#pragma once

#include <vesta_core/graph.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>

#include <string>
#include <vector>

namespace vesta_constraints
{

/**
 * @brief Abstract base class for marginalization strategies
 *
 * Provides a pluggable interface for different marginalization algorithms.
 * Subclasses implement the actual marginalization logic.
 */
class Marginalizer
{
public:
  virtual ~Marginalizer() = default;

  /**
   * @brief Generate a transaction that marginalizes out the requested variables
   *
   * @param[in] source                 The name of the sensor or motion model
   * @param[in] marginalized_variables The set of variable UUIDs to marginalize out
   * @param[in] graph                  The graph containing the variables and constraints
   * @return A transaction containing marginal constraints to add and variables/constraints to remove
   */
  virtual vesta_core::Transaction marginalize(const std::string& source,
                                              const std::vector<vesta_core::UUID>& marginalized_variables,
                                              const vesta_core::Graph& graph) = 0;
};

}  // namespace vesta_constraints
