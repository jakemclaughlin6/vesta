#pragma once

#include <vesta_constraints/common/marginalizer.h>
#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/uuid_ordering.h>
#include <vesta_core/graph.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>

#include <string>
#include <vector>

namespace vesta_constraints
{

/**
 * @brief QR decomposition-based marginalizer with optional FEJ support
 *
 * Implements marginalization via iterative QR decomposition of linearized
 * constraints. When FEJ (First Estimate Jacobian) is enabled, Jacobians are
 * evaluated at the stored linearization points while residuals use current
 * variable values, reducing linearization inconsistency in sliding-window
 * estimators.
 */
class QRMarginalizer : public Marginalizer
{
public:
  /**
   * @brief Constructor
   *
   * @param[in] use_fej If true, evaluate Jacobians at variable linearization points
   *                    instead of current values. Variables without a stored linearization
   *                    point fall back to current values.
   */
  explicit QRMarginalizer(bool use_fej = false);

  /**
   * @brief Marginalize variables using a computed elimination order
   */
  vesta_core::Transaction marginalize(const std::string& source,
                                      const std::vector<vesta_core::UUID>& marginalized_variables,
                                      const vesta_core::Graph& graph) override;

  /**
   * @brief Marginalize variables with a user-provided elimination order
   *
   * @param[in] elimination_order A sequential ordering where marginalized variables come first
   */
  vesta_core::Transaction marginalize(const std::string& source,
                                      const std::vector<vesta_core::UUID>& marginalized_variables,
                                      const vesta_core::Graph& graph,
                                      const UuidOrdering& elimination_order);

private:
  bool use_fej_;
};

}  // namespace vesta_constraints
