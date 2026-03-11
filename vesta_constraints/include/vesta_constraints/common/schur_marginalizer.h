#pragma once

#include <vesta_constraints/common/marginalizer.h>
#include <vesta_core/graph.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>

#include <string>
#include <vector>

namespace vesta_constraints
{

/**
 * @brief Block Schur complement marginalizer with optional FEJ support
 *
 * Eliminates non-stamped variables (e.g. landmarks) via block Schur complement
 * on the information matrix, then eliminates stamped variables via QR
 * decomposition. This is more efficient than iterative QR when many independent
 * non-stamped variables are being marginalized, as the block-diagonal structure
 * of H_nn avoids fill-in during elimination.
 *
 * Falls back to QRMarginalizer when no non-stamped variables are present.
 */
class SchurMarginalizer : public Marginalizer
{
public:
  /**
   * @brief Constructor
   *
   * @param[in] use_fej If true, evaluate Jacobians at variable linearization points
   *                    instead of current values.
   */
  explicit SchurMarginalizer(bool use_fej = false);

  /**
   * @brief Marginalize variables using block Schur complement for non-stamped variables
   */
  vesta_core::Transaction marginalize(const std::string& source,
                                      const std::vector<vesta_core::UUID>& marginalized_variables,
                                      const vesta_core::Graph& graph) override;

private:
  bool use_fej_;
};

}  // namespace vesta_constraints
