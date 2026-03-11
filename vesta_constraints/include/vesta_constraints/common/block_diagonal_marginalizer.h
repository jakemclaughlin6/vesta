#pragma once

#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/marginalizer.h>
#include <vesta_constraints/common/uuid_ordering.h>
#include <vesta_core/graph.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>

#include <string>
#include <vector>

namespace vesta_constraints
{

/**
 * @brief Block-diagonal marginalizer that produces independent per-variable priors
 *
 * Performs exact marginalization (Schur complement for non-stamped variables,
 * QR for stamped variables) identical to SchurMarginalizer, but instead of
 * creating a single dense marginal constraint coupling all remaining variables,
 * extracts each variable's diagonal block from the information matrix and
 * eigendecomposes independently. This produces N independent single-variable
 * MarginalConstraints, trading cross-variable correlation accuracy for reduced
 * fill-in and faster subsequent marginalizations.
 */
class BlockDiagonalMarginalizer : public Marginalizer
{
public:
  /**
   * @brief Constructor
   *
   * @param[in] use_fej If true, evaluate Jacobians at variable linearization points
   *                    instead of current values.
   */
  explicit BlockDiagonalMarginalizer(bool use_fej = false);

  /**
   * @brief Marginalize variables, producing independent per-variable priors
   */
  vesta_core::Transaction marginalize(const std::string& source,
                                      const std::vector<vesta_core::UUID>& marginalized_variables,
                                      const vesta_core::Graph& graph) override;

private:
  bool use_fej_;
};

}  // namespace vesta_constraints
