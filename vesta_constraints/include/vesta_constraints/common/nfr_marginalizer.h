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
 * @brief Nonlinear Factor Recovery (NFR) marginalizer.
 *
 * Implements the method from Usenko et al. (2019) "Visual-Inertial Mapping with Non-Linear Factor Recovery".
 * Given a set of variables to marginalize, this class computes the dense Schur complement and then
 * recovers a sparse set of pairwise relative and per-node absolute factors that optimally approximate
 * the dense information in the KL-divergence sense.
 *
 * The sparsity pattern consists of:
 * - Pairwise relative factors between variables that share the highest mutual information (Chow-Liu tree)
 * - Per-node absolute prior factors for each remaining variable
 *
 * For each factor with Jacobian J_i evaluated at the linearization point, the information matrix is
 * recovered in closed form as H_i = ({J_i * Sigma * J_i^T}_i)^{-1}, where Sigma = H_oo^{-1}.
 *
 * Unlike CLT which uses virtual measurements [-I, I] between variable pairs, NFR uses the actual
 * Jacobian structure of relative and absolute factors, producing a better approximation of the
 * original dense distribution.
 */
class NfrMarginalizer : public Marginalizer
{
public:
  /**
   * @brief Construct an NfrMarginalizer.
   *
   * @param use_fej Whether to use First Estimate Jacobians for linearization
   */
  explicit NfrMarginalizer(bool use_fej = false);

  /**
   * @brief Compute a sparse replacement graph for the marginalized variables.
   *
   * @param source                 Name of the sensor model generating the transaction
   * @param marginalized_variables UUIDs of the variables to marginalize out
   * @param graph                  The current factor graph
   * @return Transaction containing the replacement constraints and variable removals
   */
  vesta_core::Transaction marginalize(const std::string& source,
                                      const std::vector<vesta_core::UUID>& marginalized_variables,
                                      const vesta_core::Graph& graph) override;

private:
  bool use_fej_;
};

}  // namespace vesta_constraints
