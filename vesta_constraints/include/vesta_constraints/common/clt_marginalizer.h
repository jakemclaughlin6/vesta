#pragma once

#include <vesta_constraints/common/marginalizer.h>
#include <vesta_core/graph.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>

#include <string>
#include <vector>

namespace vesta_constraints
{

/// Sparsity pattern selection strategy for Chow-Liu tree sparsification.
enum class SparsityMode
{
  Tree,     ///< Chow-Liu tree approximation (closed-form solution)
  Subgraph  ///< Chow-Liu tree + additional chords (iterative PQN solver)
};

/**
 * @brief Chow-Liu tree sparsification marginalizer.
 *
 * Implements the method from Mazuran et al. (RSS 2014) "Nonlinear Graph Sparsification for SLAM". Given a set of
 * variables to marginalize, this class computes a sparse replacement factor graph that approximates the dense
 * marginalization result.
 *
 * In Tree mode, the sparsity pattern is a Chow-Liu tree over the remaining variables, and the replacement factors
 * have a closed-form solution. In Subgraph mode, additional chord edges are added to the tree and the factor
 * information matrices are optimized via a Projected Quasi-Newton (PQN) solver.
 */
class CltMarginalizer : public Marginalizer
{
public:
  /**
   * @brief Construct a CltMarginalizer.
   *
   * @param mode           Sparsity pattern selection strategy (default: Tree)
   * @param use_fej        Whether to use First Estimate Jacobians for linearization
   * @param chord_ratio    For Subgraph mode, ratio of additional chords to tree edges
   * @param max_iterations Maximum PQN iterations for Subgraph mode
   * @param gradient_tolerance Convergence tolerance for the PQN solver
   */
  explicit CltMarginalizer(SparsityMode mode = SparsityMode::Tree, bool use_fej = false, double chord_ratio = 0.5,
                           int max_iterations = 100, double gradient_tolerance = 1e-8);

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
  SparsityMode mode_;
  bool use_fej_;
  double chord_ratio_;
  int max_iterations_;
  double gradient_tolerance_;
};

}  // namespace vesta_constraints
