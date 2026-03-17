#pragma once

/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2019, Locus Robotics
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#include <vesta_constraints/common/marginal_constraint.h>
#include <vesta_constraints/common/marginalizer.h>
#include <vesta_constraints/common/uuid_ordering.h>
#include <vesta_core/constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/uuid.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/graph.h>
#include <vesta_core/manifold.h>
#include <vesta_core/transaction.h>
#include <vesta_core/variable.h>

#include <ceres/cost_function.h>
#include <boost/iterator/transform_iterator.hpp>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace vesta_constraints
{

/**
 * @brief Compute an efficient elimination order for the marginalized variables
 *
 * The marginalized_variables are guaranteed to be placed before any additional
 * connected variables. Each time a variable is eliminated from the system, the
 * resulting reduced system is independent of the eliminated variable. By
 * eliminating the "marginalized variables" first, all of the "additional
 * connected variables" will remain in the system, but they will not depend on
 * any of the "marginalized variables"...which is what we want.
 *
 * This function uses CCOLAMD to find a good elimination order that eliminates
 * all the "marginalized variables" first.
 *
 * @param[in] marginalized_variables The variable UUIDs to be marginalized out
 * @param[in] graph                  A graph containing, at least, all
 * constraints that involve at least one marginalized variable
 * @return The mapping from variable UUID to the computed elimination order
 */
UuidOrdering computeEliminationOrder(const std::vector<vesta_core::UUID>& marginalized_variables,
                                     const vesta_core::Graph& graph);

/**
 * @brief Generate a transaction that, when applied to the graph, will
 * marginalize out the requested variables
 *
 * This computes a linear approximation of the marginal information on the
 * non-marginalized variables. The current variable values in the graph are used
 * as the linearization points for the linear approximation. Thus, marginalizing
 * out a variable will introduce linearization errors as the optimal values move
 * away from the fixed linearization points.
 *
 * This version computes an efficient elimination order using
 * computeEliminationOrder().
 *
 * @param[in] source                 The name of the sensor or motion model that
 * generated this constraint
 * @param[in] marginalized_variables The set of variable UUIDs to marginalize
 * out
 * @param[in] graph                  A graph containing the variables and
 * constraints that are connected to at least one marginalized variable. The
 * graph may also contain additional variables and constraints.
 * @return A transaction object containing the computed marginal constraints to
 * be added, as well as the set of variables and constraints to be removed.
 */
vesta_core::Transaction marginalizeVariables(const std::string& source,
                                             const std::vector<vesta_core::UUID>& marginalized_variables,
                                             const vesta_core::Graph& graph);

/**
 * @brief Generate a transaction that, when applied to the graph, will
 * marginalize out the requested variables
 *
 * This computes a linear approximation of the marginal information on the
 * non-marginalized variables. The current variable values in the graph are used
 * as the linearization points for the linear approximation. Thus, marginalizing
 * out a variable will introduce linearization errors as the optimal values move
 * away from the fixed linearization points.
 *
 * This version allows the user to provide their own elimination order. The
 * marginalized_variables *must* occur before any other variables in that
 * elimination order.
 *
 * @param[in] source                 The name of the sensor or motion model that
 * generated this constraint
 * @param[in] marginalized_variables The set of variable UUIDs to marginalize
 * out
 * @param[in] graph                  A graph containing the variables and
 * constraints that are connected to at least one marginalized variable. The
 * graph may also contain additional variables and constraints.
 * @param[in] elimination_order      An sequential ordering of at least the
 * marginalized variables
 * @return A transaction object containing the computed marginal constraints to
 * be added, as well as the set of variables and constraints to be removed.
 */
vesta_core::Transaction marginalizeVariables(const std::string& source,
                                             const std::vector<vesta_core::UUID>& marginalized_variables,
                                             const vesta_core::Graph& graph,
                                             const vesta_constraints::UuidOrdering& elimination_order);

/**
 * @brief Generate a transaction that marginalizes out the requested variables
 * using the provided marginalizer
 *
 * This overload allows the caller to choose a marginalization strategy at
 * runtime (e.g. QRMarginalizer or SchurMarginalizer).
 *
 * @param[in] source                 The name of the sensor or motion model
 * @param[in] marginalized_variables The set of variable UUIDs to marginalize out
 * @param[in] graph                  The graph containing the variables and constraints
 * @param[in] marginalizer           The marginalization strategy to use
 * @return A transaction containing marginal constraints to add and variables/constraints to remove
 */
vesta_core::Transaction marginalizeVariables(const std::string& source,
                                             const std::vector<vesta_core::UUID>& marginalized_variables,
                                             const vesta_core::Graph& graph, Marginalizer& marginalizer);

namespace detail
{

/**
 * @brief Structure holding linearized Jacobian blocks
 *
 * The LinearTerm uses sequential variable indices instead of UUIDs
 */
struct LinearTerm
{
  std::vector<unsigned int> variables;
  std::vector<vesta_core::MatrixXd> A;
  vesta_core::VectorXd b;
};

/**
 * @brief Linearize the nonlinear constraint with optional FEJ support
 *
 * When \p use_fej is true, Jacobians are evaluated at the stored linearization
 * points while residuals use current variable values. Variables without a
 * stored linearization point fall back to current values.
 *
 * @param[in] constraint        The constraint to linearize
 * @param[in] graph             A graph containing the variables
 * @param[in] elimination_order A mapping from variable UUID to elimination order
 * @param[in] use_fej           If true, use First Estimate Jacobian linearization
 * @return A LinearTerm consisting of Jacobian blocks in elimination order
 */
LinearTerm linearize(const vesta_core::Constraint& constraint, const vesta_core::Graph& graph,
                     const UuidOrdering& elimination_order, bool use_fej = false);

/**
 * @brief Marginalize out the lowest-ordered variable from the provided set of
 * linear terms
 *
 * A linear marginal term is returned. This represents the information on the
 * remaining variables after marginalizing out the lowest-ordered variable.
 *
 * @param[in] linear_terms The set of LinearTerms that are connected to the
 * lowest-ordered variable index
 * @return A LinearTerm object containing the information on the remaining
 * variables
 */
LinearTerm marginalizeNext(const std::vector<LinearTerm>& linear_terms);

/**
 * @brief Convert the provided linear term into a MarginalConstraint
 *
 * @param[in] source            The name of the sensor or motion model that
 * generated this constraint
 * @param[in] linear_term       The LinearTerm object to convert
 * @param[in] graph             The graph object containing the current variable
 * values
 * @param[in] elimination_order The mapping from variable UUID to LinearTerm
 * variable index
 * @return An equivalent MarginalConstraint object
 */
MarginalConstraint::SharedPtr createMarginalConstraint(const std::string& source, const LinearTerm& linear_term,
                                                       const vesta_core::Graph& graph,
                                                       const UuidOrdering& elimination_order);
/**
 * @brief Result of classifying marginalized variables as stamped vs non-stamped
 */
struct ClassifiedVariables
{
  std::vector<vesta_core::UUID> non_stamped;
  std::vector<vesta_core::UUID> stamped;
};

/**
 * @brief Classify marginalized variables into stamped and non-stamped groups
 *
 * @param[in] marginalized_variables The variable UUIDs to classify
 * @param[in] graph                  The graph containing the variables
 * @return A ClassifiedVariables struct with the two groups
 */
ClassifiedVariables classifyVariables(const std::vector<vesta_core::UUID>& marginalized_variables,
                                      const vesta_core::Graph& graph);

/**
 * @brief Build a Schur-style elimination order with non-stamped variables first
 *
 * @param[in] non_stamped_vars Non-stamped variable UUIDs (placed first)
 * @param[in] stamped_vars     Stamped variable UUIDs (placed after non-stamped)
 * @return A UuidOrdering with non-stamped variables at indices [0, K) and stamped at [K, M)
 */
UuidOrdering buildSchurEliminationOrder(const std::vector<vesta_core::UUID>& non_stamped_vars,
                                        const std::vector<vesta_core::UUID>& stamped_vars);

/**
 * @brief Result of linearizing and bucketing constraints connected to marginalized variables
 */
struct LinearizationResult
{
  vesta_core::Transaction transaction;
  UuidOrdering variable_order;
  std::vector<std::vector<LinearTerm>> linear_terms;
};

/**
 * @brief Linearize all constraints connected to marginalized variables and bucket by lowest-ordered variable
 *
 * Creates a transaction that removes all marginalized variables and their connected constraints,
 * linearizes each constraint, and places the LinearTerms into buckets indexed by the lowest-ordered
 * variable in each term.
 *
 * @param[in] source                 The sensor/model name for the transaction
 * @param[in] marginalized_variables All variable UUIDs being marginalized
 * @param[in] num_marginalized       Total number of marginalized variables
 * @param[in] graph                  The graph containing variables and constraints
 * @param[in] variable_order         The elimination order (will be extended with connected variables)
 * @param[in] use_fej                Whether to use First Estimate Jacobians
 * @return A LinearizationResult with the transaction, extended variable order, and bucketed linear terms
 */
LinearizationResult linearizeAndBucket(const std::string& source,
                                       const std::vector<vesta_core::UUID>& marginalized_variables,
                                       size_t num_marginalized, const vesta_core::Graph& graph,
                                       UuidOrdering variable_order, bool use_fej);

/**
 * @brief Collect all LinearTerms from non-stamped variable buckets into a single vector
 *
 * Moves all linear terms from buckets [0, num_non_stamped) into the returned vector
 * and clears those buckets.
 *
 * @param[in,out] linear_terms    The bucketed linear terms (modified in place)
 * @param[in]     num_non_stamped Number of non-stamped variable buckets to collect from
 * @return A vector of all LinearTerms from the non-stamped buckets
 */
std::vector<LinearTerm> collectSchurTerms(std::vector<std::vector<LinearTerm>>& linear_terms, size_t num_non_stamped);

/**
 * @brief Result of computing the Schur complement on the non-stamped variables
 */
struct SchurResult
{
  vesta_core::MatrixXd H_oo;   ///< Information matrix on remaining ("other") variables
  vesta_core::VectorXd eta_o;  ///< Information vector on remaining ("other") variables
  std::vector<unsigned int> other_indices;                  ///< Variable indices of remaining variables
  std::unordered_map<unsigned int, int> other_offsets;      ///< Offset of each variable in H_oo
  std::unordered_map<unsigned int, int> var_tangent_sizes;  ///< Tangent dimension of each variable
  int total_other_dim;                                      ///< Total dimension of H_oo
};

/**
 * @brief Compute the Schur complement, eliminating non-stamped variables from the information system
 *
 * Accumulates an information matrix and vector from the provided linear terms, then eliminates
 * non-stamped variables via block Schur complement to produce H_oo and eta_o on the remaining
 * ("other") variables.
 *
 * @param[in] schur_terms      LinearTerms involving the non-stamped variables
 * @param[in] num_non_stamped  Number of non-stamped variables (indices 0..K-1)
 * @param[in] graph            The graph (used to check holdConstant)
 * @param[in] variable_order   The elimination order
 * @return A SchurResult if there are remaining variables, or std::nullopt if total_other_dim == 0
 */
std::optional<SchurResult> computeSchurComplement(const std::vector<LinearTerm>& schur_terms, size_t num_non_stamped,
                                                  const vesta_core::Graph& graph, const UuidOrdering& variable_order);

/**
 * @brief Run the QR marginalization phase for stamped variables
 *
 * For each stamped variable (indices [num_non_stamped, num_marginalized)), calls marginalizeNext
 * and places the result into the appropriate bucket of linear_terms.
 *
 * @param[in,out] linear_terms      Bucketed linear terms (modified in place)
 * @param[in]     num_non_stamped   Start index of stamped variables
 * @param[in]     num_marginalized  End index of marginalized variables
 */
void marginalizeStampedVariables(std::vector<std::vector<LinearTerm>>& linear_terms, size_t num_non_stamped,
                                 size_t num_marginalized);

/**
 * @brief Emit MarginalConstraints from remaining linear term buckets into a transaction
 *
 * Creates a MarginalConstraint for each LinearTerm in buckets [start_index, linear_terms.size())
 * and adds them to the transaction.
 *
 * @param[in]     source         The sensor/model name
 * @param[in]     linear_terms   Bucketed linear terms
 * @param[in]     start_index    First bucket to emit from
 * @param[in]     graph          The graph containing current variable values
 * @param[in]     variable_order The elimination order
 * @param[in,out] transaction    The transaction to add constraints to
 */
void emitRemainingConstraints(const std::string& source, const std::vector<std::vector<LinearTerm>>& linear_terms,
                              size_t start_index, const vesta_core::Graph& graph, const UuidOrdering& variable_order,
                              vesta_core::Transaction& transaction);

/**
 * @brief Create a single-variable LinearTerm from an information matrix diagonal block
 *
 * Eigendecomposes H_ii and produces a LinearTerm with Jacobian J and residual b such that
 * J^T J = H_ii (up to rank) and J^T b encodes the information vector eta_i.
 *
 * @param[in] var_idx Variable index in the elimination order
 * @param[in] H_ii    Information matrix diagonal block for the variable
 * @param[in] eta_i   Information vector segment for the variable
 * @return A LinearTerm for the variable, or an empty LinearTerm if H_ii has zero rank
 */
LinearTerm createSingleVariableTerm(unsigned int var_idx, const vesta_core::MatrixXd& H_ii,
                                    const vesta_core::VectorXd& eta_i);

}  // namespace detail

}  // namespace vesta_constraints
