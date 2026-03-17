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
#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/qr_marginalizer.h>
#include <vesta_constraints/common/uuid_ordering.h>
#include <vesta_constraints/common/variable_constraints.h>
#include <vesta_core/uuid.h>
#include <vesta_core/variable.h>
#include <vesta_variables/common/stamped.h>

#include <glog/logging.h>
#include <suitesparse/ccolamd.h>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <boost/iterator/transform_iterator.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vesta_constraints
{

UuidOrdering computeEliminationOrder(const std::vector<vesta_core::UUID>& marginalized_variables,
                                     const vesta_core::Graph& graph)
{
  // COLAMD wants a somewhat weird structure
  // Variables are numbered sequentially in some arbitrary order. We call this
  // the "variable index" order. Similarly, the Constraints are numbered
  // sequentially. We call this the "constraint index" order. 'A' contains the
  // constraint index for each connected constraint to a specific variable. The
  // connected
  //     constraints are added to 'A' in variable index order.
  // 'p' contains the boundary indices for each variable in 'A'. So variable #1
  // spans entries
  //     from A[p[0]] to A[p[1] - 1], and variable #2 is the range A[p[1]] to
  //     A[p[2] - 1], etc.
  // In order to compute A and p efficiently, we first construct a
  // VariableConstraints object We will construct sequential indices for the
  // variables and constraints while we populate the VariableConstraints object.
  // For orphan variables, i.e. variables with no constraints, p[c] == p[c+1],
  // which still satisfies CCOLAMD specs:
  // https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CCOLAMD/Source/ccolamd.c#L174
  auto variable_order = UuidOrdering();
  auto constraint_order = UuidOrdering();
  auto variable_constraints = VariableConstraints();
  for (const auto& variable_uuid : marginalized_variables)
  {
    // Get all connected constraints to this variable
    const auto constraints = graph.getConnectedConstraints(variable_uuid);

    // If the variable is orphan (it has no constraints), add it to the
    // VariableConstraints object without constraints New variable index is
    // automatically generated
    if (constraints.empty())
    {
      variable_constraints.insert(variable_order[variable_uuid]);
    }
    else
    {
      // Add each constraint to the VariableConstraints object
      // New constraint and variable indices are automatically generated
      for (const auto& constraint : constraints)
      {
        unsigned int constraint_index = constraint_order[constraint.uuid()];
        for (const auto& constraint_variable_uuid : constraint.variables())
        {
          variable_constraints.insert(constraint_index, variable_order[constraint_variable_uuid]);
        }
      }
    }
  }

  // Construct the CCOLAMD input structures
  auto recommended_size =
      ccolamd_recommended(variable_constraints.size(), constraint_order.size(), variable_order.size());
  auto A = std::vector<int>(recommended_size);
  auto p = std::vector<int>(variable_order.size() + 1);

  // Use the VariableConstraints table to construct the A and p structures
  auto A_iter = A.begin();
  auto p_iter = p.begin();
  *p_iter = 0;
  ++p_iter;
  for (unsigned int variable_index = 0u; variable_index < variable_order.size(); ++variable_index)
  {
    A_iter = variable_constraints.getConstraints(variable_index, A_iter);
    *p_iter = std::distance(A.begin(), A_iter);
    ++p_iter;
  }

  // Define the variable groups used by CCOLAMD. All of the marginalized
  // variables should be group0, all the rest should be group1.
  std::vector<int> variable_groups(variable_order.size(),
                                   1);  // Default all variables to group1
  for (const auto& variable_uuid : marginalized_variables)
  {
    // Reassign the marginalized variables to group0
    variable_groups[variable_order.at(variable_uuid)] = 0;
  }

  // Create some additional CCOLAMD required structures
  double knobs[CCOLAMD_KNOBS];
  ccolamd_set_defaults(knobs);
  int stats[CCOLAMD_STATS];

  // Finally call CCOLAMD
  auto success = ccolamd(constraint_order.size(), variable_order.size(), recommended_size, A.data(), p.data(), knobs,
                         stats, variable_groups.data());
  if (!success)
  {
    throw std::runtime_error("Failed to call CCOLAMD to generate the elimination order.");
  }

  // Extract the elimination order from CCOLAMD.
  // CCOLAMD returns the elimination order by updating the values stored in p
  // with the variable index Remember that p is larger than
  // variable_order.size()
  auto elimination_order = UuidOrdering();
  for (size_t i = 0ul; i < variable_order.size(); ++i)
  {
    elimination_order.push_back(variable_order[p[i]]);
  }

  return elimination_order;
}

vesta_core::Transaction marginalizeVariables(const std::string& source,
                                             const std::vector<vesta_core::UUID>& marginalized_variables,
                                             const vesta_core::Graph& graph)
{
  return QRMarginalizer(false).marginalize(source, marginalized_variables, graph);
}

vesta_core::Transaction marginalizeVariables(const std::string& source,
                                             const std::vector<vesta_core::UUID>& marginalized_variables,
                                             const vesta_core::Graph& graph,
                                             const vesta_constraints::UuidOrdering& elimination_order)
{
  return QRMarginalizer(false).marginalize(source, marginalized_variables, graph, elimination_order);
}

vesta_core::Transaction marginalizeVariables(const std::string& source,
                                             const std::vector<vesta_core::UUID>& marginalized_variables,
                                             const vesta_core::Graph& graph, Marginalizer& marginalizer)
{
  return marginalizer.marginalize(source, marginalized_variables, graph);
}

namespace detail
{
// TODO(swilliams) There are more graph lookups of each Variable than needed.
// Refactor so that each Variable is only
//                 accessed once. This will mean storing the current variable
//                 value and manifold in the LinearTerm.

/**
 * In order for the linearize function to work correctly, it must perform the
 * same operations as Google Ceres-Solver. Unfortunately those functions are not
 * callable from the public API, so we must replicate them here. The following
 * function is not identical to the Ceres-Solver code, but it was referenced
 * heavily during the creation of this function. As such, I'd like to
 * acknowledge the original authors.
 *
 * Ceres Solver - A fast non-linear least squares minimizer
 * http://ceres-solver.org/
 * Author: keir@google.com (Keir Mierle)
 *         sameeragarwal@google.com (Sameer Agarwal)
 *
 *  -
 * https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/residual_block.cc
 *  -
 * https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/corrector.cc
 */
LinearTerm linearize(const vesta_core::Constraint& constraint, const vesta_core::Graph& graph,
                     const UuidOrdering& elimination_order, bool use_fej)
{
  LinearTerm result;

  auto cost_function = constraint.costFunction();
  size_t row_count = cost_function->num_residuals();

  const auto& variable_uuids = constraint.variables();
  const size_t variable_count = variable_uuids.size();

  std::vector<const double*> current_values;
  std::vector<const double*> linearization_values;
  current_values.reserve(variable_count);
  if (use_fej)
  {
    linearization_values.reserve(variable_count);
  }

  std::vector<double*> jacobians;
  jacobians.reserve(variable_count);
  result.variables.reserve(variable_count);
  result.A.reserve(variable_count);

  for (const auto& variable_uuid : variable_uuids)
  {
    const auto& variable = graph.getVariable(variable_uuid);
    current_values.push_back(variable.data());
    if (use_fej)
    {
      linearization_values.push_back(variable.linearizationPoint());
    }
    result.variables.push_back(elimination_order.at(variable_uuid));
    result.A.push_back(vesta_core::MatrixXd(row_count, variable.size()));
    jacobians.push_back(result.A.back().data());
  }
  result.b = vesta_core::VectorXd(row_count);

  bool success;
  if (use_fej)
  {
    // FEJ: Jacobians at linearization points, residuals at current values
    vesta_core::VectorXd dummy_residuals(row_count);
    success = cost_function->Evaluate(linearization_values.data(), dummy_residuals.data(), jacobians.data());
    success = success && cost_function->Evaluate(current_values.data(), result.b.data(), nullptr);
  }
  else
  {
    success = cost_function->Evaluate(current_values.data(), result.b.data(), jacobians.data());
  }
  delete cost_function;

  success = success && result.b.array().isFinite().all();
  for (const auto& A : result.A)
  {
    success = success && A.array().isFinite().all();
  }
  if (!success)
  {
    throw std::runtime_error("Error in evaluating the cost function. "
                             "Either the CostFunction did not evaluate and fill all residual and jacobians "
                             "that were requested or there was a non-finite value (nan/infinite) generated "
                             "during the jacobian computation.");
  }

  // Apply manifold corrections
  const auto& eval_values = use_fej ? linearization_values : current_values;
  for (size_t index = 0ul; index < variable_count; ++index)
  {
    const auto& variable_uuid = variable_uuids[index];
    const auto& variable = graph.getVariable(variable_uuid);
    auto manifold = variable.manifold();
    auto& jacobian = result.A[index];
    if (variable.holdConstant())
    {
      if (manifold)
      {
        jacobian.resize(Eigen::NoChange, manifold->TangentSize());
      }
      jacobian.setZero();
    }
    else if (manifold)
    {
      vesta_core::MatrixXd J(manifold->AmbientSize(), manifold->TangentSize());
      manifold->PlusJacobian(eval_values[index], J.data());
      jacobian *= J;
    }
    if (manifold)
    {
      delete manifold;
    }
  }

  // Correct A and b for the effects of the loss function
  auto loss_function = constraint.lossFunction();
  if (loss_function)
  {
    double squared_norm = result.b.squaredNorm();
    double rho[3];
    loss_function->Evaluate(squared_norm, rho);
    if (vesta_core::Loss::Ownership == ceres::Ownership::TAKE_OWNERSHIP)
    {
      delete loss_function;
    }
    double sqrt_rho1 = std::sqrt(rho[1]);
    double alpha = 0.0;
    if ((squared_norm > 0.0) && (rho[2] > 0.0))
    {
      const double D = 1.0 + 2.0 * squared_norm * rho[2] / rho[1];
      alpha = 1.0 - std::sqrt(D);
    }

    for (auto& jacobian : result.A)
    {
      if (alpha == 0.0)
      {
        jacobian *= sqrt_rho1;
      }
      else
      {
        jacobian = sqrt_rho1 * (jacobian - (alpha / squared_norm) * result.b * (result.b.transpose() * jacobian));
      }
    }

    result.b *= sqrt_rho1 / (1 - alpha);
  }

  return result;
}

LinearTerm marginalizeNext(const std::vector<LinearTerm>& linear_terms)
{
  if (linear_terms.empty())
  {
    return {};
  }

  // We need to create a dense matrix from all of the provided linear terms, and
  // that matrix must order the variables in the proper elimination order. The
  // LinearTerms have the elimination order baked into the variable indices, but
  // since not all variables are necessarily present, we need to remove any gaps
  // from the variable indices. We use vector operations instead of a std::set
  // because the number of variables is assumed to be small. You need 1000s of
  // variables before the std::set outperforms the std::vector.
  auto dense_to_index = std::vector<unsigned int>();
  for (const auto& linear_term : linear_terms)
  {
    std::copy(linear_term.variables.begin(), linear_term.variables.end(), std::back_inserter(dense_to_index));
  }
  std::sort(dense_to_index.begin(), dense_to_index.end());
  dense_to_index.erase(std::unique(dense_to_index.begin(), dense_to_index.end()), dense_to_index.end());

  // Construct the inverse mapping
  auto index_to_dense = std::vector<unsigned int>(dense_to_index.back() + 1, 0);
  for (size_t dense = 0ul; dense < dense_to_index.size(); ++dense)
  {
    index_to_dense[dense_to_index[dense]] = dense;
  }

  // Compute the row offsets
  auto row_offsets = std::vector<unsigned int>();
  row_offsets.reserve(linear_terms.size() + 1ul);
  row_offsets.push_back(0u);
  for (const auto& linear_term : linear_terms)
  {
    row_offsets.push_back(row_offsets.back() + linear_term.b.rows());
  }

  // Compute the column offsets
  auto index_to_cols = std::vector<unsigned int>(dense_to_index.back() + 1u, 0u);
  for (const auto& linear_term : linear_terms)
  {
    for (size_t i = 0ul; i < linear_term.variables.size(); ++i)
    {
      auto index = linear_term.variables[i];
      index_to_cols[index] = linear_term.A[i].cols();
    }
  }

  auto column_offsets = std::vector<unsigned int>();
  column_offsets.reserve(dense_to_index.size() + 1ul);
  column_offsets.push_back(0u);
  for (size_t dense = 0; dense < dense_to_index.size(); ++dense)
  {
    column_offsets.push_back(column_offsets.back() + index_to_cols[dense_to_index[dense]]);
  }

  // Construct the Ab matrix
  vesta_core::MatrixXd Ab = vesta_core::MatrixXd::Zero(row_offsets.back(), column_offsets.back() + 1u);
  for (size_t term_index = 0ul; term_index < linear_terms.size(); ++term_index)
  {
    const auto& linear_term = linear_terms[term_index];
    auto row_offset = row_offsets[term_index];
    for (size_t i = 0ul; i < linear_term.variables.size(); ++i)
    {
      const auto& A = linear_term.A[i];
      auto dense = index_to_dense[linear_term.variables[i]];
      auto column_offset = column_offsets[dense];
      for (int row = 0; row < A.rows(); ++row)
      {
        for (int col = 0; col < A.cols(); ++col)
        {
          Ab(row_offset + row, column_offset + col) = A(row, col);
        }
      }
    }
    const auto& b = linear_term.b;
    int column_offset = column_offsets.back();
    for (int row = 0; row < b.rows(); ++row)
    {
      Ab(row_offset + row, column_offset) = b(row);
    }
  }

  // Compute the QR decomposition
  // I really want to do this "in place" instead of making a copy into the Eigen
  // QR object and a second copy back out, but Eigen does not make it easy.
  // https://eigen.tuxfamily.org/dox/HouseholderQR_8h_source.html Line 379
  // HouseholderQR<MatrixType>::computeInPlace()
  {
    using MatrixType = vesta_core::MatrixXd;
    using HCoeffsType = Eigen::internal::plain_diag_type<MatrixType>::type;
    using RowVectorType = Eigen::internal::plain_row_type<MatrixType>::type;
    auto rows = Ab.rows();
    auto cols = Ab.cols();
    auto size = std::min(rows, cols);
    auto hCoeffs = HCoeffsType(size);
    auto temp = RowVectorType(cols);
    Eigen::internal::householder_qr_inplace_blocked<MatrixType, HCoeffsType>::run(Ab, hCoeffs, 48, temp.data());
    Ab.triangularView<Eigen::StrictlyLower>().setZero();  // Zero out the below-diagonal elements
  }

  // Extract the marginal term from R (now stored in Ab)
  // The first row block is the conditional term for the marginalized variable:
  // P(x | y, z, ...) The remaining rows are the marginal on the remaining
  // variables: P(y, z, ...)
  auto min_row = column_offsets[1];
  // However, depending on the input, not all rows may be usable.
  auto max_row = std::min(Ab.rows(), Ab.cols() - 1);  // -1 for the included b vector
  auto marginal_rows = max_row - min_row;
  auto marginal_term = LinearTerm();
  if (marginal_rows > 0)
  {
    auto variable_count = dense_to_index.size() - 1;
    marginal_term.variables.reserve(variable_count);
    marginal_term.A.reserve(variable_count);
    for (size_t dense = 1ul; dense < dense_to_index.size(); ++dense)  // Skipping the marginalized variable
    {
      auto index = dense_to_index[dense];
      marginal_term.variables.push_back(index);
      vesta_core::MatrixXd A = vesta_core::MatrixXd::Zero(marginal_rows, index_to_cols[index]);
      auto column_offset = column_offsets[dense];
      for (int row = 0; row < A.rows(); ++row)
      {
        for (int col = 0; col < A.cols(); ++col)
        {
          A(row, col) = Ab(min_row + row, column_offset + col);
        }
      }
      marginal_term.A.push_back(std::move(A));
    }
    marginal_term.b = vesta_core::VectorXd::Zero(marginal_rows);
    auto column_offset = column_offsets.back();
    for (int row = 0; row < marginal_term.b.rows(); ++row)
    {
      marginal_term.b(row) = Ab(min_row + row, column_offset);
    }
  }
  return marginal_term;
}

MarginalConstraint::SharedPtr createMarginalConstraint(const std::string& source, const LinearTerm& linear_term,
                                                       const vesta_core::Graph& graph,
                                                       const UuidOrdering& elimination_order)
{
  auto index_to_variable = [&graph, &elimination_order](const unsigned int index) -> const vesta_core::Variable& {
    return graph.getVariable(elimination_order.at(index));
  };

  return MarginalConstraint::make_shared(
      source, boost::make_transform_iterator(linear_term.variables.begin(), index_to_variable),
      boost::make_transform_iterator(linear_term.variables.end(), index_to_variable), linear_term.A.begin(),
      linear_term.A.end(), linear_term.b);
}

ClassifiedVariables classifyVariables(const std::vector<vesta_core::UUID>& marginalized_variables,
                                      const vesta_core::Graph& graph)
{
  ClassifiedVariables result;
  for (const auto& uuid : marginalized_variables)
  {
    const auto& variable = graph.getVariable(uuid);
    if (dynamic_cast<const vesta_variables::Stamped*>(&variable) != nullptr)
    {
      result.stamped.push_back(uuid);
    }
    else
    {
      result.non_stamped.push_back(uuid);
    }
  }
  return result;
}

UuidOrdering buildSchurEliminationOrder(const std::vector<vesta_core::UUID>& non_stamped_vars,
                                        const std::vector<vesta_core::UUID>& stamped_vars)
{
  auto variable_order = UuidOrdering();
  for (const auto& uuid : non_stamped_vars)
  {
    variable_order.push_back(uuid);
  }
  for (const auto& uuid : stamped_vars)
  {
    variable_order.push_back(uuid);
  }
  return variable_order;
}

LinearizationResult linearizeAndBucket(const std::string& source,
                                       const std::vector<vesta_core::UUID>& marginalized_variables,
                                       size_t num_marginalized, const vesta_core::Graph& graph,
                                       UuidOrdering variable_order, bool use_fej)
{
  LinearizationResult result;
  result.variable_order = std::move(variable_order);

  for (const auto& uuid : marginalized_variables)
  {
    result.transaction.removeVariable(uuid);
  }

  auto used_constraints = std::unordered_set<vesta_core::UUID, vesta_core::uuid::hash>();
  result.linear_terms.resize(result.variable_order.size());

  for (size_t i = 0ul; i < num_marginalized; ++i)
  {
    const auto constraints = graph.getConnectedConstraints(result.variable_order[i]);
    for (const auto& constraint : constraints)
    {
      if (used_constraints.find(constraint.uuid()) == used_constraints.end())
      {
        used_constraints.insert(constraint.uuid());
        for (const auto& variable_uuid : constraint.variables())
        {
          result.variable_order.push_back(variable_uuid);
        }
        auto lt = linearize(constraint, graph, result.variable_order, use_fej);
        auto min_var = *std::min_element(lt.variables.begin(), lt.variables.end());
        result.linear_terms[min_var].push_back(std::move(lt));
        result.transaction.removeConstraint(constraint.uuid());
      }
    }
  }

  result.linear_terms.resize(result.variable_order.size());
  return result;
}

std::vector<LinearTerm> collectSchurTerms(std::vector<std::vector<LinearTerm>>& linear_terms, size_t num_non_stamped)
{
  std::vector<LinearTerm> schur_terms;
  for (size_t i = 0; i < num_non_stamped; ++i)
  {
    for (auto& lt : linear_terms[i])
    {
      schur_terms.push_back(std::move(lt));
    }
    linear_terms[i].clear();
  }
  return schur_terms;
}

std::optional<SchurResult> computeSchurComplement(const std::vector<LinearTerm>& schur_terms, size_t num_non_stamped,
                                                  const vesta_core::Graph& graph, const UuidOrdering& variable_order)
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

  if (total_other_dim == 0)
  {
    return std::nullopt;
  }

  // Build H_oo and eta_o
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

  // Accumulate information matrix contributions
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
        auto it = other_offsets.find(vi);
        if (it != other_offsets.end())
        {
          eta_o.segment(it->second, atb.size()) += atb;
        }
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
          auto it = other_offsets.find(vj);
          if (it != other_offsets.end())
          {
            ns_data[vi].h_ko.block(0, it->second, block.rows(), block.cols()) += block;
          }
        }
        else if (!vi_ns && !vj_ns)
        {
          auto it_i = other_offsets.find(vi);
          auto it_j = other_offsets.find(vj);
          if (it_i != other_offsets.end() && it_j != other_offsets.end())
          {
            H_oo.block(it_i->second, it_j->second, block.rows(), block.cols()) += block;
          }
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
      LOG(WARNING) << "computeSchurComplement: Cholesky of H_kk failed for variable " << variable_order[idx]
                   << ". Skipping.";
      continue;
    }
    vesta_core::MatrixXd Z = llt.solve(data.h_ko);
    H_oo -= data.h_ko.transpose() * Z;
    eta_o -= data.h_ko.transpose() * llt.solve(data.eta_k);
  }

  // Symmetrize H_oo
  H_oo = (H_oo + H_oo.transpose()) * 0.5;

  SchurResult result;
  result.H_oo = std::move(H_oo);
  result.eta_o = std::move(eta_o);
  result.other_indices = std::move(other_indices);
  result.other_offsets = std::move(other_offsets);
  result.var_tangent_sizes = std::move(var_tangent_sizes);
  result.total_other_dim = total_other_dim;
  return result;
}

void marginalizeStampedVariables(std::vector<std::vector<LinearTerm>>& linear_terms, size_t num_non_stamped,
                                 size_t num_marginalized)
{
  for (size_t i = num_non_stamped; i < num_marginalized; ++i)
  {
    auto linear_marginal = marginalizeNext(linear_terms[i]);
    if (!linear_marginal.variables.empty())
    {
      auto lowest = linear_marginal.variables.front();
      linear_terms[lowest].push_back(std::move(linear_marginal));
    }
  }
}

void emitRemainingConstraints(const std::string& source, const std::vector<std::vector<LinearTerm>>& linear_terms,
                              size_t start_index, const vesta_core::Graph& graph, const UuidOrdering& variable_order,
                              vesta_core::Transaction& transaction)
{
  for (size_t i = start_index; i < linear_terms.size(); ++i)
  {
    for (const auto& linear_term : linear_terms[i])
    {
      auto marginal_constraint = createMarginalConstraint(source, linear_term, graph, variable_order);
      transaction.addConstraint(std::move(marginal_constraint));
    }
  }
}

LinearTerm createSingleVariableTerm(unsigned int var_idx, const vesta_core::MatrixXd& H_ii,
                                    const vesta_core::VectorXd& eta_i)
{
  constexpr double EIGENVALUE_REL_THRESHOLD = 1e-10;
  constexpr double EIGENVALUE_ABS_THRESHOLD = 1e-14;

  Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig(H_ii);
  if (eig.info() != Eigen::Success)
  {
    return {};
  }

  const auto& eigenvalues = eig.eigenvalues();
  const auto& eigenvectors = eig.eigenvectors();

  double max_eigenvalue = eigenvalues.maxCoeff();
  double threshold = max_eigenvalue * EIGENVALUE_REL_THRESHOLD;
  if (threshold < EIGENVALUE_ABS_THRESHOLD)
  {
    threshold = EIGENVALUE_ABS_THRESHOLD;
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

  LinearTerm term;
  term.variables = { var_idx };
  term.A = { J };
  term.b = b;
  return term;
}

}  // namespace detail

}  // namespace vesta_constraints
