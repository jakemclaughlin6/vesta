#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/qr_marginalizer.h>
#include <vesta_constraints/common/schur_marginalizer.h>
#include <vesta_core/eigen.h>
#include <vesta_core/graph.h>
#include <vesta_core/uuid.h>

#include <glog/logging.h>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace vesta_constraints
{

SchurMarginalizer::SchurMarginalizer(bool use_fej) : use_fej_(use_fej)
{
}

vesta_core::Transaction SchurMarginalizer::marginalize(const std::string& source,
                                                       const std::vector<vesta_core::UUID>& marginalized_variables,
                                                       const vesta_core::Graph& graph)
{
  // 1. Classify variables
  auto classified = detail::classifyVariables(marginalized_variables, graph);

  // 2. If no non-stamped, delegate entirely to QR
  if (classified.non_stamped.empty())
  {
    return QRMarginalizer(use_fej_).marginalize(source, marginalized_variables, graph);
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
      // ---- UNIQUE SCHUR LOGIC: Dense eigendecomposition of H_oo ----
      // Recover square-root form via eigendecomposition.
      // H_oo after Schur complement is PSD (not necessarily PD), so we use
      // eigendecomposition to extract the square root, keeping only components
      // with positive eigenvalues.
      Eigen::SelfAdjointEigenSolver<vesta_core::MatrixXd> eig(schur_result->H_oo);
      if (eig.info() == Eigen::Success)
      {
        const auto& eigenvalues = eig.eigenvalues();
        const auto& eigenvectors = eig.eigenvectors();

        // Threshold for numerical zero (relative to largest eigenvalue)
        double max_eigenvalue = eigenvalues.maxCoeff();
        double threshold = max_eigenvalue * 1e-10;
        if (threshold < 1e-14)
        {
          threshold = 1e-14;
        }

        // Count positive eigenvalues to determine rank
        int rank = 0;
        for (int k = 0; k < eigenvalues.size(); ++k)
        {
          if (eigenvalues(k) > threshold)
          {
            ++rank;
          }
        }

        if (rank > 0)
        {
          // Build J = sqrt(D_pos) * V_pos^T (rank x total_other_dim)
          vesta_core::MatrixXd J(rank, schur_result->total_other_dim);
          int row = 0;
          for (int k = 0; k < eigenvalues.size(); ++k)
          {
            if (eigenvalues(k) > threshold)
            {
              J.row(row) = std::sqrt(eigenvalues(k)) * eigenvectors.col(k).transpose();
              ++row;
            }
          }

          // Recover b: J^T b = eta_o  =>  b = D^{-1/2} V^T eta_o
          vesta_core::VectorXd b(rank);
          row = 0;
          for (int k = 0; k < eigenvalues.size(); ++k)
          {
            if (eigenvalues(k) > threshold)
            {
              b(row) = eigenvectors.col(k).dot(schur_result->eta_o) / std::sqrt(eigenvalues(k));
              ++row;
            }
          }

          // Create a LinearTerm from J and b, split by variable
          detail::LinearTerm schur_lt;
          schur_lt.b = b;
          schur_lt.variables.reserve(schur_result->other_indices.size());
          schur_lt.A.reserve(schur_result->other_indices.size());
          for (auto idx : schur_result->other_indices)
          {
            int off = schur_result->other_offsets.at(idx);
            int cols = schur_result->var_tangent_sizes.at(idx);
            schur_lt.variables.push_back(idx);
            schur_lt.A.push_back(J.block(0, off, rank, cols));
          }

          // Place the Schur result into the appropriate bucket
          if (!schur_lt.variables.empty())
          {
            auto min_var = *std::min_element(schur_lt.variables.begin(), schur_lt.variables.end());
            lin_result.linear_terms[min_var].push_back(std::move(schur_lt));
          }
        }
      }
      else
      {
        // Eigendecomposition failed — fall back to QR for non-stamped variables
        LOG(ERROR) << "SchurMarginalizer: eigendecomposition of H_oo (" << schur_result->H_oo.rows() << "x"
                   << schur_result->H_oo.cols() << ") failed after Schur complement. Falling back to QR "
                   << "marginalization for " << num_non_stamped << " non-stamped variables.";
        for (auto& lt : schur_terms)
        {
          auto min_var = *std::min_element(lt.variables.begin(), lt.variables.end());
          lin_result.linear_terms[min_var].push_back(std::move(lt));
        }
        for (size_t i = 0; i < num_non_stamped; ++i)
        {
          auto linear_marginal = detail::marginalizeNext(lin_result.linear_terms[i]);
          if (!linear_marginal.variables.empty())
          {
            auto lowest = linear_marginal.variables.front();
            lin_result.linear_terms[lowest].push_back(std::move(linear_marginal));
          }
        }
      }
      // ---- END UNIQUE LOGIC ----
    }
  }

  // 7. QR phase for stamped variables
  detail::marginalizeStampedVariables(lin_result.linear_terms, num_non_stamped, num_marginalized);

  // 8. Emit remaining constraints
  detail::emitRemainingConstraints(source, lin_result.linear_terms, num_marginalized, graph,
                                   lin_result.variable_order, lin_result.transaction);

  return std::move(lin_result.transaction);
}

}  // namespace vesta_constraints
