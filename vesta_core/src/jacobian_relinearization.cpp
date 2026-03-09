/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026, Locus Robotics
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
#include <vesta_core/jacobian_relinearization.h>

#include <glog/logging.h>

#include <algorithm>
#include <cmath>

namespace vesta_core
{

// --- JacobianRelinearizationController ---

JacobianRelinearizationController::JacobianRelinearizationController(JacobianPolicy policy, int period,
                                                                     double threshold)
  : policy_(policy), period_(period), threshold_(threshold)
{
  CHECK_GT(period_, 0) << "Jacobian relinearization period must be positive.";
  CHECK_GE(threshold_, 0.0) << "Jacobian relinearization threshold must be non-negative.";
}

void JacobianRelinearizationController::prepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point)
{
  if (!evaluate_jacobians)
  {
    // Ceres is only requesting residuals (e.g., to check cost reduction).
    // No Jacobian decision needed — set recompute to false since no Jacobians
    // will be requested.
    recompute_jacobians_.store(false, std::memory_order_release);
    return;
  }

  switch (policy_)
  {
    case JacobianPolicy::kDefault:
      recompute_jacobians_.store(true, std::memory_order_release);
      break;

    case JacobianPolicy::kEveryN:
      if (new_evaluation_point)
      {
        ++jacobian_eval_count_;
      }
      // Recompute on iterations 1, 1+period, 1+2*period, etc.
      recompute_jacobians_.store(((jacobian_eval_count_ - 1) % period_) == 0, std::memory_order_release);
      break;

    case JacobianPolicy::kFirstEstimate:
      if (!first_jacobian_done_)
      {
        recompute_jacobians_.store(true, std::memory_order_release);
        first_jacobian_done_ = true;
      }
      else
      {
        recompute_jacobians_.store(false, std::memory_order_release);
      }
      break;

    case JacobianPolicy::kAdaptive:
      // Don't force global recompute — each CachedJacobianCostFunction decides
      // locally by comparing current parameters to the values at which Jacobians
      // were last computed.
      recompute_jacobians_.store(false, std::memory_order_release);
      break;
  }
}

bool JacobianRelinearizationController::shouldRecomputeJacobians() const
{
  return recompute_jacobians_.load(std::memory_order_acquire);
}

void JacobianRelinearizationController::resetForNewOptimization()
{
  if (policy_ == JacobianPolicy::kEveryN)
  {
    jacobian_eval_count_ = 0;
  }
  // For kFirstEstimate, do NOT reset — Jacobians remain frozen across
  // optimize() calls. For kDefault, nothing to reset.
}

// --- JacobianEvaluationCallback ---

JacobianEvaluationCallback::JacobianEvaluationCallback(std::shared_ptr<JacobianRelinearizationController> controller)
  : controller_(std::move(controller))
{
}

void JacobianEvaluationCallback::PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point)
{
  controller_->prepareForEvaluation(evaluate_jacobians, new_evaluation_point);
}

// --- CachedJacobianCostFunction ---

CachedJacobianCostFunction::CachedJacobianCostFunction(ceres::CostFunction* inner,
                                                       std::shared_ptr<JacobianRelinearizationController> controller)
  : inner_(inner), controller_(std::move(controller))
{
  // Copy the parameter block sizes and residual count from the inner cost
  // function
  *mutable_parameter_block_sizes() = inner_->parameter_block_sizes();
  set_num_residuals(inner_->num_residuals());
}

bool CachedJacobianCostFunction::Evaluate(const double* const* parameters, double* residuals, double** jacobians) const
{
  // If no Jacobians are requested, just evaluate residuals
  if (!jacobians)
  {
    return inner_->Evaluate(parameters, residuals, nullptr);
  }

  bool recompute = controller_->shouldRecomputeJacobians() || !has_cached_jacobians_;

  // For kAdaptive, check if parameters changed significantly since last
  // linearization
  if (!recompute && controller_->policy() == JacobianPolicy::kAdaptive && has_cached_jacobians_)
  {
    recompute = parametersChangedSignificantly(parameters);
  }

  if (recompute)
  {
    // Compute fresh residuals and Jacobians
    if (!inner_->Evaluate(parameters, residuals, jacobians))
    {
      return false;
    }

    // Cache the Jacobians and parameter values
    const auto& block_sizes = parameter_block_sizes();
    const int num_res = num_residuals();
    cached_jacobians_.resize(block_sizes.size());
    cached_parameters_.resize(block_sizes.size());
    for (size_t i = 0; i < block_sizes.size(); ++i)
    {
      if (jacobians[i])
      {
        const int jac_size = num_res * block_sizes[i];
        cached_jacobians_[i].assign(jacobians[i], jacobians[i] + jac_size);
      }
      else
      {
        cached_jacobians_[i].clear();
      }
      // Always cache parameter values for adaptive threshold checking
      cached_parameters_[i].assign(parameters[i], parameters[i] + block_sizes[i]);
    }
    has_cached_jacobians_ = true;
  }
  else
  {
    // Compute fresh residuals only
    if (!inner_->Evaluate(parameters, residuals, nullptr))
    {
      return false;
    }

    // Copy cached Jacobians to output
    for (size_t i = 0; i < cached_jacobians_.size(); ++i)
    {
      if (jacobians[i] && !cached_jacobians_[i].empty())
      {
        std::copy(cached_jacobians_[i].begin(), cached_jacobians_[i].end(), jacobians[i]);
      }
    }
  }

  return true;
}

bool CachedJacobianCostFunction::parametersChangedSignificantly(const double* const* parameters) const
{
  const auto& block_sizes = parameter_block_sizes();
  const double threshold = controller_->threshold();
  const double threshold_sq = threshold * threshold;

  for (size_t i = 0; i < block_sizes.size(); ++i)
  {
    double sum_sq = 0.0;
    for (int j = 0; j < block_sizes[i]; ++j)
    {
      const double diff = parameters[i][j] - cached_parameters_[i][j];
      sum_sq += diff * diff;
    }
    if (sum_sq > threshold_sq)
    {
      return true;
    }
  }
  return false;
}

}  // namespace vesta_core
