#pragma once

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

#include <ceres/cost_function.h>
#include <ceres/evaluation_callback.h>

#include <atomic>
#include <memory>
#include <vector>

namespace vesta_core
{

/**
 * @brief Policy controlling how often Jacobians are recomputed during optimization.
 */
enum class JacobianPolicy
{
  kDefault,         ///< Recompute every iteration (standard Ceres behavior, zero overhead)
  kEveryN,          ///< Recompute every N iterations, use cached Jacobians otherwise
  kFirstEstimate,   ///< Freeze Jacobians at first linearization (full FEJ)
  kAdaptive         ///< Relinearize only when parameter values change beyond a threshold
};

/**
 * @brief Controls when cost functions should recompute vs. use cached Jacobians.
 *
 * Shared between JacobianEvaluationCallback (which updates state) and
 * CachedJacobianCostFunction instances (which read state during parallel evaluation).
 */
class JacobianRelinearizationController
{
public:
  /**
   * @brief Constructor
   *
   * @param[in] policy The relinearization policy
   * @param[in] period Recompute period for kEveryN (ignored for other policies)
   * @param[in] threshold Change threshold for kAdaptive — relinearize when any parameter block's
   *            L2 norm of change exceeds this value (ignored for other policies)
   */
  explicit JacobianRelinearizationController(JacobianPolicy policy, int period = 1,
                                             double threshold = 0.01);

  /**
   * @brief Called by EvaluationCallback::PrepareForEvaluation before each evaluation batch.
   *
   * Updates the internal state to determine whether the upcoming Evaluate() calls should
   * recompute or use cached Jacobians. This method is called single-threaded by Ceres.
   *
   * @param[in] evaluate_jacobians Whether the upcoming evaluation will request Jacobians
   * @param[in] new_evaluation_point Whether the parameter values have changed since last call
   */
  void prepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point);

  /**
   * @brief Returns whether cost functions should recompute Jacobians in the current evaluation.
   *
   * Thread-safe (atomic read). Called from CachedJacobianCostFunction::Evaluate() which may
   * run in parallel across multiple threads.
   */
  bool shouldRecomputeJacobians() const;

  /**
   * @brief Reset iteration counter for a new optimize() call.
   *
   * For kEveryN, resets the iteration counter so the period restarts.
   * For kFirstEstimate, this is a no-op (Jacobians remain frozen across optimize calls).
   */
  void resetForNewOptimization();

  /**
   * @brief Returns the configured policy.
   */
  JacobianPolicy policy() const { return policy_; }

  /**
   * @brief Returns the configured period.
   */
  int period() const { return period_; }

  /**
   * @brief Returns the configured adaptive threshold.
   */
  double threshold() const { return threshold_; }

private:
  JacobianPolicy policy_;
  int period_;
  double threshold_;
  std::atomic<bool> recompute_jacobians_{true};
  int jacobian_eval_count_ = 0;
  bool first_jacobian_done_ = false;
};

/**
 * @brief Ceres EvaluationCallback that delegates to a JacobianRelinearizationController.
 *
 * Set on ceres::Problem::Options::evaluation_callback. Ceres calls PrepareForEvaluation()
 * once (single-threaded) before each batch of CostFunction::Evaluate() calls.
 */
class JacobianEvaluationCallback : public ceres::EvaluationCallback
{
public:
  explicit JacobianEvaluationCallback(
    std::shared_ptr<JacobianRelinearizationController> controller);

  void PrepareForEvaluation(bool evaluate_jacobians,
                            bool new_evaluation_point) override;

private:
  std::shared_ptr<JacobianRelinearizationController> controller_;
};

/**
 * @brief CostFunction wrapper that caches Jacobians and returns cached values when instructed.
 *
 * Always computes fresh residuals at the current parameter values. Jacobians are either
 * recomputed or returned from cache based on the controller's decision.
 *
 * This wrapper is transparent to the inner cost function — it works with any CostFunction
 * including AutoDiffCostFunction, NumericDiffCostFunction, and analytic cost functions.
 */
class CachedJacobianCostFunction : public ceres::CostFunction
{
public:
  /**
   * @brief Constructor
   *
   * @param[in] inner Takes ownership of the inner cost function
   * @param[in] controller Shared controller that signals cache/recompute decisions
   */
  CachedJacobianCostFunction(
    ceres::CostFunction* inner,
    std::shared_ptr<JacobianRelinearizationController> controller);

  ~CachedJacobianCostFunction() override = default;

  bool Evaluate(const double* const* parameters,
                double* residuals,
                double** jacobians) const override;

private:
  std::unique_ptr<ceres::CostFunction> inner_;
  std::shared_ptr<JacobianRelinearizationController> controller_;

  /**
   * @brief Check if any parameter block has changed more than the adaptive threshold.
   *
   * Computes L2 norm of the difference between current and cached parameter values for each block.
   * Returns true if any block's change exceeds the controller's threshold.
   */
  bool parametersChangedSignificantly(const double* const* parameters) const;

  // Thread-safety note: These mutable members are safe because each residual block in the
  // ceres::Problem gets its own CachedJacobianCostFunction instance. Ceres never shares a
  // CostFunction* across residual blocks, so parallel Evaluate() calls operate on distinct instances.
  mutable bool has_cached_jacobians_ = false;
  mutable std::vector<std::vector<double>> cached_jacobians_;
  mutable std::vector<std::vector<double>> cached_parameters_;
};

}  // namespace vesta_core

