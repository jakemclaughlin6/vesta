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
#include <vesta_core/constraint.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/jacobian_relinearization.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_core/variable.h>
#include <vesta_graphs/hash_graph.h>

#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include <cmath>
#include <memory>
#include <vector>

// ============================================================================
// Test cost functions
// ============================================================================

/**
 * @brief Simple linear cost function: r = x - target, J = 1
 */
class SimpleCostFunction : public ceres::CostFunction
{
public:
  explicit SimpleCostFunction(double target) : target_(target)
  {
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(1);
  }

  bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override
  {
    residuals[0] = parameters[0][0] - target_;
    if (jacobians && jacobians[0])
    {
      jacobians[0][0] = 1.0;
    }
    return true;
  }

private:
  double target_;
};

/**
 * @brief Quadratic cost function: r = x^2 - target, J = 2*x
 *
 * The Jacobian depends on the parameter value, making it useful for testing
 * whether Jacobians are cached vs. recomputed.
 */
class QuadraticCostFunction : public ceres::CostFunction
{
public:
  explicit QuadraticCostFunction(double target) : target_(target)
  {
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(1);
  }

  bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override
  {
    const double x = parameters[0][0];
    residuals[0] = x * x - target_;
    if (jacobians && jacobians[0])
    {
      jacobians[0][0] = 2.0 * x;
    }
    return true;
  }

private:
  double target_;
};

// ============================================================================
// Mock variable and constraint for integration tests
// ============================================================================

class TestVariable : public vesta_core::Variable
{
public:
  VESTA_VARIABLE_DEFINITIONS(TestVariable)

  TestVariable() : vesta_core::Variable(vesta_core::uuid::generate()), data_(0.0)
  {
  }

  size_t size() const override
  {
    return 1;
  }
  const double* data() const override
  {
    return &data_;
  }
  double* data() override
  {
    return &data_;
  }
  void print(std::ostream& /*stream*/) const override
  {
  }

private:
  double data_;

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Variable>(*this);
    archive & data_;
  }
};

BOOST_CLASS_EXPORT(TestVariable)

class TestConstraint : public vesta_core::Constraint
{
public:
  VESTA_CONSTRAINT_DEFINITIONS(TestConstraint)

  TestConstraint() = default;

  explicit TestConstraint(const vesta_core::UUID& var_uuid) : vesta_core::Constraint("test", { var_uuid })
  {
  }

  ceres::CostFunction* costFunction() const override
  {
    return new SimpleCostFunction(5.0);
  }

  void print(std::ostream& /*stream*/) const override
  {
  }

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Constraint>(*this);
  }
};

BOOST_CLASS_EXPORT(TestConstraint)

// ============================================================================
// 1. Controller Tests
// ============================================================================

TEST(JacobianRelinearizationController, DefaultPolicyAlwaysRecomputes)
{
  vesta_core::JacobianRelinearizationController controller(vesta_core::JacobianPolicy::kDefault);

  // Every call with evaluate_jacobians=true should signal recompute
  controller.prepareForEvaluation(true, true);
  EXPECT_TRUE(controller.shouldRecomputeJacobians());

  controller.prepareForEvaluation(true, true);
  EXPECT_TRUE(controller.shouldRecomputeJacobians());

  controller.prepareForEvaluation(true, true);
  EXPECT_TRUE(controller.shouldRecomputeJacobians());
}

TEST(JacobianRelinearizationController, FirstEstimateFreezes)
{
  vesta_core::JacobianRelinearizationController controller(vesta_core::JacobianPolicy::kFirstEstimate);

  // First evaluation: should recompute (initial linearization)
  controller.prepareForEvaluation(true, true);
  EXPECT_TRUE(controller.shouldRecomputeJacobians());

  // Second evaluation: should NOT recompute (frozen)
  controller.prepareForEvaluation(true, true);
  EXPECT_FALSE(controller.shouldRecomputeJacobians());

  // Third evaluation: still frozen
  controller.prepareForEvaluation(true, true);
  EXPECT_FALSE(controller.shouldRecomputeJacobians());

  // resetForNewOptimization should NOT unfreeze FEJ Jacobians
  controller.resetForNewOptimization();
  controller.prepareForEvaluation(true, true);
  EXPECT_FALSE(controller.shouldRecomputeJacobians());
}

TEST(JacobianRelinearizationController, EveryNRecomputes)
{
  const int period = 3;
  vesta_core::JacobianRelinearizationController controller(vesta_core::JacobianPolicy::kEveryN, period);

  // Iteration 1: always recompute first
  controller.prepareForEvaluation(true, true);
  EXPECT_TRUE(controller.shouldRecomputeJacobians());

  // Iteration 2: use cached
  controller.prepareForEvaluation(true, true);
  EXPECT_FALSE(controller.shouldRecomputeJacobians());

  // Iteration 3: use cached
  controller.prepareForEvaluation(true, true);
  EXPECT_FALSE(controller.shouldRecomputeJacobians());

  // Iteration 4: recompute (1 + period = 4, so count=4, 4%3==1)
  controller.prepareForEvaluation(true, true);
  EXPECT_TRUE(controller.shouldRecomputeJacobians());

  // After resetForNewOptimization, iteration counter resets
  controller.resetForNewOptimization();
  controller.prepareForEvaluation(true, true);
  EXPECT_TRUE(controller.shouldRecomputeJacobians());
}

TEST(JacobianRelinearizationController, EveryNWithPeriod1IsEquivalentToDefault)
{
  // period=1 means recompute every iteration (equivalent to kDefault)
  vesta_core::JacobianRelinearizationController controller(vesta_core::JacobianPolicy::kEveryN, 1);

  for (int i = 0; i < 5; ++i)
  {
    controller.prepareForEvaluation(true, true);
    EXPECT_TRUE(controller.shouldRecomputeJacobians()) << "Failed on iteration " << (i + 1);
  }
}

TEST(JacobianRelinearizationController, ResidualOnlyEvaluationReturnsFalse)
{
  vesta_core::JacobianRelinearizationController controller(vesta_core::JacobianPolicy::kDefault);

  // When evaluate_jacobians is false, no Jacobians should be recomputed
  controller.prepareForEvaluation(false, true);
  EXPECT_FALSE(controller.shouldRecomputeJacobians());
}

TEST(JacobianRelinearizationController, AdaptiveNeverForcesGlobalRecompute)
{
  vesta_core::JacobianRelinearizationController controller(vesta_core::JacobianPolicy::kAdaptive, 1, 0.1);

  // kAdaptive always sets shouldRecomputeJacobians to false (per-instance
  // decision)
  controller.prepareForEvaluation(true, true);
  EXPECT_FALSE(controller.shouldRecomputeJacobians());

  controller.prepareForEvaluation(true, true);
  EXPECT_FALSE(controller.shouldRecomputeJacobians());
}

TEST(JacobianRelinearizationController, PolicyAndPeriodAccessors)
{
  {
    vesta_core::JacobianRelinearizationController controller(vesta_core::JacobianPolicy::kDefault);
    EXPECT_EQ(vesta_core::JacobianPolicy::kDefault, controller.policy());
    EXPECT_EQ(1, controller.period());
    EXPECT_DOUBLE_EQ(0.01, controller.threshold());
  }
  {
    vesta_core::JacobianRelinearizationController controller(vesta_core::JacobianPolicy::kEveryN, 5);
    EXPECT_EQ(vesta_core::JacobianPolicy::kEveryN, controller.policy());
    EXPECT_EQ(5, controller.period());
  }
  {
    vesta_core::JacobianRelinearizationController controller(vesta_core::JacobianPolicy::kFirstEstimate);
    EXPECT_EQ(vesta_core::JacobianPolicy::kFirstEstimate, controller.policy());
  }
  {
    vesta_core::JacobianRelinearizationController controller(vesta_core::JacobianPolicy::kAdaptive, 1, 0.05);
    EXPECT_EQ(vesta_core::JacobianPolicy::kAdaptive, controller.policy());
    EXPECT_DOUBLE_EQ(0.05, controller.threshold());
  }
}

// ============================================================================
// 2. CachedJacobianCostFunction Tests
// ============================================================================

TEST(CachedJacobianCostFunction, WrapperPassesThrough)
{
  auto controller =
      std::make_shared<vesta_core::JacobianRelinearizationController>(vesta_core::JacobianPolicy::kDefault);

  // Wrap a SimpleCostFunction (target=3.0)
  vesta_core::CachedJacobianCostFunction wrapper(new SimpleCostFunction(3.0), controller);

  EXPECT_EQ(1, wrapper.num_residuals());
  ASSERT_EQ(1u, wrapper.parameter_block_sizes().size());
  EXPECT_EQ(1, wrapper.parameter_block_sizes()[0]);

  double x = 7.0;
  const double* parameters[] = { &x };
  double residual = 0.0;
  double jacobian = 0.0;
  double* jacobians[] = { &jacobian };

  // First evaluation
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, jacobians));
  EXPECT_DOUBLE_EQ(4.0, residual);  // 7 - 3
  EXPECT_DOUBLE_EQ(1.0, jacobian);

  // Change parameter and evaluate again — with kDefault, both should update
  x = 10.0;
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, jacobians));
  EXPECT_DOUBLE_EQ(7.0, residual);  // 10 - 3
  EXPECT_DOUBLE_EQ(1.0, jacobian);  // Still 1.0 (constant Jacobian)
}

TEST(CachedJacobianCostFunction, WrapperCachesJacobians)
{
  auto controller =
      std::make_shared<vesta_core::JacobianRelinearizationController>(vesta_core::JacobianPolicy::kFirstEstimate);

  // Use SimpleCostFunction (constant Jacobian = 1.0)
  vesta_core::CachedJacobianCostFunction wrapper(new SimpleCostFunction(3.0), controller);

  double x = 5.0;
  const double* parameters[] = { &x };
  double residual = 0.0;
  double jacobian = 0.0;
  double* jacobians[] = { &jacobian };

  // First evaluation: recompute
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, jacobians));
  EXPECT_DOUBLE_EQ(2.0, residual);  // 5 - 3
  EXPECT_DOUBLE_EQ(1.0, jacobian);

  // Second evaluation: FEJ frozen, but residual still updates
  x = 9.0;
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, jacobians));
  EXPECT_DOUBLE_EQ(6.0, residual);  // 9 - 3 (fresh residual)
  EXPECT_DOUBLE_EQ(1.0, jacobian);  // Still 1.0 (cached, same value as original)
}

TEST(CachedJacobianCostFunction, CacheReturnsStaleJacobian)
{
  auto controller =
      std::make_shared<vesta_core::JacobianRelinearizationController>(vesta_core::JacobianPolicy::kFirstEstimate);

  // QuadraticCostFunction: r = x^2 - target, J = 2*x
  vesta_core::CachedJacobianCostFunction wrapper(new QuadraticCostFunction(4.0), controller);

  double x = 3.0;
  const double* parameters[] = { &x };
  double residual = 0.0;
  double jacobian = 0.0;
  double* jacobians[] = { &jacobian };

  // First evaluation at x=3: r = 9 - 4 = 5, J = 2*3 = 6
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, jacobians));
  EXPECT_DOUBLE_EQ(5.0, residual);
  EXPECT_DOUBLE_EQ(6.0, jacobian);

  // Second evaluation at x=5: r = 25 - 4 = 21 (fresh), J = 6.0 (CACHED,
  // not 10.0)
  x = 5.0;
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, jacobians));
  EXPECT_DOUBLE_EQ(21.0, residual);  // Fresh residual
  EXPECT_DOUBLE_EQ(6.0, jacobian);   // Cached Jacobian from x=3
}

TEST(CachedJacobianCostFunction, NullJacobiansJustComputeResiduals)
{
  auto controller =
      std::make_shared<vesta_core::JacobianRelinearizationController>(vesta_core::JacobianPolicy::kDefault);

  vesta_core::CachedJacobianCostFunction wrapper(new SimpleCostFunction(2.0), controller);

  double x = 8.0;
  const double* parameters[] = { &x };
  double residual = 0.0;

  // Evaluate with nullptr Jacobians
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, nullptr));
  EXPECT_DOUBLE_EQ(6.0, residual);  // 8 - 2
}

// ============================================================================
// 3. JacobianEvaluationCallback Tests
// ============================================================================

TEST(JacobianEvaluationCallback, DelegatesToController)
{
  auto controller =
      std::make_shared<vesta_core::JacobianRelinearizationController>(vesta_core::JacobianPolicy::kFirstEstimate);

  vesta_core::JacobianEvaluationCallback callback(controller);

  // First call via callback should trigger recompute
  callback.PrepareForEvaluation(true, true);
  EXPECT_TRUE(controller->shouldRecomputeJacobians());

  // Second call: frozen
  callback.PrepareForEvaluation(true, true);
  EXPECT_FALSE(controller->shouldRecomputeJacobians());
}

// ============================================================================
// 4. Integration Tests with ceres::Problem
// ============================================================================

TEST(JacobianRelinearizationIntegration, DefaultPolicyConverges)
{
  auto controller =
      std::make_shared<vesta_core::JacobianRelinearizationController>(vesta_core::JacobianPolicy::kDefault);

  auto callback = std::make_unique<vesta_core::JacobianEvaluationCallback>(controller);

  ceres::Problem::Options problem_options;
  problem_options.evaluation_callback = callback.get();
  ceres::Problem problem(problem_options);

  // Variable: single scalar, initial value = 0
  double x = 0.0;
  problem.AddParameterBlock(&x, 1);

  // Constraint: r = x - 5 (should converge to x=5)
  auto* cost = new vesta_core::CachedJacobianCostFunction(new SimpleCostFunction(5.0), controller);
  problem.AddResidualBlock(cost, nullptr, &x);

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  EXPECT_NEAR(5.0, x, 1e-6);
  EXPECT_TRUE(summary.IsSolutionUsable());
}

TEST(JacobianRelinearizationIntegration, FEJPolicyProducesResult)
{
  auto controller =
      std::make_shared<vesta_core::JacobianRelinearizationController>(vesta_core::JacobianPolicy::kFirstEstimate);

  auto callback = std::make_unique<vesta_core::JacobianEvaluationCallback>(controller);

  ceres::Problem::Options problem_options;
  problem_options.evaluation_callback = callback.get();
  ceres::Problem problem(problem_options);

  // Variable: single scalar, initial value = 0
  double x = 0.0;
  problem.AddParameterBlock(&x, 1);

  // For a linear cost function (constant Jacobian), FEJ should converge
  // identically
  auto* cost = new vesta_core::CachedJacobianCostFunction(new SimpleCostFunction(5.0), controller);
  problem.AddResidualBlock(cost, nullptr, &x);

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  // Linear cost function: FEJ should give the same result as default
  EXPECT_NEAR(5.0, x, 1e-6);
  EXPECT_TRUE(summary.IsSolutionUsable());
}

TEST(JacobianRelinearizationIntegration, EveryNPolicyConverges)
{
  auto controller = std::make_shared<vesta_core::JacobianRelinearizationController>(vesta_core::JacobianPolicy::kEveryN,
                                                                                    /*period=*/2);

  auto callback = std::make_unique<vesta_core::JacobianEvaluationCallback>(controller);

  ceres::Problem::Options problem_options;
  problem_options.evaluation_callback = callback.get();
  ceres::Problem problem(problem_options);

  // Variable: single scalar, initial value = 0
  double x = 0.0;
  problem.AddParameterBlock(&x, 1);

  auto* cost = new vesta_core::CachedJacobianCostFunction(new SimpleCostFunction(5.0), controller);
  problem.AddResidualBlock(cost, nullptr, &x);

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  EXPECT_NEAR(5.0, x, 1e-6);
  EXPECT_TRUE(summary.IsSolutionUsable());
}

TEST(JacobianRelinearizationIntegration, FEJQuadraticStillConverges)
{
  // With a nonlinear cost function, FEJ freezes the Jacobian at the first
  // linearization point. For a scalar quadratic, the optimization may take more
  // iterations but should still converge because the Gauss-Newton direction
  // with a stale Jacobian still reduces the cost for a well-conditioned problem
  // near the optimum.
  auto controller =
      std::make_shared<vesta_core::JacobianRelinearizationController>(vesta_core::JacobianPolicy::kFirstEstimate);

  auto callback = std::make_unique<vesta_core::JacobianEvaluationCallback>(controller);

  ceres::Problem::Options problem_options;
  problem_options.evaluation_callback = callback.get();
  ceres::Problem problem(problem_options);

  // Variable: initial guess close to optimum to ensure FEJ convergence
  double x = 1.5;  // Optimum is x=2.0 for target=4.0
  problem.AddParameterBlock(&x, 1);

  auto* cost = new vesta_core::CachedJacobianCostFunction(new QuadraticCostFunction(4.0), controller);
  problem.AddResidualBlock(cost, nullptr, &x);

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 100;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  // With FEJ and a close initial guess, should get near the optimum
  EXPECT_NEAR(2.0, x, 0.5);
  EXPECT_TRUE(summary.IsSolutionUsable());
}

// ============================================================================
// 5. Integration Tests with HashGraph
// ============================================================================

TEST(JacobianRelinearizationHashGraph, DefaultPolicyConverges)
{
  vesta_graphs::HashGraph graph;

  auto var = TestVariable::make_shared();
  *var->data() = 0.0;
  graph.addVariable(var);

  auto constraint = TestConstraint::make_shared(var->uuid());
  graph.addConstraint(constraint);

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 50;
  auto summary = graph.optimize(solver_options);

  EXPECT_TRUE(summary.IsSolutionUsable());
  EXPECT_NEAR(5.0, *var->data(), 1e-6);
}

TEST(JacobianRelinearizationHashGraph, FEJPolicyConverges)
{
  vesta_graphs::HashGraphParams params;
  params.jacobian_policy = vesta_core::JacobianPolicy::kFirstEstimate;
  vesta_graphs::HashGraph graph(params);

  auto var = TestVariable::make_shared();
  *var->data() = 0.0;
  graph.addVariable(var);

  auto constraint = TestConstraint::make_shared(var->uuid());
  graph.addConstraint(constraint);

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 50;
  auto summary = graph.optimize(solver_options);

  // Linear cost function: FEJ has no effect since Jacobian is constant
  // (always 1.0)
  EXPECT_TRUE(summary.IsSolutionUsable());
  EXPECT_NEAR(5.0, *var->data(), 1e-6);
}

TEST(JacobianRelinearizationHashGraph, EveryNPolicyConverges)
{
  vesta_graphs::HashGraphParams params;
  params.jacobian_policy = vesta_core::JacobianPolicy::kEveryN;
  params.jacobian_relinearization_period = 3;
  vesta_graphs::HashGraph graph(params);

  auto var = TestVariable::make_shared();
  *var->data() = 0.0;
  graph.addVariable(var);

  auto constraint = TestConstraint::make_shared(var->uuid());
  graph.addConstraint(constraint);

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 50;
  auto summary = graph.optimize(solver_options);

  EXPECT_TRUE(summary.IsSolutionUsable());
  EXPECT_NEAR(5.0, *var->data(), 1e-6);
}

// ============================================================================
// 6. Adaptive Policy Tests
// ============================================================================

TEST(CachedJacobianCostFunction, AdaptiveRecomputesOnLargeChange)
{
  auto controller = std::make_shared<vesta_core::JacobianRelinearizationController>(
      vesta_core::JacobianPolicy::kAdaptive, 1, /*threshold=*/0.5);

  // QuadraticCostFunction: r = x^2 - target, J = 2*x
  vesta_core::CachedJacobianCostFunction wrapper(new QuadraticCostFunction(4.0), controller);

  double x = 3.0;
  const double* parameters[] = { &x };
  double residual = 0.0;
  double jacobian = 0.0;
  double* jacobians[] = { &jacobian };

  // First evaluation at x=3: J = 2*3 = 6 (always recomputes when no cache)
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, jacobians));
  EXPECT_DOUBLE_EQ(6.0, jacobian);

  // Move x far beyond threshold (|5-3| = 2 > 0.5): should recompute
  x = 5.0;
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, jacobians));
  EXPECT_DOUBLE_EQ(10.0, jacobian);  // Fresh J = 2*5 = 10
}

TEST(CachedJacobianCostFunction, AdaptiveReturnsCachedOnSmallChange)
{
  auto controller = std::make_shared<vesta_core::JacobianRelinearizationController>(
      vesta_core::JacobianPolicy::kAdaptive, 1, /*threshold=*/1.0);

  // QuadraticCostFunction: r = x^2 - target, J = 2*x
  vesta_core::CachedJacobianCostFunction wrapper(new QuadraticCostFunction(4.0), controller);

  double x = 3.0;
  const double* parameters[] = { &x };
  double residual = 0.0;
  double jacobian = 0.0;
  double* jacobians[] = { &jacobian };

  // First evaluation at x=3: J = 6
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, jacobians));
  EXPECT_DOUBLE_EQ(6.0, jacobian);

  // Small change: |3.5 - 3| = 0.5 < 1.0 threshold: should return CACHED
  // Jacobian
  x = 3.5;
  controller->prepareForEvaluation(true, true);
  ASSERT_TRUE(wrapper.Evaluate(parameters, &residual, jacobians));
  EXPECT_DOUBLE_EQ(8.25, residual);  // Fresh residual: 3.5^2 - 4 = 8.25
  EXPECT_DOUBLE_EQ(6.0, jacobian);   // Cached Jacobian from x=3 (not 7.0)
}

TEST(CachedJacobianCostFunction, AdaptiveUpdatesSnapshotAfterRecompute)
{
  auto controller = std::make_shared<vesta_core::JacobianRelinearizationController>(
      vesta_core::JacobianPolicy::kAdaptive, 1, /*threshold=*/0.5);

  vesta_core::CachedJacobianCostFunction wrapper(new QuadraticCostFunction(4.0), controller);

  double x = 1.0;
  const double* parameters[] = { &x };
  double residual = 0.0;
  double jacobian = 0.0;
  double* jacobians[] = { &jacobian };

  // First evaluation at x=1: J=2
  controller->prepareForEvaluation(true, true);
  wrapper.Evaluate(parameters, &residual, jacobians);
  EXPECT_DOUBLE_EQ(2.0, jacobian);

  // Move to x=2.0 (change=1.0 > 0.5): recompute, J=4, snapshot updated to x=2.0
  x = 2.0;
  controller->prepareForEvaluation(true, true);
  wrapper.Evaluate(parameters, &residual, jacobians);
  EXPECT_DOUBLE_EQ(4.0, jacobian);

  // Move to x=2.3 (change from 2.0 = 0.3 < 0.5): cached J=4 returned
  x = 2.3;
  controller->prepareForEvaluation(true, true);
  wrapper.Evaluate(parameters, &residual, jacobians);
  EXPECT_DOUBLE_EQ(4.0, jacobian);  // Still cached from x=2.0

  // Move to x=2.8 (change from 2.0 = 0.8 > 0.5): recompute, J=5.6
  x = 2.8;
  controller->prepareForEvaluation(true, true);
  wrapper.Evaluate(parameters, &residual, jacobians);
  EXPECT_DOUBLE_EQ(5.6, jacobian);  // Fresh J = 2*2.8
}

TEST(JacobianRelinearizationIntegration, AdaptivePolicyConverges)
{
  auto controller = std::make_shared<vesta_core::JacobianRelinearizationController>(
      vesta_core::JacobianPolicy::kAdaptive, 1, /*threshold=*/0.01);

  auto callback = std::make_unique<vesta_core::JacobianEvaluationCallback>(controller);

  ceres::Problem::Options problem_options;
  problem_options.evaluation_callback = callback.get();
  ceres::Problem problem(problem_options);

  double x = 0.0;
  problem.AddParameterBlock(&x, 1);

  auto* cost = new vesta_core::CachedJacobianCostFunction(new SimpleCostFunction(5.0), controller);
  problem.AddResidualBlock(cost, nullptr, &x);

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  EXPECT_NEAR(5.0, x, 1e-6);
  EXPECT_TRUE(summary.IsSolutionUsable());
}

TEST(JacobianRelinearizationHashGraph, AdaptivePolicyConverges)
{
  vesta_graphs::HashGraphParams params;
  params.jacobian_policy = vesta_core::JacobianPolicy::kAdaptive;
  params.jacobian_relinearization_threshold = 0.01;
  vesta_graphs::HashGraph graph(params);

  auto var = TestVariable::make_shared();
  *var->data() = 0.0;
  graph.addVariable(var);

  auto constraint = TestConstraint::make_shared(var->uuid());
  graph.addConstraint(constraint);

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 50;
  auto summary = graph.optimize(solver_options);

  EXPECT_TRUE(summary.IsSolutionUsable());
  EXPECT_NEAR(5.0, *var->data(), 1e-6);
}
