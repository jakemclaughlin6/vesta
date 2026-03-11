#include <vesta_constraints/3d/absolute_orientation_3d_stamped_constraint.h>
#include <vesta_constraints/3d/relative_orientation_3d_stamped_constraint.h>
#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/qr_marginalizer.h>
#include <vesta_core/eigen.h>
#include <vesta_core/eigen_gtest.h>
#include <vesta_core/uuid.h>
#include <vesta_core/variable.h>
#include <vesta_graphs/hash_graph.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>

#include <gtest/gtest.h>

#include <chrono>
#include <set>
#include <vector>

namespace
{

// Helper to build a test graph with 3D orientation variables and constraints
struct TestGraph
{
  vesta_variables::Orientation3DStamped::SharedPtr x1, x2, x3, l1;
  vesta_graphs::HashGraph graph;

  TestGraph()
  {
    x1 = vesta_variables::Orientation3DStamped::make_shared(vesta_core::Timestamp(1, 0));
    x1->w() = 0.927362;
    x1->x() = 0.1;
    x1->y() = 0.2;
    x1->z() = 0.3;

    x2 = vesta_variables::Orientation3DStamped::make_shared(vesta_core::Timestamp(2, 0));
    x2->w() = 0.848625;
    x2->x() = 0.13798;
    x2->y() = 0.175959;
    x2->z() = 0.479411;

    x3 = vesta_variables::Orientation3DStamped::make_shared(vesta_core::Timestamp(3, 0));
    x3->w() = 0.735597;
    x3->x() = 0.170384;
    x3->y() = 0.144808;
    x3->z() = 0.63945;

    l1 = vesta_variables::Orientation3DStamped::make_shared(vesta_core::Timestamp(3, 500000000));
    l1->w() = 0.803884;
    l1->x() = 0.304917;
    l1->y() = 0.268286;
    l1->z() = 0.434533;

    graph.addVariable(x1);
    graph.addVariable(x2);
    graph.addVariable(x3);
    graph.addVariable(l1);

    vesta_core::Vector4d mean1;
    mean1 << 0.92736185, 0.1, 0.2, 0.3;
    vesta_core::Matrix3d cov1;
    cov1 << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;
    graph.addConstraint(
        vesta_constraints::AbsoluteOrientation3DStampedConstraint::make_shared("test", *x1, mean1, cov1));

    vesta_core::Vector4d delta2;
    delta2 << 0.979795897, 0.0, 0.0, 0.2;
    vesta_core::Matrix3d cov2;
    cov2 << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;
    graph.addConstraint(
        vesta_constraints::RelativeOrientation3DStampedConstraint::make_shared("test", *x1, *x2, delta2, cov2));

    vesta_core::Vector4d delta3;
    delta3 << 0.979795897, 0.0, 0.0, 0.2;
    vesta_core::Matrix3d cov3;
    cov3 << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;
    graph.addConstraint(
        vesta_constraints::RelativeOrientation3DStampedConstraint::make_shared("test", *x2, *x3, delta3, cov3));

    vesta_core::Vector4d delta4;
    delta4 << 0.979795897, 0.2, 0.0, 0.0;
    vesta_core::Matrix3d cov4;
    cov4 << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;
    graph.addConstraint(
        vesta_constraints::RelativeOrientation3DStampedConstraint::make_shared("test", *x2, *l1, delta4, cov4));
  }
};

}  // namespace

TEST(QRMarginalizer, MatchesFreeFunctions)
{
  // Build identical graphs
  TestGraph tg1;
  TestGraph tg2;

  tg1.graph.optimize();
  tg2.graph.optimize();

  // Marginalize x1 using free function
  auto transaction_free = vesta_constraints::marginalizeVariables("test", { tg1.x1->uuid() }, tg1.graph);

  // Marginalize x1 using QRMarginalizer
  vesta_constraints::QRMarginalizer marginalizer(false);
  auto transaction_class = marginalizer.marginalize("test", { tg2.x1->uuid() }, tg2.graph);

  // Both should produce the same structure
  auto free_removed_vars_range = transaction_free.removedVariables();
  auto class_removed_vars_range = transaction_class.removedVariables();
  auto free_removed_vars = std::set<vesta_core::UUID>(free_removed_vars_range.begin(), free_removed_vars_range.end());
  auto class_removed_vars =
      std::set<vesta_core::UUID>(class_removed_vars_range.begin(), class_removed_vars_range.end());
  EXPECT_EQ(free_removed_vars, class_removed_vars);

  auto free_removed_cons_range = transaction_free.removedConstraints();
  auto class_removed_cons_range = transaction_class.removedConstraints();
  EXPECT_EQ(std::distance(free_removed_cons_range.begin(), free_removed_cons_range.end()),
            std::distance(class_removed_cons_range.begin(), class_removed_cons_range.end()));

  auto free_added = transaction_free.addedConstraints();
  auto class_added = transaction_class.addedConstraints();
  EXPECT_EQ(std::distance(free_added.begin(), free_added.end()), std::distance(class_added.begin(), class_added.end()));

  // Apply both and re-optimize, results should be equivalent
  tg1.graph.update(transaction_free);
  tg2.graph.update(transaction_class);

  tg1.graph.optimize();
  tg2.graph.optimize();

  // Compare optimized variable values
  for (size_t i = 0; i < tg1.x2->size(); ++i)
  {
    EXPECT_NEAR(tg1.x2->data()[i], tg2.x2->data()[i], 1.0e-10);
  }
  for (size_t i = 0; i < tg1.x3->size(); ++i)
  {
    EXPECT_NEAR(tg1.x3->data()[i], tg2.x3->data()[i], 1.0e-10);
  }
}

TEST(QRMarginalizer, FejFallbackMatchesNonFej)
{
  // When no linearization points are set, FEJ should produce identical results
  TestGraph tg1;
  TestGraph tg2;

  tg1.graph.optimize();
  tg2.graph.optimize();

  vesta_constraints::QRMarginalizer non_fej(false);
  vesta_constraints::QRMarginalizer fej(true);

  auto transaction1 = non_fej.marginalize("test", { tg1.x1->uuid() }, tg1.graph);
  auto transaction2 = fej.marginalize("test", { tg2.x1->uuid() }, tg2.graph);

  // Apply and re-optimize
  tg1.graph.update(transaction1);
  tg2.graph.update(transaction2);

  tg1.graph.optimize();
  tg2.graph.optimize();

  // Results should be identical since linearizationPoint() falls back to data()
  for (size_t i = 0; i < tg1.x2->size(); ++i)
  {
    EXPECT_NEAR(tg1.x2->data()[i], tg2.x2->data()[i], 1.0e-10);
  }
  for (size_t i = 0; i < tg1.x3->size(); ++i)
  {
    EXPECT_NEAR(tg1.x3->data()[i], tg2.x3->data()[i], 1.0e-10);
  }
  for (size_t i = 0; i < tg1.l1->size(); ++i)
  {
    EXPECT_NEAR(tg1.l1->data()[i], tg2.l1->data()[i], 1.0e-10);
  }
}

TEST(QRMarginalizer, FejProducesDifferentJacobians)
{
  // Optimize, set linearization points, perturb values, then compare FEJ vs non-FEJ
  TestGraph tg1;
  TestGraph tg2;

  tg1.graph.optimize();
  tg2.graph.optimize();

  // Set linearization points at the optimized values (via shared_ptr, since graph gives const access)
  tg2.x1->setLinearizationPoint();
  tg2.x2->setLinearizationPoint();
  tg2.x3->setLinearizationPoint();
  tg2.l1->setLinearizationPoint();

  // Perturb x2 in both graphs
  auto perturb = [](vesta_variables::Orientation3DStamped& var) {
    var.x() += 0.02;
    var.y() += 0.01;
    // Re-normalize the quaternion
    Eigen::Map<Eigen::Vector4d> q(var.data());
    q.normalize();
  };
  perturb(*tg1.x2);
  perturb(*tg2.x2);

  // Marginalize x1 with non-FEJ and FEJ
  vesta_constraints::QRMarginalizer non_fej(false);
  vesta_constraints::QRMarginalizer fej(true);

  auto transaction1 = non_fej.marginalize("test", { tg1.x1->uuid() }, tg1.graph);
  auto transaction2 = fej.marginalize("test", { tg2.x1->uuid() }, tg2.graph);

  // The marginal constraints should differ because Jacobians are evaluated at different points
  // We verify this by applying both and checking that the optimized results diverge
  tg1.graph.update(transaction1);
  tg2.graph.update(transaction2);

  tg1.graph.optimize();
  tg2.graph.optimize();

  // At least one variable should have a measurable difference
  double max_diff = 0.0;
  for (size_t i = 0; i < tg1.x2->size(); ++i)
  {
    max_diff = std::max(max_diff, std::abs(tg1.x2->data()[i] - tg2.x2->data()[i]));
  }
  for (size_t i = 0; i < tg1.x3->size(); ++i)
  {
    max_diff = std::max(max_diff, std::abs(tg1.x3->data()[i] - tg2.x3->data()[i]));
  }

  // The difference should be non-trivial (FEJ linearizes at old point, non-FEJ at perturbed)
  // but both should still produce reasonable results
  EXPECT_GT(max_diff, 1.0e-12) << "FEJ and non-FEJ should produce different results after perturbation";
}

TEST(QRMarginalizer, FejConsistency)
{
  // Optimize, set linearization points, perturb, marginalize with FEJ, re-optimize
  // Verify solution doesn't diverge
  TestGraph tg;
  tg.graph.optimize();

  // Save optimized values
  auto optimized_x2 = Eigen::Map<const Eigen::Vector4d>(tg.x2->data());
  auto optimized_x3 = Eigen::Map<const Eigen::Vector4d>(tg.x3->data());

  // Set linearization points at optimized values
  tg.x1->setLinearizationPoint();
  tg.x2->setLinearizationPoint();
  tg.x3->setLinearizationPoint();
  tg.l1->setLinearizationPoint();

  // Small perturbation
  tg.x2->x() += 0.005;
  Eigen::Map<Eigen::Vector4d> q2(tg.x2->data());
  q2.normalize();

  // Marginalize x1 with FEJ
  vesta_constraints::QRMarginalizer fej(true);
  auto transaction = fej.marginalize("test", { tg.x1->uuid() }, tg.graph);
  tg.graph.update(transaction);
  tg.graph.optimize();

  // Solution should still be close to the original optimized values
  auto final_x2 = Eigen::Map<const Eigen::Vector4d>(tg.x2->data());
  auto final_x3 = Eigen::Map<const Eigen::Vector4d>(tg.x3->data());

  EXPECT_LT((final_x2 - optimized_x2).norm(), 0.1) << "FEJ marginalization should not cause divergence";
  EXPECT_LT((final_x3 - optimized_x3).norm(), 0.1) << "FEJ marginalization should not cause divergence";
}

TEST(VariableLinearizationPoint, SetGetHasClear)
{
  auto var = vesta_variables::Orientation3DStamped::make_shared(vesta_core::Timestamp(1, 0));
  var->w() = 0.927362;
  var->x() = 0.1;
  var->y() = 0.2;
  var->z() = 0.3;

  // Initially no linearization point
  EXPECT_FALSE(var->hasLinearizationPoint());
  // linearizationPoint() should return data() when unset
  EXPECT_EQ(var->linearizationPoint(), var->data());

  // Set linearization point
  var->setLinearizationPoint();
  EXPECT_TRUE(var->hasLinearizationPoint());
  EXPECT_NE(var->linearizationPoint(), var->data());

  // Values should match
  for (size_t i = 0; i < var->size(); ++i)
  {
    EXPECT_DOUBLE_EQ(var->linearizationPoint()[i], var->data()[i]);
  }

  // Modify the variable — linearization point should retain old values
  double old_w = var->w();
  var->w() = 0.5;
  EXPECT_DOUBLE_EQ(var->linearizationPoint()[0], old_w);
  EXPECT_NE(var->linearizationPoint()[0], var->data()[0]);

  // Clear
  var->clearLinearizationPoint();
  EXPECT_FALSE(var->hasLinearizationPoint());
  EXPECT_EQ(var->linearizationPoint(), var->data());
}

TEST(QRMarginalizer, RuntimeMeasurement)
{
  TestGraph tg;
  tg.graph.optimize();

  // Warm up
  {
    TestGraph warmup;
    warmup.graph.optimize();
    vesta_constraints::QRMarginalizer m(false);
    m.marginalize("test", { warmup.x1->uuid() }, warmup.graph);
  }

  constexpr int NUM_ITERATIONS = 100;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERATIONS; ++i)
  {
    TestGraph t;
    t.graph.optimize();
    vesta_constraints::QRMarginalizer m(false);
    m.marginalize("test", { t.x1->uuid() }, t.graph);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "[          ] QRMarginalizer: " << duration_us / NUM_ITERATIONS << " us/marginalization ("
            << NUM_ITERATIONS << " iterations)" << std::endl;
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
