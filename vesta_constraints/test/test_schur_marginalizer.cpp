#include <vesta_constraints/common/absolute_constraint.h>
#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/qr_marginalizer.h>
#include <vesta_constraints/common/relative_constraint.h>
#include <vesta_constraints/common/schur_marginalizer.h>
#include <vesta_core/constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/eigen_gtest.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_core/variable.h>
#include <vesta_graphs/hash_graph.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/point_3d_landmark.h>

#include <ceres/cost_function.h>
#include <ceres/sized_cost_function.h>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <set>
#include <string>
#include <vector>

namespace
{

/**
 * @brief A cost function connecting a Position3DStamped and a Point3DLandmark
 *
 * Residual: r = (landmark - position) - delta, weighted by sqrt_information
 */
class PoseLandmarkCostFunction : public ceres::SizedCostFunction<3, 3, 3>
{
public:
  PoseLandmarkCostFunction(const Eigen::Vector3d& delta, const Eigen::Matrix3d& sqrt_information)
    : delta_(delta), sqrt_information_(sqrt_information)
  {
  }

  bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<const Eigen::Vector3d> position(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> landmark(parameters[1]);
    Eigen::Map<Eigen::Vector3d> r(residuals);

    r = sqrt_information_ * (landmark - position - delta_);

    if (jacobians)
    {
      if (jacobians[0])
      {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
        J = -sqrt_information_;
      }
      if (jacobians[1])
      {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[1]);
        J = sqrt_information_;
      }
    }
    return true;
  }

private:
  Eigen::Vector3d delta_;
  Eigen::Matrix3d sqrt_information_;
};

/**
 * @brief A constraint connecting a Position3DStamped to a Point3DLandmark
 */
class PoseLandmarkConstraint : public vesta_core::Constraint
{
public:
  VESTA_CONSTRAINT_DEFINITIONS(PoseLandmarkConstraint);

  PoseLandmarkConstraint() = default;

  PoseLandmarkConstraint(const std::string& source, const vesta_variables::Position3DStamped& position,
                         const vesta_variables::Point3DLandmark& landmark, const Eigen::Vector3d& delta,
                         const Eigen::Matrix3d& covariance)
    : vesta_core::Constraint(source, {position.uuid(), landmark.uuid()})
    , delta_(delta)
    , sqrt_information_(covariance.inverse().llt().matrixL().transpose())
  {
  }

  void print(std::ostream& /*stream*/) const override
  {
  }

  ceres::CostFunction* costFunction() const override
  {
    return new PoseLandmarkCostFunction(delta_, sqrt_information_);
  }

private:
  Eigen::Vector3d delta_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d sqrt_information_{Eigen::Matrix3d::Identity()};

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Constraint>(*this);
  }
};

/**
 * @brief Build a test graph with poses (stamped) and landmarks (non-stamped)
 *
 * Graph structure:
 *   prior -> p1 -- odom --> p2
 *             |               |
 *             obs            obs
 *             v               v
 *            l1              l2
 *             ^
 *             |
 *         prior on l1
 */
struct MixedGraph
{
  vesta_variables::Position3DStamped::SharedPtr p1, p2;
  vesta_variables::Point3DLandmark::SharedPtr l1, l2;
  vesta_graphs::HashGraph graph;

  MixedGraph()
  {
    p1 = vesta_variables::Position3DStamped::make_shared(vesta_core::Timestamp(1, 0));
    p1->x() = 1.0;
    p1->y() = 0.0;
    p1->z() = 0.0;

    p2 = vesta_variables::Position3DStamped::make_shared(vesta_core::Timestamp(2, 0));
    p2->x() = 2.0;
    p2->y() = 0.0;
    p2->z() = 0.0;

    l1 = vesta_variables::Point3DLandmark::make_shared(uint64_t{1});
    l1->x() = 1.0;
    l1->y() = 1.0;
    l1->z() = 0.0;

    l2 = vesta_variables::Point3DLandmark::make_shared(uint64_t{2});
    l2->x() = 2.0;
    l2->y() = 1.0;
    l2->z() = 0.0;

    graph.addVariable(p1);
    graph.addVariable(p2);
    graph.addVariable(l1);
    graph.addVariable(l2);

    // Prior on p1
    Eigen::Vector3d mean_p1;
    mean_p1 << 1.0, 0.0, 0.0;
    Eigen::Matrix3d cov_p1 = Eigen::Matrix3d::Identity() * 0.1;
    graph.addConstraint(
        vesta_constraints::AbsoluteConstraint<vesta_variables::Position3DStamped>::make_shared("test", *p1, mean_p1, cov_p1));

    // Prior on l1
    Eigen::Vector3d mean_l1;
    mean_l1 << 1.0, 1.0, 0.0;
    Eigen::Matrix3d cov_l1 = Eigen::Matrix3d::Identity() * 0.5;
    graph.addConstraint(
        vesta_constraints::AbsoluteConstraint<vesta_variables::Point3DLandmark>::make_shared("test", *l1, mean_l1, cov_l1));

    // Odometry p1 -> p2
    Eigen::Vector3d odom_delta;
    odom_delta << 1.0, 0.0, 0.0;
    Eigen::Matrix3d cov_odom = Eigen::Matrix3d::Identity() * 0.2;
    graph.addConstraint(
        vesta_constraints::RelativeConstraint<vesta_variables::Position3DStamped>::make_shared("test", *p1, *p2, odom_delta, cov_odom));

    // Observation p1 -> l1
    Eigen::Vector3d obs_delta1;
    obs_delta1 << 0.0, 1.0, 0.0;
    Eigen::Matrix3d cov_obs = Eigen::Matrix3d::Identity() * 0.3;
    graph.addConstraint(PoseLandmarkConstraint::make_shared("test", *p1, *l1, obs_delta1, cov_obs));

    // Observation p2 -> l2
    Eigen::Vector3d obs_delta2;
    obs_delta2 << 0.0, 1.0, 0.0;
    graph.addConstraint(PoseLandmarkConstraint::make_shared("test", *p2, *l2, obs_delta2, cov_obs));
  }
};

}  // namespace

BOOST_CLASS_EXPORT(PoseLandmarkConstraint);

TEST(SchurMarginalizer, MatchesQR_NoNonStamped)
{
  // When all marginalized variables are stamped, SchurMarginalizer should
  // produce equivalent results to QRMarginalizer
  MixedGraph tg1;
  MixedGraph tg2;

  tg1.graph.optimize();
  tg2.graph.optimize();

  vesta_constraints::QRMarginalizer qr(false);
  vesta_constraints::SchurMarginalizer schur(false);

  auto txn_qr = qr.marginalize("test", {tg1.p1->uuid()}, tg1.graph);
  auto txn_schur = schur.marginalize("test", {tg2.p1->uuid()}, tg2.graph);

  tg1.graph.update(txn_qr);
  tg2.graph.update(txn_schur);

  tg1.graph.optimize();
  tg2.graph.optimize();

  // Compare variable values
  for (size_t i = 0; i < tg1.p2->size(); ++i)
  {
    EXPECT_NEAR(tg1.p2->data()[i], tg2.p2->data()[i], 1.0e-8);
  }
  for (size_t i = 0; i < tg1.l1->size(); ++i)
  {
    EXPECT_NEAR(tg1.l1->data()[i], tg2.l1->data()[i], 1.0e-8);
  }
  for (size_t i = 0; i < tg1.l2->size(); ++i)
  {
    EXPECT_NEAR(tg1.l2->data()[i], tg2.l2->data()[i], 1.0e-8);
  }
}

TEST(SchurMarginalizer, MatchesQR_MixedVariables)
{
  // Marginalize both a stamped variable (p1) and a non-stamped variable (l1)
  MixedGraph tg1;
  MixedGraph tg2;

  tg1.graph.optimize();
  tg2.graph.optimize();

  std::vector<vesta_core::UUID> to_marginalize = {tg1.p1->uuid(), tg1.l1->uuid()};

  vesta_constraints::QRMarginalizer qr(false);
  auto txn_qr = qr.marginalize("test", to_marginalize, tg1.graph);

  std::vector<vesta_core::UUID> to_marginalize2 = {tg2.p1->uuid(), tg2.l1->uuid()};
  vesta_constraints::SchurMarginalizer schur(false);
  auto txn_schur = schur.marginalize("test", to_marginalize2, tg2.graph);

  tg1.graph.update(txn_qr);
  tg2.graph.update(txn_schur);

  tg1.graph.optimize();
  tg2.graph.optimize();

  // Compare remaining variable values
  for (size_t i = 0; i < tg1.p2->size(); ++i)
  {
    EXPECT_NEAR(tg1.p2->data()[i], tg2.p2->data()[i], 1.0e-6);
  }
  for (size_t i = 0; i < tg1.l2->size(); ++i)
  {
    EXPECT_NEAR(tg1.l2->data()[i], tg2.l2->data()[i], 1.0e-6);
  }
}

TEST(SchurMarginalizer, MatchesQR_OnlyNonStamped)
{
  // Marginalize only non-stamped variables
  MixedGraph tg1;
  MixedGraph tg2;

  tg1.graph.optimize();
  tg2.graph.optimize();

  std::vector<vesta_core::UUID> to_marginalize = {tg1.l1->uuid()};

  vesta_constraints::QRMarginalizer qr(false);
  auto txn_qr = qr.marginalize("test", to_marginalize, tg1.graph);

  std::vector<vesta_core::UUID> to_marginalize2 = {tg2.l1->uuid()};
  vesta_constraints::SchurMarginalizer schur(false);
  auto txn_schur = schur.marginalize("test", to_marginalize2, tg2.graph);

  tg1.graph.update(txn_qr);
  tg2.graph.update(txn_schur);

  tg1.graph.optimize();
  tg2.graph.optimize();

  for (size_t i = 0; i < tg1.p1->size(); ++i)
  {
    EXPECT_NEAR(tg1.p1->data()[i], tg2.p1->data()[i], 1.0e-6);
  }
  for (size_t i = 0; i < tg1.p2->size(); ++i)
  {
    EXPECT_NEAR(tg1.p2->data()[i], tg2.p2->data()[i], 1.0e-6);
  }
}

TEST(SchurMarginalizer, FejFallbackMatchesNonFej)
{
  // When no linearization points are set, FEJ should produce identical results
  MixedGraph tg1;
  MixedGraph tg2;

  tg1.graph.optimize();
  tg2.graph.optimize();

  std::vector<vesta_core::UUID> to_marginalize1 = {tg1.l1->uuid()};
  std::vector<vesta_core::UUID> to_marginalize2 = {tg2.l1->uuid()};

  vesta_constraints::SchurMarginalizer non_fej(false);
  vesta_constraints::SchurMarginalizer fej(true);

  auto txn1 = non_fej.marginalize("test", to_marginalize1, tg1.graph);
  auto txn2 = fej.marginalize("test", to_marginalize2, tg2.graph);

  tg1.graph.update(txn1);
  tg2.graph.update(txn2);

  tg1.graph.optimize();
  tg2.graph.optimize();

  for (size_t i = 0; i < tg1.p1->size(); ++i)
  {
    EXPECT_NEAR(tg1.p1->data()[i], tg2.p1->data()[i], 1.0e-10);
  }
  for (size_t i = 0; i < tg1.p2->size(); ++i)
  {
    EXPECT_NEAR(tg1.p2->data()[i], tg2.p2->data()[i], 1.0e-10);
  }
}

TEST(SchurMarginalizer, StructuralCorrectness)
{
  // Verify the transaction contains the correct removals and additions
  MixedGraph tg;
  tg.graph.optimize();

  std::vector<vesta_core::UUID> to_marginalize = {tg.l1->uuid()};

  vesta_constraints::SchurMarginalizer schur(false);
  auto txn = schur.marginalize("test", to_marginalize, tg.graph);

  // Should remove l1
  auto removed_vars = txn.removedVariables();
  auto removed_set = std::set<vesta_core::UUID>(removed_vars.begin(), removed_vars.end());
  EXPECT_EQ(removed_set.count(tg.l1->uuid()), 1u);
  EXPECT_EQ(removed_set.size(), 1u);

  // Should remove constraints connected to l1 (prior on l1 + observation p1->l1)
  auto removed_cons = txn.removedConstraints();
  EXPECT_EQ(std::distance(removed_cons.begin(), removed_cons.end()), 2);

  // Should add marginal constraint(s) on the remaining variables
  auto added_cons = txn.addedConstraints();
  EXPECT_GE(std::distance(added_cons.begin(), added_cons.end()), 1);
}

TEST(SchurMarginalizer, RuntimeComparison)
{
  constexpr int NUM_ITERATIONS = 50;

  // Measure QR
  auto start_qr = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUM_ITERATIONS; ++i)
  {
    MixedGraph tg;
    tg.graph.optimize();
    vesta_constraints::QRMarginalizer qr(false);
    qr.marginalize("test", {tg.l1->uuid(), tg.l2->uuid()}, tg.graph);
  }
  auto end_qr = std::chrono::high_resolution_clock::now();
  auto qr_us = std::chrono::duration_cast<std::chrono::microseconds>(end_qr - start_qr).count();

  // Measure Schur
  auto start_schur = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUM_ITERATIONS; ++i)
  {
    MixedGraph tg;
    tg.graph.optimize();
    vesta_constraints::SchurMarginalizer schur(false);
    schur.marginalize("test", {tg.l1->uuid(), tg.l2->uuid()}, tg.graph);
  }
  auto end_schur = std::chrono::high_resolution_clock::now();
  auto schur_us = std::chrono::duration_cast<std::chrono::microseconds>(end_schur - start_schur).count();

  std::cout << "[          ] QRMarginalizer:    " << qr_us / NUM_ITERATIONS << " us/marginalization" << std::endl;
  std::cout << "[          ] SchurMarginalizer: " << schur_us / NUM_ITERATIONS << " us/marginalization" << std::endl;
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
