#include <vesta_constraints/inertial/absolute_imu_state_3d_stamped_constraint.h>
#include <vesta_constraints/inertial/imu_preintegrator.h>
#include <vesta_constraints/inertial/relative_imu_state_3d_stamped_constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/eigen_gtest.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/acceleration_bias_3d_stamped.h>
#include <vesta_variables/3d/gyroscope_bias_3d_stamped.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/3d/velocity_linear_3d_stamped.h>

#include <ceres/problem.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>

#include <vector>

using vesta_constraints::AbsoluteImuState3DStampedConstraint;
using vesta_constraints::ImuData;
using vesta_constraints::ImuPreintegrator;
using vesta_constraints::kGravityNominal;
using vesta_constraints::kGravityWorld;
using vesta_constraints::RelativeImuState3DStampedConstraint;
using vesta_variables::AccelerationBias3DStamped;
using vesta_variables::GyroscopeBias3DStamped;
using vesta_variables::Orientation3DStamped;
using vesta_variables::Position3DStamped;
using vesta_variables::VelocityLinear3DStamped;

TEST(RelativeImuState3DStampedConstraint, Constructor)
{
  auto device_id = vesta_core::uuid::generate("imu_rel");
  auto stamp1 = vesta_core::Timestamp(1, 0);
  auto stamp2 = vesta_core::Timestamp(2, 0);

  Orientation3DStamped orientation1(stamp1, device_id);
  Position3DStamped position1(stamp1, device_id);
  VelocityLinear3DStamped velocity1(stamp1, device_id);
  GyroscopeBias3DStamped gyro_bias1(stamp1, device_id);
  AccelerationBias3DStamped accel_bias1(stamp1, device_id);

  Orientation3DStamped orientation2(stamp2, device_id);
  Position3DStamped position2(stamp2, device_id);
  VelocityLinear3DStamped velocity2(stamp2, device_id);
  GyroscopeBias3DStamped gyro_bias2(stamp2, device_id);
  AccelerationBias3DStamped accel_bias2(stamp2, device_id);

  // Create a preintegrator with some data
  ImuPreintegrator preintegrator;
  const double dt = 0.01;
  const int num_steps = 100;
  const Eigen::Vector3d zero_gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d gravity_compensating_accel(0.0, 0.0, kGravityNominal);

  for (int i = 0; i <= num_steps; ++i)
  {
    int64_t ns = static_cast<int64_t>(1e9 + i * dt * 1e9);
    ImuData imu(vesta_core::Timestamp(ns), zero_gyro, gravity_compensating_accel);
    preintegrator.data.emplace(imu.stamp, imu);
  }

  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  preintegrator.integrate(stamp2, bg, ba, true, true, true);

  EXPECT_NO_THROW(RelativeImuState3DStampedConstraint constraint("test", orientation1, position1, velocity1, gyro_bias1,
                                                                 accel_bias1, orientation2, position2, velocity2,
                                                                 gyro_bias2, accel_bias2, preintegrator, bg, ba));
}

TEST(RelativeImuState3DStampedConstraint, Optimization)
{
  // Two-state optimization test:
  // State1 is fixed at identity/origin with zero velocity.
  // IMU data simulates constant acceleration of [1,0,0] for 1 second.
  // After optimization, state2 should have position ~ [0.5,0,0], velocity ~
  // [1,0,0].
  auto device_id = vesta_core::uuid::generate("imu_opt");
  auto stamp1 = vesta_core::Timestamp(0);
  auto stamp2 = vesta_core::Timestamp(static_cast<int64_t>(1e9));

  // State 1 variables (at identity/origin)
  auto orientation1 = Orientation3DStamped::make_shared(stamp1, device_id);
  orientation1->w() = 1.0;
  orientation1->x() = 0.0;
  orientation1->y() = 0.0;
  orientation1->z() = 0.0;

  auto position1 = Position3DStamped::make_shared(stamp1, device_id);
  position1->x() = 0.0;
  position1->y() = 0.0;
  position1->z() = 0.0;

  auto velocity1 = VelocityLinear3DStamped::make_shared(stamp1, device_id);
  velocity1->x() = 0.0;
  velocity1->y() = 0.0;
  velocity1->z() = 0.0;

  auto gyro_bias1 = GyroscopeBias3DStamped::make_shared(stamp1, device_id);
  gyro_bias1->x() = 0.0;
  gyro_bias1->y() = 0.0;
  gyro_bias1->z() = 0.0;

  auto accel_bias1 = AccelerationBias3DStamped::make_shared(stamp1, device_id);
  accel_bias1->x() = 0.0;
  accel_bias1->y() = 0.0;
  accel_bias1->z() = 0.0;

  // Build preintegrator with constant acceleration [1,0,0] + gravity
  // compensation
  ImuPreintegrator preintegrator;
  const double dt = 0.01;
  const int num_steps = 100;
  const Eigen::Vector3d zero_gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d accel(1.0, 0.0, kGravityNominal);

  for (int i = 0; i <= num_steps; ++i)
  {
    int64_t ns = static_cast<int64_t>(i * dt * 1e9);
    ImuData imu(vesta_core::Timestamp(ns), zero_gyro, accel);
    preintegrator.data.emplace(imu.stamp, imu);
  }

  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  preintegrator.integrate(stamp2, bg, ba, true, true, true);

  // Predict state2 from preintegration
  // p2 = p1 + v1*dt + 0.5*g*dt^2 + R1*delta_p
  // v2 = v1 + g*dt + R1*delta_v
  const double T = preintegrator.delta.dt;
  Eigen::Vector3d expected_pos =
      Eigen::Vector3d::Zero() + Eigen::Vector3d::Zero() * T + 0.5 * kGravityWorld * T * T + preintegrator.delta.p;
  Eigen::Vector3d expected_vel = Eigen::Vector3d::Zero() + kGravityWorld * T + preintegrator.delta.v;

  // State 2 variables (initialize to predicted values with small perturbation)
  auto orientation2 = Orientation3DStamped::make_shared(stamp2, device_id);
  orientation2->w() = 1.0;
  orientation2->x() = 0.0;
  orientation2->y() = 0.0;
  orientation2->z() = 0.0;

  auto position2 = Position3DStamped::make_shared(stamp2, device_id);
  position2->x() = expected_pos.x() + 0.1;
  position2->y() = expected_pos.y() + 0.1;
  position2->z() = expected_pos.z() + 0.1;

  auto velocity2 = VelocityLinear3DStamped::make_shared(stamp2, device_id);
  velocity2->x() = expected_vel.x() + 0.1;
  velocity2->y() = expected_vel.y() + 0.1;
  velocity2->z() = expected_vel.z() + 0.1;

  auto gyro_bias2 = GyroscopeBias3DStamped::make_shared(stamp2, device_id);
  gyro_bias2->x() = 0.0;
  gyro_bias2->y() = 0.0;
  gyro_bias2->z() = 0.0;

  auto accel_bias2 = AccelerationBias3DStamped::make_shared(stamp2, device_id);
  accel_bias2->x() = 0.0;
  accel_bias2->y() = 0.0;
  accel_bias2->z() = 0.0;

  // Create relative constraint
  auto relative = RelativeImuState3DStampedConstraint::make_shared(
      "test", *orientation1, *position1, *velocity1, *gyro_bias1, *accel_bias1, *orientation2, *position2, *velocity2,
      *gyro_bias2, *accel_bias2, preintegrator, bg, ba);

  // Build the problem
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = vesta_core::Loss::Ownership;
  ceres::Problem problem(problem_options);

  // Add parameter blocks
  problem.AddParameterBlock(orientation1->data(), orientation1->size(), orientation1->manifold());
  problem.AddParameterBlock(position1->data(), position1->size(), position1->manifold());
  problem.AddParameterBlock(velocity1->data(), velocity1->size());
  problem.AddParameterBlock(gyro_bias1->data(), gyro_bias1->size());
  problem.AddParameterBlock(accel_bias1->data(), accel_bias1->size());
  problem.AddParameterBlock(orientation2->data(), orientation2->size(), orientation2->manifold());
  problem.AddParameterBlock(position2->data(), position2->size(), position2->manifold());
  problem.AddParameterBlock(velocity2->data(), velocity2->size());
  problem.AddParameterBlock(gyro_bias2->data(), gyro_bias2->size());
  problem.AddParameterBlock(accel_bias2->data(), accel_bias2->size());

  // Fix state1
  problem.SetParameterBlockConstant(orientation1->data());
  problem.SetParameterBlockConstant(position1->data());
  problem.SetParameterBlockConstant(velocity1->data());
  problem.SetParameterBlockConstant(gyro_bias1->data());
  problem.SetParameterBlockConstant(accel_bias1->data());

  // Add relative constraint
  std::vector<double*> relative_blocks;
  relative_blocks.push_back(orientation1->data());
  relative_blocks.push_back(position1->data());
  relative_blocks.push_back(velocity1->data());
  relative_blocks.push_back(gyro_bias1->data());
  relative_blocks.push_back(accel_bias1->data());
  relative_blocks.push_back(orientation2->data());
  relative_blocks.push_back(position2->data());
  relative_blocks.push_back(velocity2->data());
  relative_blocks.push_back(gyro_bias2->data());
  relative_blocks.push_back(accel_bias2->data());

  problem.AddResidualBlock(relative->costFunction(), relative->lossFunction(), relative_blocks);

  // Solve
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Verify state2 converged to expected values
  EXPECT_NEAR(expected_pos.x(), position2->x(), 1e-3);
  EXPECT_NEAR(expected_pos.y(), position2->y(), 1e-3);
  EXPECT_NEAR(expected_pos.z(), position2->z(), 1e-3);

  EXPECT_NEAR(expected_vel.x(), velocity2->x(), 1e-3);
  EXPECT_NEAR(expected_vel.y(), velocity2->y(), 1e-3);
  EXPECT_NEAR(expected_vel.z(), velocity2->z(), 1e-3);

  EXPECT_NEAR(1.0, orientation2->w(), 1e-3);
  EXPECT_NEAR(0.0, orientation2->x(), 1e-3);
  EXPECT_NEAR(0.0, orientation2->y(), 1e-3);
  EXPECT_NEAR(0.0, orientation2->z(), 1e-3);
}

TEST(RelativeImuState3DStampedConstraint, Serialization)
{
  auto device_id = vesta_core::uuid::generate("imu_ser");
  auto stamp1 = vesta_core::Timestamp(1, 0);
  auto stamp2 = vesta_core::Timestamp(2, 0);

  Orientation3DStamped orientation1(stamp1, device_id);
  Position3DStamped position1(stamp1, device_id);
  VelocityLinear3DStamped velocity1(stamp1, device_id);
  GyroscopeBias3DStamped gyro_bias1(stamp1, device_id);
  AccelerationBias3DStamped accel_bias1(stamp1, device_id);

  Orientation3DStamped orientation2(stamp2, device_id);
  Position3DStamped position2(stamp2, device_id);
  VelocityLinear3DStamped velocity2(stamp2, device_id);
  GyroscopeBias3DStamped gyro_bias2(stamp2, device_id);
  AccelerationBias3DStamped accel_bias2(stamp2, device_id);

  // Create a preintegrator with data
  ImuPreintegrator preintegrator;
  const double dt_step = 0.01;
  const int num_steps = 100;
  const Eigen::Vector3d gyro(0.0, 0.0, 0.1);
  const Eigen::Vector3d accel(1.0, 0.0, kGravityNominal);

  for (int i = 0; i <= num_steps; ++i)
  {
    int64_t ns = static_cast<int64_t>(1e9 + i * dt_step * 1e9);
    ImuData imu(vesta_core::Timestamp(ns), gyro, accel);
    preintegrator.data.emplace(imu.stamp, imu);
  }

  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  preintegrator.integrate(stamp2, bg, ba, true, true, true);

  RelativeImuState3DStampedConstraint expected("test", orientation1, position1, velocity1, gyro_bias1, accel_bias1,
                                               orientation2, position2, velocity2, gyro_bias2, accel_bias2,
                                               preintegrator, bg, ba);

  // Serialize
  std::stringstream stream;
  {
    vesta_core::TextOutputArchive archive(stream);
    expected.serialize(archive);
  }

  // Deserialize
  RelativeImuState3DStampedConstraint actual;
  {
    vesta_core::TextInputArchive archive(stream);
    actual.deserialize(archive);
  }

  // Compare
  EXPECT_EQ(expected.uuid(), actual.uuid());
  EXPECT_EQ(expected.variables(), actual.variables());
  EXPECT_NEAR(expected.dt(), actual.dt(), 1e-12);
  EXPECT_MATRIX_NEAR(expected.sqrtInformation(), actual.sqrtInformation(), 1e-9);
  EXPECT_NEAR(expected.deltaQ().w(), actual.deltaQ().w(), 1e-12);
  EXPECT_NEAR(expected.deltaQ().x(), actual.deltaQ().x(), 1e-12);
  EXPECT_NEAR(expected.deltaQ().y(), actual.deltaQ().y(), 1e-12);
  EXPECT_NEAR(expected.deltaQ().z(), actual.deltaQ().z(), 1e-12);
  EXPECT_MATRIX_NEAR(expected.deltaP(), actual.deltaP(), 1e-12);
  EXPECT_MATRIX_NEAR(expected.deltaV(), actual.deltaV(), 1e-12);
  EXPECT_MATRIX_NEAR(expected.gravity(), actual.gravity(), 1e-12);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
