#include <vesta_constraints/inertial/imu_preintegrator.h>
#include <vesta_core/timestamp.h>

#include <gtest/gtest.h>

#include <cmath>

using vesta_constraints::ImuData;
using vesta_constraints::ImuPreintegrator;
using vesta_constraints::kGravityNominal;

TEST(ImuPreintegrator, ZeroMotion) {
  // Feed IMU data with zero angular velocity and gravity-compensating
  // acceleration. After integration, delta should be approximately identity
  // rotation, zero velocity, zero position.
  ImuPreintegrator preintegrator;

  const double dt = 0.01;
  const int num_steps = 100; // 1 second total
  const Eigen::Vector3d zero_gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d gravity_compensating_accel(0.0, 0.0, kGravityNominal);

  // Insert IMU data into the buffer
  for (int i = 0; i <= num_steps; ++i) {
    int64_t ns = static_cast<int64_t>(i * dt * 1e9);
    ImuData imu(vesta_core::Timestamp(ns), zero_gyro,
                gravity_compensating_accel);
    preintegrator.data.emplace(imu.stamp, imu);
  }

  // Integrate for 1 second
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  bool success = preintegrator.integrate(
      vesta_core::Timestamp(static_cast<int64_t>(1.0 * 1e9)), bg, ba, true,
      true, true);
  ASSERT_TRUE(success);

  // Check delta time
  EXPECT_NEAR(1.0, preintegrator.delta.dt, 1e-9);

  // Check identity rotation
  EXPECT_NEAR(1.0, preintegrator.delta.q.w(), 1e-6);
  EXPECT_NEAR(0.0, preintegrator.delta.q.x(), 1e-6);
  EXPECT_NEAR(0.0, preintegrator.delta.q.y(), 1e-6);
  EXPECT_NEAR(0.0, preintegrator.delta.q.z(), 1e-6);

  // The preintegrator integrates in the body frame without gravity subtraction.
  // With acceleration = [0,0,g] and zero bias, the preintegrated delta includes
  // the gravity contribution. Gravity is subtracted at the constraint level.
  // delta.v should be approximately [0, 0, g*dt] = [0, 0, 9.80665]
  EXPECT_NEAR(0.0, preintegrator.delta.v.x(), 1e-4);
  EXPECT_NEAR(0.0, preintegrator.delta.v.y(), 1e-4);
  EXPECT_NEAR(kGravityNominal * 1.0, preintegrator.delta.v.z(), 1e-3);

  // delta.p should be approximately [0, 0, 0.5*g*dt^2] = [0, 0, 4.903]
  EXPECT_NEAR(0.0, preintegrator.delta.p.x(), 1e-4);
  EXPECT_NEAR(0.0, preintegrator.delta.p.y(), 1e-4);
  EXPECT_NEAR(0.5 * kGravityNominal * 1.0 * 1.0, preintegrator.delta.p.z(),
              1e-2);
}

TEST(ImuPreintegrator, ConstantVelocity) {
  // Feed IMU data with zero angular velocity, acceleration = [0,0,g] + [1,0,0].
  // Integrate for 1 second with dt=0.01.
  // Check that: delta.v ~ [1,0,0], delta.p ~ [0.5,0,0]
  ImuPreintegrator preintegrator;

  const double dt = 0.01;
  const int num_steps = 100;
  const Eigen::Vector3d zero_gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d accel(1.0, 0.0, kGravityNominal);

  for (int i = 0; i <= num_steps; ++i) {
    int64_t ns = static_cast<int64_t>(i * dt * 1e9);
    ImuData imu(vesta_core::Timestamp(ns), zero_gyro, accel);
    preintegrator.data.emplace(imu.stamp, imu);
  }

  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  bool success = preintegrator.integrate(
      vesta_core::Timestamp(static_cast<int64_t>(1.0 * 1e9)), bg, ba, true,
      true, true);
  ASSERT_TRUE(success);

  EXPECT_NEAR(1.0, preintegrator.delta.dt, 1e-9);

  // The preintegrator integrates in body frame without gravity subtraction.
  // With acceleration = [1,0,g] and zero bias:
  // delta.v ~ [1, 0, g] after 1 second
  EXPECT_NEAR(1.0, preintegrator.delta.v.x(), 1e-3);
  EXPECT_NEAR(0.0, preintegrator.delta.v.y(), 1e-3);
  EXPECT_NEAR(kGravityNominal * 1.0, preintegrator.delta.v.z(), 1e-3);

  // delta.p ~ [0.5, 0, 0.5*g] after 1 second
  EXPECT_NEAR(0.5, preintegrator.delta.p.x(), 1e-3);
  EXPECT_NEAR(0.0, preintegrator.delta.p.y(), 1e-3);
  EXPECT_NEAR(0.5 * kGravityNominal * 1.0 * 1.0, preintegrator.delta.p.z(),
              1e-2);

  // Rotation should still be identity
  EXPECT_NEAR(1.0, preintegrator.delta.q.w(), 1e-6);
  EXPECT_NEAR(0.0, preintegrator.delta.q.x(), 1e-6);
  EXPECT_NEAR(0.0, preintegrator.delta.q.y(), 1e-6);
  EXPECT_NEAR(0.0, preintegrator.delta.q.z(), 1e-6);
}

TEST(ImuPreintegrator, ConstantRotation) {
  // Feed IMU data with constant angular velocity [0,0,0.1] rad/s,
  // gravity-compensating acceleration. After 1 second, check rotation angle ~
  // 0.1 rad about z-axis.
  ImuPreintegrator preintegrator;

  const double dt = 0.01;
  const int num_steps = 100;
  const Eigen::Vector3d gyro(0.0, 0.0, 0.1);
  const Eigen::Vector3d gravity_compensating_accel(0.0, 0.0, kGravityNominal);

  for (int i = 0; i <= num_steps; ++i) {
    int64_t ns = static_cast<int64_t>(i * dt * 1e9);
    ImuData imu(vesta_core::Timestamp(ns), gyro, gravity_compensating_accel);
    preintegrator.data.emplace(imu.stamp, imu);
  }

  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  bool success = preintegrator.integrate(
      vesta_core::Timestamp(static_cast<int64_t>(1.0 * 1e9)), bg, ba, true,
      true, true);
  ASSERT_TRUE(success);

  // Extract the rotation angle from the quaternion
  Eigen::AngleAxisd aa(preintegrator.delta.q);
  double angle = aa.angle();
  Eigen::Vector3d axis = aa.axis();

  // The angle should be approximately 0.1 rad
  EXPECT_NEAR(0.1, angle, 1e-3);

  // The rotation axis should be approximately [0, 0, 1]
  EXPECT_NEAR(0.0, axis.x(), 1e-3);
  EXPECT_NEAR(0.0, axis.y(), 1e-3);
  EXPECT_NEAR(1.0, std::abs(axis.z()), 1e-3);
}

TEST(ImuPreintegrator, CovarianceGrowth) {
  // Verify covariance grows over time (norm increases with more IMU data).
  ImuPreintegrator preintegrator;

  const double dt = 0.01;
  const int num_steps = 200;
  const Eigen::Vector3d zero_gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d gravity_compensating_accel(0.0, 0.0, kGravityNominal);

  for (int i = 0; i <= num_steps; ++i) {
    int64_t ns = static_cast<int64_t>(i * dt * 1e9);
    ImuData imu(vesta_core::Timestamp(ns), zero_gyro,
                gravity_compensating_accel);
    preintegrator.data.emplace(imu.stamp, imu);
  }

  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();

  // Integrate for 0.5 seconds
  preintegrator.integrate(
      vesta_core::Timestamp(static_cast<int64_t>(0.5 * 1e9)), bg, ba, false,
      true, false);
  double norm_half = preintegrator.delta.covariance.norm();

  // Integrate for 1.0 seconds
  preintegrator.integrate(
      vesta_core::Timestamp(static_cast<int64_t>(1.0 * 1e9)), bg, ba, false,
      true, false);
  double norm_full = preintegrator.delta.covariance.norm();

  // Integrate for 2.0 seconds
  preintegrator.integrate(
      vesta_core::Timestamp(static_cast<int64_t>(2.0 * 1e9)), bg, ba, false,
      true, false);
  double norm_double = preintegrator.delta.covariance.norm();

  // Covariance norm should increase with time
  EXPECT_GT(norm_full, norm_half);
  EXPECT_GT(norm_double, norm_full);
}

TEST(ImuPreintegrator, ResetClearsState) {
  // Verify reset() zeros everything.
  ImuPreintegrator preintegrator;

  const double dt = 0.01;
  const int num_steps = 50;
  const Eigen::Vector3d gyro(0.1, 0.2, 0.3);
  const Eigen::Vector3d accel(1.0, 2.0, kGravityNominal);

  for (int i = 0; i <= num_steps; ++i) {
    int64_t ns = static_cast<int64_t>(i * dt * 1e9);
    ImuData imu(vesta_core::Timestamp(ns), gyro, accel);
    preintegrator.data.emplace(imu.stamp, imu);
  }

  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();

  preintegrator.integrate(
      vesta_core::Timestamp(static_cast<int64_t>(0.5 * 1e9)), bg, ba, true,
      true, true);

  // Verify state is non-zero before reset
  EXPECT_GT(preintegrator.delta.dt, 0.0);
  EXPECT_GT(preintegrator.delta.p.norm(), 0.0);
  EXPECT_GT(preintegrator.delta.v.norm(), 0.0);

  // Reset
  preintegrator.reset();

  // Verify everything is zeroed
  EXPECT_DOUBLE_EQ(0.0, preintegrator.delta.dt);
  EXPECT_DOUBLE_EQ(0.0, preintegrator.delta.p.norm());
  EXPECT_DOUBLE_EQ(0.0, preintegrator.delta.v.norm());
  EXPECT_NEAR(1.0, preintegrator.delta.q.w(), 1e-12);
  EXPECT_NEAR(0.0, preintegrator.delta.q.x(), 1e-12);
  EXPECT_NEAR(0.0, preintegrator.delta.q.y(), 1e-12);
  EXPECT_NEAR(0.0, preintegrator.delta.q.z(), 1e-12);
  EXPECT_DOUBLE_EQ(0.0, preintegrator.delta.covariance.norm());
  EXPECT_DOUBLE_EQ(0.0, preintegrator.delta.sqrt_information.norm());

  // Jacobians should also be zeroed
  EXPECT_DOUBLE_EQ(0.0, preintegrator.jacobian.dq_dbg.norm());
  EXPECT_DOUBLE_EQ(0.0, preintegrator.jacobian.dp_dbg.norm());
  EXPECT_DOUBLE_EQ(0.0, preintegrator.jacobian.dp_dba.norm());
  EXPECT_DOUBLE_EQ(0.0, preintegrator.jacobian.dv_dbg.norm());
  EXPECT_DOUBLE_EQ(0.0, preintegrator.jacobian.dv_dba.norm());
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
