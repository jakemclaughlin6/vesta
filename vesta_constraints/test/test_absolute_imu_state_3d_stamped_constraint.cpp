#include <vesta_constraints/inertial/absolute_imu_state_3d_stamped_constraint.h>
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
using vesta_variables::AccelerationBias3DStamped;
using vesta_variables::GyroscopeBias3DStamped;
using vesta_variables::Orientation3DStamped;
using vesta_variables::Position3DStamped;
using vesta_variables::VelocityLinear3DStamped;

TEST(AbsoluteImuState3DStampedConstraint, Constructor)
{
  auto stamp = vesta_core::Timestamp(1234, 5678);
  auto device_id = vesta_core::uuid::generate("imu_test");

  Orientation3DStamped orientation(stamp, device_id);
  Position3DStamped position(stamp, device_id);
  VelocityLinear3DStamped velocity(stamp, device_id);
  GyroscopeBias3DStamped gyro_bias(stamp, device_id);
  AccelerationBias3DStamped accel_bias(stamp, device_id);

  Eigen::Matrix<double, 16, 1> mean;
  mean << 1.0, 0.0, 0.0, 0.0,  // quaternion (w, x, y, z)
      1.0, 2.0, 3.0,           // position
      0.1, 0.2, 0.3,           // velocity
      0.01, 0.02, 0.03,        // gyro bias
      0.04, 0.05, 0.06;        // accel bias

  Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Identity();

  EXPECT_NO_THROW(AbsoluteImuState3DStampedConstraint constraint("test", orientation, position, velocity, gyro_bias,
                                                                 accel_bias, mean, cov));
}

TEST(AbsoluteImuState3DStampedConstraint, Covariance)
{
  auto stamp = vesta_core::Timestamp(1234, 5678);
  auto device_id = vesta_core::uuid::generate("imu_test");

  Orientation3DStamped orientation(stamp, device_id);
  Position3DStamped position(stamp, device_id);
  VelocityLinear3DStamped velocity(stamp, device_id);
  GyroscopeBias3DStamped gyro_bias(stamp, device_id);
  AccelerationBias3DStamped accel_bias(stamp, device_id);

  Eigen::Matrix<double, 16, 1> mean;
  mean << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  // Create a diagonal covariance with distinct values
  Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Identity();
  for (int i = 0; i < 15; ++i)
  {
    cov(i, i) = static_cast<double>(i + 1);
  }

  AbsoluteImuState3DStampedConstraint constraint("test", orientation, position, velocity, gyro_bias, accel_bias, mean,
                                                 cov);

  // Verify round-trip: covariance -> sqrt_information -> covariance
  Eigen::Matrix<double, 15, 15> recovered_cov = constraint.covariance();
  EXPECT_MATRIX_NEAR(cov, recovered_cov, 1e-9);
}

TEST(AbsoluteImuState3DStampedConstraint, Optimization)
{
  // Create variables with perturbed initial values, add absolute constraint
  // with known mean, optimize with Ceres, verify variables converge to mean.
  auto stamp = vesta_core::Timestamp(1, 0);
  auto device_id = vesta_core::uuid::generate("imu_opt");

  auto orientation = Orientation3DStamped::make_shared(stamp, device_id);
  orientation->w() = 0.952;
  orientation->x() = 0.038;
  orientation->y() = -0.189;
  orientation->z() = 0.239;

  auto position = Position3DStamped::make_shared(stamp, device_id);
  position->x() = 1.5;
  position->y() = -3.0;
  position->z() = 10.0;

  auto velocity = VelocityLinear3DStamped::make_shared(stamp, device_id);
  velocity->x() = 0.5;
  velocity->y() = -0.3;
  velocity->z() = 1.2;

  auto gyro_bias = GyroscopeBias3DStamped::make_shared(stamp, device_id);
  gyro_bias->x() = 0.05;
  gyro_bias->y() = -0.03;
  gyro_bias->z() = 0.02;

  auto accel_bias = AccelerationBias3DStamped::make_shared(stamp, device_id);
  accel_bias->x() = 0.1;
  accel_bias->y() = -0.2;
  accel_bias->z() = 0.15;

  // Define the mean (target values)
  Eigen::Matrix<double, 16, 1> mean;
  mean << 1.0, 0.0, 0.0, 0.0,  // identity quaternion
      1.0, 2.0, 3.0,           // position
      0.0, 0.0, 0.0,           // velocity
      0.0, 0.0, 0.0,           // gyro bias
      0.0, 0.0, 0.0;           // accel bias

  Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Identity();

  auto constraint = AbsoluteImuState3DStampedConstraint::make_shared("test", *orientation, *position, *velocity,
                                                                     *gyro_bias, *accel_bias, mean, cov);

  // Build the problem
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = vesta_core::Loss::Ownership;
  ceres::Problem problem(problem_options);

  problem.AddParameterBlock(orientation->data(), orientation->size(), orientation->manifold());
  problem.AddParameterBlock(position->data(), position->size(), position->manifold());
  problem.AddParameterBlock(velocity->data(), velocity->size());
  problem.AddParameterBlock(gyro_bias->data(), gyro_bias->size());
  problem.AddParameterBlock(accel_bias->data(), accel_bias->size());

  std::vector<double*> parameter_blocks;
  parameter_blocks.push_back(orientation->data());
  parameter_blocks.push_back(position->data());
  parameter_blocks.push_back(velocity->data());
  parameter_blocks.push_back(gyro_bias->data());
  parameter_blocks.push_back(accel_bias->data());

  problem.AddResidualBlock(constraint->costFunction(), constraint->lossFunction(), parameter_blocks);

  // Run the solver
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Check convergence
  EXPECT_NEAR(1.0, position->x(), 1e-3);
  EXPECT_NEAR(2.0, position->y(), 1e-3);
  EXPECT_NEAR(3.0, position->z(), 1e-3);

  EXPECT_NEAR(1.0, orientation->w(), 1e-3);
  EXPECT_NEAR(0.0, orientation->x(), 1e-3);
  EXPECT_NEAR(0.0, orientation->y(), 1e-3);
  EXPECT_NEAR(0.0, orientation->z(), 1e-3);

  EXPECT_NEAR(0.0, velocity->x(), 1e-3);
  EXPECT_NEAR(0.0, velocity->y(), 1e-3);
  EXPECT_NEAR(0.0, velocity->z(), 1e-3);

  EXPECT_NEAR(0.0, gyro_bias->x(), 1e-3);
  EXPECT_NEAR(0.0, gyro_bias->y(), 1e-3);
  EXPECT_NEAR(0.0, gyro_bias->z(), 1e-3);

  EXPECT_NEAR(0.0, accel_bias->x(), 1e-3);
  EXPECT_NEAR(0.0, accel_bias->y(), 1e-3);
  EXPECT_NEAR(0.0, accel_bias->z(), 1e-3);
}

TEST(AbsoluteImuState3DStampedConstraint, Serialization)
{
  auto stamp = vesta_core::Timestamp(1234, 5678);
  auto device_id = vesta_core::uuid::generate("imu_ser");

  Orientation3DStamped orientation(stamp, device_id);
  Position3DStamped position(stamp, device_id);
  VelocityLinear3DStamped velocity(stamp, device_id);
  GyroscopeBias3DStamped gyro_bias(stamp, device_id);
  AccelerationBias3DStamped accel_bias(stamp, device_id);

  Eigen::Matrix<double, 16, 1> mean;
  mean << 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06;

  Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Identity();

  AbsoluteImuState3DStampedConstraint expected("test", orientation, position, velocity, gyro_bias, accel_bias, mean,
                                               cov);

  // Serialize
  std::stringstream stream;
  {
    vesta_core::TextOutputArchive archive(stream);
    expected.serialize(archive);
  }

  // Deserialize
  AbsoluteImuState3DStampedConstraint actual;
  {
    vesta_core::TextInputArchive archive(stream);
    actual.deserialize(archive);
  }

  // Compare
  EXPECT_EQ(expected.uuid(), actual.uuid());
  EXPECT_EQ(expected.variables(), actual.variables());
  EXPECT_MATRIX_EQ(expected.mean(), actual.mean());
  EXPECT_MATRIX_EQ(expected.sqrtInformation(), actual.sqrtInformation());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
