#include <vesta_constraints/inertial/imu_preintegration.h>
#include <vesta_constraints/inertial/imu_preintegrator.h>
#include <vesta_core/eigen.h>
#include <vesta_core/eigen_gtest.h>
#include <vesta_core/uuid.h>

#include <ceres/problem.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>

#include <vector>

using vesta_constraints::ImuPreintegration;
using vesta_constraints::ImuPreintegrationParams;
using vesta_constraints::ImuData;
using vesta_constraints::kGravityNominal;
using vesta_constraints::kGravityWorld;


TEST(ImuPreintegration, BasicWorkflow)
{
  // Test the basic API workflow:
  // Create, setStart, addImuData, createPreintegratedFactor
  ImuPreintegrationParams params;
  ImuPreintegration imu_preint(params);

  auto stamp0 = vesta_core::Timestamp(0);
  auto stamp1 = vesta_core::Timestamp(static_cast<int64_t>(1e9));

  // Add IMU data for 1 second (dt=0.01, constant gravity-compensating accel)
  const double dt = 0.01;
  const int num_steps = 100;
  const Eigen::Vector3d zero_gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d gravity_compensating_accel(0.0, 0.0, kGravityNominal);

  for (int i = 0; i <= num_steps; ++i)
  {
    int64_t ns = static_cast<int64_t>(i * dt * 1e9);
    imu_preint.addImuData(ImuData(vesta_core::Timestamp(ns), zero_gyro, gravity_compensating_accel));
  }

  // Set start at t=0
  imu_preint.setStart(stamp0);

  // Create preintegrated factor at t=1.0
  auto result = imu_preint.createPreintegratedFactor(stamp1);

  // Verify non-null constraints and variables
  EXPECT_NE(nullptr, result.relative_constraint);
  EXPECT_NE(nullptr, result.prior_constraint);  // first window should have prior
  EXPECT_NE(nullptr, result.orientation);
  EXPECT_NE(nullptr, result.position);
  EXPECT_NE(nullptr, result.velocity);
  EXPECT_NE(nullptr, result.gyro_bias);
  EXPECT_NE(nullptr, result.accel_bias);

  // Verify predicted state is reasonable (zero motion, so position/velocity should be near zero)
  EXPECT_NEAR(0.0, result.predicted_state.position.x(), 1e-3);
  EXPECT_NEAR(0.0, result.predicted_state.position.y(), 1e-3);
  EXPECT_NEAR(0.0, result.predicted_state.position.z(), 1e-3);

  EXPECT_NEAR(0.0, result.predicted_state.velocity.x(), 1e-3);
  EXPECT_NEAR(0.0, result.predicted_state.velocity.y(), 1e-3);
  EXPECT_NEAR(0.0, result.predicted_state.velocity.z(), 1e-3);
}

TEST(ImuPreintegration, ConstantVelocityTrajectory)
{
  // Feed IMU data simulating constant velocity motion with accel=[1,0,g] for 1 second.
  // Build an optimization problem with prior + relative constraints.
  // Fix first state, optimize second, verify convergence.
  ImuPreintegrationParams params;
  ImuPreintegration imu_preint(params);

  auto stamp0 = vesta_core::Timestamp(0);
  auto stamp1 = vesta_core::Timestamp(static_cast<int64_t>(1e9));
  auto device_id = vesta_core::uuid::NIL;

  // Add IMU data
  const double dt = 0.01;
  const int num_steps = 100;
  const Eigen::Vector3d zero_gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d accel(1.0, 0.0, kGravityNominal);

  for (int i = 0; i <= num_steps; ++i)
  {
    int64_t ns = static_cast<int64_t>(i * dt * 1e9);
    imu_preint.addImuData(ImuData(vesta_core::Timestamp(ns), zero_gyro, accel));
  }

  // Initialize at origin
  imu_preint.setStart(stamp0, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(),
                       Eigen::Vector3d::Zero(), device_id);

  // Create preintegrated factor
  auto result = imu_preint.createPreintegratedFactor(stamp1, device_id, true);

  // Expected state2: p = [0.5, 0, 0], v = [1, 0, 0] (since gravity cancels in world frame)
  // In the full prediction: p = v0*dt + 0.5*g*dt^2 + R*delta_p
  // With v0=0, R=I, g=[0,0,-9.80665]: p = 0.5*g + delta_p
  // delta_p integrates acceleration [1,0,g] - bias(0) in body frame
  // Since orientation is identity throughout, body=world
  // delta_p ~ [0.5, 0, 0.5*g]
  // predicted p = 0.5*[0,0,-g] + [0.5, 0, 0.5*g] = [0.5, 0, 0]

  // Perturb state2 initial values
  result.position->x() += 0.2;
  result.position->y() += 0.1;
  result.position->z() += 0.1;
  result.velocity->x() += 0.3;
  result.velocity->y() += 0.1;
  result.velocity->z() += 0.1;

  // Build Ceres problem
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = vesta_core::Loss::Ownership;
  ceres::Problem problem(problem_options);

  // Get the prior constraint's variables (state1)
  const auto& ori1 = imu_preint.currentOrientation();
  const auto& pos1 = imu_preint.currentPosition();
  const auto& vel1 = imu_preint.currentVelocity();
  const auto& gbias1 = imu_preint.currentGyroBias();
  const auto& abias1 = imu_preint.currentAccelBias();

  // Wait -- after createPreintegratedFactor, the internal state has advanced to state2.
  // The prior and relative constraints reference the OLD state1 variables by UUID.
  // We need to use the variables that were current BEFORE the factor was created.
  // The prior constraint references orientation_i_, position_i_, etc. from BEFORE the call.
  // But those internal variables have been overwritten. The constraint stores UUIDs only.
  // For testing, we need to create fresh variables matching the UUIDs.

  // Actually, looking at the code more carefully, the prior was created with the old orientation_i_
  // etc., and the relative constraint connects old state to new state. The UUIDs in the constraints
  // refer to the variable objects that existed at creation time. We need to create a new set
  // of variables for state1 and use result.* for state2.

  // Let's rebuild with a simpler approach: manually create state1 variables and use them directly.
  auto ori1_var = vesta_variables::Orientation3DStamped::make_shared(stamp0, device_id);
  ori1_var->w() = 1.0;
  ori1_var->x() = 0.0;
  ori1_var->y() = 0.0;
  ori1_var->z() = 0.0;

  auto pos1_var = vesta_variables::Position3DStamped::make_shared(stamp0, device_id);
  pos1_var->x() = 0.0;
  pos1_var->y() = 0.0;
  pos1_var->z() = 0.0;

  auto vel1_var = vesta_variables::VelocityLinear3DStamped::make_shared(stamp0, device_id);
  vel1_var->x() = 0.0;
  vel1_var->y() = 0.0;
  vel1_var->z() = 0.0;

  auto gbias1_var = vesta_variables::GyroscopeBias3DStamped::make_shared(stamp0, device_id);
  gbias1_var->x() = 0.0;
  gbias1_var->y() = 0.0;
  gbias1_var->z() = 0.0;

  auto abias1_var = vesta_variables::AccelerationBias3DStamped::make_shared(stamp0, device_id);
  abias1_var->x() = 0.0;
  abias1_var->y() = 0.0;
  abias1_var->z() = 0.0;

  // Add state1 parameter blocks
  problem.AddParameterBlock(ori1_var->data(), ori1_var->size(), ori1_var->manifold());
  problem.AddParameterBlock(pos1_var->data(), pos1_var->size(), pos1_var->manifold());
  problem.AddParameterBlock(vel1_var->data(), vel1_var->size());
  problem.AddParameterBlock(gbias1_var->data(), gbias1_var->size());
  problem.AddParameterBlock(abias1_var->data(), abias1_var->size());

  // Fix state1
  problem.SetParameterBlockConstant(ori1_var->data());
  problem.SetParameterBlockConstant(pos1_var->data());
  problem.SetParameterBlockConstant(vel1_var->data());
  problem.SetParameterBlockConstant(gbias1_var->data());
  problem.SetParameterBlockConstant(abias1_var->data());

  // Add state2 parameter blocks
  problem.AddParameterBlock(result.orientation->data(), result.orientation->size(),
                             result.orientation->manifold());
  problem.AddParameterBlock(result.position->data(), result.position->size(),
                             result.position->manifold());
  problem.AddParameterBlock(result.velocity->data(), result.velocity->size());
  problem.AddParameterBlock(result.gyro_bias->data(), result.gyro_bias->size());
  problem.AddParameterBlock(result.accel_bias->data(), result.accel_bias->size());

  // Add prior constraint on state1
  {
    std::vector<double*> prior_blocks;
    prior_blocks.push_back(ori1_var->data());
    prior_blocks.push_back(pos1_var->data());
    prior_blocks.push_back(vel1_var->data());
    prior_blocks.push_back(gbias1_var->data());
    prior_blocks.push_back(abias1_var->data());

    problem.AddResidualBlock(
      result.prior_constraint->costFunction(),
      result.prior_constraint->lossFunction(),
      prior_blocks);
  }

  // Add relative constraint
  {
    std::vector<double*> rel_blocks;
    rel_blocks.push_back(ori1_var->data());
    rel_blocks.push_back(pos1_var->data());
    rel_blocks.push_back(vel1_var->data());
    rel_blocks.push_back(gbias1_var->data());
    rel_blocks.push_back(abias1_var->data());
    rel_blocks.push_back(result.orientation->data());
    rel_blocks.push_back(result.position->data());
    rel_blocks.push_back(result.velocity->data());
    rel_blocks.push_back(result.gyro_bias->data());
    rel_blocks.push_back(result.accel_bias->data());

    problem.AddResidualBlock(
      result.relative_constraint->costFunction(),
      result.relative_constraint->lossFunction(),
      rel_blocks);
  }

  // Solve
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Verify state2 converged: expected position ~ [0.5, 0, 0], velocity ~ [1, 0, 0]
  EXPECT_NEAR(0.5, result.position->x(), 0.05);
  EXPECT_NEAR(0.0, result.position->y(), 0.05);
  EXPECT_NEAR(0.0, result.position->z(), 0.05);

  EXPECT_NEAR(1.0, result.velocity->x(), 0.05);
  EXPECT_NEAR(0.0, result.velocity->y(), 0.05);
  EXPECT_NEAR(0.0, result.velocity->z(), 0.05);
}

TEST(ImuPreintegration, BiasEstimation)
{
  // Add a systematic gyro bias to synthetic IMU data.
  // Create constraints, build optimization, verify bias converges near truth.
  ImuPreintegrationParams params;
  params.cov_prior_noise = 1e-2;  // Looser prior to allow bias to be estimated
  ImuPreintegration imu_preint(params);

  auto stamp0 = vesta_core::Timestamp(0);
  auto stamp1 = vesta_core::Timestamp(static_cast<int64_t>(1e9));
  auto device_id = vesta_core::uuid::NIL;

  // True gyro bias
  const Eigen::Vector3d true_gyro_bias(0.01, 0.0, 0.0);
  const Eigen::Vector3d true_accel_bias = Eigen::Vector3d::Zero();

  // IMU data: zero real angular velocity, but measured as bias (gyro reads bias when stationary)
  // Acceleration: gravity compensation (stationary)
  const double dt = 0.01;
  const int num_steps = 100;
  const Eigen::Vector3d measured_gyro = true_gyro_bias;  // stationary + bias
  const Eigen::Vector3d measured_accel(0.0, 0.0, kGravityNominal);

  for (int i = 0; i <= num_steps; ++i)
  {
    int64_t ns = static_cast<int64_t>(i * dt * 1e9);
    imu_preint.addImuData(ImuData(vesta_core::Timestamp(ns), measured_gyro, measured_accel));
  }

  // Initialize at origin with zero biases (wrong bias estimate)
  imu_preint.setStart(stamp0, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(),
                       Eigen::Vector3d::Zero(), device_id);

  auto result = imu_preint.createPreintegratedFactor(stamp1, device_id, true);

  // Build the optimization problem
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = vesta_core::Loss::Ownership;
  ceres::Problem problem(problem_options);

  // State1 variables
  auto ori1 = vesta_variables::Orientation3DStamped::make_shared(stamp0, device_id);
  ori1->w() = 1.0; ori1->x() = 0.0; ori1->y() = 0.0; ori1->z() = 0.0;

  auto pos1 = vesta_variables::Position3DStamped::make_shared(stamp0, device_id);
  pos1->x() = 0.0; pos1->y() = 0.0; pos1->z() = 0.0;

  auto vel1 = vesta_variables::VelocityLinear3DStamped::make_shared(stamp0, device_id);
  vel1->x() = 0.0; vel1->y() = 0.0; vel1->z() = 0.0;

  auto gbias1 = vesta_variables::GyroscopeBias3DStamped::make_shared(stamp0, device_id);
  gbias1->x() = 0.0; gbias1->y() = 0.0; gbias1->z() = 0.0;

  auto abias1 = vesta_variables::AccelerationBias3DStamped::make_shared(stamp0, device_id);
  abias1->x() = 0.0; abias1->y() = 0.0; abias1->z() = 0.0;

  // Add state1 param blocks and fix pose/velocity (we know the robot is stationary)
  problem.AddParameterBlock(ori1->data(), ori1->size(), ori1->manifold());
  problem.AddParameterBlock(pos1->data(), pos1->size(), pos1->manifold());
  problem.AddParameterBlock(vel1->data(), vel1->size());
  problem.AddParameterBlock(gbias1->data(), gbias1->size());
  problem.AddParameterBlock(abias1->data(), abias1->size());

  problem.SetParameterBlockConstant(ori1->data());
  problem.SetParameterBlockConstant(pos1->data());
  problem.SetParameterBlockConstant(vel1->data());

  // Add state2 param blocks
  // We know state2 should also be stationary (identity orientation, zero pos, zero vel)
  result.orientation->w() = 1.0;
  result.orientation->x() = 0.0;
  result.orientation->y() = 0.0;
  result.orientation->z() = 0.0;
  result.position->x() = 0.0;
  result.position->y() = 0.0;
  result.position->z() = 0.0;
  result.velocity->x() = 0.0;
  result.velocity->y() = 0.0;
  result.velocity->z() = 0.0;

  problem.AddParameterBlock(result.orientation->data(), result.orientation->size(),
                             result.orientation->manifold());
  problem.AddParameterBlock(result.position->data(), result.position->size(),
                             result.position->manifold());
  problem.AddParameterBlock(result.velocity->data(), result.velocity->size());
  problem.AddParameterBlock(result.gyro_bias->data(), result.gyro_bias->size());
  problem.AddParameterBlock(result.accel_bias->data(), result.accel_bias->size());

  // Fix state2 pose and velocity as well (stationary)
  problem.SetParameterBlockConstant(result.orientation->data());
  problem.SetParameterBlockConstant(result.position->data());
  problem.SetParameterBlockConstant(result.velocity->data());

  // Add prior on state1
  {
    std::vector<double*> prior_blocks;
    prior_blocks.push_back(ori1->data());
    prior_blocks.push_back(pos1->data());
    prior_blocks.push_back(vel1->data());
    prior_blocks.push_back(gbias1->data());
    prior_blocks.push_back(abias1->data());

    problem.AddResidualBlock(
      result.prior_constraint->costFunction(),
      result.prior_constraint->lossFunction(),
      prior_blocks);
  }

  // Add relative constraint
  {
    std::vector<double*> rel_blocks;
    rel_blocks.push_back(ori1->data());
    rel_blocks.push_back(pos1->data());
    rel_blocks.push_back(vel1->data());
    rel_blocks.push_back(gbias1->data());
    rel_blocks.push_back(abias1->data());
    rel_blocks.push_back(result.orientation->data());
    rel_blocks.push_back(result.position->data());
    rel_blocks.push_back(result.velocity->data());
    rel_blocks.push_back(result.gyro_bias->data());
    rel_blocks.push_back(result.accel_bias->data());

    problem.AddResidualBlock(
      result.relative_constraint->costFunction(),
      result.relative_constraint->lossFunction(),
      rel_blocks);
  }

  // Solve
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // The bias variables should move towards the true bias.
  // With a short window and limited observability, we may not get exact convergence,
  // but the direction should be correct (gbias1.x moving positive).
  // Check that at least one of the bias states has moved in the right direction.
  // The biases are linked via the random-walk model (bg2 - bg1 ~ 0), so both should be similar.
  double avg_gyro_bias_x = (gbias1->x() + result.gyro_bias->x()) / 2.0;
  EXPECT_GT(avg_gyro_bias_x, 0.0);  // Should be positive (true bias is 0.01)
  EXPECT_LT(avg_gyro_bias_x, 0.05);  // Should be reasonable (not diverged)
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
