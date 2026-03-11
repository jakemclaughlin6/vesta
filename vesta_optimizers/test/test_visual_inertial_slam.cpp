#include <vesta_constraints/3d/absolute_pose_3d_stamped_constraint.h>
#include <vesta_constraints/inertial/absolute_imu_state_3d_stamped_constraint.h>
#include <vesta_constraints/inertial/imu_preintegrator.h>
#include <vesta_constraints/inertial/relative_imu_state_3d_stamped_constraint.h>
#include <vesta_constraints/vision/stereo_reprojection_error_constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/timestamp.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>
#include <vesta_graphs/hash_graph.h>
#include <vesta_optimizers/batch_optimizer.h>
#include <vesta_optimizers/batch_optimizer_params.h>
#include <vesta_variables/3d/acceleration_bias_3d_stamped.h>
#include <vesta_variables/3d/gyroscope_bias_3d_stamped.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/3d/velocity_linear_3d_stamped.h>
#include <vesta_variables/vision/point_3d_landmark.h>
#include <vesta_variables/vision/stereo_camera_fixed.h>

#include "common.h"

#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

static constexpr double kGravity = 9.80665;

// Camera intrinsics
static constexpr double kFx = 500.0;
static constexpr double kFy = 500.0;
static constexpr double kCx = 320.0;
static constexpr double kCy = 240.0;
static constexpr double kBaseline = 0.12;
static constexpr double kImageW = 640.0;
static constexpr double kImageH = 480.0;

// Ground truth keyframe state
struct KeyframeGT
{
  double t;
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
  Eigen::Vector3d velocity;
};

// Project a world point through a stereo camera at a given world-frame pose.
// Uses the world-frame convention: p_cam = R_wc^{-1} * (X_world - p_world)
// Returns (u_left, v_left, u_right, v_right) and sets z_cam for depth check.
static Eigen::Vector4d projectStereo(const Eigen::Vector3d& cam_world_pos, const Eigen::Quaterniond& cam_world_ori,
                                     const Eigen::Vector3d& landmark, double& z_cam)
{
  // p_cam = R_wc^{-1} * (X - p) = R_cw * (X - p)
  Eigen::Vector3d diff = landmark - cam_world_pos;
  Eigen::Vector3d p_cam = cam_world_ori.inverse() * diff;
  z_cam = p_cam.z();

  double inv_z = 1.0 / p_cam.z();
  double u_left = kFx * p_cam.x() * inv_z + kCx;
  double v_left = kFy * p_cam.y() * inv_z + kCy;
  double u_right = kFx * (p_cam.x() - kBaseline) * inv_z + kCx;
  double v_right = kFy * p_cam.y() * inv_z + kCy;
  return { u_left, v_left, u_right, v_right };
}

// Generate random 3D landmarks visible from the trajectory
static std::vector<Eigen::Vector3d> generateLandmarks(const std::vector<KeyframeGT>& gt, int num_landmarks,
                                                      std::mt19937& rng)
{
  std::vector<Eigen::Vector3d> landmarks;
  landmarks.reserve(num_landmarks);

  // Spread landmarks in a volume around the trajectory
  // x: slightly beyond trajectory range, y: +/- 3m, z: 3-15m in front
  double x_min = gt.front().position.x() - 2.0;
  double x_max = gt.back().position.x() + 2.0;
  std::uniform_real_distribution<double> x_dist(x_min, x_max);
  std::uniform_real_distribution<double> y_dist(-3.0, 3.0);
  std::uniform_real_distribution<double> z_dist(3.0, 15.0);

  for (int i = 0; i < num_landmarks; ++i)
  {
    Eigen::Vector3d lm(x_dist(rng), y_dist(rng), z_dist(rng));

    // Verify it's visible from at least 2 keyframes
    int visible_count = 0;
    for (const auto& kf : gt)
    {
      double z_cam;
      auto obs = projectStereo(kf.position, kf.orientation, lm, z_cam);
      if (z_cam > 0.5 && obs[0] >= 0 && obs[0] <= kImageW && obs[1] >= 0 && obs[1] <= kImageH)
      {
        ++visible_count;
      }
    }
    if (visible_count >= 2)
    {
      landmarks.push_back(lm);
    }
    else
    {
      --i;  // Retry
    }
  }
  return landmarks;
}

// Helper: create IMU state variables for a keyframe
struct ImuStateVars
{
  std::shared_ptr<vesta_variables::Orientation3DStamped> ori;
  std::shared_ptr<vesta_variables::Position3DStamped> pos;
  std::shared_ptr<vesta_variables::VelocityLinear3DStamped> vel;
  std::shared_ptr<vesta_variables::GyroscopeBias3DStamped> gbias;
  std::shared_ptr<vesta_variables::AccelerationBias3DStamped> abias;
};

static ImuStateVars createImuState(const KeyframeGT& kf, const vesta_core::UUID& device_id, std::mt19937& rng,
                                   double pos_sigma, double vel_sigma)
{
  std::normal_distribution<double> pn(0.0, pos_sigma);
  std::normal_distribution<double> vn(0.0, vel_sigma);

  auto stamp = vesta_core::Timestamp(static_cast<int64_t>(kf.t * 1e9));

  ImuStateVars s;
  s.ori = std::make_shared<vesta_variables::Orientation3DStamped>(stamp, device_id);
  s.pos = std::make_shared<vesta_variables::Position3DStamped>(stamp, device_id);
  s.vel = std::make_shared<vesta_variables::VelocityLinear3DStamped>(stamp, device_id);
  s.gbias = std::make_shared<vesta_variables::GyroscopeBias3DStamped>(stamp, device_id);
  s.abias = std::make_shared<vesta_variables::AccelerationBias3DStamped>(stamp, device_id);

  s.ori->w() = kf.orientation.w();
  s.ori->x() = kf.orientation.x();
  s.ori->y() = kf.orientation.y();
  s.ori->z() = kf.orientation.z();
  s.pos->x() = kf.position.x() + pn(rng);
  s.pos->y() = kf.position.y() + pn(rng);
  s.pos->z() = kf.position.z() + pn(rng);
  s.vel->x() = kf.velocity.x() + vn(rng);
  s.vel->y() = kf.velocity.y() + vn(rng);
  s.vel->z() = kf.velocity.z() + vn(rng);
  s.gbias->x() = 0.0;
  s.gbias->y() = 0.0;
  s.gbias->z() = 0.0;
  s.abias->x() = 0.0;
  s.abias->y() = 0.0;
  s.abias->z() = 0.0;

  return s;
}

// Helper: build IMU preintegrators between consecutive keyframes
static std::vector<vesta_constraints::ImuPreintegrator> buildPreintegrators(const std::vector<KeyframeGT>& gt)
{
  const int num_segments = static_cast<int>(gt.size()) - 1;
  std::vector<vesta_constraints::ImuPreintegrator> preintegrators(num_segments);

  for (int seg = 0; seg < num_segments; ++seg)
  {
    auto& preint = preintegrators[seg];
    preint.cov_gyro = Eigen::Matrix3d::Identity() * 1e-4;
    preint.cov_accel = Eigen::Matrix3d::Identity() * 1e-3;
    preint.cov_gyro_bias = Eigen::Matrix3d::Identity() * 1e-6;
    preint.cov_accel_bias = Eigen::Matrix3d::Identity() * 1e-4;

    double t_start = gt[seg].t;
    double t_end = gt[seg + 1].t;
    const int num_imu_samples = 100;
    double dt = (t_end - t_start) / num_imu_samples;

    for (int k = 0; k <= num_imu_samples; ++k)
    {
      double t = t_start + k * dt;
      vesta_constraints::ImuData sample;
      sample.stamp = vesta_core::Timestamp(static_cast<int64_t>(t * 1e9));
      sample.angular_velocity = Eigen::Vector3d::Zero();
      // For constant velocity with identity orientation, the accelerometer
      // reads gravity in the body frame (compensating for gravity).
      sample.linear_acceleration = Eigen::Vector3d(0.0, 0.0, kGravity);
      preint.data[sample.stamp] = sample;
    }

    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    auto end_stamp = vesta_core::Timestamp(static_cast<int64_t>(t_end * 1e9));
    bool ok = preint.integrate(end_stamp, bg, ba, true, true, true);
    EXPECT_TRUE(ok) << "IMU preintegration failed for segment " << seg;
  }

  return preintegrators;
}

// =============================================================================
// Test 1: IMU + AbsolutePose priors (original test, kept for reference)
//
// Demonstrates that IMU preintegration works with world-frame pose priors
// representing a visual frontend.
// =============================================================================
TEST(VisualInertialSlam, BatchWithPosePriors)
{
  const int num_keyframes = 3;
  const auto device_id = vesta_core::uuid::generate("imu");

  std::vector<KeyframeGT> gt = {
    { 0.0, { 0.0, 0.0, 0.0 }, Eigen::Quaterniond::Identity(), { 1.0, 0.0, 0.0 } },
    { 1.0, { 1.0, 0.0, 0.0 }, Eigen::Quaterniond::Identity(), { 1.0, 0.0, 0.0 } },
    { 2.0, { 2.0, 0.0, 0.0 }, Eigen::Quaterniond::Identity(), { 1.0, 0.0, 0.0 } },
  };

  std::mt19937 rng(42);
  std::vector<ImuStateVars> states;
  for (int i = 0; i < num_keyframes; ++i)
  {
    states.push_back(createImuState(gt[i], device_id, rng, 0.05, 0.05));
  }

  auto preintegrators = buildPreintegrators(gt);

  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(static_cast<int64_t>(gt.back().t * 1e9)));

  for (int i = 0; i < num_keyframes; ++i)
  {
    txn->addVariable(states[i].ori);
    txn->addVariable(states[i].pos);
    txn->addVariable(states[i].vel);
    txn->addVariable(states[i].gbias);
    txn->addVariable(states[i].abias);
  }

  // Prior on first IMU state
  {
    Eigen::Matrix<double, 16, 1> mean;
    mean << gt[0].orientation.w(), gt[0].orientation.x(), gt[0].orientation.y(), gt[0].orientation.z(),
        gt[0].position.x(), gt[0].position.y(), gt[0].position.z(), gt[0].velocity.x(), gt[0].velocity.y(),
        gt[0].velocity.z(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Identity() * 1e-6;
    auto prior = std::make_shared<vesta_constraints::AbsoluteImuState3DStampedConstraint>(
        "prior", *states[0].ori, *states[0].pos, *states[0].vel, *states[0].gbias, *states[0].abias, mean, cov);
    txn->addConstraint(prior);
  }

  // IMU preintegration constraints
  Eigen::Vector3d gravity_world(0.0, 0.0, -kGravity);
  for (int seg = 0; seg < num_keyframes - 1; ++seg)
  {
    Eigen::Vector3d bg_lin = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba_lin = Eigen::Vector3d::Zero();
    auto imu_constraint = std::make_shared<vesta_constraints::RelativeImuState3DStampedConstraint>(
        "imu", *states[seg].ori, *states[seg].pos, *states[seg].vel, *states[seg].gbias, *states[seg].abias,
        *states[seg + 1].ori, *states[seg + 1].pos, *states[seg + 1].vel, *states[seg + 1].gbias,
        *states[seg + 1].abias, preintegrators[seg], bg_lin, ba_lin, gravity_world);
    txn->addConstraint(imu_constraint);
  }

  // Absolute pose priors at keyframes 1 and 2
  std::normal_distribution<double> visual_noise(0.0, 0.02);
  for (int i = 1; i < num_keyframes; ++i)
  {
    vesta_core::Vector7d mean;
    mean << gt[i].position.x() + visual_noise(rng), gt[i].position.y() + visual_noise(rng),
        gt[i].position.z() + visual_noise(rng), gt[i].orientation.w(), gt[i].orientation.x(), gt[i].orientation.y(),
        gt[i].orientation.z();
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 0.01;
    auto visual_prior = std::make_shared<vesta_constraints::AbsolutePose3DStampedConstraint>("visual", *states[i].pos,
                                                                                             *states[i].ori, mean, cov);
    txn->addConstraint(visual_prior);
  }

  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 200;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));
  optimizer.addTransaction("vio", txn);
  auto summary = optimizer.optimize();
  logSolverSummary("VIO::BatchWithPosePriors", summary);

  ASSERT_TRUE(summary.IsSolutionUsable());

  const auto& g = optimizer.graph();
  for (int i = 0; i < num_keyframes; ++i)
  {
    const auto& pos = dynamic_cast<const vesta_variables::Position3DStamped&>(g.getVariable(states[i].pos->uuid()));
    EXPECT_NEAR(pos.x(), gt[i].position.x(), 0.05) << "KF " << i << " position x";
    EXPECT_NEAR(pos.y(), gt[i].position.y(), 0.05) << "KF " << i << " position y";
    EXPECT_NEAR(pos.z(), gt[i].position.z(), 0.05) << "KF " << i << " position z";

    const auto& vel =
        dynamic_cast<const vesta_variables::VelocityLinear3DStamped&>(g.getVariable(states[i].vel->uuid()));
    EXPECT_NEAR(vel.x(), gt[i].velocity.x(), 0.05) << "KF " << i << " velocity x";
    EXPECT_NEAR(vel.y(), gt[i].velocity.y(), 0.05) << "KF " << i << " velocity y";
    EXPECT_NEAR(vel.z(), gt[i].velocity.z(), 0.05) << "KF " << i << " velocity z";
  }
}

// =============================================================================
// Test 2: IMU + Stereo Reprojection (world-frame convention)
//
// This test uses StereoReprojectionErrorConstraint which computes
// p_cam = R_wc^{-1} * (X_world - p_world), sharing the same position and
// orientation variables as the IMU preintegration constraint.
// =============================================================================
TEST(VisualInertialSlam, BatchWithStereoReprojection)
{
  const int num_keyframes = 5;
  const int num_landmarks = 50;
  const auto device_id = vesta_core::uuid::generate("imu");

  // Ground truth: constant velocity along x-axis
  std::vector<KeyframeGT> gt;
  for (int i = 0; i < num_keyframes; ++i)
  {
    gt.push_back({ static_cast<double>(i),
                   { static_cast<double>(i), 0.0, 0.0 },
                   Eigen::Quaterniond::Identity(),
                   { 1.0, 0.0, 0.0 } });
  }

  std::mt19937 rng(42);

  // Generate random landmarks visible from multiple keyframes
  auto gt_landmarks = generateLandmarks(gt, num_landmarks, rng);
  const int actual_num_landmarks = static_cast<int>(gt_landmarks.size());

  // Create IMU state variables
  std::vector<ImuStateVars> states;
  for (int i = 0; i < num_keyframes; ++i)
  {
    states.push_back(createImuState(gt[i], device_id, rng, 0.05, 0.05));
  }

  // Create landmark variables
  std::normal_distribution<double> lm_noise(0.0, 0.2);
  std::vector<std::shared_ptr<vesta_variables::Point3DLandmark>> landmarks;
  for (int j = 0; j < actual_num_landmarks; ++j)
  {
    auto lm = std::make_shared<vesta_variables::Point3DLandmark>(static_cast<uint64_t>(j));
    lm->x() = gt_landmarks[j].x() + lm_noise(rng);
    lm->y() = gt_landmarks[j].y() + lm_noise(rng);
    lm->z() = gt_landmarks[j].z() + lm_noise(rng);
    landmarks.push_back(lm);
  }

  // Stereo camera intrinsics (held constant)
  auto stereo_cam = std::make_shared<vesta_variables::StereoCameraFixed>(uint64_t{ 0 });
  stereo_cam->fx() = kFx;
  stereo_cam->fy() = kFy;
  stereo_cam->cx() = kCx;
  stereo_cam->cy() = kCy;
  stereo_cam->baseline() = kBaseline;

  auto preintegrators = buildPreintegrators(gt);

  // Build transaction
  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(static_cast<int64_t>(gt.back().t * 1e9)));

  // Add all variables
  for (int i = 0; i < num_keyframes; ++i)
  {
    txn->addVariable(states[i].ori);
    txn->addVariable(states[i].pos);
    txn->addVariable(states[i].vel);
    txn->addVariable(states[i].gbias);
    txn->addVariable(states[i].abias);
  }
  txn->addVariable(stereo_cam);
  for (int j = 0; j < actual_num_landmarks; ++j)
  {
    txn->addVariable(landmarks[j]);
  }

  // Prior on first IMU state (tight)
  {
    Eigen::Matrix<double, 16, 1> mean;
    mean << gt[0].orientation.w(), gt[0].orientation.x(), gt[0].orientation.y(), gt[0].orientation.z(),
        gt[0].position.x(), gt[0].position.y(), gt[0].position.z(), gt[0].velocity.x(), gt[0].velocity.y(),
        gt[0].velocity.z(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Identity() * 1e-6;
    auto prior = std::make_shared<vesta_constraints::AbsoluteImuState3DStampedConstraint>(
        "prior", *states[0].ori, *states[0].pos, *states[0].vel, *states[0].gbias, *states[0].abias, mean, cov);
    txn->addConstraint(prior);
  }

  // IMU preintegration constraints
  Eigen::Vector3d gravity_world(0.0, 0.0, -kGravity);
  for (int seg = 0; seg < num_keyframes - 1; ++seg)
  {
    Eigen::Vector3d bg_lin = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba_lin = Eigen::Vector3d::Zero();
    auto imu_constraint = std::make_shared<vesta_constraints::RelativeImuState3DStampedConstraint>(
        "imu", *states[seg].ori, *states[seg].pos, *states[seg].vel, *states[seg].gbias, *states[seg].abias,
        *states[seg + 1].ori, *states[seg + 1].pos, *states[seg + 1].vel, *states[seg + 1].gbias,
        *states[seg + 1].abias, preintegrators[seg], bg_lin, ba_lin, gravity_world);
    txn->addConstraint(imu_constraint);
  }

  // Stereo reprojection constraints using world-frame convention
  // Project each landmark into each keyframe camera
  std::normal_distribution<double> pixel_noise(0.0, 0.5);
  int total_observations = 0;
  for (int i = 0; i < num_keyframes; ++i)
  {
    for (int j = 0; j < actual_num_landmarks; ++j)
    {
      double z_cam;
      auto obs = projectStereo(gt[i].position, gt[i].orientation, gt_landmarks[j], z_cam);
      if (z_cam <= 0.5)
        continue;
      if (obs[0] < 0 || obs[0] > kImageW)
        continue;
      if (obs[1] < 0 || obs[1] > kImageH)
        continue;

      vesta_core::Vector4d obs_noisy;
      obs_noisy << obs[0] + pixel_noise(rng), obs[1] + pixel_noise(rng), obs[2] + pixel_noise(rng),
          obs[3] + pixel_noise(rng);
      vesta_core::Matrix4d cov = vesta_core::Matrix4d::Identity() * 1.0;

      auto reproj = std::make_shared<vesta_constraints::StereoReprojectionErrorConstraint>(
          "stereo", *states[i].pos, *states[i].ori, *stereo_cam, *landmarks[j], obs_noisy, cov);
      txn->addConstraint(reproj);
      ++total_observations;
    }
  }

  ASSERT_GT(total_observations, 100) << "Not enough stereo observations for a well-constrained problem";

  // Optimize
  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 200;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));
  optimizer.addTransaction("vio", txn);
  auto summary = optimizer.optimize();
  logSolverSummary("VIO::BatchWithStereoReprojection", summary);

  ASSERT_TRUE(summary.IsSolutionUsable());

  // Verify convergence
  const auto& g = optimizer.graph();

  // Check positions
  for (int i = 0; i < num_keyframes; ++i)
  {
    const auto& pos = dynamic_cast<const vesta_variables::Position3DStamped&>(g.getVariable(states[i].pos->uuid()));
    EXPECT_NEAR(pos.x(), gt[i].position.x(), 0.05) << "KF " << i << " position x";
    EXPECT_NEAR(pos.y(), gt[i].position.y(), 0.05) << "KF " << i << " position y";
    EXPECT_NEAR(pos.z(), gt[i].position.z(), 0.05) << "KF " << i << " position z";
  }

  // Check velocities
  for (int i = 0; i < num_keyframes; ++i)
  {
    const auto& vel =
        dynamic_cast<const vesta_variables::VelocityLinear3DStamped&>(g.getVariable(states[i].vel->uuid()));
    EXPECT_NEAR(vel.x(), gt[i].velocity.x(), 0.03) << "KF " << i << " velocity x";
    EXPECT_NEAR(vel.y(), gt[i].velocity.y(), 0.03) << "KF " << i << " velocity y";
    EXPECT_NEAR(vel.z(), gt[i].velocity.z(), 0.03) << "KF " << i << " velocity z";
  }

  // Check biases
  for (int i = 0; i < num_keyframes; ++i)
  {
    const auto& gb =
        dynamic_cast<const vesta_variables::GyroscopeBias3DStamped&>(g.getVariable(states[i].gbias->uuid()));
    EXPECT_NEAR(gb.x(), 0.0, 0.005) << "KF " << i << " gyro bias x";
    EXPECT_NEAR(gb.y(), 0.0, 0.005) << "KF " << i << " gyro bias y";
    EXPECT_NEAR(gb.z(), 0.0, 0.005) << "KF " << i << " gyro bias z";

    const auto& ab =
        dynamic_cast<const vesta_variables::AccelerationBias3DStamped&>(g.getVariable(states[i].abias->uuid()));
    EXPECT_NEAR(ab.x(), 0.0, 0.01) << "KF " << i << " accel bias x";
    EXPECT_NEAR(ab.y(), 0.0, 0.01) << "KF " << i << " accel bias y";
    EXPECT_NEAR(ab.z(), 0.0, 0.01) << "KF " << i << " accel bias z";
  }

  // Check landmarks
  for (int j = 0; j < actual_num_landmarks; ++j)
  {
    const auto& lm = dynamic_cast<const vesta_variables::Point3DLandmark&>(g.getVariable(landmarks[j]->uuid()));
    EXPECT_NEAR(lm.x(), gt_landmarks[j].x(), 0.35) << "Landmark " << j << " x";
    EXPECT_NEAR(lm.y(), gt_landmarks[j].y(), 0.35) << "Landmark " << j << " y";
    EXPECT_NEAR(lm.z(), gt_landmarks[j].z(), 0.35) << "Landmark " << j << " z";
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
