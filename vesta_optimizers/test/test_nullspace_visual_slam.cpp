/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2024, Locus Robotics
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

/**
 * @file test_nullspace_visual_slam.cpp
 * @brief End-to-end visual SLAM tests using nullspace projection constraints.
 *
 * These tests mirror the structure of test_visual_slam.cpp but use the structureless
 * NullspaceProjectionConstraint and StereoNullspaceProjectionConstraint, which
 * analytically eliminate landmark variables via nullspace projection of the
 * landmark Jacobian. The graph contains only camera pose variables.
 *
 * Additionally, marginalization tests verify that QR, Schur, and BlockDiagonal
 * marginalizers produce correct results when operating on graphs containing
 * nullspace constraints (which have no non-stamped variables).
 */

#include <vesta_constraints/3d/absolute_pose_3d_stamped_constraint.h>
#include <vesta_constraints/3d/relative_pose_3d_stamped_constraint.h>
#include <vesta_constraints/common/block_diagonal_marginalizer.h>
#include <vesta_constraints/common/marginalizer.h>
#include <vesta_constraints/common/qr_marginalizer.h>
#include <vesta_constraints/common/schur_marginalizer.h>
#include <vesta_constraints/vision/nullspace_projection_constraint.h>
#include <vesta_constraints/vision/stereo_nullspace_projection_constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/timestamp.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>
#include <vesta_graphs/hash_graph.h>
#include <vesta_optimizers/batch_optimizer.h>
#include <vesta_optimizers/batch_optimizer_params.h>
#include <vesta_optimizers/fixed_lag_smoother.h>
#include <vesta_optimizers/fixed_lag_smoother_params.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/pinhole_camera.h>
#include <vesta_variables/vision/pinhole_camera_fixed.h>
#include <vesta_variables/vision/stereo_camera.h>
#include <vesta_variables/vision/stereo_camera_fixed.h>

#include "common.h"

#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

// Ground truth camera positions along x-axis (world frame)
static const std::vector<Eigen::Vector3d> kCamPositions = {
  { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 2.0, 0.0, 0.0 }, { 3.0, 0.0, 0.0 }, { 4.0, 0.0, 0.0 },
};

// Ground truth 3D landmarks spread in front of cameras
static const std::vector<Eigen::Vector3d> kLandmarks = {
  { 1.0, 1.0, 5.0 },  { 2.0, -1.0, 6.0 }, { 3.0, 0.5, 7.0 },  { -1.0, 2.0, 8.0 },
  { 0.0, -1.5, 5.5 }, { 4.0, 1.0, 6.5 },  { 2.5, -0.5, 9.0 }, { 1.5, 1.5, 10.0 },
};

// Camera intrinsics
static constexpr double kFx = 500.0;
static constexpr double kFy = 500.0;
static constexpr double kCx = 320.0;
static constexpr double kCy = 240.0;
static constexpr double kBaseline = 0.12;

static constexpr size_t kNumCameras = 5;
static constexpr size_t kNumLandmarks = 8;

// Project a world point into a camera with identity orientation at the given
// world position. Returns (u, v) and sets z_cam for depth check.
static Eigen::Vector2d projectMono(const Eigen::Vector3d& cam_world_pos, const Eigen::Vector3d& landmark, double& z_cam)
{
  Eigen::Vector3d p = landmark - cam_world_pos;
  z_cam = p.z();
  double u = kFx * p.x() / p.z() + kCx;
  double v = kFy * p.y() / p.z() + kCy;
  return { u, v };
}

// Returns (u_left, v_left, u_right, v_right) stereo observation
static Eigen::Vector4d projectStereo(const Eigen::Vector3d& cam_world_pos, const Eigen::Vector3d& landmark,
                                     double& z_cam)
{
  Eigen::Vector3d p = landmark - cam_world_pos;
  z_cam = p.z();
  double u_left = kFx * p.x() / p.z() + kCx;
  double v_left = kFy * p.y() / p.z() + kCy;
  double u_right = kFx * (p.x() - kBaseline) / p.z() + kCx;
  double v_right = kFy * p.y() / p.z() + kCy;
  return { u_left, v_left, u_right, v_right };
}

/**
 * @brief Helper: create a PinholeCamera variable with the test intrinsics.
 *
 * The nullspace constraint constructor takes a PinholeCamera (non-fixed) for
 * calibration extraction, but the calibration is stored internally and not optimized.
 * We use PinholeCameraFixed so it's also held constant if accidentally added to a graph.
 */
static vesta_variables::PinholeCameraFixed::SharedPtr makeMonoCalibration()
{
  auto cam = vesta_variables::PinholeCameraFixed::make_shared(uint64_t{ 0 });
  cam->fx() = kFx;
  cam->fy() = kFy;
  cam->cx() = kCx;
  cam->cy() = kCy;
  return cam;
}

/**
 * @brief Helper: create a StereoCameraFixed variable with the test intrinsics.
 */
static vesta_variables::StereoCameraFixed::SharedPtr makeStereoCalibration()
{
  auto cam = vesta_variables::StereoCameraFixed::make_shared(uint64_t{ 0 });
  cam->fx() = kFx;
  cam->fy() = kFy;
  cam->cx() = kCx;
  cam->cy() = kCy;
  cam->baseline() = kBaseline;
  return cam;
}

/**
 * @brief Helper: create camera pose variables for all keyframes.
 *
 * Positions are initialized at ground truth + noise; orientations at identity + small noise.
 */
struct CameraPoses
{
  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;
};

static CameraPoses createCameraPoses(const vesta_core::UUID& cam_device_id, std::mt19937& rng, double pos_sigma)
{
  std::normal_distribution<double> pos_noise(0.0, pos_sigma);
  CameraPoses poses;
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    auto pos = vesta_variables::Position3DStamped::make_shared(vesta_core::Timestamp(i, 0), cam_device_id);
    auto ori = vesta_variables::Orientation3DStamped::make_shared(vesta_core::Timestamp(i, 0), cam_device_id);

    pos->x() = kCamPositions[i].x() + pos_noise(rng);
    pos->y() = kCamPositions[i].y() + pos_noise(rng);
    pos->z() = kCamPositions[i].z() + pos_noise(rng);

    ori->w() = 1.0;
    ori->x() = pos_noise(rng) * 0.01;
    ori->y() = pos_noise(rng) * 0.01;
    ori->z() = pos_noise(rng) * 0.01;
    double norm = std::sqrt(ori->w() * ori->w() + ori->x() * ori->x() + ori->y() * ori->y() + ori->z() * ori->z());
    ori->w() /= norm;
    ori->x() /= norm;
    ori->y() /= norm;
    ori->z() /= norm;

    poses.positions.push_back(pos);
    poses.orientations.push_back(ori);
  }
  return poses;
}

/**
 * @brief Helper: add a pose prior constraint on the first camera.
 */
static void addPosePrior(vesta_core::Transaction& txn, const vesta_variables::Position3DStamped& pos,
                         const vesta_variables::Orientation3DStamped& ori, double cov_scale)
{
  vesta_core::Vector7d prior_mean;
  prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * cov_scale;
  auto prior = vesta_constraints::AbsolutePose3DStampedConstraint::make_shared("prior", pos, ori, prior_mean, prior_cov);
  txn.addConstraint(prior);
}

/**
 * @brief Helper: add relative pose (odometry) constraints between consecutive cameras.
 */
static void addOdometryConstraints(vesta_core::Transaction& txn, const CameraPoses& poses)
{
  for (size_t i = 0; i < kNumCameras - 1; ++i)
  {
    Eigen::Vector3d dt = kCamPositions[i + 1] - kCamPositions[i];
    vesta_core::Vector7d delta;
    delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 0.01;
    auto rel = vesta_constraints::RelativePose3DStampedConstraint::make_shared(
        "odom", *poses.positions[i], *poses.orientations[i], *poses.positions[i + 1], *poses.orientations[i + 1], delta,
        cov);
    txn.addConstraint(rel);
  }
}

/**
 * @brief Helper: create mono nullspace constraints for all landmarks.
 *
 * For each landmark, collects all camera observations, dereferences the
 * pose variables, and creates a NullspaceProjectionConstraint.
 */
static void addMonoNullspaceConstraints(vesta_core::Transaction& txn, const CameraPoses& poses,
                                        const vesta_variables::PinholeCamera& calibration, std::mt19937& rng,
                                        double pixel_sigma, bool add_noise)
{
  std::normal_distribution<double> pixel_noise(0.0, pixel_sigma);
  vesta_core::Matrix2d pixel_cov = vesta_core::Matrix2d::Identity() * (pixel_sigma * pixel_sigma);

  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    std::vector<vesta_variables::Position3DStamped> obs_positions;
    std::vector<vesta_variables::Orientation3DStamped> obs_orientations;
    std::vector<Eigen::Vector2d> observations;

    for (size_t i = 0; i < kNumCameras; ++i)
    {
      double z_cam = 0.0;
      Eigen::Vector2d uv = projectMono(kCamPositions[i], kLandmarks[j], z_cam);
      if (z_cam <= 0.0)
      {
        continue;
      }

      Eigen::Vector2d obs;
      if (add_noise)
      {
        obs << uv.x() + pixel_noise(rng), uv.y() + pixel_noise(rng);
      }
      else
      {
        obs = uv;
      }

      obs_positions.push_back(*poses.positions[i]);
      obs_orientations.push_back(*poses.orientations[i]);
      observations.push_back(obs);
    }

    if (observations.size() >= 2)
    {
      auto constraint = vesta_constraints::NullspaceProjectionConstraint::make_shared(
          "camera", obs_positions, obs_orientations, calibration, observations, pixel_cov);
      txn.addConstraint(constraint);
    }
  }
}

/**
 * @brief Helper: create stereo nullspace constraints for all landmarks.
 */
static void addStereoNullspaceConstraints(vesta_core::Transaction& txn, const CameraPoses& poses,
                                          const vesta_variables::StereoCamera& calibration, std::mt19937& rng,
                                          double pixel_sigma, bool add_noise)
{
  std::normal_distribution<double> pixel_noise(0.0, pixel_sigma);
  vesta_core::Matrix4d pixel_cov = vesta_core::Matrix4d::Identity() * (pixel_sigma * pixel_sigma);

  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    std::vector<vesta_variables::Position3DStamped> obs_positions;
    std::vector<vesta_variables::Orientation3DStamped> obs_orientations;
    std::vector<Eigen::Vector4d> observations;

    for (size_t i = 0; i < kNumCameras; ++i)
    {
      double z_cam = 0.0;
      Eigen::Vector4d obs_gt = projectStereo(kCamPositions[i], kLandmarks[j], z_cam);
      if (z_cam <= 0.0)
      {
        continue;
      }

      Eigen::Vector4d obs;
      if (add_noise)
      {
        obs << obs_gt[0] + pixel_noise(rng), obs_gt[1] + pixel_noise(rng), obs_gt[2] + pixel_noise(rng),
            obs_gt[3] + pixel_noise(rng);
      }
      else
      {
        obs = obs_gt;
      }

      obs_positions.push_back(*poses.positions[i]);
      obs_orientations.push_back(*poses.orientations[i]);
      observations.push_back(obs);
    }

    if (observations.size() >= 2)
    {
      auto constraint = vesta_constraints::StereoNullspaceProjectionConstraint::make_shared(
          "stereo_camera", obs_positions, obs_orientations, calibration, observations, pixel_cov);
      txn.addConstraint(constraint);
    }
  }
}

// ===========================================================================
// Test: Mono batch optimization with nullspace constraints
// ===========================================================================
TEST(NullspaceVisualSlamTest, NullspaceVisualSlam_MonoBatch)
{
  std::mt19937 rng(42);
  const auto cam_device_id = vesta_core::uuid::generate("cam");
  auto calibration = makeMonoCalibration();
  auto poses = createCameraPoses(cam_device_id, rng, 0.05);

  // Build a single transaction with pose variables and nullspace constraints
  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(kNumCameras - 1, 0));
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    txn->addInvolvedStamp(vesta_core::Timestamp(i, 0));
    txn->addVariable(poses.positions[i]);
    txn->addVariable(poses.orientations[i]);
  }

  // Pose prior on first camera
  addPosePrior(*txn, *poses.positions[0], *poses.orientations[0], 1e-4);

  // Odometry constraints
  addOdometryConstraints(*txn, poses);

  // Nullspace projection constraints (one per landmark, linking all observing cameras)
  addMonoNullspaceConstraints(*txn, poses, *calibration, rng, 1.0, true);

  // Set up batch optimizer
  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

  optimizer.addTransaction("nullspace_vslam", txn);
  auto summary = optimizer.optimize();
  logSolverSummary("NullspaceVisualSlam::MonoBatch", summary);

  ASSERT_TRUE(summary.IsSolutionUsable());

  // Check camera positions — nullspace + odometry + prior should anchor scale
  const auto& result_graph = optimizer.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(poses.positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 0.15) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 0.15) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 0.15) << "Camera " << i << " z";
  }

  // Verify no landmark variables exist in the graph
  // (nullspace constraints are structureless)
  size_t num_vars = 0;
  for (auto it = result_graph.getVariables().begin(); it != result_graph.getVariables().end(); ++it)
  {
    ++num_vars;
  }
  EXPECT_EQ(num_vars, 2 * kNumCameras) << "Graph should contain only pose variables (pos + ori per camera)";
}

// ===========================================================================
// Test: Stereo batch optimization with nullspace constraints
// ===========================================================================
TEST(NullspaceVisualSlamTest, NullspaceVisualSlam_StereoBatch)
{
  std::mt19937 rng(123);
  const auto cam_device_id = vesta_core::uuid::generate("cam");
  auto calibration = makeStereoCalibration();
  auto poses = createCameraPoses(cam_device_id, rng, 0.05);

  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(kNumCameras - 1, 0));
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    txn->addInvolvedStamp(vesta_core::Timestamp(i, 0));
    txn->addVariable(poses.positions[i]);
    txn->addVariable(poses.orientations[i]);
  }

  addPosePrior(*txn, *poses.positions[0], *poses.orientations[0], 1e-4);
  addOdometryConstraints(*txn, poses);
  addStereoNullspaceConstraints(*txn, poses, *calibration, rng, 1.0, true);

  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

  optimizer.addTransaction("nullspace_vslam", txn);
  auto summary = optimizer.optimize();
  logSolverSummary("NullspaceVisualSlam::StereoBatch", summary);

  ASSERT_TRUE(summary.IsSolutionUsable());

  // Stereo nullspace has scale observability — check absolute positions
  const auto& result_graph = optimizer.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(poses.positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 0.1) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 0.1) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 0.1) << "Camera " << i << " z";
  }
}

// ===========================================================================
// Test: Noiseless mono batch — verifies convergence (with odometry for scale)
// ===========================================================================
TEST(NullspaceVisualSlamTest, NullspaceVisualSlam_MonoBatchNoiseless)
{
  std::mt19937 rng(42);
  const auto cam_device_id = vesta_core::uuid::generate("cam");
  auto calibration = makeMonoCalibration();
  auto poses = createCameraPoses(cam_device_id, rng, 0.05);

  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(kNumCameras - 1, 0));
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    txn->addInvolvedStamp(vesta_core::Timestamp(i, 0));
    txn->addVariable(poses.positions[i]);
    txn->addVariable(poses.orientations[i]);
  }

  // Tight prior to anchor first pose
  addPosePrior(*txn, *poses.positions[0], *poses.orientations[0], 1e-8);
  addOdometryConstraints(*txn, poses);
  addMonoNullspaceConstraints(*txn, poses, *calibration, rng, 1.0, false);  // noiseless observations

  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

  optimizer.addTransaction("nullspace_vslam", txn);
  auto summary = optimizer.optimize();
  logSolverSummary("NullspaceVisualSlam::MonoBatchNoiseless", summary);

  ASSERT_TRUE(summary.IsSolutionUsable());

  const auto& result_graph = optimizer.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(poses.positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 1e-3) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 1e-3) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 1e-3) << "Camera " << i << " z";
  }
}

// ===========================================================================
// Test: Noiseless stereo batch — verifies exact convergence
// ===========================================================================
TEST(NullspaceVisualSlamTest, NullspaceVisualSlam_StereoBatchNoiseless)
{
  std::mt19937 rng(42);
  const auto cam_device_id = vesta_core::uuid::generate("cam");
  auto calibration = makeStereoCalibration();
  auto poses = createCameraPoses(cam_device_id, rng, 0.05);

  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(kNumCameras - 1, 0));
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    txn->addInvolvedStamp(vesta_core::Timestamp(i, 0));
    txn->addVariable(poses.positions[i]);
    txn->addVariable(poses.orientations[i]);
  }

  addPosePrior(*txn, *poses.positions[0], *poses.orientations[0], 1e-8);
  addOdometryConstraints(*txn, poses);
  addStereoNullspaceConstraints(*txn, poses, *calibration, rng, 1.0, false);

  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

  optimizer.addTransaction("nullspace_vslam", txn);
  auto summary = optimizer.optimize();
  logSolverSummary("NullspaceVisualSlam::StereoBatchNoiseless", summary);

  ASSERT_TRUE(summary.IsSolutionUsable());

  const auto& result_graph = optimizer.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(poses.positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 1e-3) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 1e-3) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 1e-3) << "Camera " << i << " z";
  }
}

// ===========================================================================
// Test: Mono nullspace fixed-lag smoother
// ===========================================================================
TEST(NullspaceVisualSlamTest, NullspaceVisualSlam_MonoFixedLag)
{
  std::mt19937 rng(77);
  std::normal_distribution<double> pixel_noise_dist(0.0, 1.0);
  std::normal_distribution<double> pos_noise(0.0, 0.05);

  const auto cam_device_id = vesta_core::uuid::generate("cam");
  auto calibration = makeMonoCalibration();

  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::FixedLagSmootherParams params;
  params.lag_duration = vesta_core::Duration(10, 0);
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::FixedLagSmoother smoother(params, std::move(graph));

  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;

  vesta_core::Matrix2d pixel_cov = vesta_core::Matrix2d::Identity();

  for (size_t i = 0; i < kNumCameras; ++i)
  {
    auto txn = std::make_shared<vesta_core::Transaction>();
    vesta_core::Timestamp stamp(i, 0);
    txn->stamp(stamp);
    txn->addInvolvedStamp(stamp);

    auto pos = vesta_variables::Position3DStamped::make_shared(stamp, cam_device_id);
    auto ori = vesta_variables::Orientation3DStamped::make_shared(stamp, cam_device_id);

    pos->x() = kCamPositions[i].x() + pos_noise(rng);
    pos->y() = kCamPositions[i].y() + pos_noise(rng);
    pos->z() = kCamPositions[i].z() + pos_noise(rng);

    ori->w() = 1.0;
    ori->x() = pos_noise(rng) * 0.01;
    ori->y() = pos_noise(rng) * 0.01;
    ori->z() = pos_noise(rng) * 0.01;
    double norm = std::sqrt(ori->w() * ori->w() + ori->x() * ori->x() + ori->y() * ori->y() + ori->z() * ori->z());
    ori->w() /= norm;
    ori->x() /= norm;
    ori->y() /= norm;
    ori->z() /= norm;

    positions.push_back(pos);
    orientations.push_back(ori);
    txn->addVariable(pos);
    txn->addVariable(ori);

    // Pose prior on first camera
    if (i == 0)
    {
      addPosePrior(*txn, *pos, *ori, 1e-4);
    }

    // Odometry from previous camera
    if (i > 0)
    {
      txn->addInvolvedStamp(vesta_core::Timestamp(i - 1, 0));
      Eigen::Vector3d dt = kCamPositions[i] - kCamPositions[i - 1];
      vesta_core::Vector7d delta;
      delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d odom_cov = vesta_core::Matrix6d::Identity() * 0.01;
      auto rel = vesta_constraints::RelativePose3DStampedConstraint::make_shared(
          "odom", *positions[i - 1], *orientations[i - 1], *pos, *ori, delta, odom_cov);
      txn->addConstraint(rel);
    }

    // Create nullspace constraints for landmarks visible from this camera and
    // at least one previous camera. Each nullspace constraint links all cameras
    // that observe a given landmark up to (and including) the current one.
    for (size_t j = 0; j < kNumLandmarks; ++j)
    {
      // Collect observations from all cameras up to and including i
      std::vector<vesta_variables::Position3DStamped> obs_positions;
      std::vector<vesta_variables::Orientation3DStamped> obs_orientations;
      std::vector<Eigen::Vector2d> observations;

      for (size_t k = 0; k <= i; ++k)
      {
        double z_cam = 0.0;
        Eigen::Vector2d uv = projectMono(kCamPositions[k], kLandmarks[j], z_cam);
        if (z_cam <= 0.0)
        {
          continue;
        }

        Eigen::Vector2d obs;
        obs << uv.x() + pixel_noise_dist(rng), uv.y() + pixel_noise_dist(rng);
        obs_positions.push_back(*positions[k]);
        obs_orientations.push_back(*orientations[k]);
        observations.push_back(obs);

        // Add involved stamps for previous cameras referenced by this constraint
        if (k < i)
        {
          txn->addInvolvedStamp(vesta_core::Timestamp(k, 0));
        }
      }

      if (observations.size() >= 2)
      {
        auto constraint = vesta_constraints::NullspaceProjectionConstraint::make_shared(
            "camera", obs_positions, obs_orientations, *calibration, observations, pixel_cov);
        txn->addConstraint(constraint);
      }
    }

    smoother.addTransaction("nullspace_vslam", txn);
    auto summary = smoother.optimize();
    logSolverSummary("NullspaceVisualSlam::MonoFixedLag [step " + std::to_string(i) + "]", summary);
    ASSERT_TRUE(summary.IsSolutionUsable()) << "Fixed-lag optimize failed at step " << i;
  }

  // Check final results
  const auto& result_graph = smoother.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 0.15) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 0.15) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 0.15) << "Camera " << i << " z";
  }
}

// ===========================================================================
// Test: Stereo nullspace fixed-lag smoother
// ===========================================================================
TEST(NullspaceVisualSlamTest, NullspaceVisualSlam_StereoFixedLag)
{
  std::mt19937 rng(99);
  std::normal_distribution<double> pixel_noise_dist(0.0, 1.0);
  std::normal_distribution<double> pos_noise(0.0, 0.05);

  const auto cam_device_id = vesta_core::uuid::generate("cam");
  auto calibration = makeStereoCalibration();

  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::FixedLagSmootherParams params;
  params.lag_duration = vesta_core::Duration(10, 0);
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::FixedLagSmoother smoother(params, std::move(graph));

  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;

  vesta_core::Matrix4d pixel_cov = vesta_core::Matrix4d::Identity();

  for (size_t i = 0; i < kNumCameras; ++i)
  {
    auto txn = std::make_shared<vesta_core::Transaction>();
    vesta_core::Timestamp stamp(i, 0);
    txn->stamp(stamp);
    txn->addInvolvedStamp(stamp);

    auto pos = vesta_variables::Position3DStamped::make_shared(stamp, cam_device_id);
    auto ori = vesta_variables::Orientation3DStamped::make_shared(stamp, cam_device_id);

    pos->x() = kCamPositions[i].x() + pos_noise(rng);
    pos->y() = kCamPositions[i].y() + pos_noise(rng);
    pos->z() = kCamPositions[i].z() + pos_noise(rng);

    ori->w() = 1.0;
    ori->x() = pos_noise(rng) * 0.01;
    ori->y() = pos_noise(rng) * 0.01;
    ori->z() = pos_noise(rng) * 0.01;
    double norm = std::sqrt(ori->w() * ori->w() + ori->x() * ori->x() + ori->y() * ori->y() + ori->z() * ori->z());
    ori->w() /= norm;
    ori->x() /= norm;
    ori->y() /= norm;
    ori->z() /= norm;

    positions.push_back(pos);
    orientations.push_back(ori);
    txn->addVariable(pos);
    txn->addVariable(ori);

    if (i == 0)
    {
      addPosePrior(*txn, *pos, *ori, 1e-4);
    }

    if (i > 0)
    {
      txn->addInvolvedStamp(vesta_core::Timestamp(i - 1, 0));
      Eigen::Vector3d dt = kCamPositions[i] - kCamPositions[i - 1];
      vesta_core::Vector7d delta;
      delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d odom_cov = vesta_core::Matrix6d::Identity() * 0.01;
      auto rel = vesta_constraints::RelativePose3DStampedConstraint::make_shared(
          "odom", *positions[i - 1], *orientations[i - 1], *pos, *ori, delta, odom_cov);
      txn->addConstraint(rel);
    }

    // Create stereo nullspace constraints for all landmarks with >= 2 observations
    for (size_t j = 0; j < kNumLandmarks; ++j)
    {
      std::vector<vesta_variables::Position3DStamped> obs_positions;
      std::vector<vesta_variables::Orientation3DStamped> obs_orientations;
      std::vector<Eigen::Vector4d> observations;

      for (size_t k = 0; k <= i; ++k)
      {
        double z_cam = 0.0;
        Eigen::Vector4d obs_gt = projectStereo(kCamPositions[k], kLandmarks[j], z_cam);
        if (z_cam <= 0.0)
        {
          continue;
        }

        Eigen::Vector4d obs;
        obs << obs_gt[0] + pixel_noise_dist(rng), obs_gt[1] + pixel_noise_dist(rng),
            obs_gt[2] + pixel_noise_dist(rng), obs_gt[3] + pixel_noise_dist(rng);
        obs_positions.push_back(*positions[k]);
        obs_orientations.push_back(*orientations[k]);
        observations.push_back(obs);

        if (k < i)
        {
          txn->addInvolvedStamp(vesta_core::Timestamp(k, 0));
        }
      }

      if (observations.size() >= 2)
      {
        auto constraint = vesta_constraints::StereoNullspaceProjectionConstraint::make_shared(
            "stereo_camera", obs_positions, obs_orientations, *calibration, observations, pixel_cov);
        txn->addConstraint(constraint);
      }
    }

    smoother.addTransaction("nullspace_vslam", txn);
    auto summary = smoother.optimize();
    logSolverSummary("NullspaceVisualSlam::StereoFixedLag [step " + std::to_string(i) + "]", summary);
    ASSERT_TRUE(summary.IsSolutionUsable()) << "Fixed-lag optimize failed at step " << i;
  }

  const auto& result_graph = smoother.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 0.2) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 0.2) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 0.2) << "Camera " << i << " z";
  }
}

// ===========================================================================
// Test: QR marginalization with nullspace constraints
//
// Builds a graph with nullspace constraints (no landmarks), marginalizes the
// oldest pose, and verifies the remaining graph optimizes correctly.
// ===========================================================================
TEST(NullspaceVisualSlamTest, NullspaceVisualSlam_MarginalizeQR)
{
  std::mt19937 rng(42);
  const auto cam_device_id = vesta_core::uuid::generate("cam");
  auto calibration = makeStereoCalibration();

  std::normal_distribution<double> pos_noise_dist(0.0, 0.05);
  std::normal_distribution<double> pixel_noise_dist(0.0, 1.0);

  // Build a graph directly (not via optimizer) so we can control marginalization
  vesta_graphs::HashGraph graph;

  // Create camera poses
  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    auto pos = vesta_variables::Position3DStamped::make_shared(vesta_core::Timestamp(i, 0), cam_device_id);
    auto ori = vesta_variables::Orientation3DStamped::make_shared(vesta_core::Timestamp(i, 0), cam_device_id);

    pos->x() = kCamPositions[i].x() + pos_noise_dist(rng);
    pos->y() = kCamPositions[i].y() + pos_noise_dist(rng);
    pos->z() = kCamPositions[i].z() + pos_noise_dist(rng);

    ori->w() = 1.0;
    ori->x() = pos_noise_dist(rng) * 0.01;
    ori->y() = pos_noise_dist(rng) * 0.01;
    ori->z() = pos_noise_dist(rng) * 0.01;
    double norm = std::sqrt(ori->w() * ori->w() + ori->x() * ori->x() + ori->y() * ori->y() + ori->z() * ori->z());
    ori->w() /= norm;
    ori->x() /= norm;
    ori->y() /= norm;
    ori->z() /= norm;

    positions.push_back(pos);
    orientations.push_back(ori);
    graph.addVariable(pos);
    graph.addVariable(ori);
  }

  // Add pose prior
  {
    vesta_core::Vector7d prior_mean;
    prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-4;
    graph.addConstraint(vesta_constraints::AbsolutePose3DStampedConstraint::make_shared(
        "prior", *positions[0], *orientations[0], prior_mean, prior_cov));
  }

  // Add odometry
  for (size_t i = 0; i < kNumCameras - 1; ++i)
  {
    Eigen::Vector3d dt = kCamPositions[i + 1] - kCamPositions[i];
    vesta_core::Vector7d delta;
    delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d odom_cov = vesta_core::Matrix6d::Identity() * 0.01;
    graph.addConstraint(vesta_constraints::RelativePose3DStampedConstraint::make_shared(
        "odom", *positions[i], *orientations[i], *positions[i + 1], *orientations[i + 1], delta, odom_cov));
  }

  // Add stereo nullspace constraints
  vesta_core::Matrix4d pixel_cov = vesta_core::Matrix4d::Identity();
  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    std::vector<vesta_variables::Position3DStamped> obs_positions;
    std::vector<vesta_variables::Orientation3DStamped> obs_orientations;
    std::vector<Eigen::Vector4d> observations;

    for (size_t i = 0; i < kNumCameras; ++i)
    {
      double z_cam = 0.0;
      Eigen::Vector4d obs_gt = projectStereo(kCamPositions[i], kLandmarks[j], z_cam);
      if (z_cam <= 0.0)
      {
        continue;
      }
      Eigen::Vector4d obs;
      obs << obs_gt[0] + pixel_noise_dist(rng), obs_gt[1] + pixel_noise_dist(rng),
          obs_gt[2] + pixel_noise_dist(rng), obs_gt[3] + pixel_noise_dist(rng);
      obs_positions.push_back(*positions[i]);
      obs_orientations.push_back(*orientations[i]);
      observations.push_back(obs);
    }

    if (observations.size() >= 2)
    {
      graph.addConstraint(vesta_constraints::StereoNullspaceProjectionConstraint::make_shared(
          "stereo_camera", obs_positions, obs_orientations, *calibration, observations, pixel_cov));
    }
  }

  // Optimize to get a good linearization point before marginalization
  graph.optimize();

  // Marginalize the first pose
  std::vector<vesta_core::UUID> to_marginalize;
  to_marginalize.push_back(positions[0]->uuid());
  to_marginalize.push_back(orientations[0]->uuid());

  vesta_constraints::QRMarginalizer qr(false);
  auto txn = qr.marginalize("test", to_marginalize, graph);
  graph.update(txn);

  // Verify marginalized variables are gone
  EXPECT_FALSE(graph.variableExists(positions[0]->uuid()));
  EXPECT_FALSE(graph.variableExists(orientations[0]->uuid()));

  // Verify remaining poses still exist
  for (size_t i = 1; i < kNumCameras; ++i)
  {
    EXPECT_TRUE(graph.variableExists(positions[i]->uuid())) << "Position " << i << " should still exist";
    EXPECT_TRUE(graph.variableExists(orientations[i]->uuid())) << "Orientation " << i << " should still exist";
  }

  // Re-optimize and check the solution is still good
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  auto summary = graph.optimize(options);

  ASSERT_TRUE(summary.IsSolutionUsable());

  for (size_t i = 1; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(graph.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 0.15) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 0.15) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 0.15) << "Camera " << i << " z";
  }
}

// ===========================================================================
// Test: Marginalization comparison (QR vs Schur vs BlockDiagonal) with
// nullspace constraints.
//
// Since nullspace constraints contain no non-stamped variables, the Schur
// marginalizer should delegate to QR and produce identical results.
// ===========================================================================
TEST(NullspaceVisualSlamTest, NullspaceVisualSlam_MarginalizeComparison)
{
  std::mt19937 rng(42);
  const auto cam_device_id = vesta_core::uuid::generate("cam");
  auto calibration = makeStereoCalibration();

  std::normal_distribution<double> pos_noise_dist(0.0, 0.05);
  std::normal_distribution<double> pixel_noise_dist(0.0, 1.0);

  // Build a graph with nullspace constraints
  auto buildGraph = [&]() -> vesta_graphs::HashGraph {
    std::mt19937 local_rng(42);  // deterministic per call
    std::normal_distribution<double> local_pos_noise(0.0, 0.05);
    std::normal_distribution<double> local_pixel_noise(0.0, 1.0);

    vesta_graphs::HashGraph g;

    std::vector<vesta_variables::Position3DStamped::SharedPtr> pos_vec;
    std::vector<vesta_variables::Orientation3DStamped::SharedPtr> ori_vec;
    for (size_t i = 0; i < kNumCameras; ++i)
    {
      auto pos = vesta_variables::Position3DStamped::make_shared(vesta_core::Timestamp(i, 0), cam_device_id);
      auto ori = vesta_variables::Orientation3DStamped::make_shared(vesta_core::Timestamp(i, 0), cam_device_id);

      pos->x() = kCamPositions[i].x() + local_pos_noise(local_rng);
      pos->y() = kCamPositions[i].y() + local_pos_noise(local_rng);
      pos->z() = kCamPositions[i].z() + local_pos_noise(local_rng);

      ori->w() = 1.0;
      ori->x() = local_pos_noise(local_rng) * 0.01;
      ori->y() = local_pos_noise(local_rng) * 0.01;
      ori->z() = local_pos_noise(local_rng) * 0.01;
      double norm =
          std::sqrt(ori->w() * ori->w() + ori->x() * ori->x() + ori->y() * ori->y() + ori->z() * ori->z());
      ori->w() /= norm;
      ori->x() /= norm;
      ori->y() /= norm;
      ori->z() /= norm;

      pos_vec.push_back(pos);
      ori_vec.push_back(ori);
      g.addVariable(pos);
      g.addVariable(ori);
    }

    // Prior
    {
      vesta_core::Vector7d prior_mean;
      prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-4;
      g.addConstraint(vesta_constraints::AbsolutePose3DStampedConstraint::make_shared(
          "prior", *pos_vec[0], *ori_vec[0], prior_mean, prior_cov));
    }

    // Odometry
    for (size_t i = 0; i < kNumCameras - 1; ++i)
    {
      Eigen::Vector3d dt = kCamPositions[i + 1] - kCamPositions[i];
      vesta_core::Vector7d delta;
      delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d odom_cov = vesta_core::Matrix6d::Identity() * 0.01;
      g.addConstraint(vesta_constraints::RelativePose3DStampedConstraint::make_shared(
          "odom", *pos_vec[i], *ori_vec[i], *pos_vec[i + 1], *ori_vec[i + 1], delta, odom_cov));
    }

    // Stereo nullspace constraints
    vesta_core::Matrix4d pixel_cov = vesta_core::Matrix4d::Identity();
    for (size_t j = 0; j < kNumLandmarks; ++j)
    {
      std::vector<vesta_variables::Position3DStamped> obs_positions;
      std::vector<vesta_variables::Orientation3DStamped> obs_orientations;
      std::vector<Eigen::Vector4d> observations;

      for (size_t i = 0; i < kNumCameras; ++i)
      {
        double z_cam = 0.0;
        Eigen::Vector4d obs_gt = projectStereo(kCamPositions[i], kLandmarks[j], z_cam);
        if (z_cam <= 0.0)
        {
          continue;
        }
        Eigen::Vector4d obs;
        obs << obs_gt[0] + local_pixel_noise(local_rng), obs_gt[1] + local_pixel_noise(local_rng),
            obs_gt[2] + local_pixel_noise(local_rng), obs_gt[3] + local_pixel_noise(local_rng);
        obs_positions.push_back(*pos_vec[i]);
        obs_orientations.push_back(*ori_vec[i]);
        observations.push_back(obs);
      }

      if (observations.size() >= 2)
      {
        g.addConstraint(vesta_constraints::StereoNullspaceProjectionConstraint::make_shared(
            "stereo_camera", obs_positions, obs_orientations, *calibration, observations, pixel_cov));
      }
    }

    g.optimize();
    return g;
  };

  // Build three identical graphs
  auto graph_qr = buildGraph();
  auto graph_schur = buildGraph();
  auto graph_bd = buildGraph();

  // Marginalize the first pose from each
  auto pos0_uuid = vesta_variables::Position3DStamped(vesta_core::Timestamp(0, 0), cam_device_id).uuid();
  auto ori0_uuid = vesta_variables::Orientation3DStamped(vesta_core::Timestamp(0, 0), cam_device_id).uuid();

  std::vector<vesta_core::UUID> to_marginalize = { pos0_uuid, ori0_uuid };

  vesta_constraints::QRMarginalizer qr(false);
  vesta_constraints::SchurMarginalizer schur(false);
  vesta_constraints::BlockDiagonalMarginalizer bd(false);

  auto txn_qr = qr.marginalize("test", to_marginalize, graph_qr);
  auto txn_schur = schur.marginalize("test", to_marginalize, graph_schur);
  auto txn_bd = bd.marginalize("test", to_marginalize, graph_bd);

  graph_qr.update(txn_qr);
  graph_schur.update(txn_schur);
  graph_bd.update(txn_bd);

  // Optimize all three
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;

  auto summary_qr = graph_qr.optimize(options);
  auto summary_schur = graph_schur.optimize(options);
  auto summary_bd = graph_bd.optimize(options);

  ASSERT_TRUE(summary_qr.IsSolutionUsable());
  ASSERT_TRUE(summary_schur.IsSolutionUsable());
  ASSERT_TRUE(summary_bd.IsSolutionUsable());

  // Compare results: QR and Schur should be nearly identical (Schur delegates to QR
  // when there are no non-stamped variables). BD may differ more.
  for (size_t i = 1; i < kNumCameras; ++i)
  {
    auto pos_uuid = vesta_variables::Position3DStamped(vesta_core::Timestamp(i, 0), cam_device_id).uuid();

    const auto& pos_qr =
        dynamic_cast<const vesta_variables::Position3DStamped&>(graph_qr.getVariable(pos_uuid));
    const auto& pos_schur =
        dynamic_cast<const vesta_variables::Position3DStamped&>(graph_schur.getVariable(pos_uuid));
    const auto& pos_bd =
        dynamic_cast<const vesta_variables::Position3DStamped&>(graph_bd.getVariable(pos_uuid));

    // QR vs Schur: should be nearly identical (Schur delegates to QR for pose-only)
    EXPECT_NEAR(pos_qr.x(), pos_schur.x(), 1e-10) << "QR vs Schur Camera " << i << " x";
    EXPECT_NEAR(pos_qr.y(), pos_schur.y(), 1e-10) << "QR vs Schur Camera " << i << " y";
    EXPECT_NEAR(pos_qr.z(), pos_schur.z(), 1e-10) << "QR vs Schur Camera " << i << " z";

    // BlockDiagonal drops cross-correlations, so looser tolerance
    EXPECT_NEAR(pos_qr.x(), pos_bd.x(), 0.5) << "QR vs BD Camera " << i << " x";
    EXPECT_NEAR(pos_qr.y(), pos_bd.y(), 0.5) << "QR vs BD Camera " << i << " y";
    EXPECT_NEAR(pos_qr.z(), pos_bd.z(), 0.5) << "QR vs BD Camera " << i << " z";

    // All should be close to ground truth
    EXPECT_NEAR(pos_qr.x(), kCamPositions[i].x(), 0.15) << "QR Camera " << i << " x vs gt";
    EXPECT_NEAR(pos_schur.x(), kCamPositions[i].x(), 0.15) << "Schur Camera " << i << " x vs gt";
    EXPECT_NEAR(pos_bd.x(), kCamPositions[i].x(), 0.5) << "BD Camera " << i << " x vs gt";
  }

  // QR and Schur final costs should match
  EXPECT_NEAR(summary_qr.final_cost, summary_schur.final_cost, 1e-10);

  std::cout << "\n=== Nullspace Marginalization Comparison ==="
            << "\n  QR    final cost:  " << summary_qr.final_cost
            << "\n  Schur final cost:  " << summary_schur.final_cost
            << "\n  BD    final cost:  " << summary_bd.final_cost << "\n"
            << std::endl;
}

// ===========================================================================
// Test: Incremental marginalization with nullspace constraints
//
// Simulates a fixed-lag smoother workflow: builds a larger graph, then
// incrementally marginalizes old poses one at a time, re-optimizing after
// each step. Verifies all three marginalizers remain stable.
// ===========================================================================
TEST(NullspaceVisualSlamTest, NullspaceVisualSlam_IncrementalMarginalize)
{
  constexpr size_t kExtNumCameras = 8;
  constexpr size_t kWindowSize = 4;

  // Extended camera positions
  std::vector<Eigen::Vector3d> cam_positions;
  for (size_t i = 0; i < kExtNumCameras; ++i)
  {
    cam_positions.push_back({ static_cast<double>(i), 0.0, 0.0 });
  }

  // More landmarks for better constraint coverage
  std::vector<Eigen::Vector3d> landmarks = {
    { 1.0, 1.0, 5.0 },   { 2.0, -1.0, 6.0 },  { 3.0, 0.5, 7.0 },   { -1.0, 2.0, 8.0 },
    { 0.0, -1.5, 5.5 },  { 4.0, 1.0, 6.5 },   { 2.5, -0.5, 9.0 },  { 1.5, 1.5, 10.0 },
    { 5.0, 0.5, 6.0 },   { 6.0, -0.5, 7.0 },  { 3.5, 1.0, 5.5 },   { 4.5, -1.0, 8.5 },
  };
  const size_t num_landmarks = landmarks.size();

  const auto cam_device_id = vesta_core::uuid::generate("cam");
  auto calibration = makeStereoCalibration();

  auto buildExtGraph = [&]() -> std::pair<vesta_graphs::HashGraph,
                                          std::pair<std::vector<vesta_variables::Position3DStamped::SharedPtr>,
                                                    std::vector<vesta_variables::Orientation3DStamped::SharedPtr>>> {
    std::mt19937 local_rng(99);
    std::normal_distribution<double> local_pos_noise(0.0, 0.05);
    std::normal_distribution<double> local_pixel_noise(0.0, 1.0);

    vesta_graphs::HashGraph g;

    std::vector<vesta_variables::Position3DStamped::SharedPtr> pos_vec;
    std::vector<vesta_variables::Orientation3DStamped::SharedPtr> ori_vec;
    for (size_t i = 0; i < kExtNumCameras; ++i)
    {
      auto pos = vesta_variables::Position3DStamped::make_shared(vesta_core::Timestamp(i, 0), cam_device_id);
      auto ori = vesta_variables::Orientation3DStamped::make_shared(vesta_core::Timestamp(i, 0), cam_device_id);

      pos->x() = cam_positions[i].x() + local_pos_noise(local_rng);
      pos->y() = cam_positions[i].y() + local_pos_noise(local_rng);
      pos->z() = cam_positions[i].z() + local_pos_noise(local_rng);

      ori->w() = 1.0;
      ori->x() = local_pos_noise(local_rng) * 0.01;
      ori->y() = local_pos_noise(local_rng) * 0.01;
      ori->z() = local_pos_noise(local_rng) * 0.01;
      double norm =
          std::sqrt(ori->w() * ori->w() + ori->x() * ori->x() + ori->y() * ori->y() + ori->z() * ori->z());
      ori->w() /= norm;
      ori->x() /= norm;
      ori->y() /= norm;
      ori->z() /= norm;

      pos_vec.push_back(pos);
      ori_vec.push_back(ori);
      g.addVariable(pos);
      g.addVariable(ori);
    }

    // Prior
    {
      vesta_core::Vector7d prior_mean;
      prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-4;
      g.addConstraint(vesta_constraints::AbsolutePose3DStampedConstraint::make_shared(
          "prior", *pos_vec[0], *ori_vec[0], prior_mean, prior_cov));
    }

    // Odometry
    for (size_t i = 0; i < kExtNumCameras - 1; ++i)
    {
      Eigen::Vector3d dt = cam_positions[i + 1] - cam_positions[i];
      vesta_core::Vector7d delta;
      delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d odom_cov = vesta_core::Matrix6d::Identity() * 0.01;
      g.addConstraint(vesta_constraints::RelativePose3DStampedConstraint::make_shared(
          "odom", *pos_vec[i], *ori_vec[i], *pos_vec[i + 1], *ori_vec[i + 1], delta, odom_cov));
    }

    // Stereo nullspace constraints
    vesta_core::Matrix4d pixel_cov = vesta_core::Matrix4d::Identity();
    for (size_t j = 0; j < num_landmarks; ++j)
    {
      std::vector<vesta_variables::Position3DStamped> obs_positions;
      std::vector<vesta_variables::Orientation3DStamped> obs_orientations;
      std::vector<Eigen::Vector4d> observations;

      for (size_t i = 0; i < kExtNumCameras; ++i)
      {
        Eigen::Vector3d p = landmarks[j] - cam_positions[i];
        if (p.z() <= 0.0)
        {
          continue;
        }
        Eigen::Vector4d obs_gt;
        obs_gt << kFx * p.x() / p.z() + kCx, kFy * p.y() / p.z() + kCy,
            kFx * (p.x() - kBaseline) / p.z() + kCx, kFy * p.y() / p.z() + kCy;

        Eigen::Vector4d obs;
        obs << obs_gt[0] + local_pixel_noise(local_rng), obs_gt[1] + local_pixel_noise(local_rng),
            obs_gt[2] + local_pixel_noise(local_rng), obs_gt[3] + local_pixel_noise(local_rng);
        obs_positions.push_back(*pos_vec[i]);
        obs_orientations.push_back(*ori_vec[i]);
        observations.push_back(obs);
      }

      if (observations.size() >= 2)
      {
        g.addConstraint(vesta_constraints::StereoNullspaceProjectionConstraint::make_shared(
            "stereo_camera", obs_positions, obs_orientations, *calibration, observations, pixel_cov));
      }
    }

    g.optimize();
    return { std::move(g), { pos_vec, ori_vec } };
  };

  auto [graph_qr, vars_qr] = buildExtGraph();
  auto [graph_schur, vars_schur] = buildExtGraph();
  auto [graph_bd, vars_bd] = buildExtGraph();

  auto& [pos_qr, ori_qr] = vars_qr;
  auto& [pos_schur, ori_schur] = vars_schur;
  auto& [pos_bd, ori_bd] = vars_bd;

  int num_marginalizations = 0;

  // Incrementally marginalize old poses
  for (size_t step = 0; step + kWindowSize < kExtNumCameras; ++step)
  {
    std::vector<vesta_core::UUID> to_marg_qr = { pos_qr[step]->uuid(), ori_qr[step]->uuid() };
    std::vector<vesta_core::UUID> to_marg_schur = { pos_schur[step]->uuid(), ori_schur[step]->uuid() };
    std::vector<vesta_core::UUID> to_marg_bd = { pos_bd[step]->uuid(), ori_bd[step]->uuid() };

    vesta_constraints::QRMarginalizer qr(false);
    auto txn_qr = qr.marginalize("test", to_marg_qr, graph_qr);
    graph_qr.update(txn_qr);
    graph_qr.optimize();

    vesta_constraints::SchurMarginalizer schur(false);
    auto txn_schur = schur.marginalize("test", to_marg_schur, graph_schur);
    graph_schur.update(txn_schur);
    graph_schur.optimize();

    vesta_constraints::BlockDiagonalMarginalizer bd(false);
    auto txn_bd = bd.marginalize("test", to_marg_bd, graph_bd);
    graph_bd.update(txn_bd);
    graph_bd.optimize();

    ++num_marginalizations;
  }

  // Verify remaining poses are close to ground truth for all marginalizers
  size_t first_remaining = kExtNumCameras - kWindowSize;
  for (size_t i = first_remaining; i < kExtNumCameras; ++i)
  {
    const auto& pq =
        dynamic_cast<const vesta_variables::Position3DStamped&>(graph_qr.getVariable(pos_qr[i]->uuid()));
    const auto& ps =
        dynamic_cast<const vesta_variables::Position3DStamped&>(graph_schur.getVariable(pos_schur[i]->uuid()));
    const auto& pb =
        dynamic_cast<const vesta_variables::Position3DStamped&>(graph_bd.getVariable(pos_bd[i]->uuid()));

    // QR vs Schur: identical (no non-stamped variables for Schur to use)
    EXPECT_NEAR(pq.x(), ps.x(), 0.05) << "Camera " << i << " x";
    EXPECT_NEAR(pq.y(), ps.y(), 0.05) << "Camera " << i << " y";
    EXPECT_NEAR(pq.z(), ps.z(), 0.05) << "Camera " << i << " z";

    // BlockDiagonal: looser
    EXPECT_NEAR(pq.x(), pb.x(), 0.5) << "BD Camera " << i << " x";
    EXPECT_NEAR(pq.y(), pb.y(), 0.5) << "BD Camera " << i << " y";
    EXPECT_NEAR(pq.z(), pb.z(), 0.5) << "BD Camera " << i << " z";

    // All vs ground truth (incremental marginalization loses information, so tolerances are looser)
    EXPECT_NEAR(pq.x(), cam_positions[i].x(), 0.25) << "QR Camera " << i << " x vs gt";
    EXPECT_NEAR(ps.x(), cam_positions[i].x(), 0.25) << "Schur Camera " << i << " x vs gt";
    EXPECT_NEAR(pb.x(), cam_positions[i].x(), 0.5) << "BD Camera " << i << " x vs gt";
  }

  std::cout << "\n=== Incremental Nullspace Marginalization (" << num_marginalizations << " steps) ===" << std::endl;
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
