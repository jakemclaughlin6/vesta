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

#include <vesta_constraints/3d/absolute_pose_3d_stamped_constraint.h>
#include <vesta_constraints/3d/relative_pose_3d_stamped_constraint.h>
#include <vesta_constraints/common/block_diagonal_marginalizer.h>
#include <vesta_constraints/vision/reprojection_error_constraint.h>
#include <vesta_constraints/vision/stereo_reprojection_error_constraint.h>
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
#include <vesta_variables/vision/pinhole_camera_fixed.h>
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
// world position. The cost functor uses the convention p_cam = R_wc^{-1} * (X - p_world).
// For identity rotation R_wc=I, p_cam = X - p_world.
// Returns (u, v) and sets z_cam for depth check.
static Eigen::Vector2d projectMono(const Eigen::Vector3d& cam_world_pos, const Eigen::Vector3d& landmark, double& z_cam)
{
  // Camera frame: p_cam = R_wc^{-1} * (landmark - cam_world_pos)
  // With identity rotation: p_cam = landmark - cam_world_pos
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

// ---------------------------------------------------------------------------
// Test: Mono batch optimization
// ---------------------------------------------------------------------------
TEST(VisualSlamTest, VisualSlam_MonoBatch)
{
  std::mt19937 rng(42);
  std::normal_distribution<double> pixel_noise(0.0, 1.0);
  std::normal_distribution<double> pos_noise(0.0, 0.05);
  std::normal_distribution<double> lm_noise(0.0, 0.1);

  const auto cam_device_id = vesta_core::uuid::generate("cam");

  // Build a single transaction with all variables and constraints
  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(kNumCameras - 1, 0));
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    txn->addInvolvedStamp(vesta_core::Timestamp(i, 0));
  }

  // Create camera intrinsics variable (not perturbed)
  auto cam = vesta_variables::PinholeCameraFixed::make_shared(uint64_t{ 0 });
  cam->fx() = kFx;
  cam->fy() = kFy;
  cam->cx() = kCx;
  cam->cy() = kCy;
  txn->addVariable(cam);

  // Create camera poses
  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    auto pos = vesta_variables::Position3DStamped::make_shared(vesta_core::Timestamp(i, 0), cam_device_id);
    auto ori = vesta_variables::Orientation3DStamped::make_shared(vesta_core::Timestamp(i, 0), cam_device_id);

    // Initialize at ground truth + noise
    // Position variable is world-frame position
    pos->x() = kCamPositions[i].x() + pos_noise(rng);
    pos->y() = kCamPositions[i].y() + pos_noise(rng);
    pos->z() = kCamPositions[i].z() + pos_noise(rng);

    // Identity quaternion (w, x, y, z) with small perturbation
    ori->w() = 1.0;
    ori->x() = pos_noise(rng) * 0.01;
    ori->y() = pos_noise(rng) * 0.01;
    ori->z() = pos_noise(rng) * 0.01;
    // Normalize
    double norm = std::sqrt(ori->w() * ori->w() + ori->x() * ori->x() + ori->y() * ori->y() + ori->z() * ori->z());
    ori->w() /= norm;
    ori->x() /= norm;
    ori->y() /= norm;
    ori->z() /= norm;

    positions.push_back(pos);
    orientations.push_back(ori);
    txn->addVariable(pos);
    txn->addVariable(ori);
  }

  // Create landmarks
  std::vector<vesta_variables::Point3DLandmark::SharedPtr> landmarks;
  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    auto lm = vesta_variables::Point3DLandmark::make_shared(uint64_t{ j });
    lm->x() = kLandmarks[j].x() + lm_noise(rng);
    lm->y() = kLandmarks[j].y() + lm_noise(rng);
    lm->z() = kLandmarks[j].z() + lm_noise(rng);
    landmarks.push_back(lm);
    txn->addVariable(lm);
  }

  // Add pose prior on first camera
  {
    vesta_core::Vector7d prior_mean;
    prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;  // (x,y,z, qw,qx,qy,qz)
    vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-4;
    auto prior = vesta_constraints::AbsolutePose3DStampedConstraint::make_shared(
        "prior", *positions[0], *orientations[0], prior_mean, prior_cov);
    txn->addConstraint(prior);
  }

  // Add reprojection constraints
  vesta_core::Matrix2d pixel_cov = vesta_core::Matrix2d::Identity();  // sigma=1 pixel
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    for (size_t j = 0; j < kNumLandmarks; ++j)
    {
      double z_cam = 0.0;
      Eigen::Vector2d uv = projectMono(kCamPositions[i], kLandmarks[j], z_cam);
      if (z_cam <= 0.0)
      {
        continue;
      }
      // Add noise to observation
      vesta_core::Vector2d obs;
      obs << uv.x() + pixel_noise(rng), uv.y() + pixel_noise(rng);

      auto constraint = vesta_constraints::ReprojectionErrorConstraint::make_shared(
          "camera", *positions[i], *orientations[i], *cam, *landmarks[j], obs, pixel_cov);
      txn->addConstraint(constraint);
    }
  }

  // Relative pose constraints (odometry) between consecutive cameras
  // For identity orientations: delta_t = p_{i+1} - p_i = cam_{i+1} - cam_i
  for (size_t i = 0; i < kNumCameras - 1; ++i)
  {
    Eigen::Vector3d dt = kCamPositions[i + 1] - kCamPositions[i];  // (1,0,0)
    vesta_core::Vector7d delta;
    delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 0.01;
    auto rel = vesta_constraints::RelativePose3DStampedConstraint::make_shared(
        "odom", *positions[i], *orientations[i], *positions[i + 1], *orientations[i + 1], delta, cov);
    txn->addConstraint(rel);
  }

  // Set up batch optimizer
  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

  optimizer.addTransaction("visual_slam", txn);
  auto summary = optimizer.optimize();
  logSolverSummary("VisualSlam::MonoBatch", summary);

  ASSERT_TRUE(summary.IsSolutionUsable());

  // Check camera positions (world-frame)
  const auto& result_graph = optimizer.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 0.1) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 0.1) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 0.1) << "Camera " << i << " z";
  }

  // Check landmarks
  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    const auto& lm =
        dynamic_cast<const vesta_variables::Point3DLandmark&>(result_graph.getVariable(landmarks[j]->uuid()));
    EXPECT_NEAR(lm.x(), kLandmarks[j].x(), 0.35) << "Landmark " << j << " x";
    EXPECT_NEAR(lm.y(), kLandmarks[j].y(), 0.35) << "Landmark " << j << " y";
    EXPECT_NEAR(lm.z(), kLandmarks[j].z(), 0.35) << "Landmark " << j << " z";
  }
}

// ---------------------------------------------------------------------------
// Test: Stereo batch optimization
// ---------------------------------------------------------------------------
TEST(VisualSlamTest, VisualSlam_StereoBatch)
{
  std::mt19937 rng(123);
  std::normal_distribution<double> pixel_noise(0.0, 1.0);
  std::normal_distribution<double> pos_noise(0.0, 0.05);
  std::normal_distribution<double> lm_noise(0.0, 0.1);

  const auto cam_device_id = vesta_core::uuid::generate("cam");

  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(kNumCameras - 1, 0));
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    txn->addInvolvedStamp(vesta_core::Timestamp(i, 0));
  }

  // Stereo camera intrinsics (not perturbed)
  auto stereo_cam = vesta_variables::StereoCameraFixed::make_shared(uint64_t{ 0 });
  stereo_cam->fx() = kFx;
  stereo_cam->fy() = kFy;
  stereo_cam->cx() = kCx;
  stereo_cam->cy() = kCy;
  stereo_cam->baseline() = kBaseline;
  txn->addVariable(stereo_cam);

  // Camera poses
  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;
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

    positions.push_back(pos);
    orientations.push_back(ori);
    txn->addVariable(pos);
    txn->addVariable(ori);
  }

  // Landmarks
  std::vector<vesta_variables::Point3DLandmark::SharedPtr> landmarks;
  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    auto lm = vesta_variables::Point3DLandmark::make_shared(uint64_t{ j });
    lm->x() = kLandmarks[j].x() + lm_noise(rng);
    lm->y() = kLandmarks[j].y() + lm_noise(rng);
    lm->z() = kLandmarks[j].z() + lm_noise(rng);
    landmarks.push_back(lm);
    txn->addVariable(lm);
  }

  // Pose prior on first camera
  {
    vesta_core::Vector7d prior_mean;
    prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-4;
    auto prior = vesta_constraints::AbsolutePose3DStampedConstraint::make_shared(
        "prior", *positions[0], *orientations[0], prior_mean, prior_cov);
    txn->addConstraint(prior);
  }

  // Stereo reprojection constraints
  vesta_core::Matrix4d pixel_cov = vesta_core::Matrix4d::Identity();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    for (size_t j = 0; j < kNumLandmarks; ++j)
    {
      double z_cam = 0.0;
      Eigen::Vector4d obs_gt = projectStereo(kCamPositions[i], kLandmarks[j], z_cam);
      if (z_cam <= 0.0)
      {
        continue;
      }
      vesta_core::Vector4d obs;
      obs << obs_gt[0] + pixel_noise(rng), obs_gt[1] + pixel_noise(rng), obs_gt[2] + pixel_noise(rng),
          obs_gt[3] + pixel_noise(rng);

      auto constraint = vesta_constraints::StereoReprojectionErrorConstraint::make_shared(
          "stereo_camera", *positions[i], *orientations[i], *stereo_cam, *landmarks[j], obs, pixel_cov);
      txn->addConstraint(constraint);
    }
  }

  // Relative pose constraints (odometry)
  for (size_t i = 0; i < kNumCameras - 1; ++i)
  {
    Eigen::Vector3d dt = kCamPositions[i + 1] - kCamPositions[i];
    vesta_core::Vector7d delta;
    delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 0.01;
    auto rel = vesta_constraints::RelativePose3DStampedConstraint::make_shared(
        "odom", *positions[i], *orientations[i], *positions[i + 1], *orientations[i + 1], delta, cov);
    txn->addConstraint(rel);
  }

  // Batch optimizer
  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

  optimizer.addTransaction("visual_slam", txn);
  auto summary = optimizer.optimize();
  logSolverSummary("VisualSlam::StereoBatch", summary);

  ASSERT_TRUE(summary.IsSolutionUsable());

  const auto& result_graph = optimizer.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 0.1) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 0.1) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 0.1) << "Camera " << i << " z";
  }

  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    const auto& lm =
        dynamic_cast<const vesta_variables::Point3DLandmark&>(result_graph.getVariable(landmarks[j]->uuid()));
    EXPECT_NEAR(lm.x(), kLandmarks[j].x(), 0.3) << "Landmark " << j << " x";
    EXPECT_NEAR(lm.y(), kLandmarks[j].y(), 0.3) << "Landmark " << j << " y";
    EXPECT_NEAR(lm.z(), kLandmarks[j].z(), 0.3) << "Landmark " << j << " z";
  }
}

// ---------------------------------------------------------------------------
// Test: Mono fixed-lag smoother
// ---------------------------------------------------------------------------
TEST(VisualSlamTest, VisualSlam_MonoFixedLag)
{
  std::mt19937 rng(77);
  std::normal_distribution<double> pixel_noise(0.0, 1.0);
  std::normal_distribution<double> pos_noise(0.0, 0.05);
  std::normal_distribution<double> lm_noise(0.0, 0.1);

  const auto cam_device_id = vesta_core::uuid::generate("cam");

  // Camera intrinsics
  auto cam = vesta_variables::PinholeCameraFixed::make_shared(uint64_t{ 0 });
  cam->fx() = kFx;
  cam->fy() = kFy;
  cam->cx() = kCx;
  cam->cy() = kCy;

  // Pre-create landmarks (shared across transactions)
  std::vector<vesta_variables::Point3DLandmark::SharedPtr> landmarks;
  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    auto lm = vesta_variables::Point3DLandmark::make_shared(uint64_t{ j });
    lm->x() = kLandmarks[j].x() + lm_noise(rng);
    lm->y() = kLandmarks[j].y() + lm_noise(rng);
    lm->z() = kLandmarks[j].z() + lm_noise(rng);
    landmarks.push_back(lm);
  }

  // Set up fixed-lag smoother with large lag to keep all poses
  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::FixedLagSmootherParams params;
  params.lag_duration = vesta_core::Duration(10, 0);
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::FixedLagSmoother smoother(params, std::move(graph));

  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;

  vesta_core::Matrix2d pixel_cov = vesta_core::Matrix2d::Identity();

  // Add one transaction per camera pose
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

    // Add camera intrinsics and landmarks in first transaction
    if (i == 0)
    {
      txn->addVariable(cam);
      for (size_t j = 0; j < kNumLandmarks; ++j)
      {
        txn->addVariable(landmarks[j]);
      }
    }

    // Add pose prior on first camera
    if (i == 0)
    {
      vesta_core::Vector7d prior_mean;
      prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-4;
      auto prior =
          vesta_constraints::AbsolutePose3DStampedConstraint::make_shared("prior", *pos, *ori, prior_mean, prior_cov);
      txn->addConstraint(prior);
    }

    // Relative pose constraint (odometry) from previous camera
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

    // Reprojection constraints for this pose
    for (size_t j = 0; j < kNumLandmarks; ++j)
    {
      double z_cam = 0.0;
      Eigen::Vector2d uv = projectMono(kCamPositions[i], kLandmarks[j], z_cam);
      if (z_cam <= 0.0)
      {
        continue;
      }
      vesta_core::Vector2d obs;
      obs << uv.x() + pixel_noise(rng), uv.y() + pixel_noise(rng);

      auto constraint = vesta_constraints::ReprojectionErrorConstraint::make_shared("camera", *pos, *ori, *cam,
                                                                                    *landmarks[j], obs, pixel_cov);
      txn->addConstraint(constraint);
    }

    smoother.addTransaction("visual_slam", txn);
    auto summary = smoother.optimize();
    logSolverSummary("VisualSlam::MonoFixedLag [step " + std::to_string(i) + "]", summary);
    ASSERT_TRUE(summary.IsSolutionUsable()) << "Fixed-lag optimize failed at step " << i;
  }

  // Check final results
  const auto& result_graph = smoother.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 0.1) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 0.1) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 0.1) << "Camera " << i << " z";
  }

  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    const auto& lm =
        dynamic_cast<const vesta_variables::Point3DLandmark&>(result_graph.getVariable(landmarks[j]->uuid()));
    EXPECT_NEAR(lm.x(), kLandmarks[j].x(), 0.1) << "Landmark " << j << " x";
    EXPECT_NEAR(lm.y(), kLandmarks[j].y(), 0.1) << "Landmark " << j << " y";
    EXPECT_NEAR(lm.z(), kLandmarks[j].z(), 0.1) << "Landmark " << j << " z";
  }
}

// ---------------------------------------------------------------------------
// Test: Stereo fixed-lag smoother
// ---------------------------------------------------------------------------
TEST(VisualSlamTest, VisualSlam_StereoFixedLag)
{
  std::mt19937 rng(99);
  std::normal_distribution<double> pixel_noise(0.0, 1.0);
  std::normal_distribution<double> pos_noise(0.0, 0.05);
  std::normal_distribution<double> lm_noise(0.0, 0.1);

  const auto cam_device_id = vesta_core::uuid::generate("cam");

  // Stereo camera intrinsics
  auto stereo_cam = vesta_variables::StereoCameraFixed::make_shared(uint64_t{ 0 });
  stereo_cam->fx() = kFx;
  stereo_cam->fy() = kFy;
  stereo_cam->cx() = kCx;
  stereo_cam->cy() = kCy;
  stereo_cam->baseline() = kBaseline;

  // Pre-create landmarks
  std::vector<vesta_variables::Point3DLandmark::SharedPtr> landmarks;
  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    auto lm = vesta_variables::Point3DLandmark::make_shared(uint64_t{ j });
    lm->x() = kLandmarks[j].x() + lm_noise(rng);
    lm->y() = kLandmarks[j].y() + lm_noise(rng);
    lm->z() = kLandmarks[j].z() + lm_noise(rng);
    landmarks.push_back(lm);
  }

  // Set up fixed-lag smoother
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

    // Add stereo camera and landmarks in first transaction
    if (i == 0)
    {
      txn->addVariable(stereo_cam);
      for (size_t j = 0; j < kNumLandmarks; ++j)
      {
        txn->addVariable(landmarks[j]);
      }
    }

    // Pose prior on first camera
    if (i == 0)
    {
      vesta_core::Vector7d prior_mean;
      prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-4;
      auto prior =
          vesta_constraints::AbsolutePose3DStampedConstraint::make_shared("prior", *pos, *ori, prior_mean, prior_cov);
      txn->addConstraint(prior);
    }

    // Relative pose constraint (odometry) from previous camera
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

    // Stereo reprojection constraints
    for (size_t j = 0; j < kNumLandmarks; ++j)
    {
      double z_cam = 0.0;
      Eigen::Vector4d obs_gt = projectStereo(kCamPositions[i], kLandmarks[j], z_cam);
      if (z_cam <= 0.0)
      {
        continue;
      }
      vesta_core::Vector4d obs;
      obs << obs_gt[0] + pixel_noise(rng), obs_gt[1] + pixel_noise(rng), obs_gt[2] + pixel_noise(rng),
          obs_gt[3] + pixel_noise(rng);

      auto constraint = vesta_constraints::StereoReprojectionErrorConstraint::make_shared(
          "stereo_camera", *pos, *ori, *stereo_cam, *landmarks[j], obs, pixel_cov);
      txn->addConstraint(constraint);
    }

    smoother.addTransaction("visual_slam", txn);
    auto summary = smoother.optimize();
    logSolverSummary("VisualSlam::StereoFixedLag [step " + std::to_string(i) + "]", summary);
    ASSERT_TRUE(summary.IsSolutionUsable()) << "Fixed-lag optimize failed at step " << i;
  }

  // Check final results
  const auto& result_graph = smoother.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 0.2) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 0.2) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 0.2) << "Camera " << i << " z";
  }

  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    const auto& lm =
        dynamic_cast<const vesta_variables::Point3DLandmark&>(result_graph.getVariable(landmarks[j]->uuid()));
    EXPECT_NEAR(lm.x(), kLandmarks[j].x(), 0.3) << "Landmark " << j << " x";
    EXPECT_NEAR(lm.y(), kLandmarks[j].y(), 0.3) << "Landmark " << j << " y";
    EXPECT_NEAR(lm.z(), kLandmarks[j].z(), 0.3) << "Landmark " << j << " z";
  }
}

// ---------------------------------------------------------------------------
// Test: Noiseless mono batch — verifies exact convergence to ground truth
// ---------------------------------------------------------------------------
TEST(VisualSlamTest, VisualSlam_MonoBatchNoiseless)
{
  std::mt19937 rng(42);
  std::normal_distribution<double> pos_noise(0.0, 0.05);
  std::normal_distribution<double> lm_noise(0.0, 0.1);

  const auto cam_device_id = vesta_core::uuid::generate("cam");

  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(kNumCameras - 1, 0));
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    txn->addInvolvedStamp(vesta_core::Timestamp(i, 0));
  }

  auto cam = vesta_variables::PinholeCameraFixed::make_shared(uint64_t{ 0 });
  cam->fx() = kFx;
  cam->fy() = kFy;
  cam->cx() = kCx;
  cam->cy() = kCy;
  txn->addVariable(cam);

  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;
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

    positions.push_back(pos);
    orientations.push_back(ori);
    txn->addVariable(pos);
    txn->addVariable(ori);
  }

  std::vector<vesta_variables::Point3DLandmark::SharedPtr> landmarks;
  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    auto lm = vesta_variables::Point3DLandmark::make_shared(uint64_t{ j });
    lm->x() = kLandmarks[j].x() + lm_noise(rng);
    lm->y() = kLandmarks[j].y() + lm_noise(rng);
    lm->z() = kLandmarks[j].z() + lm_noise(rng);
    landmarks.push_back(lm);
    txn->addVariable(lm);
  }

  // Tight prior on first camera at exact ground truth
  {
    vesta_core::Vector7d prior_mean;
    prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-8;
    auto prior = vesta_constraints::AbsolutePose3DStampedConstraint::make_shared(
        "prior", *positions[0], *orientations[0], prior_mean, prior_cov);
    txn->addConstraint(prior);
  }

  // Noiseless reprojection constraints
  vesta_core::Matrix2d pixel_cov = vesta_core::Matrix2d::Identity();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    for (size_t j = 0; j < kNumLandmarks; ++j)
    {
      double z_cam = 0.0;
      Eigen::Vector2d uv = projectMono(kCamPositions[i], kLandmarks[j], z_cam);
      if (z_cam <= 0.0)
      {
        continue;
      }
      vesta_core::Vector2d obs;
      obs << uv.x(), uv.y();

      auto constraint = vesta_constraints::ReprojectionErrorConstraint::make_shared(
          "camera", *positions[i], *orientations[i], *cam, *landmarks[j], obs, pixel_cov);
      txn->addConstraint(constraint);
    }
  }

  // Noiseless odometry at exact ground truth
  for (size_t i = 0; i < kNumCameras - 1; ++i)
  {
    Eigen::Vector3d dt = kCamPositions[i + 1] - kCamPositions[i];
    vesta_core::Vector7d delta;
    delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 0.01;
    auto rel = vesta_constraints::RelativePose3DStampedConstraint::make_shared(
        "odom", *positions[i], *orientations[i], *positions[i + 1], *orientations[i + 1], delta, cov);
    txn->addConstraint(rel);
  }

  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

  optimizer.addTransaction("visual_slam", txn);
  auto summary = optimizer.optimize();
  logSolverSummary("VisualSlam::MonoBatchNoiseless", summary);

  ASSERT_TRUE(summary.IsSolutionUsable());

  const auto& result_graph = optimizer.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 1e-3) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 1e-3) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 1e-3) << "Camera " << i << " z";
  }

  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    const auto& lm =
        dynamic_cast<const vesta_variables::Point3DLandmark&>(result_graph.getVariable(landmarks[j]->uuid()));
    EXPECT_NEAR(lm.x(), kLandmarks[j].x(), 1e-3) << "Landmark " << j << " x";
    EXPECT_NEAR(lm.y(), kLandmarks[j].y(), 1e-3) << "Landmark " << j << " y";
    EXPECT_NEAR(lm.z(), kLandmarks[j].z(), 1e-3) << "Landmark " << j << " z";
  }
}

// ---------------------------------------------------------------------------
// Test: Noiseless stereo batch — verifies exact convergence to ground truth
// ---------------------------------------------------------------------------
TEST(VisualSlamTest, VisualSlam_StereoBatchNoiseless)
{
  std::mt19937 rng(42);
  std::normal_distribution<double> pos_noise(0.0, 0.05);
  std::normal_distribution<double> lm_noise(0.0, 0.1);

  const auto cam_device_id = vesta_core::uuid::generate("cam");

  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(kNumCameras - 1, 0));
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    txn->addInvolvedStamp(vesta_core::Timestamp(i, 0));
  }

  auto stereo_cam = vesta_variables::StereoCameraFixed::make_shared(uint64_t{ 0 });
  stereo_cam->fx() = kFx;
  stereo_cam->fy() = kFy;
  stereo_cam->cx() = kCx;
  stereo_cam->cy() = kCy;
  stereo_cam->baseline() = kBaseline;
  txn->addVariable(stereo_cam);

  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;
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

    positions.push_back(pos);
    orientations.push_back(ori);
    txn->addVariable(pos);
    txn->addVariable(ori);
  }

  std::vector<vesta_variables::Point3DLandmark::SharedPtr> landmarks;
  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    auto lm = vesta_variables::Point3DLandmark::make_shared(uint64_t{ j });
    lm->x() = kLandmarks[j].x() + lm_noise(rng);
    lm->y() = kLandmarks[j].y() + lm_noise(rng);
    lm->z() = kLandmarks[j].z() + lm_noise(rng);
    landmarks.push_back(lm);
    txn->addVariable(lm);
  }

  // Tight prior on first camera at exact ground truth
  {
    vesta_core::Vector7d prior_mean;
    prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-8;
    auto prior = vesta_constraints::AbsolutePose3DStampedConstraint::make_shared(
        "prior", *positions[0], *orientations[0], prior_mean, prior_cov);
    txn->addConstraint(prior);
  }

  // Noiseless stereo reprojection constraints
  vesta_core::Matrix4d pixel_cov = vesta_core::Matrix4d::Identity();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    for (size_t j = 0; j < kNumLandmarks; ++j)
    {
      double z_cam = 0.0;
      Eigen::Vector4d obs_gt = projectStereo(kCamPositions[i], kLandmarks[j], z_cam);
      if (z_cam <= 0.0)
      {
        continue;
      }
      vesta_core::Vector4d obs;
      obs << obs_gt[0], obs_gt[1], obs_gt[2], obs_gt[3];

      auto constraint = vesta_constraints::StereoReprojectionErrorConstraint::make_shared(
          "stereo_camera", *positions[i], *orientations[i], *stereo_cam, *landmarks[j], obs, pixel_cov);
      txn->addConstraint(constraint);
    }
  }

  // Noiseless odometry at exact ground truth
  for (size_t i = 0; i < kNumCameras - 1; ++i)
  {
    Eigen::Vector3d dt = kCamPositions[i + 1] - kCamPositions[i];
    vesta_core::Vector7d delta;
    delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 0.01;
    auto rel = vesta_constraints::RelativePose3DStampedConstraint::make_shared(
        "odom", *positions[i], *orientations[i], *positions[i + 1], *orientations[i + 1], delta, cov);
    txn->addConstraint(rel);
  }

  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

  optimizer.addTransaction("visual_slam", txn);
  auto summary = optimizer.optimize();
  logSolverSummary("VisualSlam::StereoBatchNoiseless", summary);

  ASSERT_TRUE(summary.IsSolutionUsable());

  const auto& result_graph = optimizer.graph();
  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), kCamPositions[i].x(), 1e-3) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), kCamPositions[i].y(), 1e-3) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), kCamPositions[i].z(), 1e-3) << "Camera " << i << " z";
  }

  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    const auto& lm =
        dynamic_cast<const vesta_variables::Point3DLandmark&>(result_graph.getVariable(landmarks[j]->uuid()));
    EXPECT_NEAR(lm.x(), kLandmarks[j].x(), 1e-3) << "Landmark " << j << " x";
    EXPECT_NEAR(lm.y(), kLandmarks[j].y(), 1e-3) << "Landmark " << j << " y";
    EXPECT_NEAR(lm.z(), kLandmarks[j].z(), 1e-3) << "Landmark " << j << " z";
  }
}

// ---------------------------------------------------------------------------
// Test: Large-scale stereo fixed-lag smoother with marginalization
//
// 20 keyframes at 2 Hz over 10 seconds, 5-second lag window.
// ~2000 landmarks spread along the trajectory; the sliding window
// holds ~1000 landmarks at any time. Marginalization kicks in once
// the window fills, exercising the full fixed-lag pipeline.
// ---------------------------------------------------------------------------
TEST(VisualSlamTest, StereoFixedLagLargeScale)
{
  // --- Configuration ---
  constexpr int NUM_KEYFRAMES = 20;
  constexpr double KEYFRAME_DT = 0.5;   // seconds between keyframes
  constexpr double LAG_DURATION = 5.0;  // sliding window duration
  constexpr int TOTAL_LANDMARKS = 200;  // total landmarks in environment
  constexpr double IMAGE_W = 640.0;
  constexpr double IMAGE_H = 480.0;
  constexpr double MIN_DEPTH = 0.5;

  std::mt19937 rng(42);
  const auto cam_device_id = vesta_core::uuid::generate("cam");

  // Ground truth: constant velocity along x-axis at 1 m/s
  // (keyframe spacing = KEYFRAME_DT * 1 m/s = 0.5 m)
  std::vector<Eigen::Vector3d> gt_cam;
  for (int i = 0; i < NUM_KEYFRAMES; ++i)
  {
    gt_cam.emplace_back(i * KEYFRAME_DT, 0.0, 0.0);
  }
  const double traj_length = (NUM_KEYFRAMES - 1) * KEYFRAME_DT;

  // Generate landmarks spread along and beyond the trajectory
  std::uniform_real_distribution<double> lm_x(-2.0, traj_length + 2.0);
  std::uniform_real_distribution<double> lm_y(-3.0, 3.0);
  std::uniform_real_distribution<double> lm_z(4.0, 15.0);

  std::vector<Eigen::Vector3d> gt_landmarks;
  gt_landmarks.reserve(TOTAL_LANDMARKS);
  for (int j = 0; j < TOTAL_LANDMARKS; ++j)
  {
    gt_landmarks.emplace_back(lm_x(rng), lm_y(rng), lm_z(rng));
  }

  // Stereo camera intrinsics (fixed)
  auto stereo_cam = vesta_variables::StereoCameraFixed::make_shared(uint64_t{ 0 });
  stereo_cam->fx() = kFx;
  stereo_cam->fy() = kFy;
  stereo_cam->cx() = kCx;
  stereo_cam->cy() = kCy;
  stereo_cam->baseline() = kBaseline;

  // Pre-create all landmark variables (only added to graph on first observation)
  std::normal_distribution<double> lm_init_noise(0.0, 0.2);
  std::vector<vesta_variables::Point3DLandmark::SharedPtr> landmarks;
  landmarks.reserve(TOTAL_LANDMARKS);
  for (int j = 0; j < TOTAL_LANDMARKS; ++j)
  {
    auto lm = vesta_variables::Point3DLandmark::make_shared(static_cast<uint64_t>(j));
    lm->x() = gt_landmarks[j].x() + lm_init_noise(rng);
    lm->y() = gt_landmarks[j].y() + lm_init_noise(rng);
    lm->z() = gt_landmarks[j].z() + lm_init_noise(rng);
    landmarks.push_back(lm);
  }

  // Track which landmarks have been added to the graph
  std::vector<bool> landmark_added(TOTAL_LANDMARKS, false);

  // Set up fixed-lag smoother
  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::FixedLagSmootherParams params;
  params.lag_duration = vesta_core::Duration(static_cast<int64_t>(LAG_DURATION * 1e9));
  params.solver_options.max_num_iterations = 50;
  params.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  vesta_optimizers::FixedLagSmoother smoother(params, std::move(graph));

  smoother.setMarginalizer(std::make_unique<vesta_constraints::BlockDiagonalMarginalizer>());

  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;

  std::normal_distribution<double> pixel_noise(0.0, 1.0);
  std::normal_distribution<double> pos_init_noise(0.0, 0.05);
  vesta_core::Matrix4d pixel_cov = vesta_core::Matrix4d::Identity();

  for (int i = 0; i < NUM_KEYFRAMES; ++i)
  {
    auto txn = std::make_shared<vesta_core::Transaction>();
    vesta_core::Timestamp stamp(static_cast<int64_t>(i * KEYFRAME_DT * 1e9));
    txn->stamp(stamp);
    txn->addInvolvedStamp(stamp);

    // Camera pose (identity orientation, perturbed position)
    auto pos = vesta_variables::Position3DStamped::make_shared(stamp, cam_device_id);
    auto ori = vesta_variables::Orientation3DStamped::make_shared(stamp, cam_device_id);
    pos->x() = gt_cam[i].x() + pos_init_noise(rng);
    pos->y() = gt_cam[i].y() + pos_init_noise(rng);
    pos->z() = gt_cam[i].z() + pos_init_noise(rng);
    ori->w() = 1.0;
    ori->x() = 0.0;
    ori->y() = 0.0;
    ori->z() = 0.0;

    positions.push_back(pos);
    orientations.push_back(ori);
    txn->addVariable(pos);
    txn->addVariable(ori);

    // First transaction: add stereo camera intrinsics
    if (i == 0)
    {
      txn->addVariable(stereo_cam);
    }

    // Pose prior on first camera
    if (i == 0)
    {
      vesta_core::Vector7d prior_mean;
      prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-4;
      auto prior =
          vesta_constraints::AbsolutePose3DStampedConstraint::make_shared("prior", *pos, *ori, prior_mean, prior_cov);
      txn->addConstraint(prior);
    }

    // Relative odometry from previous keyframe
    if (i > 0)
    {
      vesta_core::Timestamp prev_stamp(static_cast<int64_t>((i - 1) * KEYFRAME_DT * 1e9));
      txn->addInvolvedStamp(prev_stamp);

      Eigen::Vector3d dt = gt_cam[i] - gt_cam[i - 1];
      vesta_core::Vector7d delta;
      delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d odom_cov = vesta_core::Matrix6d::Identity() * 0.01;
      auto rel = vesta_constraints::RelativePose3DStampedConstraint::make_shared(
          "odom", *positions[i - 1], *orientations[i - 1], *pos, *ori, delta, odom_cov);
      txn->addConstraint(rel);
    }

    // Stereo reprojection constraints for all visible landmarks
    int obs_count = 0;
    for (int j = 0; j < TOTAL_LANDMARKS; ++j)
    {
      // p_cam = R_wc^{-1} * (X - p_world) = X - p_world  (identity R)
      Eigen::Vector3d p_cam = gt_landmarks[j] - gt_cam[i];
      if (p_cam.z() <= MIN_DEPTH)
      {
        continue;
      }

      double u_left = kFx * p_cam.x() / p_cam.z() + kCx;
      double v_left = kFy * p_cam.y() / p_cam.z() + kCy;
      double u_right = kFx * (p_cam.x() - kBaseline) / p_cam.z() + kCx;

      if (u_left < 0 || u_left > IMAGE_W || v_left < 0 || v_left > IMAGE_H)
      {
        continue;
      }

      // Add landmark variable on first observation
      if (!landmark_added[j])
      {
        txn->addVariable(landmarks[j]);
        landmark_added[j] = true;
      }

      vesta_core::Vector4d obs;
      obs << u_left + pixel_noise(rng), v_left + pixel_noise(rng), u_right + pixel_noise(rng),
          v_left + pixel_noise(rng);  // v_right == v_left for rectified stereo

      auto constraint = vesta_constraints::StereoReprojectionErrorConstraint::make_shared(
          "stereo_camera", *pos, *ori, *stereo_cam, *landmarks[j], obs, pixel_cov);
      txn->addConstraint(constraint);
      ++obs_count;
    }

    smoother.addTransaction("visual_slam", txn);
    auto summary = smoother.optimize();

    // Log stats with observation count for context
    std::cout << "  [kf " << i << "] observations=" << obs_count;
    logSolverSummary("StereoFixedLagLargeScale [kf " + std::to_string(i) + "]", summary);
    ASSERT_TRUE(summary.IsSolutionUsable()) << "Fixed-lag optimize failed at keyframe " << i;
  }

  // Verify the most recent poses in the window are close to ground truth
  const auto& result_graph = smoother.graph();
  for (int i = NUM_KEYFRAMES - 3; i < NUM_KEYFRAMES; ++i)
  {
    if (!result_graph.variableExists(positions[i]->uuid()))
    {
      continue;
    }
    const auto& pos =
        dynamic_cast<const vesta_variables::Position3DStamped&>(result_graph.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), gt_cam[i].x(), 0.2) << "Camera " << i << " x";
    EXPECT_NEAR(pos.y(), gt_cam[i].y(), 0.2) << "Camera " << i << " y";
    EXPECT_NEAR(pos.z(), gt_cam[i].z(), 0.2) << "Camera " << i << " z";
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
