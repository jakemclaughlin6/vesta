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
#include <vesta_constraints/common/marginalizer.h>
#include <vesta_constraints/common/qr_marginalizer.h>
#include <vesta_constraints/common/schur_marginalizer.h>
#include <vesta_constraints/vision/reprojection_error_constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/timestamp.h>
#include <vesta_core/uuid.h>
#include <vesta_graphs/hash_graph.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/pinhole_camera_fixed.h>
#include <vesta_variables/vision/point_3d_landmark.h>

#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

namespace
{

// Camera intrinsics
constexpr double kFx = 500.0;
constexpr double kFy = 500.0;
constexpr double kCx = 320.0;
constexpr double kCy = 240.0;

Eigen::Vector2d projectMono(const Eigen::Vector3d& cam_pos, const Eigen::Vector3d& landmark, double& z_cam)
{
  Eigen::Vector3d p = landmark - cam_pos;
  z_cam = p.z();
  return { kFx * p.x() / p.z() + kCx, kFy * p.y() / p.z() + kCy };
}

/**
 * @brief Build a visual SLAM graph with cameras (stamped) and landmarks (non-stamped)
 *
 * Creates a trajectory of camera poses along the x-axis observing a grid of landmarks
 * placed in front of the cameras. This mirrors a typical fixed-lag visual SLAM scenario
 * where old poses and their observed-only landmarks need to be marginalized.
 */
struct VisualSlamGraph
{
  std::vector<Eigen::Vector3d> gt_cam_positions;
  std::vector<Eigen::Vector3d> gt_landmarks;
  std::vector<vesta_variables::Position3DStamped::SharedPtr> positions;
  std::vector<vesta_variables::Orientation3DStamped::SharedPtr> orientations;
  std::vector<vesta_variables::Point3DLandmark::SharedPtr> landmarks;
  vesta_variables::PinholeCameraFixed::SharedPtr camera;
  vesta_graphs::HashGraph graph;

  VisualSlamGraph(size_t num_cameras, size_t num_landmarks, unsigned int seed)
  {
    std::mt19937 rng(seed);
    std::normal_distribution<double> pixel_noise(0.0, 1.0);
    std::normal_distribution<double> pos_noise(0.0, 0.05);
    std::normal_distribution<double> lm_noise(0.0, 0.1);

    const auto cam_device_id = vesta_core::uuid::generate("cam");

    // Generate ground truth
    for (size_t i = 0; i < num_cameras; ++i)
    {
      gt_cam_positions.push_back({ static_cast<double>(i), 0.0, 0.0 });
    }
    for (size_t j = 0; j < num_landmarks; ++j)
    {
      double x = static_cast<double>(j % 5) - 1.0;
      double y = (static_cast<double>(j / 5) - 1.0) * 0.8;
      double z = 5.0 + static_cast<double>(j % 3) * 2.0;
      gt_landmarks.push_back({ x, y, z });
    }

    // Camera intrinsics (fixed, not optimized)
    camera = vesta_variables::PinholeCameraFixed::make_shared(uint64_t{ 0 });
    camera->fx() = kFx;
    camera->fy() = kFy;
    camera->cx() = kCx;
    camera->cy() = kCy;
    graph.addVariable(camera);

    // Camera poses
    for (size_t i = 0; i < num_cameras; ++i)
    {
      auto stamp = vesta_core::Timestamp(i, 0);
      auto pos = vesta_variables::Position3DStamped::make_shared(stamp, cam_device_id);
      auto ori = vesta_variables::Orientation3DStamped::make_shared(stamp, cam_device_id);

      pos->x() = gt_cam_positions[i].x() + pos_noise(rng);
      pos->y() = gt_cam_positions[i].y() + pos_noise(rng);
      pos->z() = gt_cam_positions[i].z() + pos_noise(rng);

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
      graph.addVariable(pos);
      graph.addVariable(ori);
    }

    // Landmarks
    for (size_t j = 0; j < num_landmarks; ++j)
    {
      auto lm = vesta_variables::Point3DLandmark::make_shared(uint64_t{ j });
      lm->x() = gt_landmarks[j].x() + lm_noise(rng);
      lm->y() = gt_landmarks[j].y() + lm_noise(rng);
      lm->z() = gt_landmarks[j].z() + lm_noise(rng);
      landmarks.push_back(lm);
      graph.addVariable(lm);
    }

    // Prior on first camera pose
    {
      vesta_core::Vector7d prior_mean;
      prior_mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d prior_cov = vesta_core::Matrix6d::Identity() * 1e-4;
      graph.addConstraint(vesta_constraints::AbsolutePose3DStampedConstraint::make_shared(
          "prior", *positions[0], *orientations[0], prior_mean, prior_cov));
    }

    // Odometry between consecutive cameras
    for (size_t i = 0; i < num_cameras - 1; ++i)
    {
      Eigen::Vector3d dt = gt_cam_positions[i + 1] - gt_cam_positions[i];
      vesta_core::Vector7d delta;
      delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
      vesta_core::Matrix6d odom_cov = vesta_core::Matrix6d::Identity() * 0.01;
      graph.addConstraint(vesta_constraints::RelativePose3DStampedConstraint::make_shared(
          "odom", *positions[i], *orientations[i], *positions[i + 1], *orientations[i + 1], delta, odom_cov));
    }

    // Reprojection constraints
    vesta_core::Matrix2d pixel_cov = vesta_core::Matrix2d::Identity();
    for (size_t i = 0; i < num_cameras; ++i)
    {
      for (size_t j = 0; j < num_landmarks; ++j)
      {
        double z_cam = 0.0;
        Eigen::Vector2d uv = projectMono(gt_cam_positions[i], gt_landmarks[j], z_cam);
        if (z_cam <= 0.0)
        {
          continue;
        }
        vesta_core::Vector2d obs;
        obs << uv.x() + pixel_noise(rng), uv.y() + pixel_noise(rng);
        graph.addConstraint(vesta_constraints::ReprojectionErrorConstraint::make_shared(
            "camera", *positions[i], *orientations[i], *camera, *landmarks[j], obs, pixel_cov));
      }
    }
  }
};

struct MarginalizationResult
{
  vesta_graphs::HashGraph graph;
  double marginalize_us;
  double optimize_us;
  ceres::Solver::Summary summary;
};

MarginalizationResult runMarginalization(const vesta_graphs::HashGraph& source_graph,
                                         const std::vector<vesta_core::UUID>& vars_to_marginalize,
                                         vesta_constraints::Marginalizer& marginalizer)
{
  MarginalizationResult result;
  result.graph = source_graph;

  auto t0 = std::chrono::high_resolution_clock::now();
  auto txn = marginalizer.marginalize("test", vars_to_marginalize, result.graph);
  auto t1 = std::chrono::high_resolution_clock::now();
  result.marginalize_us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

  result.graph.update(txn);

  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;

  auto t2 = std::chrono::high_resolution_clock::now();
  result.summary = result.graph.optimize(options);
  auto t3 = std::chrono::high_resolution_clock::now();
  result.optimize_us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());

  return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test: QR vs Schur accuracy when marginalizing landmarks only
// ---------------------------------------------------------------------------
TEST(MarginalizerComparison, LandmarkOnly_Accuracy)
{
  VisualSlamGraph vslam(5, 8, 42);
  vslam.graph.optimize();

  std::vector<vesta_core::UUID> to_marginalize;
  for (size_t j = 0; j < 3; ++j)
  {
    to_marginalize.push_back(vslam.landmarks[j]->uuid());
  }

  vesta_constraints::QRMarginalizer qr(false);
  vesta_constraints::SchurMarginalizer schur(false);

  auto result_qr = runMarginalization(vslam.graph, to_marginalize, qr);
  auto result_schur = runMarginalization(vslam.graph, to_marginalize, schur);

  ASSERT_TRUE(result_qr.summary.IsSolutionUsable());
  ASSERT_TRUE(result_schur.summary.IsSolutionUsable());

  for (size_t i = 0; i < 5; ++i)
  {
    const auto& pos_qr = dynamic_cast<const vesta_variables::Position3DStamped&>(
        result_qr.graph.getVariable(vslam.positions[i]->uuid()));
    const auto& pos_schur = dynamic_cast<const vesta_variables::Position3DStamped&>(
        result_schur.graph.getVariable(vslam.positions[i]->uuid()));

    EXPECT_NEAR(pos_qr.x(), pos_schur.x(), 1e-6) << "Camera " << i << " x";
    EXPECT_NEAR(pos_qr.y(), pos_schur.y(), 1e-6) << "Camera " << i << " y";
    EXPECT_NEAR(pos_qr.z(), pos_schur.z(), 1e-6) << "Camera " << i << " z";
  }

  for (size_t j = 3; j < 8; ++j)
  {
    const auto& lm_qr =
        dynamic_cast<const vesta_variables::Point3DLandmark&>(result_qr.graph.getVariable(vslam.landmarks[j]->uuid()));
    const auto& lm_schur = dynamic_cast<const vesta_variables::Point3DLandmark&>(
        result_schur.graph.getVariable(vslam.landmarks[j]->uuid()));

    EXPECT_NEAR(lm_qr.x(), lm_schur.x(), 1e-6) << "Landmark " << j << " x";
    EXPECT_NEAR(lm_qr.y(), lm_schur.y(), 1e-6) << "Landmark " << j << " y";
    EXPECT_NEAR(lm_qr.z(), lm_schur.z(), 1e-6) << "Landmark " << j << " z";
  }

  EXPECT_NEAR(result_qr.summary.final_cost, result_schur.summary.final_cost, 1e-6);

  std::cout << "\n=== Landmark-Only Marginalization ==="
            << "\n  QR   marginalize: " << result_qr.marginalize_us << " us"
            << "\n  Schur marginalize: " << result_schur.marginalize_us << " us"
            << "\n  QR   final cost:  " << result_qr.summary.final_cost
            << "\n  Schur final cost: " << result_schur.summary.final_cost << "\n"
            << std::endl;
}

// ---------------------------------------------------------------------------
// Test: QR vs Schur accuracy when marginalizing a mix of poses and landmarks
// ---------------------------------------------------------------------------
TEST(MarginalizerComparison, Mixed_Accuracy)
{
  VisualSlamGraph vslam(6, 10, 77);
  vslam.graph.optimize();

  std::vector<vesta_core::UUID> to_marginalize;
  to_marginalize.push_back(vslam.positions[0]->uuid());
  to_marginalize.push_back(vslam.orientations[0]->uuid());
  to_marginalize.push_back(vslam.landmarks[0]->uuid());
  to_marginalize.push_back(vslam.landmarks[1]->uuid());

  vesta_constraints::QRMarginalizer qr(false);
  vesta_constraints::SchurMarginalizer schur(false);

  auto result_qr = runMarginalization(vslam.graph, to_marginalize, qr);
  auto result_schur = runMarginalization(vslam.graph, to_marginalize, schur);

  ASSERT_TRUE(result_qr.summary.IsSolutionUsable());
  ASSERT_TRUE(result_schur.summary.IsSolutionUsable());

  for (size_t i = 1; i < 6; ++i)
  {
    const auto& pos_qr = dynamic_cast<const vesta_variables::Position3DStamped&>(
        result_qr.graph.getVariable(vslam.positions[i]->uuid()));
    const auto& pos_schur = dynamic_cast<const vesta_variables::Position3DStamped&>(
        result_schur.graph.getVariable(vslam.positions[i]->uuid()));

    EXPECT_NEAR(pos_qr.x(), pos_schur.x(), 1e-4) << "Camera " << i << " x";
    EXPECT_NEAR(pos_qr.y(), pos_schur.y(), 1e-4) << "Camera " << i << " y";
    EXPECT_NEAR(pos_qr.z(), pos_schur.z(), 1e-4) << "Camera " << i << " z";
  }

  for (size_t j = 2; j < 10; ++j)
  {
    const auto& lm_qr =
        dynamic_cast<const vesta_variables::Point3DLandmark&>(result_qr.graph.getVariable(vslam.landmarks[j]->uuid()));
    const auto& lm_schur = dynamic_cast<const vesta_variables::Point3DLandmark&>(
        result_schur.graph.getVariable(vslam.landmarks[j]->uuid()));

    EXPECT_NEAR(lm_qr.x(), lm_schur.x(), 1e-4) << "Landmark " << j << " x";
    EXPECT_NEAR(lm_qr.y(), lm_schur.y(), 1e-4) << "Landmark " << j << " y";
    EXPECT_NEAR(lm_qr.z(), lm_schur.z(), 1e-4) << "Landmark " << j << " z";
  }

  EXPECT_NEAR(result_qr.summary.final_cost, result_schur.summary.final_cost, 1e-4);

  std::cout << "\n=== Mixed (Pose + Landmark) Marginalization ==="
            << "\n  QR   marginalize: " << result_qr.marginalize_us << " us"
            << "\n  Schur marginalize: " << result_schur.marginalize_us << " us"
            << "\n  QR   final cost:  " << result_qr.summary.final_cost
            << "\n  Schur final cost: " << result_schur.summary.final_cost << "\n"
            << std::endl;
}

// ---------------------------------------------------------------------------
// Test: QR vs Schur accuracy on pose-only marginalization (no non-stamped)
// Schur should delegate to QR internally and produce identical results.
// ---------------------------------------------------------------------------
TEST(MarginalizerComparison, PoseOnly_Accuracy)
{
  VisualSlamGraph vslam(5, 6, 123);
  vslam.graph.optimize();

  std::vector<vesta_core::UUID> to_marginalize;
  to_marginalize.push_back(vslam.positions[0]->uuid());
  to_marginalize.push_back(vslam.orientations[0]->uuid());

  vesta_constraints::QRMarginalizer qr(false);
  vesta_constraints::SchurMarginalizer schur(false);

  auto result_qr = runMarginalization(vslam.graph, to_marginalize, qr);
  auto result_schur = runMarginalization(vslam.graph, to_marginalize, schur);

  ASSERT_TRUE(result_qr.summary.IsSolutionUsable());
  ASSERT_TRUE(result_schur.summary.IsSolutionUsable());

  for (size_t i = 1; i < 5; ++i)
  {
    const auto& pos_qr = dynamic_cast<const vesta_variables::Position3DStamped&>(
        result_qr.graph.getVariable(vslam.positions[i]->uuid()));
    const auto& pos_schur = dynamic_cast<const vesta_variables::Position3DStamped&>(
        result_schur.graph.getVariable(vslam.positions[i]->uuid()));

    EXPECT_NEAR(pos_qr.x(), pos_schur.x(), 1e-10) << "Camera " << i << " x";
    EXPECT_NEAR(pos_qr.y(), pos_schur.y(), 1e-10) << "Camera " << i << " y";
    EXPECT_NEAR(pos_qr.z(), pos_schur.z(), 1e-10) << "Camera " << i << " z";
  }

  EXPECT_NEAR(result_qr.summary.final_cost, result_schur.summary.final_cost, 1e-10);
}

// ---------------------------------------------------------------------------
// Test: Runtime comparison with many landmarks
// ---------------------------------------------------------------------------
TEST(MarginalizerComparison, Runtime_ManyLandmarks)
{
  constexpr size_t kNumCameras = 5;
  constexpr size_t kNumLandmarks = 700;
  constexpr int kNumTrials = 20;

  VisualSlamGraph vslam(kNumCameras, kNumLandmarks, 42);
  vslam.graph.optimize();

  std::vector<vesta_core::UUID> to_marginalize;
  for (size_t j = 0; j < kNumLandmarks; ++j)
  {
    to_marginalize.push_back(vslam.landmarks[j]->uuid());
  }

  // Warm up
  {
    vesta_constraints::QRMarginalizer qr(false);
    auto g = vslam.graph;
    qr.marginalize("test", to_marginalize, g);
  }
  {
    vesta_constraints::SchurMarginalizer schur(false);
    auto g = vslam.graph;
    schur.marginalize("test", to_marginalize, g);
  }

  // Benchmark QR
  double qr_total_us = 0.0;
  for (int t = 0; t < kNumTrials; ++t)
  {
    vesta_constraints::QRMarginalizer qr(false);
    auto g = vslam.graph;
    auto t0 = std::chrono::high_resolution_clock::now();
    qr.marginalize("test", to_marginalize, g);
    auto t1 = std::chrono::high_resolution_clock::now();
    qr_total_us += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
  }

  // Benchmark Schur
  double schur_total_us = 0.0;
  for (int t = 0; t < kNumTrials; ++t)
  {
    vesta_constraints::SchurMarginalizer schur(false);
    auto g = vslam.graph;
    auto t0 = std::chrono::high_resolution_clock::now();
    schur.marginalize("test", to_marginalize, g);
    auto t1 = std::chrono::high_resolution_clock::now();
    schur_total_us += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
  }

  std::cout << "\n=== Runtime: " << kNumLandmarks << " landmarks, " << kNumCameras << " cameras ==="
            << "\n  QR   avg: " << qr_total_us / kNumTrials << " us"
            << "\n  Schur avg: " << schur_total_us / kNumTrials << " us"
            << "\n  Speedup:   " << qr_total_us / schur_total_us << "x\n"
            << std::endl;

  // Verify accuracy on the last trial
  vesta_constraints::QRMarginalizer qr(false);
  vesta_constraints::SchurMarginalizer schur(false);

  auto result_qr = runMarginalization(vslam.graph, to_marginalize, qr);
  auto result_schur = runMarginalization(vslam.graph, to_marginalize, schur);

  ASSERT_TRUE(result_qr.summary.IsSolutionUsable());
  ASSERT_TRUE(result_schur.summary.IsSolutionUsable());

  for (size_t i = 0; i < kNumCameras; ++i)
  {
    const auto& pos_qr = dynamic_cast<const vesta_variables::Position3DStamped&>(
        result_qr.graph.getVariable(vslam.positions[i]->uuid()));
    const auto& pos_schur = dynamic_cast<const vesta_variables::Position3DStamped&>(
        result_schur.graph.getVariable(vslam.positions[i]->uuid()));

    // With 700 landmarks, numerical differences between QR and Schur grow due
    // to different elimination paths — both stay close to ground truth
    EXPECT_NEAR(pos_qr.x(), pos_schur.x(), 0.01) << "Camera " << i << " x";
    EXPECT_NEAR(pos_qr.y(), pos_schur.y(), 0.01) << "Camera " << i << " y";
    EXPECT_NEAR(pos_qr.z(), pos_schur.z(), 0.01) << "Camera " << i << " z";
  }
}

// ---------------------------------------------------------------------------
// Test: Incremental marginalization (simulating fixed-lag smoother workflow)
// ---------------------------------------------------------------------------
TEST(MarginalizerComparison, IncrementalFixedLag)
{
  constexpr size_t kNumCameras = 8;
  constexpr size_t kNumLandmarks = 12;
  constexpr size_t kWindowSize = 4;

  VisualSlamGraph vslam_qr(kNumCameras, kNumLandmarks, 99);
  vslam_qr.graph.optimize();
  VisualSlamGraph vslam_schur(kNumCameras, kNumLandmarks, 99);
  vslam_schur.graph.optimize();

  double total_qr_us = 0.0;
  double total_schur_us = 0.0;
  int num_marginalizations = 0;

  for (size_t step = 0; step + kWindowSize < kNumCameras; ++step)
  {
    std::vector<vesta_core::UUID> to_marg_qr;
    std::vector<vesta_core::UUID> to_marg_schur;
    to_marg_qr.push_back(vslam_qr.positions[step]->uuid());
    to_marg_qr.push_back(vslam_qr.orientations[step]->uuid());
    to_marg_schur.push_back(vslam_schur.positions[step]->uuid());
    to_marg_schur.push_back(vslam_schur.orientations[step]->uuid());

    size_t landmarks_to_remove = std::min(step + 1, kNumLandmarks);
    for (size_t j = step; j < landmarks_to_remove; ++j)
    {
      if (vslam_qr.graph.variableExists(vslam_qr.landmarks[j]->uuid()))
      {
        to_marg_qr.push_back(vslam_qr.landmarks[j]->uuid());
      }
      if (vslam_schur.graph.variableExists(vslam_schur.landmarks[j]->uuid()))
      {
        to_marg_schur.push_back(vslam_schur.landmarks[j]->uuid());
      }
    }

    vesta_constraints::QRMarginalizer qr(false);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto txn_qr = qr.marginalize("test", to_marg_qr, vslam_qr.graph);
    auto t1 = std::chrono::high_resolution_clock::now();
    total_qr_us += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    vesta_constraints::SchurMarginalizer schur(false);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto txn_schur = schur.marginalize("test", to_marg_schur, vslam_schur.graph);
    auto t3 = std::chrono::high_resolution_clock::now();
    total_schur_us += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());

    vslam_qr.graph.update(txn_qr);
    vslam_schur.graph.update(txn_schur);

    vslam_qr.graph.optimize();
    vslam_schur.graph.optimize();

    ++num_marginalizations;
  }

  size_t first_remaining = kNumCameras - kWindowSize;
  for (size_t i = first_remaining; i < kNumCameras; ++i)
  {
    const auto& pos_qr = dynamic_cast<const vesta_variables::Position3DStamped&>(
        vslam_qr.graph.getVariable(vslam_qr.positions[i]->uuid()));
    const auto& pos_schur = dynamic_cast<const vesta_variables::Position3DStamped&>(
        vslam_schur.graph.getVariable(vslam_schur.positions[i]->uuid()));

    EXPECT_NEAR(pos_qr.x(), pos_schur.x(), 0.05) << "Camera " << i << " x";
    EXPECT_NEAR(pos_qr.y(), pos_schur.y(), 0.05) << "Camera " << i << " y";
    EXPECT_NEAR(pos_qr.z(), pos_schur.z(), 0.05) << "Camera " << i << " z";

    EXPECT_NEAR(pos_qr.x(), vslam_qr.gt_cam_positions[i].x(), 0.15) << "QR Camera " << i << " x vs gt";
    EXPECT_NEAR(pos_schur.x(), vslam_schur.gt_cam_positions[i].x(), 0.15) << "Schur Camera " << i << " x vs gt";
  }

  std::cout << "\n=== Incremental Fixed-Lag (" << num_marginalizations << " steps) ==="
            << "\n  QR   total: " << total_qr_us << " us (avg " << total_qr_us / num_marginalizations << " us/step)"
            << "\n  Schur total: " << total_schur_us << " us (avg " << total_schur_us / num_marginalizations
            << " us/step)"
            << "\n  Speedup:   " << total_qr_us / total_schur_us << "x\n"
            << std::endl;
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
