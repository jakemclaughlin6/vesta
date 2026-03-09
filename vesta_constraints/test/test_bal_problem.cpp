/*
 * Software License Agreement (BSD License)
 *
 *  Author: Oscar Mendez
 *  Created on Dec 12 2023
 *
 *  Copyright (c) 2023 Locus Robotics
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
#include <vesta_constraints/vision/reprojection_error_constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/eigen_gtest.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/pinhole_camera_fixed.h>
#include <vesta_variables/vision/point_3d_landmark.h>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

using vesta_constraints::ReprojectionErrorConstraint;
using vesta_variables::Orientation3DStamped;
using vesta_variables::PinholeCameraFixed;
using vesta_variables::Point3DLandmark;
using vesta_variables::Position3DStamped;

// BALProblem adapted from:
// https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/simple_bundle_adjuster.cc
//
// The BAL format stores cameras using the Snavelly convention:
//   - Rotation: angle-axis (converted to quaternion q_cw on load)
//   - Translation: t (camera-frame translation, p_cam = R_cw * X + t)
//   - Intrinsics: focal, k1, k2
//
// We convert to world-frame convention on load:
//   - q_wc = conjugate(q_cw)
//   - p_world = -R_cw^T * t
//   - fx = fy = -focal (negated to compensate for Snavelly's negative-z axis)
//   - cx = cy = 0 (BAL has no principal point)
//
// Radial distortion (k1, k2) is ignored since ReprojectionErrorConstraint
// uses a standard pinhole model without distortion.
class BALProblem {
public:
  BALProblem() = default;
  ~BALProblem() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }
  int num_observations() const { return num_observations_; }
  int num_cameras() const { return num_cameras_; }
  int num_points() const { return num_points_; }
  const double *observations() const { return observations_; }

  // Access world-frame camera parameters: [q_wc(4), p_world(3), fx, fy, cx, cy]
  // = 11 doubles per camera
  double *camera(int i) { return parameters_ + i * 11; }
  double *points(int i) {
    return parameters_ + 11 * num_cameras_ + i * 3;
  }

  int camera_for_observation(int i) const { return camera_index_[i]; }
  int point_for_observation(int i) const { return point_index_[i]; }

  bool LoadFile(const char *filename) {
    FILE *fptr = fopen(filename, "r");
    if (fptr == nullptr) {
      return false;
    }
    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    for (int i = 0; i < num_observations_; ++i) {
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 2; ++j) {
        FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
      }
    }

    // Read raw BAL parameters: 9 per camera (angle-axis(3), t(3), f, k1, k2)
    // + 3 per point
    int num_raw = 9 * num_cameras_ + 3 * num_points_;
    auto *raw = new double[num_raw];
    for (int i = 0; i < num_raw; ++i) {
      FscanfOrDie(fptr, "%lf", raw + i);
    }
    fclose(fptr);

    // Convert to world-frame convention:
    // [q_wc(4), p_world(3), fx, fy, cx, cy] = 11 per camera
    int num_params = 11 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_params];

    double *src = raw;
    double *dst = parameters_;
    for (int i = 0; i < num_cameras_; ++i) {
      // Convert angle-axis to quaternion (gives q_cw)
      double q_cw[4];
      ceres::AngleAxisToQuaternion(src, q_cw);
      src += 3;

      // q_wc = conjugate(q_cw)
      dst[0] = q_cw[0];
      dst[1] = -q_cw[1];
      dst[2] = -q_cw[2];
      dst[3] = -q_cw[3];

      // p_world = -R_cw^T * t = R_wc * (-t)
      double neg_t[3] = {-src[0], -src[1], -src[2]};
      // R_wc rotates from camera to world, and q_wc is exactly that
      ceres::QuaternionRotatePoint(dst, neg_t, dst + 4);
      src += 3;

      // fx = fy = -focal (Snavelly uses negative-z projection)
      // cx = cy = 0
      double focal = src[0];
      dst[7] = -focal;  // fx
      dst[8] = -focal;  // fy
      dst[9] = 0.0;     // cx
      dst[10] = 0.0;    // cy
      src += 3; // skip focal, k1, k2

      dst += 11;
    }

    // Copy points as-is (world-frame coordinates)
    for (int i = 0; i < 3 * num_points_; ++i) {
      *dst++ = *src++;
    }

    delete[] raw;
    return true;
  }

private:
  template <typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }
  int num_cameras_ = 0;
  int num_points_ = 0;
  int num_observations_ = 0;
  int *point_index_ = nullptr;
  int *camera_index_ = nullptr;
  double *observations_ = nullptr;
  double *parameters_ = nullptr;
};

// Inline world-frame pinhole reprojection error for raw Ceres comparison.
// Matches the convention in ReprojectionErrorCostFunctor:
//   p_cam = R_wc^{-1} * (X - p_world)
//   u = fx * p_cam[0] / p_cam[2] + cx
//   v = fy * p_cam[1] / p_cam[2] + cy
struct WorldFramePinholeReprojectionError {
  WorldFramePinholeReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T *const position, const T *const orientation,
                  const T *const calibration, const T *const point,
                  T *residuals) const {
    T diff[3];
    diff[0] = point[0] - position[0];
    diff[1] = point[1] - position[1];
    diff[2] = point[2] - position[2];

    T q_inv[4];
    q_inv[0] = orientation[0];
    q_inv[1] = -orientation[1];
    q_inv[2] = -orientation[2];
    q_inv[3] = -orientation[3];

    T p[3];
    ceres::QuaternionRotatePoint(q_inv, diff, p);

    T u = calibration[0] * p[0] / p[2] + calibration[2];
    T v = calibration[1] * p[1] / p[2] + calibration[3];

    residuals[0] = u - T(observed_x);
    residuals[1] = v - T(observed_y);
    return true;
  }

  static ceres::CostFunction *Create(double observed_x, double observed_y) {
    return new ceres::AutoDiffCostFunction<WorldFramePinholeReprojectionError, 2,
                                           3, 4, 4, 3>(
        new WorldFramePinholeReprojectionError(observed_x, observed_y));
  }

  double observed_x;
  double observed_y;
};

TEST(ReprojectionErrorConstraint, BAL) {
  std::string filename = "problem-21-11315-pre.txt";

  // ---- Solve with raw Ceres ----
  BALProblem bal_ceres;
  ASSERT_TRUE(bal_ceres.LoadFile(filename.c_str()))
      << "Unable to open file " << filename;

  ceres::Problem problem_ceres;
  const double *obs_ceres = bal_ceres.observations();
  for (int i = 0; i < bal_ceres.num_observations(); ++i) {
    double *cam = bal_ceres.camera(bal_ceres.camera_for_observation(i));
    double *pt = bal_ceres.points(bal_ceres.point_for_observation(i));

    // cam layout: [q_wc(4), p_world(3), fx, fy, cx, cy]
    double *position = cam + 4;
    double *orientation = cam;
    double *calibration = cam + 7;

    ceres::CostFunction *cost_function =
        WorldFramePinholeReprojectionError::Create(obs_ceres[2 * i + 0],
                                                   obs_ceres[2 * i + 1]);
    problem_ceres.AddResidualBlock(cost_function, nullptr, position,
                                   orientation, calibration, pt);
  }

  // Hold calibration constant for cameras that have observations
  for (int i = 0; i < bal_ceres.num_cameras(); ++i) {
    double *calibration = bal_ceres.camera(i) + 7;
    if (problem_ceres.HasParameterBlock(calibration)) {
      problem_ceres.SetParameterBlockConstant(calibration);
    }
  }

  ceres::Solver::Options options_ceres;
  options_ceres.linear_solver_type = ceres::SPARSE_SCHUR;
  options_ceres.max_num_iterations = 10;
  ceres::Solver::Summary summary_ceres;
  ceres::Solve(options_ceres, &problem_ceres, &summary_ceres);

  // ---- Solve with Vesta constraints ----
  BALProblem bal_vesta;
  ASSERT_TRUE(bal_vesta.LoadFile(filename.c_str()))
      << "Unable to open file " << filename;

  // Create vesta variables from loaded data
  std::vector<Position3DStamped> cams_p;
  std::vector<Orientation3DStamped> cams_q;
  std::vector<PinholeCameraFixed> cams_k;
  cams_p.reserve(bal_vesta.num_cameras());
  cams_q.reserve(bal_vesta.num_cameras());
  cams_k.reserve(bal_vesta.num_cameras());

  for (int i = 0; i < bal_vesta.num_cameras(); ++i) {
    double *cam = bal_vesta.camera(i);
    // cam layout: [q_wc(4), p_world(3), fx, fy, cx, cy]

    cams_q.emplace_back(vesta_core::Timestamp(i, 0),
                        vesta_core::uuid::generate("bal"));
    cams_q[i].w() = cam[0];
    cams_q[i].x() = cam[1];
    cams_q[i].y() = cam[2];
    cams_q[i].z() = cam[3];

    cams_p.emplace_back(vesta_core::Timestamp(i, 0),
                        vesta_core::uuid::generate("bal"));
    cams_p[i].x() = cam[4];
    cams_p[i].y() = cam[5];
    cams_p[i].z() = cam[6];

    cams_k.emplace_back(static_cast<uint64_t>(i));
    cams_k[i].fx() = cam[7];
    cams_k[i].fy() = cam[8];
    cams_k[i].cx() = cam[9];
    cams_k[i].cy() = cam[10];
  }

  std::vector<Point3DLandmark> pts;
  pts.reserve(bal_vesta.num_points());
  for (int i = 0; i < bal_vesta.num_points(); ++i) {
    double *pt = bal_vesta.points(i);
    pts.emplace_back(static_cast<uint64_t>(i));
    pts[i].x() = pt[0];
    pts[i].y() = pt[1];
    pts[i].z() = pt[2];
  }

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = vesta_core::Loss::Ownership;
  ceres::Problem problem;
  const double *obs = bal_vesta.observations();
  for (int i = 0; i < bal_vesta.num_observations(); ++i) {
    int c = bal_vesta.camera_for_observation(i);
    int p = bal_vesta.point_for_observation(i);

    vesta_core::Vector2d mean;
    mean << obs[2 * i + 0], obs[2 * i + 1];

    vesta_core::Matrix2d cov;
    cov << 1e-5, 0.0, // NOLINT
        0.0, 1e-5;     // NOLINT

    auto constraint = ReprojectionErrorConstraint::make_shared(
        "test", cams_p[c], cams_q[c], cams_k[c], pts[p], mean, cov);

    problem.AddParameterBlock(pts[p].data(), pts[p].size(), pts[p].manifold());

    std::vector<double *> parameter_blocks;
    parameter_blocks.push_back(cams_p[c].data());
    parameter_blocks.push_back(cams_q[c].data());
    parameter_blocks.push_back(cams_k[c].data());
    parameter_blocks.push_back(pts[p].data());

    problem.AddResidualBlock(constraint->costFunction(),
                             constraint->lossFunction(), parameter_blocks);

    if (cams_k[c].holdConstant()) {
      problem.SetParameterBlockConstant(cams_k[c].data());
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.max_num_iterations = 10;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // ---- Compare results ----
  for (int i = 0; i < bal_vesta.num_cameras(); ++i) {
    double *cam = bal_ceres.camera(i);

    EXPECT_NEAR(cams_q[i].w(), cam[0], 1e-2);
    EXPECT_NEAR(cams_q[i].x(), cam[1], 1e-2);
    EXPECT_NEAR(cams_q[i].y(), cam[2], 1e-2);
    EXPECT_NEAR(cams_q[i].z(), cam[3], 1e-2);

    EXPECT_NEAR(cams_p[i].x(), cam[4], 1e-2);
    EXPECT_NEAR(cams_p[i].y(), cam[5], 1e-2);
    EXPECT_NEAR(cams_p[i].z(), cam[6], 1e-2);

    EXPECT_NEAR(cams_k[i].fx(), cam[7], 1e-2);
    EXPECT_NEAR(cams_k[i].fy(), cam[8], 1e-2);
    EXPECT_NEAR(cams_k[i].cx(), cam[9], 1e-2);
    EXPECT_NEAR(cams_k[i].cy(), cam[10], 1e-2);
  }

  for (int i = 0; i < bal_vesta.num_points(); ++i) {
    double *pt = bal_ceres.points(i);
    EXPECT_NEAR(pts[i].x(), pt[0], 1e-2);
    EXPECT_NEAR(pts[i].y(), pt[1], 1e-2);
    EXPECT_NEAR(pts[i].z(), pt[2], 1e-2);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
