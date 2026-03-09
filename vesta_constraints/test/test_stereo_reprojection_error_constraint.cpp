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
#include <vesta_constraints/vision/stereo_reprojection_error_constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/eigen_gtest.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/stereo_camera.h>
#include <vesta_variables/vision/point_3d_landmark.h>

#include <ceres/covariance.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>

#include <utility>
#include <vector>

using vesta_constraints::StereoReprojectionErrorConstraint;
using vesta_variables::Orientation3DStamped;
using vesta_variables::Point3DLandmark;
using vesta_variables::Position3DStamped;
using vesta_variables::StereoCamera;

TEST(StereoReprojectionErrorConstraint, Constructor)
{
  // Construct a constraint just to make sure it compiles.
  Position3DStamped position_variable(vesta_core::Timestamp(1234, 5678), vesta_core::uuid::generate("walle"));
  Orientation3DStamped orientation_variable(vesta_core::Timestamp(1234, 5678), vesta_core::uuid::generate("walle"));
  Point3DLandmark point(0);
  StereoCamera calibration_variable(0);

  vesta_core::Vector4d mean;
  mean << 320.0, 240.0, 310.0, 240.0;

  vesta_core::Matrix4d cov = vesta_core::Matrix4d::Identity() * 0.25;

  EXPECT_NO_THROW(StereoReprojectionErrorConstraint constraint(
      "test", position_variable, orientation_variable, calibration_variable, point, mean, cov));
}

TEST(StereoReprojectionErrorConstraint, Covariance)
{
  // Verify the covariance <--> sqrt information conversions are correct
  Position3DStamped position_variable(vesta_core::Timestamp(1234, 5678), vesta_core::uuid::generate("mo"));
  Orientation3DStamped orientation_variable(vesta_core::Timestamp(1234, 5678), vesta_core::uuid::generate("mo"));
  Point3DLandmark point(0);
  StereoCamera calibration_variable(0);

  vesta_core::Vector4d mean;
  mean << 320.0, 240.0, 310.0, 240.0;

  vesta_core::Matrix4d cov = vesta_core::Matrix4d::Identity() * 0.25;

  StereoReprojectionErrorConstraint constraint(
      "test", position_variable, orientation_variable, calibration_variable, point, mean, cov);

  // Define the expected matrices (chol(inv(0.25*I)) = 2*I)
  vesta_core::Matrix4d expected_sqrt_info = vesta_core::Matrix4d::Identity() * 2.0;
  vesta_core::Matrix4d expected_cov = cov;

  // Compare
  EXPECT_MATRIX_NEAR(expected_cov, constraint.covariance(), 1.0e-9);
  EXPECT_MATRIX_NEAR(expected_sqrt_info, constraint.sqrtInformation(), 1.0e-9);
}

TEST(StereoReprojectionErrorConstraint, Optimization)
{
  // Optimize camera pose given stereo observations of known 3D points.
  //
  // Ground truth: camera at origin with identity orientation.
  // Camera intrinsics: fx=500, fy=500, cx=320, cy=240, baseline=0.12
  //
  // We place 4 known 3D points in front of the camera and generate the
  // stereo observations (u_l, v_l, u_r, v_r) from the ground truth pose.
  // Then we perturb the pose and optimize to recover it.

  const double fx = 500.0;
  const double fy = 500.0;
  const double cx = 320.0;
  const double cy = 240.0;
  const double baseline = 0.12;

  // 3D points (in world frame, which equals camera frame at GT pose)
  std::vector<Point3DLandmark::SharedPtr> point_variables;
  point_variables.push_back(Point3DLandmark::make_shared(0));
  point_variables.push_back(Point3DLandmark::make_shared(1));
  point_variables.push_back(Point3DLandmark::make_shared(2));
  point_variables.push_back(Point3DLandmark::make_shared(3));
  point_variables[0]->x() = -1.0; point_variables[0]->y() = -1.0; point_variables[0]->z() = 5.0;
  point_variables[1]->x() = -1.0; point_variables[1]->y() =  1.0; point_variables[1]->z() = 5.0;
  point_variables[2]->x() =  1.0; point_variables[2]->y() = -1.0; point_variables[2]->z() = 5.0;
  point_variables[3]->x() =  1.0; point_variables[3]->y() =  1.0; point_variables[3]->z() = 5.0;

  // Generate stereo observations from GT pose (identity rotation, zero translation)
  // For identity pose: p = R*X + t = X (point in camera frame = point in world frame)
  // u_l = fx * X/Z + cx, v_l = fy * Y/Z + cy
  // u_r = fx * (X - baseline)/Z + cx, v_r = fy * Y/Z + cy
  std::vector<vesta_core::Vector4d> means(4);
  for (size_t i = 0; i < 4; ++i)
  {
    double px = point_variables[i]->x();
    double py = point_variables[i]->y();
    double pz = point_variables[i]->z();
    double u_l = fx * px / pz + cx;
    double v_l = fy * py / pz + cy;
    double u_r = fx * (px - baseline) / pz + cx;
    double v_r = fy * py / pz + cy;
    means[i] << u_l, v_l, u_r, v_r;
  }

  // Create perturbed camera pose (initial guess)
  auto position_variable = Position3DStamped::make_shared(
      vesta_core::Timestamp(1, 0), vesta_core::uuid::generate("stereo"));
  position_variable->x() = 0.5;
  position_variable->y() = -0.3;
  position_variable->z() = 0.2;

  auto orientation_variable = Orientation3DStamped::make_shared(
      vesta_core::Timestamp(1, 0), vesta_core::uuid::generate("stereo"));
  orientation_variable->w() = 0.98;
  orientation_variable->x() = 0.05;
  orientation_variable->y() = -0.1;
  orientation_variable->z() = 0.15;

  // Create camera calibration (held constant)
  auto calibration_variable = StereoCamera::make_shared(0);
  calibration_variable->fx() = fx;
  calibration_variable->fy() = fy;
  calibration_variable->cx() = cx;
  calibration_variable->cy() = cy;
  calibration_variable->baseline() = baseline;

  // Define observation covariance
  vesta_core::Matrix4d cov = vesta_core::Matrix4d::Identity() * 0.25;

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = vesta_core::Loss::Ownership;
  ceres::Problem problem(problem_options);

  // Add parameter blocks
  problem.AddParameterBlock(position_variable->data(), position_variable->size(),
                            position_variable->manifold());
  problem.AddParameterBlock(orientation_variable->data(), orientation_variable->size(),
                            orientation_variable->manifold());
  problem.AddParameterBlock(calibration_variable->data(), calibration_variable->size(),
                            calibration_variable->manifold());

  // Hold calibration constant
  problem.SetParameterBlockConstant(calibration_variable->data());

  for (size_t i = 0; i < point_variables.size(); ++i)
  {
    auto constraint = StereoReprojectionErrorConstraint::make_shared(
        "test", *position_variable, *orientation_variable, *calibration_variable,
        *point_variables[i], means[i], cov);

    problem.AddParameterBlock(point_variables[i]->data(), point_variables[i]->size(),
                              point_variables[i]->manifold());

    // Hold landmarks constant (known)
    problem.SetParameterBlockConstant(point_variables[i]->data());

    std::vector<double*> parameter_blocks;
    parameter_blocks.push_back(position_variable->data());
    parameter_blocks.push_back(orientation_variable->data());
    parameter_blocks.push_back(calibration_variable->data());
    parameter_blocks.push_back(point_variables[i]->data());

    problem.AddResidualBlock(constraint->costFunction(), constraint->lossFunction(), parameter_blocks);
  }

  // Run the solver
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Check that the pose converges to the ground truth (identity pose at origin)
  EXPECT_NEAR(0.0, position_variable->x(), 1.0e-5);
  EXPECT_NEAR(0.0, position_variable->y(), 1.0e-5);
  EXPECT_NEAR(0.0, position_variable->z(), 1.0e-5);

  EXPECT_NEAR(1.0, orientation_variable->w(), 5.0e-3);
  EXPECT_NEAR(0.0, orientation_variable->x(), 5.0e-3);
  EXPECT_NEAR(0.0, orientation_variable->y(), 5.0e-3);
  EXPECT_NEAR(0.0, orientation_variable->z(), 5.0e-3);
}

TEST(StereoReprojectionErrorConstraint, Serialization)
{
  // Construct a constraint
  Position3DStamped position_variable(vesta_core::Timestamp(1234, 5678), vesta_core::uuid::generate("walle"));
  Orientation3DStamped orientation_variable(vesta_core::Timestamp(1234, 5678), vesta_core::uuid::generate("walle"));
  Point3DLandmark point(0);

  StereoCamera calibration_variable(0);
  calibration_variable.fx() = 500.0;
  calibration_variable.fy() = 500.0;
  calibration_variable.cx() = 320.0;
  calibration_variable.cy() = 240.0;
  calibration_variable.baseline() = 0.12;

  vesta_core::Vector4d mean;
  mean << 320.0, 240.0, 308.0, 240.0;

  vesta_core::Matrix4d cov = vesta_core::Matrix4d::Identity() * 0.25;

  StereoReprojectionErrorConstraint expected(
      "test", position_variable, orientation_variable, calibration_variable, point, mean, cov);

  // Serialize the constraint into an archive
  std::stringstream stream;
  {
    vesta_core::TextOutputArchive archive(stream);
    expected.serialize(archive);
  }

  // Deserialize a new constraint from that same stream
  StereoReprojectionErrorConstraint actual;
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
