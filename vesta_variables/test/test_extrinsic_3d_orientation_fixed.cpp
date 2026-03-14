/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026, Vesta Contributors
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
#include <vesta_core/serialization.h>
#include <vesta_variables/3d/extrinsic_3d_orientation_fixed.h>

#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <vector>

using vesta_variables::Extrinsic3DOrientationFixed;

TEST(Extrinsic3DOrientationFixed, Type)
{
  Extrinsic3DOrientationFixed variable(0);
  EXPECT_EQ("vesta_variables::Extrinsic3DOrientationFixed", variable.type());
}

TEST(Extrinsic3DOrientationFixed, UUID)
{
  // Verify two variables with the same extrinsic ids produce the same uuids
  {
    Extrinsic3DOrientationFixed variable1(0);
    Extrinsic3DOrientationFixed variable2(0);
    EXPECT_EQ(variable1.uuid(), variable2.uuid());
  }

  // Verify two variables with different extrinsic ids produce different uuids
  {
    Extrinsic3DOrientationFixed variable1(0);
    Extrinsic3DOrientationFixed variable2(1);
    EXPECT_NE(variable1.uuid(), variable2.uuid());
  }
}

TEST(Extrinsic3DOrientationFixed, HoldConstant)
{
  Extrinsic3DOrientationFixed variable(0);
  EXPECT_TRUE(variable.holdConstant());
}

struct QuaternionCostFunctor
{
  QuaternionCostFunctor()
  {
  }

  template <typename T>
  bool operator()(const T* const q, T* residual) const
  {
    // Target: identity quaternion (1, 0, 0, 0)
    residual[0] = T(2.0) * q[1];
    residual[1] = T(2.0) * q[2];
    residual[2] = T(2.0) * q[3];
    return true;
  }
};

TEST(Extrinsic3DOrientationFixed, Optimization)
{
  Extrinsic3DOrientationFixed variable(0);
  // Start at a small rotation around z-axis
  double angle = 0.1;
  variable.w() = std::cos(angle / 2.0);
  variable.x() = 0.0;
  variable.y() = 0.0;
  variable.z() = std::sin(angle / 2.0);

  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<QuaternionCostFunctor, 3, 4>(new QuaternionCostFunctor());

  ceres::Problem problem;
  problem.AddParameterBlock(variable.data(), variable.size(), variable.manifold());
  std::vector<double*> parameter_blocks;
  parameter_blocks.push_back(variable.data());
  problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);
  if (variable.holdConstant())
  {
    problem.SetParameterBlockConstant(variable.data());
  }

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Check the variable value has not changed since it is fixed
  EXPECT_NEAR(std::cos(angle / 2.0), variable.w(), 1.0e-5);
  EXPECT_NEAR(0.0, variable.x(), 1.0e-5);
  EXPECT_NEAR(0.0, variable.y(), 1.0e-5);
  EXPECT_NEAR(std::sin(angle / 2.0), variable.z(), 1.0e-5);
}

TEST(Extrinsic3DOrientationFixed, Serialization)
{
  Extrinsic3DOrientationFixed expected(0);
  expected.w() = 1.0;
  expected.x() = 0.0;
  expected.y() = 0.0;
  expected.z() = 0.0;

  std::stringstream stream;
  {
    vesta_core::TextOutputArchive archive(stream);
    expected.serialize(archive);
  }

  Extrinsic3DOrientationFixed actual;
  {
    vesta_core::TextInputArchive archive(stream);
    actual.deserialize(archive);
  }

  EXPECT_EQ(expected.id(), actual.id());
  EXPECT_EQ(expected.uuid(), actual.uuid());
  EXPECT_EQ(expected.w(), actual.w());
  EXPECT_EQ(expected.x(), actual.x());
  EXPECT_EQ(expected.y(), actual.y());
  EXPECT_EQ(expected.z(), actual.z());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
