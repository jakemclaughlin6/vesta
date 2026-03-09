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
#include <vesta_core/serialization.h>
#include <vesta_variables/vision/stereo_camera_fixed.h>

#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>

#include <sstream>

using vesta_variables::StereoCameraFixed;

TEST(StereoCameraFixed, Type)
{
  StereoCameraFixed variable(0);
  EXPECT_EQ("vesta_variables::StereoCameraFixed", variable.type());
}

TEST(StereoCameraFixed, UUID)
{
  // Same camera id produces same UUID
  {
    StereoCameraFixed variable1(0);
    StereoCameraFixed variable2(0);
    EXPECT_EQ(variable1.uuid(), variable2.uuid());
  }
  // Different camera id produces different UUID
  {
    StereoCameraFixed variable1(0);
    StereoCameraFixed variable2(1);
    EXPECT_NE(variable1.uuid(), variable2.uuid());
  }
  // Fixed and non-fixed produce different UUIDs (different types)
  {
    vesta_variables::StereoCamera variable1(0);
    StereoCameraFixed variable2(0);
    EXPECT_NE(variable1.uuid(), variable2.uuid());
  }
}

TEST(StereoCameraFixed, HoldConstant)
{
  StereoCameraFixed variable(0);
  EXPECT_TRUE(variable.holdConstant());
}

TEST(StereoCameraFixed, Accessors)
{
  StereoCameraFixed variable(0, 500.0, 500.0, 320.0, 240.0, 0.12);
  EXPECT_DOUBLE_EQ(500.0, variable.fx());
  EXPECT_DOUBLE_EQ(500.0, variable.fy());
  EXPECT_DOUBLE_EQ(320.0, variable.cx());
  EXPECT_DOUBLE_EQ(240.0, variable.cy());
  EXPECT_DOUBLE_EQ(0.12, variable.baseline());
}

struct StereoCostFunctor
{
  template <typename T>
  bool operator()(const T* const k, T* residual) const
  {
    residual[0] = k[0] - T(500.0);
    residual[1] = k[1] - T(500.0);
    residual[2] = k[2] - T(320.0);
    residual[3] = k[3] - T(240.0);
    residual[4] = k[4] - T(0.12);
    return true;
  }
};

TEST(StereoCameraFixed, Optimization)
{
  // Create a fixed stereo camera with initial values
  StereoCameraFixed K(0, 640.0, 480.0, 300.0, 200.0, 0.10);

  // Build the problem
  ceres::CostFunction* cost_function =
    new ceres::AutoDiffCostFunction<StereoCostFunctor, 5, 5>(new StereoCostFunctor());

  ceres::Problem problem;
  problem.AddParameterBlock(K.data(), K.size());

  if (K.holdConstant())
  {
    problem.SetParameterBlockConstant(K.data());
  }
  problem.AddResidualBlock(cost_function, nullptr, K.data());

  // Solve
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Camera should NOT have changed because it's held constant
  EXPECT_DOUBLE_EQ(640.0, K.fx());
  EXPECT_DOUBLE_EQ(480.0, K.fy());
  EXPECT_DOUBLE_EQ(300.0, K.cx());
  EXPECT_DOUBLE_EQ(200.0, K.cy());
  EXPECT_DOUBLE_EQ(0.10, K.baseline());
}

TEST(StereoCameraFixed, Serialization)
{
  StereoCameraFixed expected(0, 500.0, 500.0, 320.0, 240.0, 0.12);

  // Serialize
  std::stringstream stream;
  {
    vesta_core::TextOutputArchive archive(stream);
    expected.serialize(archive);
  }

  // Deserialize
  StereoCameraFixed actual;
  {
    vesta_core::TextInputArchive archive(stream);
    actual.deserialize(archive);
  }

  // Compare
  EXPECT_EQ(expected.id(), actual.id());
  EXPECT_EQ(expected.uuid(), actual.uuid());
  EXPECT_DOUBLE_EQ(expected.fx(), actual.fx());
  EXPECT_DOUBLE_EQ(expected.fy(), actual.fy());
  EXPECT_DOUBLE_EQ(expected.cx(), actual.cx());
  EXPECT_DOUBLE_EQ(expected.cy(), actual.cy());
  EXPECT_DOUBLE_EQ(expected.baseline(), actual.baseline());
  EXPECT_TRUE(actual.holdConstant());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
