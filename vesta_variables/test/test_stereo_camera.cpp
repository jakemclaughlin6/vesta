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
#include <vesta_variables/vision/stereo_camera.h>

#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>

#include <sstream>

using vesta_variables::StereoCamera;

TEST(StereoCamera, Type)
{
  StereoCamera variable(0);
  EXPECT_EQ("vesta_variables::StereoCamera", variable.type());
}

TEST(StereoCamera, UUID)
{
  // Verify two cameras with the same ids produce the same uuids
  {
    StereoCamera variable1(0);
    StereoCamera variable2(0);
    EXPECT_EQ(variable1.uuid(), variable2.uuid());
  }

  // Verify two cameras with different ids produce different uuids
  {
    StereoCamera variable1(0);
    StereoCamera variable2(1);
    EXPECT_NE(variable1.uuid(), variable2.uuid());
  }
}

TEST(StereoCamera, Accessors)
{
  StereoCamera K(0);
  K.fx() = 500.0;
  K.fy() = 500.0;
  K.cx() = 320.0;
  K.cy() = 240.0;
  K.baseline() = 0.12;

  EXPECT_EQ(500.0, K.fx());
  EXPECT_EQ(500.0, K.fy());
  EXPECT_EQ(320.0, K.cx());
  EXPECT_EQ(240.0, K.cy());
  EXPECT_EQ(0.12, K.baseline());
  EXPECT_EQ(5u, K.size());
  EXPECT_EQ(0u, K.id());
}

TEST(StereoCamera, ConstructorWithParameters)
{
  vesta_core::UUID uuid = vesta_core::uuid::generate("test", 42);
  StereoCamera K(uuid, 42, 500.0, 500.0, 320.0, 240.0, 0.12);

  EXPECT_EQ(500.0, K.fx());
  EXPECT_EQ(500.0, K.fy());
  EXPECT_EQ(320.0, K.cx());
  EXPECT_EQ(240.0, K.cy());
  EXPECT_EQ(0.12, K.baseline());
  EXPECT_EQ(42u, K.id());
}

struct StereoCostFunctor
{
  StereoCostFunctor()
  {
  }

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

TEST(StereoCamera, Optimization)
{
  // Create a StereoCamera
  StereoCamera K(0);
  K.fx() = 510.0;
  K.fy() = 490.0;
  K.cx() = 325.0;
  K.cy() = 245.0;
  K.baseline() = 0.15;

  // Create a simple constraint
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<StereoCostFunctor, 5, 5>(new StereoCostFunctor());

  // Build the problem
  ceres::Problem problem;
  problem.AddParameterBlock(K.data(), K.size());
  std::vector<double*> parameter_blocks;
  parameter_blocks.push_back(K.data());
  problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

  // Run the solver
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Check
  EXPECT_NEAR(500.0, K.fx(), 1.0e-5);
  EXPECT_NEAR(500.0, K.fy(), 1.0e-5);
  EXPECT_NEAR(320.0, K.cx(), 1.0e-5);
  EXPECT_NEAR(240.0, K.cy(), 1.0e-5);
  EXPECT_NEAR(0.12, K.baseline(), 1.0e-5);
}

TEST(StereoCamera, Serialization)
{
  // Create a StereoCamera
  StereoCamera expected(0);
  expected.fx() = 500.0;
  expected.fy() = 500.0;
  expected.cx() = 320.0;
  expected.cy() = 240.0;
  expected.baseline() = 0.12;

  // Serialize the variable into an archive
  std::stringstream stream;
  {
    vesta_core::TextOutputArchive archive(stream);
    expected.serialize(archive);
  }

  // Deserialize a new variable from that same stream
  StereoCamera actual;
  {
    vesta_core::TextInputArchive archive(stream);
    actual.deserialize(archive);
  }

  // Compare
  EXPECT_EQ(expected.id(), actual.id());
  EXPECT_EQ(expected.uuid(), actual.uuid());
  EXPECT_EQ(expected.fx(), actual.fx());
  EXPECT_EQ(expected.fy(), actual.fy());
  EXPECT_EQ(expected.cx(), actual.cx());
  EXPECT_EQ(expected.cy(), actual.cy());
  EXPECT_EQ(expected.baseline(), actual.baseline());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
