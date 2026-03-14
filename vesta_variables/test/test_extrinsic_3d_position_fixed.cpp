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
#include <vesta_variables/3d/extrinsic_3d_position_fixed.h>

#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>

#include <sstream>
#include <vector>

using vesta_variables::Extrinsic3DPositionFixed;

TEST(Extrinsic3DPositionFixed, Type)
{
  Extrinsic3DPositionFixed variable(0);
  EXPECT_EQ("vesta_variables::Extrinsic3DPositionFixed", variable.type());
}

TEST(Extrinsic3DPositionFixed, UUID)
{
  // Verify two variables with the same extrinsic ids produce the same uuids
  {
    Extrinsic3DPositionFixed variable1(0);
    Extrinsic3DPositionFixed variable2(0);
    EXPECT_EQ(variable1.uuid(), variable2.uuid());
  }

  // Verify two variables with different extrinsic ids produce different uuids
  {
    Extrinsic3DPositionFixed variable1(0);
    Extrinsic3DPositionFixed variable2(1);
    EXPECT_NE(variable1.uuid(), variable2.uuid());
  }
}

TEST(Extrinsic3DPositionFixed, HoldConstant)
{
  Extrinsic3DPositionFixed variable(0);
  EXPECT_TRUE(variable.holdConstant());
}

struct CostFunctor
{
  CostFunctor()
  {
  }

  template <typename T>
  bool operator()(const T* const x, T* residual) const
  {
    residual[0] = x[0] - T(0.1);
    residual[1] = x[1] - T(0.2);
    residual[2] = x[2] - T(0.3);
    return true;
  }
};

TEST(Extrinsic3DPositionFixed, Optimization)
{
  Extrinsic3DPositionFixed variable(0);
  variable.x() = 1.5;
  variable.y() = -3.0;
  variable.z() = 0.8;

  ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 3, 3>(new CostFunctor());

  ceres::Problem problem;
  problem.AddParameterBlock(variable.data(), variable.size());
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
  EXPECT_NEAR(1.5, variable.x(), 1.0e-5);
  EXPECT_NEAR(-3.0, variable.y(), 1.0e-5);
  EXPECT_NEAR(0.8, variable.z(), 1.0e-5);
}

TEST(Extrinsic3DPositionFixed, Serialization)
{
  Extrinsic3DPositionFixed expected(0);
  expected.x() = 0.1;
  expected.y() = 0.2;
  expected.z() = 0.3;

  std::stringstream stream;
  {
    vesta_core::TextOutputArchive archive(stream);
    expected.serialize(archive);
  }

  Extrinsic3DPositionFixed actual;
  {
    vesta_core::TextInputArchive archive(stream);
    actual.deserialize(archive);
  }

  EXPECT_EQ(expected.id(), actual.id());
  EXPECT_EQ(expected.uuid(), actual.uuid());
  EXPECT_EQ(expected.x(), actual.x());
  EXPECT_EQ(expected.y(), actual.y());
  EXPECT_EQ(expected.z(), actual.z());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
