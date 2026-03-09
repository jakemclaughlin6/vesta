/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Locus Robotics
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
#include <vesta_core/timestamp.h>
#include <vesta_variables/3d/gyroscope_bias_3d_stamped.h>
#include <vesta_variables/common/stamped.h>

#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>

#include <sstream>
#include <vector>

using vesta_variables::GyroscopeBias3DStamped;

TEST(GyroscopeBias3DStamped, Type) {
  GyroscopeBias3DStamped variable(vesta_core::Timestamp(12345678, 910111213));
  EXPECT_EQ("vesta_variables::GyroscopeBias3DStamped", variable.type());
}

TEST(GyroscopeBias3DStamped, UUID) {
  // Verify two biases at the same timestamp produce the same UUID
  {
    GyroscopeBias3DStamped variable1(
        vesta_core::Timestamp(12345678, 910111213));
    GyroscopeBias3DStamped variable2(
        vesta_core::Timestamp(12345678, 910111213));
    EXPECT_EQ(variable1.uuid(), variable2.uuid());

    GyroscopeBias3DStamped variable3(vesta_core::Timestamp(12345678, 910111213),
                                     vesta_core::uuid::generate("c3po"));
    GyroscopeBias3DStamped variable4(vesta_core::Timestamp(12345678, 910111213),
                                     vesta_core::uuid::generate("c3po"));
    EXPECT_EQ(variable3.uuid(), variable4.uuid());
  }

  // Verify two biases at different timestamps produce different UUIDs
  {
    GyroscopeBias3DStamped variable1(
        vesta_core::Timestamp(12345678, 910111213));
    GyroscopeBias3DStamped variable2(
        vesta_core::Timestamp(12345678, 910111214));
    GyroscopeBias3DStamped variable3(
        vesta_core::Timestamp(12345679, 910111213));
    EXPECT_NE(variable1.uuid(), variable2.uuid());
    EXPECT_NE(variable1.uuid(), variable3.uuid());
    EXPECT_NE(variable2.uuid(), variable3.uuid());
  }

  // Verify two biases with different hardware IDs produce different UUIDs
  {
    GyroscopeBias3DStamped variable1(vesta_core::Timestamp(12345678, 910111213),
                                     vesta_core::uuid::generate("8d8"));
    GyroscopeBias3DStamped variable2(vesta_core::Timestamp(12345678, 910111213),
                                     vesta_core::uuid::generate("r4-p17"));
    EXPECT_NE(variable1.uuid(), variable2.uuid());
  }
}

TEST(GyroscopeBias3DStamped, Stamped) {
  vesta_core::Variable::SharedPtr base = GyroscopeBias3DStamped::make_shared(
      vesta_core::Timestamp(12345678, 910111213),
      vesta_core::uuid::generate("mo"));
  auto derived = std::dynamic_pointer_cast<GyroscopeBias3DStamped>(base);
  ASSERT_TRUE(static_cast<bool>(derived));
  EXPECT_EQ(vesta_core::Timestamp(12345678, 910111213), derived->stamp());
  EXPECT_EQ(vesta_core::uuid::generate("mo"), derived->deviceId());

  auto stamped = std::dynamic_pointer_cast<vesta_variables::Stamped>(base);
  ASSERT_TRUE(static_cast<bool>(stamped));
  EXPECT_EQ(vesta_core::Timestamp(12345678, 910111213), stamped->stamp());
  EXPECT_EQ(vesta_core::uuid::generate("mo"), stamped->deviceId());
}

struct CostFunctor {
  CostFunctor() {}

  template <typename T> bool operator()(const T *const x, T *residual) const {
    residual[0] = x[0] - T(3.0);
    residual[1] = x[1] + T(8.0);
    residual[2] = x[2] - T(17.0);
    return true;
  }
};

TEST(GyroscopeBias3DStamped, Optimization) {
  // Create a GyroscopeBias3DStamped
  GyroscopeBias3DStamped bias(vesta_core::Timestamp(12345678, 910111213),
                              vesta_core::uuid::generate("hal9000"));
  bias.x() = 1.5;
  bias.y() = -3.0;
  bias.z() = 14.0;

  // Create a simple a constraint
  ceres::CostFunction *cost_function =
      new ceres::AutoDiffCostFunction<CostFunctor, 3, 3>(new CostFunctor());

  // Build the problem.
  ceres::Problem problem;
  problem.AddParameterBlock(bias.data(), bias.size(), bias.manifold());
  std::vector<double *> parameter_blocks;
  parameter_blocks.push_back(bias.data());
  problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

  // Run the solver
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Check
  EXPECT_NEAR(3.0, bias.x(), 1.0e-5);
  EXPECT_NEAR(-8.0, bias.y(), 1.0e-5);
  EXPECT_NEAR(17.0, bias.z(), 1.0e-5);
}

TEST(GyroscopeBias3DStamped, Serialization) {
  // Create a GyroscopeBias3DStamped
  GyroscopeBias3DStamped expected(vesta_core::Timestamp(12345678, 910111213),
                                  vesta_core::uuid::generate("hal9000"));
  expected.x() = 1.5;
  expected.y() = -3.0;
  expected.z() = 14.0;

  // Serialize the variable into an archive
  std::stringstream stream;
  {
    vesta_core::TextOutputArchive archive(stream);
    expected.serialize(archive);
  }

  // Deserialize a new variable from that same stream
  GyroscopeBias3DStamped actual;
  {
    vesta_core::TextInputArchive archive(stream);
    actual.deserialize(archive);
  }

  // Compare
  EXPECT_EQ(expected.deviceId(), actual.deviceId());
  EXPECT_EQ(expected.stamp(), actual.stamp());
  EXPECT_EQ(expected.x(), actual.x());
  EXPECT_EQ(expected.y(), actual.y());
  EXPECT_EQ(expected.z(), actual.z());
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
