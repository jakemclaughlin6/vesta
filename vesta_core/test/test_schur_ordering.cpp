/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026, Locus Robotics
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
#include <vesta_core/fuse_macros.h>
#include <vesta_core/schur_ordering.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_core/variable.h>
#include <vesta_graphs/hash_graph.h>

#include <ceres/ordered_groups.h>
#include <gtest/gtest.h>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include <memory>

/**
 * @brief Mock variable with default schurGroup() == -1 (acts like a
 * camera/pose)
 */
class DefaultGroupVariable : public vesta_core::Variable
{
public:
  VESTA_VARIABLE_DEFINITIONS(DefaultGroupVariable)

  DefaultGroupVariable() : vesta_core::Variable(vesta_core::uuid::generate()), data_(0.0)
  {
  }

  size_t size() const override
  {
    return 1;
  }
  const double* data() const override
  {
    return &data_;
  }
  double* data() override
  {
    return &data_;
  }
  void print(std::ostream& /*stream = std::cout*/) const override
  {
  }

private:
  double data_;

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Variable>(*this);
    archive & data_;
  }
};

BOOST_CLASS_EXPORT(DefaultGroupVariable)

/**
 * @brief Mock variable with schurGroup() == 0 (acts like a landmark, eliminated
 * first)
 */
class LandmarkVariable : public vesta_core::Variable
{
public:
  VESTA_VARIABLE_DEFINITIONS(LandmarkVariable)

  LandmarkVariable() : vesta_core::Variable(vesta_core::uuid::generate()), data_(0.0)
  {
  }

  size_t size() const override
  {
    return 1;
  }
  const double* data() const override
  {
    return &data_;
  }
  double* data() override
  {
    return &data_;
  }
  void print(std::ostream& /*stream = std::cout*/) const override
  {
  }

  int schurGroup() const override
  {
    return 0;
  }

private:
  double data_;

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Variable>(*this);
    archive & data_;
  }
};

BOOST_CLASS_EXPORT(LandmarkVariable)

/**
 * @brief Mock variable with explicit schurGroup() == 1 (acts like a
 * camera/pose, kept in reduced system)
 */
class ExplicitCameraVariable : public vesta_core::Variable
{
public:
  VESTA_VARIABLE_DEFINITIONS(ExplicitCameraVariable)

  ExplicitCameraVariable() : vesta_core::Variable(vesta_core::uuid::generate()), data_(0.0)
  {
  }

  size_t size() const override
  {
    return 1;
  }
  const double* data() const override
  {
    return &data_;
  }
  double* data() override
  {
    return &data_;
  }
  void print(std::ostream& /*stream = std::cout*/) const override
  {
  }

  int schurGroup() const override
  {
    return 1;
  }

private:
  double data_;

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Variable>(*this);
    archive & data_;
  }
};

BOOST_CLASS_EXPORT(ExplicitCameraVariable)

// Test: buildSchurOrdering on an empty graph returns nullptr
TEST(SchurOrdering, EmptyGraph)
{
  vesta_graphs::HashGraph graph;
  auto ordering = vesta_core::buildSchurOrdering(graph);
  EXPECT_EQ(nullptr, ordering);
}

// Test: graph with only default-group variables (schurGroup == -1) returns
// nullptr
TEST(SchurOrdering, NoLandmarks)
{
  vesta_graphs::HashGraph graph;

  auto var1 = DefaultGroupVariable::make_shared();
  auto var2 = DefaultGroupVariable::make_shared();

  graph.addVariable(var1);
  graph.addVariable(var2);

  auto ordering = vesta_core::buildSchurOrdering(graph);
  EXPECT_EQ(nullptr, ordering);
}

// Test: graph with only group-0 (landmark) variables returns valid ordering
// with all in group 0
TEST(SchurOrdering, OnlyLandmarks)
{
  vesta_graphs::HashGraph graph;

  auto lm1 = LandmarkVariable::make_shared();
  auto lm2 = LandmarkVariable::make_shared();
  auto lm3 = LandmarkVariable::make_shared();

  graph.addVariable(lm1);
  graph.addVariable(lm2);
  graph.addVariable(lm3);

  auto ordering = vesta_core::buildSchurOrdering(graph);
  ASSERT_NE(nullptr, ordering);

  // All landmarks should be in group 0
  EXPECT_EQ(0, ordering->GroupId(lm1->data()));
  EXPECT_EQ(0, ordering->GroupId(lm2->data()));
  EXPECT_EQ(0, ordering->GroupId(lm3->data()));

  // Verify the total number of elements
  EXPECT_EQ(3, ordering->NumElements());
}

// Test: graph with both landmarks and poses returns valid ordering with correct
// group assignments
TEST(SchurOrdering, MixedVariables)
{
  vesta_graphs::HashGraph graph;

  auto lm1 = LandmarkVariable::make_shared();
  auto lm2 = LandmarkVariable::make_shared();
  auto cam1 = DefaultGroupVariable::make_shared();
  auto cam2 = DefaultGroupVariable::make_shared();

  graph.addVariable(lm1);
  graph.addVariable(lm2);
  graph.addVariable(cam1);
  graph.addVariable(cam2);

  auto ordering = vesta_core::buildSchurOrdering(graph);
  ASSERT_NE(nullptr, ordering);

  // Landmarks should be in group 0 (eliminated first)
  EXPECT_EQ(0, ordering->GroupId(lm1->data()));
  EXPECT_EQ(0, ordering->GroupId(lm2->data()));

  // Camera/pose variables (default group -1) should be in group 1 (kept)
  EXPECT_EQ(1, ordering->GroupId(cam1->data()));
  EXPECT_EQ(1, ordering->GroupId(cam2->data()));

  // Verify the total number of elements
  EXPECT_EQ(4, ordering->NumElements());
}

// Test: variables with explicit schurGroup()==1 behave same as default -1
// (placed in group 1)
TEST(SchurOrdering, ExplicitGroup1)
{
  vesta_graphs::HashGraph graph;

  auto lm1 = LandmarkVariable::make_shared();
  auto cam_default = DefaultGroupVariable::make_shared();
  auto cam_explicit = ExplicitCameraVariable::make_shared();

  graph.addVariable(lm1);
  graph.addVariable(cam_default);
  graph.addVariable(cam_explicit);

  auto ordering = vesta_core::buildSchurOrdering(graph);
  ASSERT_NE(nullptr, ordering);

  // Landmark in group 0
  EXPECT_EQ(0, ordering->GroupId(lm1->data()));

  // Both default (-1) and explicit (1) camera variables should end up in group
  // 1
  EXPECT_EQ(1, ordering->GroupId(cam_default->data()));
  EXPECT_EQ(1, ordering->GroupId(cam_explicit->data()));

  EXPECT_EQ(3, ordering->NumElements());
}

// Test: verify the base Variable class returns -1 for schurGroup()
TEST(SchurOrdering, DefaultSchurGroup)
{
  DefaultGroupVariable var;
  EXPECT_EQ(-1, var.schurGroup());
}

// Test: verify LandmarkVariable returns 0 for schurGroup()
TEST(SchurOrdering, LandmarkSchurGroup)
{
  LandmarkVariable var;
  EXPECT_EQ(0, var.schurGroup());
}

// Test: verify ExplicitCameraVariable returns 1 for schurGroup()
TEST(SchurOrdering, ExplicitCameraSchurGroup)
{
  ExplicitCameraVariable var;
  EXPECT_EQ(1, var.schurGroup());
}
