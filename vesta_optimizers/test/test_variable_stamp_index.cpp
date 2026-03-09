/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2019, Locus Robotics
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
#include <vesta_core/constraint.h>
#include <vesta_core/serialization.h>
#include <vesta_core/timestamp.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>
#include <vesta_core/variable.h>
#include <vesta_optimizers/variable_stamp_index.h>
#include <vesta_variables/common/stamped.h>

#include <gtest/gtest.h>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Create a simple stamped Variable for testing
 */
class StampedVariable : public vesta_core::Variable, public vesta_variables::Stamped
{
public:
  VESTA_VARIABLE_DEFINITIONS(StampedVariable);

  explicit StampedVariable(const vesta_core::Timestamp& stamp = vesta_core::Timestamp(0, 0))
    : vesta_core::Variable(vesta_core::uuid::generate()), vesta_variables::Stamped(stamp), data_{}
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

  // Allow Boost Serialization access to private methods
  friend class boost::serialization::access;

  /**
   * @brief The Boost Serialize method that serializes all of the data members
   * in to/out of the archive
   *
   * @param[in/out] archive - The archive object that holds the serialized class
   * members
   * @param[in] version - The version of the archive being read/written.
   * Generally unused.
   */
  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Variable>(*this);
    archive& boost::serialization::base_object<vesta_variables::Stamped>(*this);
    archive & data_;
  }
};

BOOST_CLASS_EXPORT(StampedVariable);

/**
 * @brief Create a simple unstamped Variable for testing
 */
class UnstampedVariable : public vesta_core::Variable
{
public:
  VESTA_VARIABLE_DEFINITIONS(UnstampedVariable);

  UnstampedVariable() : vesta_core::Variable(vesta_core::uuid::generate()), data_{}
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

  // Allow Boost Serialization access to private methods
  friend class boost::serialization::access;

  /**
   * @brief The Boost Serialize method that serializes all of the data members
   * in to/out of the archive
   *
   * @param[in/out] archive - The archive object that holds the serialized class
   * members
   * @param[in] version - The version of the archive being read/written.
   * Generally unused.
   */
  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Variable>(*this);
    archive & data_;
  }
};

BOOST_CLASS_EXPORT(UnstampedVariable);

/**
 * @brief Create a simple Constraint for testing
 */
class GenericConstraint : public vesta_core::Constraint
{
public:
  VESTA_CONSTRAINT_DEFINITIONS(GenericConstraint);

  GenericConstraint() = default;

  GenericConstraint(const std::string& source, std::initializer_list<vesta_core::UUID> variable_uuids)
    : Constraint(source, variable_uuids)
  {
  }

  explicit GenericConstraint(const std::string& source, const vesta_core::UUID& variable1)
    : vesta_core::Constraint(source, { variable1 })
  {
  }

  GenericConstraint(const std::string& source, const vesta_core::UUID& variable1, const vesta_core::UUID& variable2)
    : vesta_core::Constraint(source, { variable1, variable2 })
  {
  }

  GenericConstraint(const std::string& source, const vesta_core::UUID& variable1, const vesta_core::UUID& variable2,
                    const vesta_core::UUID& variable3)
    : vesta_core::Constraint(source, { variable1, variable2, variable3 })
  {
  }

  void print(std::ostream& /*stream = std::cout*/) const override
  {
  }

  ceres::CostFunction* costFunction() const override
  {
    return nullptr;
  }

private:
  // Allow Boost Serialization access to private methods
  friend class boost::serialization::access;

  /**
   * @brief The Boost Serialize method that serializes all of the data members
   * in to/out of the archive
   *
   * @param[in/out] archive - The archive object that holds the serialized class
   * members
   * @param[in] version - The version of the archive being read/written.
   * Generally unused.
   */
  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Constraint>(*this);
  }
};

BOOST_CLASS_EXPORT(GenericConstraint);

TEST(VariableStampIndex, Size)
{
  // Create an empty index
  auto index = vesta_optimizers::VariableStampIndex();
  EXPECT_TRUE(index.empty());
  EXPECT_EQ(0u, index.size());

  // Add some variables/constraints to the index
  auto x1 = UnstampedVariable::make_shared();
  auto transaction1 = vesta_core::Transaction();
  transaction1.addVariable(x1);
  index.addNewTransaction(transaction1);

  EXPECT_FALSE(index.empty());
  EXPECT_EQ(1u, index.size());

  // Remove the same variables
  auto transaction2 = vesta_core::Transaction();
  transaction2.removeVariable(x1->uuid());
  index.addNewTransaction(transaction2);

  EXPECT_TRUE(index.empty());
  EXPECT_EQ(0u, index.size());
}

TEST(VariableStampIndex, CurrentStamp)
{
  // Create an empty index
  auto index = vesta_optimizers::VariableStampIndex();

  // Verify the current stamp is 0
  EXPECT_EQ(vesta_core::Timestamp(0, 0), index.currentStamp());

  // Add an unstamped variable
  auto x1 = UnstampedVariable::make_shared();
  auto transaction1 = vesta_core::Transaction();
  transaction1.addVariable(x1);
  index.addNewTransaction(transaction1);

  // Verify the current stamp is still 0
  EXPECT_EQ(vesta_core::Timestamp(0, 0), index.currentStamp());

  // Add a stamped variable
  auto x2 = StampedVariable::make_shared(vesta_core::Timestamp(1, 0));
  auto transaction2 = vesta_core::Transaction();
  transaction2.addVariable(x2);
  index.addNewTransaction(transaction2);

  // Verify the current stamp is now Time(1, 0)
  EXPECT_EQ(vesta_core::Timestamp(1, 0), index.currentStamp());
}

TEST(VariableStampIndex, Query)
{
  // Create an empty index
  auto index = vesta_optimizers::VariableStampIndex();

  // Add some variables and constraints
  auto x1 = StampedVariable::make_shared(vesta_core::Timestamp(1, 0));
  auto x2 = StampedVariable::make_shared(vesta_core::Timestamp(2, 0));
  auto x3 = StampedVariable::make_shared(vesta_core::Timestamp(3, 0));
  auto l1 = UnstampedVariable::make_shared();
  auto l2 = UnstampedVariable::make_shared();

  auto c1 = GenericConstraint::make_shared("test", x1->uuid(), x2->uuid());
  auto c2 = GenericConstraint::make_shared("test", x2->uuid(), x3->uuid());
  auto c3 = GenericConstraint::make_shared("test", x1->uuid(), l1->uuid());
  auto c4 = GenericConstraint::make_shared("test", x2->uuid(), l1->uuid());
  auto c5 = GenericConstraint::make_shared("test", x3->uuid(), l2->uuid());

  auto transaction = vesta_core::Transaction();
  transaction.addVariable(x1);
  transaction.addVariable(x2);
  transaction.addVariable(x3);
  transaction.addVariable(l1);
  transaction.addVariable(l2);
  transaction.addConstraint(c1);
  transaction.addConstraint(c2);
  transaction.addConstraint(c3);
  transaction.addConstraint(c4);
  transaction.addConstraint(c5);
  index.addNewTransaction(transaction);

  auto expected1 = std::vector<vesta_core::UUID>{};
  auto actual1 = std::vector<vesta_core::UUID>();
  index.query(vesta_core::Timestamp(1, 500000), std::back_inserter(actual1));
  EXPECT_EQ(expected1, actual1);

  auto expected2 = std::vector<vesta_core::UUID>{ x1->uuid(), l1->uuid() };
  std::sort(expected2.begin(), expected2.end());
  auto actual2 = std::vector<vesta_core::UUID>();
  index.query(vesta_core::Timestamp(2, 500000), std::back_inserter(actual2));
  std::sort(actual2.begin(), actual2.end());
  EXPECT_EQ(expected2, actual2);
}

TEST(VariableStampIndex, MarginalTransaction)
{
  // Create an empty index
  auto index = vesta_optimizers::VariableStampIndex();

  // Add some variables and constraints
  auto x1 = StampedVariable::make_shared(vesta_core::Timestamp(1, 0));
  auto x2 = StampedVariable::make_shared(vesta_core::Timestamp(2, 0));
  auto x3 = StampedVariable::make_shared(vesta_core::Timestamp(3, 0));
  auto l1 = UnstampedVariable::make_shared();
  auto l2 = UnstampedVariable::make_shared();

  auto c1 = GenericConstraint::make_shared("test", x1->uuid(), x2->uuid());
  auto c2 = GenericConstraint::make_shared("test", x2->uuid(), x3->uuid());
  auto c3 = GenericConstraint::make_shared("test", x1->uuid(), l1->uuid());
  auto c4 = GenericConstraint::make_shared("test", x2->uuid(), l1->uuid());
  auto c5 = GenericConstraint::make_shared("test", x3->uuid(), l2->uuid());

  auto transaction = vesta_core::Transaction();
  transaction.addVariable(x1);
  transaction.addVariable(x2);
  transaction.addVariable(x3);
  transaction.addVariable(l1);
  transaction.addVariable(l2);
  transaction.addConstraint(c1);
  transaction.addConstraint(c2);
  transaction.addConstraint(c3);
  transaction.addConstraint(c4);
  transaction.addConstraint(c5);
  index.addNewTransaction(transaction);

  // Now create a fake marginal transaction. The constraint connections should
  // *not* change the unstamped variable timestamps
  auto marginal = vesta_core::Transaction();
  marginal.removeVariable(x1->uuid());
  marginal.removeConstraint(c1->uuid());
  marginal.removeConstraint(c3->uuid());
  auto m1 = GenericConstraint::make_shared("test", x3->uuid(), l1->uuid());
  marginal.addConstraint(m1);
  index.addMarginalTransaction(marginal);

  // The x1 variable should be removed
  EXPECT_EQ(4u, index.size());

  // And the marginal constraint x3->l1 should not affect future queries
  auto expected = std::vector<vesta_core::UUID>{ l1->uuid() };
  std::sort(expected.begin(), expected.end());
  auto actual = std::vector<vesta_core::UUID>();
  index.query(vesta_core::Timestamp(2, 500000), std::back_inserter(actual));
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(expected, actual);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
