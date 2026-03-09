#pragma once

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
#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/manifold.h>
#include <vesta_core/serialization.h>
#include <vesta_core/variable.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <ceres/cost_function.h>

#include <algorithm>
#include <cassert>
#include <ostream>
#include <string>
#include <vector>

namespace vesta_constraints {

/**
 * @brief A constraint that represents remaining marginal information on a set
 * of variables
 *
 * The marginal constraint cost function is of the form:
 *   cost = A1 * (x1 - x1_bar) + A2 * (x2 - x2_bar) + ... + b
 * where x_bar is the linearization point of the variable taken from the
 * variable value at the time of construction, and the minus operator is
 * implemented in the variable's manifold.
 */
class MarginalConstraint : public vesta_core::Constraint {
public:
  VESTA_CONSTRAINT_DEFINITIONS(MarginalConstraint);

  /**
   * @brief Default constructor
   */
  MarginalConstraint() = default;

  /**
   * @brief Create a linear/marginal constraint
   *
   * The variable iterators and matrix iterators must be the same size. Further,
   * all A matrices and the b vector must have the same number of rows, and the
   * number of columns of each A matrix must match the \p localSize() of its
   * associated variable.
   *
   * @param[in] source         The name of the sensor or motion model that
   * generated this constraint
   * @param[in] first_variable Iterator pointing to the first involved variable
   * for this constraint
   * @param[in] last_variable  Iterator pointing to one past the last involved
   * variable for this constraint
   * @param[in] first_A        Iterator pointing to the first A matrix,
   * associated with the first variable
   * @param[in] last_A         Iterator pointing to one past the last A matrix
   * @param[in] b              The b vector of the marginal cost (of the form
   * A*(x - x_bar) + b)
   */
  template <typename VariableIterator, typename MatrixIterator>
  MarginalConstraint(const std::string &source, VariableIterator first_variable,
                     VariableIterator last_variable, MatrixIterator first_A,
                     MatrixIterator last_A, const vesta_core::VectorXd &b);

  /**
   * @brief Destructor
   */
  virtual ~MarginalConstraint() = default;

  /**
   * @brief Read-only access to the A matrices of the marginal constraint
   */
  const std::vector<vesta_core::MatrixXd> &A() const { return A_; }

  /**
   * @brief Read-only access to the b vector of the marginal constraint
   */
  const vesta_core::VectorXd &b() const { return b_; }

  /**
   * @brief Read-only access to the variable linearization points, x_bar
   */
  const std::vector<vesta_core::VectorXd> &x_bar() const { return x_bar_; }

  /**
   * @brief Read-only access to the variable manifolds
   */
  const std::vector<vesta_core::Manifold::SharedPtr> &manifolds() const {
    return manifolds_;
  }

  /**
   * @brief Print a human-readable description of the constraint to the provided
   * stream.
   *
   * @param[out] stream The stream to write to. Defaults to stdout.
   */
  void print(std::ostream &stream = std::cout) const override;

  /**
   * @brief Construct an instance of this constraint's cost function
   *
   * The function caller will own the new cost function instance. It is the
   * responsibility of the caller to delete the cost function object when it is
   * no longer needed. If the pointer is provided to a Ceres::Problem object,
   * the Ceres::Problem object will take ownership of the pointer and delete it
   * during destruction.
   *
   * @return A base pointer to an instance of a derived CostFunction.
   */
  ceres::CostFunction *costFunction() const override;

protected:
  std::vector<vesta_core::MatrixXd>
      A_;                  //!< The A matrices of the marginal constraint
  vesta_core::VectorXd b_; //!< The b vector of the marginal constraint
  std::vector<vesta_core::Manifold::SharedPtr> manifolds_; //!< The manifolds
  std::vector<vesta_core::VectorXd>
      x_bar_; //!< The linearization point of each involved variable

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
  void serialize(Archive &archive, const unsigned int /* version */) {
    archive &boost::serialization::base_object<vesta_core::Constraint>(*this);
    archive & A_;
    archive & b_;
    archive & manifolds_;
    archive & x_bar_;
  }
};

namespace detail {

/**
 * @brief Return the UUID of the provided variable
 */
inline const vesta_core::UUID getUuid(const vesta_core::Variable &variable) {
  return variable.uuid();
}

/**
 * @brief Return the current value of the provided variable
 */
inline const vesta_core::VectorXd
getCurrentValue(const vesta_core::Variable &variable) {
  return Eigen::Map<const vesta_core::VectorXd>(variable.data(),
                                                variable.size());
}

/**
 * @brief Return the manifold of the provided variable
 */
inline vesta_core::Manifold::SharedPtr const
getManifold(const vesta_core::Variable &variable) {
  return vesta_core::Manifold::SharedPtr(variable.manifold());
}

// Simple transform iterator to avoid boost::make_transform_iterator
template <typename Iterator, typename Func> class TransformIterator {
public:
  using value_type =
      std::invoke_result_t<Func,
                           typename std::iterator_traits<Iterator>::reference>;
  using reference = value_type;
  using pointer = void;
  using difference_type =
      typename std::iterator_traits<Iterator>::difference_type;
  using iterator_category = std::forward_iterator_tag;

  TransformIterator(Iterator it, Func func) : it_(it), func_(func) {}
  reference operator*() const { return func_(*it_); }
  TransformIterator &operator++() {
    ++it_;
    return *this;
  }
  TransformIterator operator++(int) {
    auto tmp = *this;
    ++it_;
    return tmp;
  }
  bool operator==(const TransformIterator &other) const {
    return it_ == other.it_;
  }
  bool operator!=(const TransformIterator &other) const {
    return it_ != other.it_;
  }

private:
  Iterator it_;
  Func func_;
};

template <typename Iterator, typename Func>
TransformIterator<Iterator, Func> makeTransformIterator(Iterator it,
                                                        Func func) {
  return TransformIterator<Iterator, Func>(it, func);
}

} // namespace detail

template <typename VariableIterator, typename MatrixIterator>
MarginalConstraint::MarginalConstraint(const std::string &source,
                                       VariableIterator first_variable,
                                       VariableIterator last_variable,
                                       MatrixIterator first_A,
                                       MatrixIterator last_A,
                                       const vesta_core::VectorXd &b)
    : Constraint(source,
                 vesta_constraints::detail::makeTransformIterator(
                     first_variable, &vesta_constraints::detail::getUuid),
                 vesta_constraints::detail::makeTransformIterator(
                     last_variable, &vesta_constraints::detail::getUuid)),
      A_(first_A, last_A), b_(b) {
  // Build manifold and x_bar vectors from the variable range
  for (auto it = first_variable; it != last_variable; ++it) {
    manifolds_.push_back(vesta_constraints::detail::getManifold(*it));
    x_bar_.push_back(vesta_constraints::detail::getCurrentValue(*it));
  }

  assert(!A_.empty());
  assert(A_.size() == x_bar_.size());
  assert(A_.size() == manifolds_.size());
  assert(b_.rows() > 0);
  assert(std::all_of(A_.begin(), A_.end(), [this](const auto &A) {
    return A.rows() == this->b_.rows();
  })); // NOLINT
  // Verify A matrix columns match variable tangent sizes
  {
    auto var_it = first_variable;
    for (size_t i = 0; i < A_.size() && var_it != last_variable;
         ++i, ++var_it) {
      assert(static_cast<size_t>(A_[i].cols()) == (*var_it).tangentSize());
    }
  }
}

} // namespace vesta_constraints

BOOST_CLASS_EXPORT_KEY(vesta_constraints::MarginalConstraint);
