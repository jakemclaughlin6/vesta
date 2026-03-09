#pragma once

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

#include <vesta_core/fuse_macros.h>
#include <vesta_core/manifold.h>
#include <vesta_core/serialization.h>
#include <vesta_core/type_name.h>
#include <vesta_core/uuid.h>

#include <boost/serialization/access.hpp>

#include <iostream>
#include <limits>
#include <ostream>
#include <string>

/**
 * @brief Implementation of the clone() member function for derived classes
 *
 * Usage:
 * @code{.cpp}
 * class Derived : public Variable
 * {
 * public:
 *   VESTA_VARIABLE_CLONE_DEFINITION(Derived);
 *   // The rest of the derived variable implementation
 * }
 * @endcode
 */
#define VESTA_VARIABLE_CLONE_DEFINITION(...)                                                                           \
  vesta_core::Variable::UniquePtr clone() const override                                                               \
  {                                                                                                                    \
    return __VA_ARGS__::make_unique(*this);                                                                            \
  }

/**
 * @brief Implementation of the serialize() and deserialize() member functions
 * for derived classes
 *
 * Usage:
 * @code{.cpp}
 * class Derived : public Variable
 * {
 * public:
 *   VESTA_VARIABLE_SERIALIZE_DEFINITION(Derived);
 *   // The rest of the derived variable implementation
 * }
 * @endcode
 */
#define VESTA_VARIABLE_SERIALIZE_DEFINITION(...)                                                                       \
  void serialize(vesta_core::BinaryOutputArchive& archive) const override                                              \
  {                                                                                                                    \
    archive << *this;                                                                                                  \
  } /* NOLINT */                                                                                                       \
  void serialize(vesta_core::TextOutputArchive& archive) const override                                                \
  {                                                                                                                    \
    archive << *this;                                                                                                  \
  } /* NOLINT */                                                                                                       \
  void deserialize(vesta_core::BinaryInputArchive& archive) override                                                   \
  {                                                                                                                    \
    archive >> *this;                                                                                                  \
  } /* NOLINT */                                                                                                       \
  void deserialize(vesta_core::TextInputArchive& archive) override                                                     \
  {                                                                                                                    \
    archive >> *this;                                                                                                  \
  }

/**
 * @brief Implements the type() member function using the suggested
 * implementation
 *
 * Also creates a static detail::type() function that may be used without an
 * object instance
 *
 * Usage:
 * @code{.cpp}
 * class Derived : public Variable
 * {
 * public:
 *   VESTA_VARIABLE_TYPE_DEFINITION(Derived);
 *   // The rest of the derived variable implementation
 * }
 * @endcode
 */
#define VESTA_VARIABLE_TYPE_DEFINITION(...)                                                                            \
  struct detail                                                                                                        \
  {                                                                                                                    \
    static std::string type()                                                                                          \
    {                                                                                                                  \
      return vesta_core::typeName<__VA_ARGS__>();                                                                      \
    } /* NOLINT */                                                                                                     \
  }; /* NOLINT */                                                                                                      \
  std::string type() const override                                                                                    \
  {                                                                                                                    \
    return detail::type();                                                                                             \
  }

/**
 * @brief Convenience function that creates the required pointer aliases,
 * clone() method, and type() method
 *
 * Usage:
 * @code{.cpp}
 * class Derived : public Variable
 * {
 * public:
 *   VESTA_VARIABLE_DEFINITIONS(Derived);
 *   // The rest of the derived variable implementation
 * }
 * @endcode
 */
#define VESTA_VARIABLE_DEFINITIONS(...)                                                                                \
  VESTA_SMART_PTR_DEFINITIONS(__VA_ARGS__)                                                                             \
  VESTA_VARIABLE_TYPE_DEFINITION(__VA_ARGS__)                                                                          \
  VESTA_VARIABLE_CLONE_DEFINITION(__VA_ARGS__)                                                                         \
  VESTA_VARIABLE_SERIALIZE_DEFINITION(__VA_ARGS__)

/**
 * @brief Convenience function that creates the required pointer aliases,
 * clone() method, and type() method for derived Variable classes that have
 * fixed-sized Eigen member objects.
 *
 * Usage:
 * @code{.cpp}
 * class Derived : public Variable
 * {
 * public:
 *   VESTA_VARIABLE_DEFINITIONS_WTIH_EIGEN(Derived);
 *   // The rest of the derived variable implementation
 * }
 * @endcode
 */
#define VESTA_VARIABLE_DEFINITIONS_WITH_EIGEN(...)                                                                     \
  VESTA_SMART_PTR_DEFINITIONS_WITH_EIGEN(__VA_ARGS__)                                                                  \
  VESTA_VARIABLE_TYPE_DEFINITION(__VA_ARGS__)                                                                          \
  VESTA_VARIABLE_CLONE_DEFINITION(__VA_ARGS__)                                                                         \
  VESTA_VARIABLE_SERIALIZE_DEFINITION(__VA_ARGS__)

namespace vesta_core
{

/**
 * @brief The Variable interface definition.
 *
 * A Variable defines some semantically meaningful group of one or more
 * individual scale values. Each variable is treated as a block by the
 * optimization engine, as the values of all of its dimensions are likely to be
 * involved in the same constraints. Some common examples of variable groupings
 * are a 2D point (x, y), 3D point (x, y, z), or camera calibration parameters
 * (fx, fy, cx, cy).
 *
 * To support the Ceres optimization engine, the Variable must hold the scalar
 * values of each dimension in a _contiguous_ memory space, and must provide
 * access to that memory location via the Variable::data() methods.
 *
 * Some Variables may require special update rules, either because they are
 * over-parameterized, as is the case with 3D rotations represented as
 * quaternions, or because the update of the individual dimensions exhibit some
 * nonlinear properties, as is the case with rotations in general (e.g. 2D
 * rotations have a discontinuity around &pi;). To support these situations,
 * Ceres uses an optional "manifold". See the Ceres documentation for more
 * details. http://ceres-solver.org/nnls_modeling.html#manifold
 */
class Variable
{
public:
  VESTA_SMART_PTR_ALIASES_ONLY(Variable);

  /**
   * @brief Default constructor
   */
  Variable() = default;

  /**
   * @brief Constructor
   *
   * The implemented UUID generation should be deterministic such that a
   * variable with the same metadata will always return the same UUID. Identical
   * UUIDs produced by sensors will be treated as the same variable by the
   * optimizer, and different UUIDs will be treated as different variables. So,
   * two derived variables representing robot poses with the same timestamp but
   * different UUIDs will incorrectly be treated as different variables, and two
   * robot poses with different timestamps but the same UUID will be incorrectly
   * treated as the same variable.
   *
   * One method of producing UUIDs that adhere to this requirement is to use the
   * boost::uuid::name_generator() function. The type() string can be used to
   * generate a UUID namespace for all variables of a given derived type, and
   * the variable metadata of consequence can be converted into a
   * carefully-formatted string or byte array and provided to the generator to
   * create the UUID for a specific variable instance.
   *
   * @param[in] uuid The unique ID number for this variable
   */
  explicit Variable(const UUID& uuid);

  /**
   * @brief Destructor
   */
  virtual ~Variable() = default;

  /**
   * @brief Returns a UUID for this variable.
   */
  const UUID& uuid() const
  {
    return uuid_;
  }

  /**
   * @brief Returns a unique name for this variable type.
   *
   * The variable type string must be unique for each class. As such, the
   * fully-qualified class name is an excellent choice for the type string.
   *
   * The suggested implementation for all derived classes is:
   * @code{.cpp}
   * return vesta_core::typeName<Derived>();
   * @endcode
   *
   * To make this easy to implement in all derived classes, the
   * VESTA_VARIABLE_TYPE_DEFINITION() and VESTA_VARIABLE_DEFINITIONS() macro
   * functions have been provided.
   */
  virtual std::string type() const = 0;

  /**
   * @brief Returns the number of elements of this variable.
   *
   * In most cases, this will be the number of degrees of freedom this variable
   * represents. For example, a 2D pose has an x, y, and theta value, so the
   * size will be 3. A notable exception is a 3D rotation represented as a
   * quaternion. It only has 3 degrees of freedom, but it is represented as four
   * elements, (w, x, y, z), so it's size will be 4.
   */
  virtual size_t size() const = 0;

  /**
   * @brief Returns the number of elements of the tangent space.
   *
   * If you override the \p manifold() method, it is good practice to also
   * override the \p tangentSize() method. By default, the \p size() method is
   * used for \p tangentSize() as well.
   */
  virtual size_t tangentSize() const
  {
    return size();
  }

  /**
   * @brief Read-only access to the variable data
   *
   * The data elements must be contiguous (such as a C-style array double[3] or
   * std::vector<double>), and it must contain at least Variable::size()
   * elements. Only Variable::size() elements will be accessed externally. This
   * interface is provided for integration with Ceres, which uses raw pointers.
   */
  virtual const double* data() const = 0;

  /**
   * @brief Read-write access to the variable data
   *
   * The data elements must be contiguous (such as a C-style array double[3] or
   * std::vector<double>), and it must contain at least Variable::size()
   * elements. Only Variable::size() elements will be accessed externally. This
   * interface is provided for integration with Ceres, which uses raw pointers.
   */
  virtual double* data() = 0;

  /**
   * @brief Print a human-readable description of the variable to the provided
   * stream.
   *
   * @param[out] stream The stream to write to. Defaults to stdout.
   */
  virtual void print(std::ostream& stream = std::cout) const = 0;

  /**
   * @brief Perform a deep copy of the Variable and return a unique pointer to
   * the copy
   *
   * Unique pointers can be implicitly upgraded to shared pointers if needed.
   *
   * This can/should be implemented as follows in all derived classes:
   * @code{.cpp}
   * return Derived::make_unique(*this);
   * @endcode
   *
   * To make this easy to implement in all derived classes, the
   * VESTA_VARIABLE_CLONE_DEFINITION() and VESTA_VARIABLE_DEFINITIONS() macros
   * functions have been provided.
   *
   * @return A unique pointer to a new instance of the most-derived Variable
   */
  virtual Variable::UniquePtr clone() const = 0;

  /**
   * @brief Create a new Ceres manifold object to apply to updates of this
   * variable
   *
   * If a manifold is not needed, a null pointer should be returned. If a
   * manifold is needed, remember to also override the \p tangentSize() method
   * to return the appropriate tangent space size.
   *
   * The Ceres interface requires a raw pointer. Ceres will take ownership of
   * the pointer and promises to properly delete the manifold when it is done.
   * Additionally, vesta promises that the Variable object will outlive any
   * generated manifold (i.e. the Ceres objects will be destroyed before the
   * Variable objects). This guarantee may allow optimizations for the creation
   * of the manifold objects.
   *
   * @return A base pointer to an instance of a derived Manifold
   */
  virtual vesta_core::Manifold* manifold() const
  {
    return nullptr;
  }

  /**
   * @brief Specifies the lower bound value of each variable dimension
   *
   * Defaults to -max.
   *
   * @param[in] index The variable dimension of interest
   * @return The lower bound for the requested variable dimension
   */
  virtual double lowerBound(size_t index) const
  {
    return std::numeric_limits<double>::lowest();
  }

  /**
   * @brief Specifies the upper bound value of each variable dimension
   *
   * Defaults to +max.
   *
   * @param[in] index The variable dimension of interest
   * @return The upper bound for the requested variable dimension
   */
  virtual double upperBound(size_t index) const
  {
    return std::numeric_limits<double>::max();
  }

  /**
   * @brief Specifies if the value of the variable should not be changed during
   * optimization
   */
  virtual bool holdConstant() const
  {
    return false;
  }

  /**
   * @brief Returns the Schur elimination group for this variable.
   *
   * Used to build a ceres::ParameterBlockOrdering for Schur complement-based
   * linear solvers (DENSE_SCHUR, SPARSE_SCHUR, ITERATIVE_SCHUR). Variables in
   * group 0 are eliminated first (e.g. 3D landmarks in bundle adjustment),
   * while variables in group 1 are kept in the reduced camera system (e.g.
   * camera poses and intrinsics).
   *
   * @return 0 for variables to eliminate first (landmarks), 1 for variables to
   * keep, or -1 (default) for unclassified variables which are placed in
   * group 1.
   */
  virtual int schurGroup() const
  {
    return -1;
  }

  /**
   * @brief Serialize this Variable into the provided binary archive
   *
   * This can/should be implemented as follows in all derived classes:
   * @code{.cpp}
   * archive << *this;
   * @endcode
   *
   * @param[out] archive - The archive to serialize this variable into
   */
  virtual void serialize(vesta_core::BinaryOutputArchive& /* archive */) const = 0;

  /**
   * @brief Serialize this Variable into the provided text archive
   *
   * This can/should be implemented as follows in all derived classes:
   * @code{.cpp}
   * archive << *this;
   * @endcode
   *
   * @param[out] archive - The archive to serialize this variable into
   */
  virtual void serialize(vesta_core::TextOutputArchive& /* archive */) const = 0;

  /**
   * @brief Deserialize data from the provided binary archive into this Variable
   *
   * This can/should be implemented as follows in all derived classes:
   * @code{.cpp}
   * archive >> *this;
   * @endcode
   *
   * @param[in] archive - The archive holding serialized Variable data
   */
  virtual void deserialize(vesta_core::BinaryInputArchive& /* archive */) = 0;

  /**
   * @brief Deserialize data from the provided text archive into this Variable
   *
   * This can/should be implemented as follows in all derived classes:
   * @code{.cpp}
   * archive >> *this;
   * @endcode
   *
   * @param[in] archive - The archive holding serialized Variable data
   */
  virtual void deserialize(vesta_core::TextInputArchive& /* archive */) = 0;

private:
  vesta_core::UUID uuid_;  //!< The unique ID number for this variable

  // Allow Boost Serialization access to private methods
  friend class boost::serialization::access;

  /**
   * @brief The Boost Serialize method that serializes all of the data members
   * in to/out of the archive
   *
   * This method, or a combination of save() and load() methods, must be
   * implemented by all derived classes. See documentation on Boost
   * Serialization for information on how to implement the serialize() method.
   * https://www.boost.org/doc/libs/1_70_0/libs/serialization/doc/
   *
   * @param[in/out] archive - The archive object that holds the serialized class
   * members
   * @param[in] version - The version of the archive being read/written.
   * Generally unused.
   */
  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive & uuid_;
  }
};

/**
 * Stream operator implementation used for all derived Variable classes.
 */
std::ostream& operator<<(std::ostream& stream, const Variable& variable);

}  // namespace vesta_core
