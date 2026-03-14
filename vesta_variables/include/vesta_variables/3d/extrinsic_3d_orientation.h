#pragma once

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

#include <vesta_core/fuse_macros.h>
#include <vesta_core/manifold.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/common/fixed_size_variable.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include <ostream>

namespace vesta_variables
{
/**
 * @brief Variable representing the rotation component of a 3D body-to-sensor extrinsic transform
 * as a quaternion.
 *
 * This is a non-stamped variable representing the static rotation from sensor frame to body frame.
 * The internal representation is a quaternion with w as the first component (w, x, y, z), matching
 * the Ceres quaternion convention. The UUID is constant after construction and dependent on a
 * user-provided extrinsic id.
 */
class Extrinsic3DOrientation : public FixedSizeVariable<4>
{
public:
  VESTA_VARIABLE_DEFINITIONS(Extrinsic3DOrientation);

  /**
   * @brief Can be used to directly index variables in the quaternion
   */
  enum : size_t
  {
    W = 0,
    X = 1,
    Y = 2,
    Z = 3
  };

  /**
   * @brief Default constructor
   */
  Extrinsic3DOrientation() = default;

  /**
   * @brief Construct an extrinsic 3D orientation variable given an extrinsic id
   *
   * @param[in] extrinsic_id  The id associated with this extrinsic calibration
   */
  explicit Extrinsic3DOrientation(const uint64_t& extrinsic_id);

  /**
   * @brief Read-write access to the quaternion w component
   */
  double& w()
  {
    return data_[W];
  }

  /**
   * @brief Read-only access to the quaternion w component
   */
  const double& w() const
  {
    return data_[W];
  }

  /**
   * @brief Read-write access to the quaternion x component
   */
  double& x()
  {
    return data_[X];
  }

  /**
   * @brief Read-only access to the quaternion x component
   */
  const double& x() const
  {
    return data_[X];
  }

  /**
   * @brief Read-write access to the quaternion y component
   */
  double& y()
  {
    return data_[Y];
  }

  /**
   * @brief Read-only access to the quaternion y component
   */
  const double& y() const
  {
    return data_[Y];
  }

  /**
   * @brief Read-write access to the quaternion z component
   */
  double& z()
  {
    return data_[Z];
  }

  /**
   * @brief Read-only access to the quaternion z component
   */
  const double& z() const
  {
    return data_[Z];
  }

  /**
   * @brief Read-only access to the id
   */
  const uint64_t& id() const
  {
    return id_;
  }

  /**
   * @brief Print a human-readable description of the variable to the provided
   * stream.
   *
   * @param[out] stream The stream to write to. Defaults to stdout.
   */
  void print(std::ostream& stream = std::cout) const override;

  /**
   * @brief Returns the number of elements of the tangent space.
   *
   * While a quaternion has 4 parameters, a 3D rotation only has 3 degrees of
   * freedom. Hence, the tangent space is only size 3.
   */
  size_t tangentSize() const override
  {
    return 3u;
  }

  /**
   * @brief Provides a Ceres manifold for the quaternion
   *
   * @return A pointer to a manifold object that indicates how to "add"
   * increments to the quaternion
   */
  vesta_core::Manifold* manifold() const override;

protected:
  /**
   * @brief Construct an extrinsic 3D orientation variable given a UUID and extrinsic id
   *
   * @param[in] uuid          The UUID for this variable
   * @param[in] extrinsic_id  The id associated with this extrinsic calibration
   */
  Extrinsic3DOrientation(const vesta_core::UUID& uuid, const uint64_t& extrinsic_id);

private:
  // Allow Boost Serialization access to private methods
  friend class boost::serialization::access;
  uint64_t id_{ 0 };

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
    archive& boost::serialization::base_object<FixedSizeVariable<SIZE>>(*this);
    archive & id_;
  }
};

}  // namespace vesta_variables

BOOST_CLASS_EXPORT_KEY(vesta_variables::Extrinsic3DOrientation);
