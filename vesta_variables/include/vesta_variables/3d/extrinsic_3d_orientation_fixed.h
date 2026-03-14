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
#include <vesta_core/serialization.h>
#include <vesta_variables/3d/extrinsic_3d_orientation.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

namespace vesta_variables
{
/**
 * @brief Variable representing the rotation component of a 3D body-to-sensor extrinsic transform
 * that is held constant during optimization.
 *
 * This is appropriate when the extrinsic rotation is known or was previously estimated to
 * sufficient accuracy. The UUID is constant after construction and dependent on a user-provided
 * extrinsic id.
 */
class Extrinsic3DOrientationFixed : public Extrinsic3DOrientation
{
public:
  VESTA_VARIABLE_DEFINITIONS(Extrinsic3DOrientationFixed);

  /**
   * @brief Default constructor
   */
  Extrinsic3DOrientationFixed() = default;

  /**
   * @brief Construct an extrinsic 3D orientation fixed variable given an extrinsic id
   *
   * @param[in] extrinsic_id  The id associated with this extrinsic calibration
   */
  explicit Extrinsic3DOrientationFixed(const uint64_t& extrinsic_id);

  /**
   * @brief Specifies if the value of the variable should not be changed during
   * optimization
   */
  bool holdConstant() const override
  {
    return true;
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
    archive& boost::serialization::base_object<Extrinsic3DOrientation>(*this);
  }
};

}  // namespace vesta_variables

BOOST_CLASS_EXPORT_KEY(vesta_variables::Extrinsic3DOrientationFixed);
