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
#include <vesta_core/serialization.h>
#include <vesta_core/timestamp.h>
#include <vesta_core/uuid.h>

#include <boost/serialization/access.hpp>

namespace vesta_variables
{

/**
 * @brief A class that provides a timestamp and device id
 *
 * This is intended to be used as secondary base class (multiple inheritance)
 * for variables that are time-varying. Some common examples include robot poses
 * or velocities. This is in contrast to variables that represent unknown but
 * fixed quantities, such as the world position of landmarks, or possibly
 * calibration values that are assumed constant.
 */
class Stamped
{
public:
  VESTA_SMART_PTR_ALIASES_ONLY(Stamped);

  /**
   * @brief Default constructor
   */
  Stamped() = default;

  /**
   * @brief Constructor
   */
  explicit Stamped(const vesta_core::Timestamp& stamp, const vesta_core::UUID& device_id = vesta_core::uuid::NIL)
    : device_id_(device_id), stamp_(stamp)
  {
  }

  /**
   * @brief Destructor
   */
  virtual ~Stamped() = default;

  /**
   * @brief Read-only access to the associated timestamp.
   */
  const vesta_core::Timestamp& stamp() const
  {
    return stamp_;
  }

  /**
   * @brief Read-only access to the associated device ID.
   */
  const vesta_core::UUID& deviceId() const
  {
    return device_id_;
  }

private:
  vesta_core::UUID device_id_;   //!< The UUID associated with this specific device or hardware
  vesta_core::Timestamp stamp_;  //!< The timestamp associated with this variable instance

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
    archive & device_id_;
    archive & stamp_;
  }
};

}  // namespace vesta_variables
