#pragma once

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

#include <vesta_core/fuse_macros.h>
#include <vesta_core/serialization.h>
#include <vesta_variables/vision/stereo_camera.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

namespace vesta_variables
{
/**
 * @brief Variable representing a stereo camera with fixed intrinsic parameters.
 *
 * This variable is held constant during optimization, meaning the camera
 * intrinsics (fx, fy, cx, cy, baseline) will not be modified by the solver.
 */
class StereoCameraFixed : public StereoCamera
{
public:
  VESTA_VARIABLE_DEFINITIONS(StereoCameraFixed);

  /**
   * @brief Default constructor
   */
  StereoCameraFixed() = default;

  /**
   * @brief Construct a fixed stereo camera variable given a camera id
   *
   * @param[in] camera_id  The id associated to a camera
   */
  explicit StereoCameraFixed(uint64_t camera_id);

  /**
   * @brief Construct a fixed stereo camera variable given a camera id and
   * intrinsic parameters
   *
   * @param[in] camera_id  The id associated to a camera
   * @param[in] fx         Focal length in x
   * @param[in] fy         Focal length in y
   * @param[in] cx         Principal point x
   * @param[in] cy         Principal point y
   * @param[in] baseline   Stereo baseline (distance between left and right
   * camera centers)
   */
  explicit StereoCameraFixed(uint64_t camera_id, double fx, double fy, double cx, double cy, double baseline);

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

  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<StereoCamera>(*this);
  }
};

}  // namespace vesta_variables

BOOST_CLASS_EXPORT_KEY(vesta_variables::StereoCameraFixed);
