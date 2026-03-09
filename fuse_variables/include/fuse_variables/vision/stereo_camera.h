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
#ifndef FUSE_VARIABLES_VISION_STEREO_CAMERA_H
#define FUSE_VARIABLES_VISION_STEREO_CAMERA_H

#include <fuse_core/fuse_macros.h>
#include <fuse_core/serialization.h>
#include <fuse_core/uuid.h>
#include <fuse_variables/vision/base_camera.h>
#include <fuse_variables/common/fixed_size_variable.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include <ostream>

namespace fuse_variables
{
/**
 * @brief Variable representing intrinsic parameters of a stereo camera.
 *
 * The stereo camera model extends the pinhole model with a baseline parameter
 * representing the distance between the left and right camera optical centers.
 * Parameters are (fx, fy, cx, cy, baseline).
 *
 * The UUID of this class is constant after construction and dependent on a user
 * input database id. As such, the database id cannot be altered after construction.
 */
class StereoCamera : public BaseCamera<5>
{
public:
  FUSE_VARIABLE_DEFINITIONS(StereoCamera);

  /**
   * @brief Can be used to directly index variables in the data array
   */
  enum : size_t
  {
    FX = 0,
    FY = 1,
    CX = 2,
    CY = 3,
    BASELINE = 4
  };

  /**
   * @brief Default constructor
   */
  StereoCamera() = default;

  /**
   * @brief Construct a stereo camera variable given a camera id
   *
   * @param[in] camera_id  The id associated to a camera
   */
  explicit StereoCamera(const uint64_t& camera_id);

  /**
   * @brief Construct a stereo camera variable given a uuid, camera id and intrinsic parameters
   *
   * @param[in] uuid       The UUID for this variable
   * @param[in] camera_id  The id associated to a camera
   * @param[in] fx         Focal length in x
   * @param[in] fy         Focal length in y
   * @param[in] cx         Principal point x
   * @param[in] cy         Principal point y
   * @param[in] baseline   Stereo baseline (distance between left and right camera centers)
   */
  explicit StereoCamera(const fuse_core::UUID& uuid, const uint64_t& camera_id,
                        const double& fx, const double& fy,
                        const double& cx, const double& cy,
                        const double& baseline);

  /**
   * @brief Read-write access to the fx parameter.
   */
  double& fx() { return data_[FX]; }

  /**
   * @brief Read-only access to the fx parameter.
   */
  const double& fx() const { return data_[FX]; }

  /**
   * @brief Read-write access to the fy parameter.
   */
  double& fy() { return data_[FY]; }

  /**
   * @brief Read-only access to the fy parameter.
   */
  const double& fy() const { return data_[FY]; }

  /**
   * @brief Read-write access to the cx parameter.
   */
  double& cx() { return data_[CX]; }

  /**
   * @brief Read-only access to the cx parameter.
   */
  const double& cx() const { return data_[CX]; }

  /**
   * @brief Read-write access to the cy parameter.
   */
  double& cy() { return data_[CY]; }

  /**
   * @brief Read-only access to the cy parameter.
   */
  const double& cy() const { return data_[CY]; }

  /**
   * @brief Read-write access to the baseline parameter.
   */
  double& baseline() { return data_[BASELINE]; }

  /**
   * @brief Read-only access to the baseline parameter.
   */
  const double& baseline() const { return data_[BASELINE]; }

  /**
   * @brief Print a human-readable description of the variable to the provided
   * stream.
   *
   * @param[out] stream The stream to write to. Defaults to stdout.
   */
  void print(std::ostream& stream = std::cout) const override;

protected:
  /**
   * @brief Construct a stereo camera given a uuid and camera_id
   *
   * @param[in] uuid       The UUID for this variable
   * @param[in] camera_id  The id associated to a camera_id
   */
  StereoCamera(const fuse_core::UUID& uuid, const uint64_t& camera_id);

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
    archive& boost::serialization::base_object<BaseCamera<SIZE>>(*this);
  }
};

}  // namespace fuse_variables

BOOST_CLASS_EXPORT_KEY(fuse_variables::StereoCamera);

#endif  // FUSE_VARIABLES_VISION_STEREO_CAMERA_H
