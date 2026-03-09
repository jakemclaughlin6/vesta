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

#include <vesta_core/constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/stereo_camera.h>
#include <vesta_variables/vision/point_3d_landmark.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <Eigen/Dense>

#include <ostream>
#include <string>
#include <vector>

namespace vesta_constraints
{

/**
 * @brief A constraint that represents a stereo observation of a 3D point.
 *
 * A stereo camera model contains 5 parameters (fx, fy, cx, cy, baseline).
 * This class takes the location of the 3D landmark and applies a stereo
 * reprojection-error based constraint on the position, orientation, calibration
 * of the camera, and the 3D point that was observed.
 *
 * The observation is a 4D vector (u_left, v_left, u_right, v_right).
 */
class StereoReprojectionErrorConstraint : public vesta_core::Constraint
{
public:
  VESTA_CONSTRAINT_DEFINITIONS_WITH_EIGEN(StereoReprojectionErrorConstraint);

  /**
   * @brief Default constructor
   */
  StereoReprojectionErrorConstraint() = default;

  /**
   * @brief Create a constraint
   *
   * @param[in] source        The name of the sensor or motion model that generated this constraint
   * @param[in] position      The variable representing the position components of the camera pose
   * @param[in] orientation   The variable representing the orientation components of the camera pose
   * @param[in] calibration   The stereo calibration parameters (5x1 vector: fx, fy, cx, cy, baseline)
   * @param[in] point         The 3D landmark point variable
   * @param[in] mean          The measured stereo observation (4x1 vector: u_left, v_left, u_right, v_right)
   * @param[in] covariance    The observation covariance (4x4 matrix)
   */
  StereoReprojectionErrorConstraint(const std::string& source,
                                     const vesta_variables::Position3DStamped& position,
                                     const vesta_variables::Orientation3DStamped& orientation,
                                     const vesta_variables::StereoCamera& calibration,
                                     const vesta_variables::Point3DLandmark& point,
                                     const vesta_core::Vector4d& mean,
                                     const vesta_core::Matrix4d& covariance);

  /**
   * @brief Destructor
   */
  ~StereoReprojectionErrorConstraint() override = default;

  /**
   * @brief Read-only access to the square root information matrix.
   *
   * Order is (u_left, v_left, u_right, v_right)
   */
  const vesta_core::Matrix4d& sqrtInformation() const
  {
    return sqrt_information_;
  }

  /**
   * @brief Read-only access to the measured/prior vector of mean values.
   *
   * Order is (u_left, v_left, u_right, v_right)
   */
  const vesta_core::Vector4d& mean() const
  {
    return mean_;
  }

  /**
   * @brief Compute the measurement covariance matrix.
   *
   * Order is (u_left, v_left, u_right, v_right)
   */
  vesta_core::Matrix4d covariance() const
  {
    return (sqrt_information_.transpose() * sqrt_information_).inverse();
  }

  /**
   * @brief Print a human-readable description of the constraint to the provided stream.
   *
   * @param[out] stream The stream to write to. Defaults to stdout.
   */
  void print(std::ostream& stream = std::cout) const override;

  /**
   * @brief Construct an instance of this constraint's cost function
   *
   * The function caller will own the new cost function instance. It is the responsibility of the caller to delete
   * the cost function object when it is no longer needed. If the pointer is provided to a Ceres::Problem object, the
   * Ceres::Problem object will takes ownership of the pointer and delete it during destruction.
   *
   * @return A base pointer to an instance of a derived CostFunction.
   */
  ceres::CostFunction* costFunction() const override;

protected:
  vesta_core::Vector4d mean_;              //!< The 4D stereo observations (in pixel space)
  vesta_core::Matrix4d sqrt_information_;  //!< The square root information matrix

private:
  // Allow Boost Serialization access to private methods
  friend class boost::serialization::access;

  /**
   * @brief The Boost Serialize method that serializes all of the data members in to/out of the archive
   *
   * @param[in/out] archive - The archive object that holds the serialized class members
   * @param[in] version - The version of the archive being read/written. Generally unused.
   */
  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Constraint>(*this);
    archive& mean_;
    archive& sqrt_information_;
  }
};

}  // namespace vesta_constraints

BOOST_CLASS_EXPORT_KEY(vesta_constraints::StereoReprojectionErrorConstraint);

