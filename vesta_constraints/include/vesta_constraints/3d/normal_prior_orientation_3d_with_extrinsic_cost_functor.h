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

#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>

#include <ceres/rotation.h>
#include <Eigen/Core>

namespace vesta_constraints
{

/**
 * @brief Cost function for an absolute 3D orientation prior with an extrinsic calibration.
 *
 * This extends NormalPriorOrientation3DCostFunctor to accept a body-frame orientation variable
 * and an extrinsic transform. The sensor orientation is computed as q_sensor = q_body * q_ext,
 * and the prior residual is computed on q_sensor.
 *
 * The ext_position parameter block is present for graph consistency but is unused in the math,
 * since translation does not affect orientation.
 *
 * Parameter blocks: body_orientation (4), ext_position (3), ext_orientation (4)
 *
 * Residuals: 3 (qx, qy, qz)
 */
class NormalPriorOrientation3DWithExtrinsicCostFunctor
{
public:
  VESTA_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A The residual weighting matrix (3x3), typically the square root information matrix
   * @param[in] b The orientation measurement or prior (4x1: w, x, y, z)
   */
  NormalPriorOrientation3DWithExtrinsicCostFunctor(const vesta_core::Matrix3d& A, const vesta_core::Vector4d& b)
    : A_(A), b_(b)
  {
  }

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   */
  template <typename T>
  bool operator()(const T* const body_orientation, const T* const /* ext_position */,
                  const T* const ext_orientation, T* residuals) const
  {
    // Compute sensor orientation: q_sensor = q_body * q_ext
    T sensor_orientation[4];
    ceres::QuaternionProduct(body_orientation, ext_orientation, sensor_orientation);

    // Compute the delta quaternion: b^-1 * q_sensor
    T observation_inverse[4] = { T(b_(0)), T(-b_(1)), T(-b_(2)), T(-b_(3)) };

    T difference[4];
    ceres::QuaternionProduct(observation_inverse, sensor_orientation, difference);
    ceres::QuaternionToAngleAxis(difference, residuals);

    // Scale the residuals by the square root information matrix to account for
    // the measurement uncertainty.
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(residuals);
    residuals_map.applyOnTheLeft(A_.template cast<T>());

    return true;
  }

private:
  vesta_core::Matrix3d A_;  //!< The residual weighting matrix
  vesta_core::Vector4d b_;  //!< The measured 3D orientation (quaternion) value
};

}  // namespace vesta_constraints
