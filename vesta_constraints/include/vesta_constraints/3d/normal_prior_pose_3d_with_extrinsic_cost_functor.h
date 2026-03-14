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

#include <vesta_constraints/3d/extrinsic_pose_3d.h>
#include <vesta_constraints/3d/normal_prior_orientation_3d_cost_functor.h>
#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>

#include <ceres/rotation.h>

namespace vesta_constraints
{

/**
 * @brief Cost function for an absolute 3D pose prior with an extrinsic calibration.
 *
 * This extends NormalPriorPose3DCostFunctor to accept body-frame pose variables and an extrinsic
 * transform T_body_sensor. The extrinsic is applied to compute the sensor-frame pose before
 * computing the prior residual.
 *
 * Parameter blocks: body_position (3), body_orientation (4), ext_position (3), ext_orientation (4)
 *
 * Residuals: 6 (x, y, z, qx, qy, qz)
 */
class NormalPriorPose3DWithExtrinsicCostFunctor
{
public:
  VESTA_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Constructor
   *
   * @param[in] A The residual weighting matrix (6x6), typically the square root information matrix
   * @param[in] b The measured pose prior (7x1: x, y, z, qw, qx, qy, qz)
   */
  NormalPriorPose3DWithExtrinsicCostFunctor(const vesta_core::Matrix6d& A, const vesta_core::Vector7d& b);

  /**
   * @brief Compute the cost values/residuals
   */
  template <typename T>
  bool operator()(const T* const body_position, const T* const body_orientation, const T* const ext_position,
                  const T* const ext_orientation, T* residual) const;

private:
  vesta_core::Matrix6d A_;
  vesta_core::Vector7d b_;

  NormalPriorOrientation3DCostFunctor orientation_functor_;
};

NormalPriorPose3DWithExtrinsicCostFunctor::NormalPriorPose3DWithExtrinsicCostFunctor(const vesta_core::Matrix6d& A,
                                                                                     const vesta_core::Vector7d& b)
  : A_(A), b_(b), orientation_functor_(vesta_core::Matrix3d::Identity(), b_.tail<4>())
{
}

template <typename T>
bool NormalPriorPose3DWithExtrinsicCostFunctor::operator()(const T* const body_position,
                                                            const T* const body_orientation,
                                                            const T* const ext_position,
                                                            const T* const ext_orientation, T* residual) const
{
  // Compute sensor-frame pose from body-frame pose + extrinsic
  T sensor_position[3];
  T sensor_orientation[4];
  computeSensorPose(body_position, body_orientation, ext_position, ext_orientation, sensor_position,
                    sensor_orientation);

  // Compute the position error
  residual[0] = sensor_position[0] - T(b_(0));
  residual[1] = sensor_position[1] - T(b_(1));
  residual[2] = sensor_position[2] - T(b_(2));

  // Use the 3D orientation cost functor to compute the orientation delta
  orientation_functor_(sensor_orientation, &residual[3]);

  // Scale the residuals by the square root information matrix to account for
  // the measurement uncertainty.
  Eigen::Map<Eigen::Matrix<T, 6, 1>> residual_map(residual);
  residual_map.applyOnTheLeft(A_.template cast<T>());

  return true;
}

}  // namespace vesta_constraints
