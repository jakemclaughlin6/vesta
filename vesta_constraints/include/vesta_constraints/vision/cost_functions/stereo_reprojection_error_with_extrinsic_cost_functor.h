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
#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>

#include <ceres/rotation.h>
#include <Eigen/Core>

namespace vesta_constraints
{

/**
 * @brief Stereo reprojection error cost function with extrinsic calibration.
 *
 * Extends StereoReprojectionErrorCostFunctor to accept body-frame pose variables and an extrinsic
 * transform T_body_sensor. The extrinsic is applied to compute the sensor-frame (camera-frame)
 * pose before performing the stereo projection.
 *
 * Parameter blocks: body_position (3), body_orientation (4), calibration (5), point (3),
 *                   ext_position (3), ext_orientation (4)
 *
 * The projection model is:
 *   T_ws = T_wb * T_bs                (sensor pose in world frame)
 *   p_cam = R_ws^{-1} * (X_world - p_ws)
 *   u_l = fx * p_cam.x / p_cam.z + cx
 *   v_l = fy * p_cam.y / p_cam.z + cy
 *   u_r = fx * (p_cam.x - baseline) / p_cam.z + cx
 *   v_r = fy * p_cam.y / p_cam.z + cy
 *
 * Residuals: 4 (u_left, v_left, u_right, v_right)
 */
class StereoReprojectionErrorWithExtrinsicCostFunctor
{
public:
  VESTA_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A The residual weighting matrix (4x4)
   * @param[in] b The 4D observation (u_left, v_left, u_right, v_right)
   */
  StereoReprojectionErrorWithExtrinsicCostFunctor(const vesta_core::Matrix4d& A, const vesta_core::Vector4d& b);

  /**
   * @brief Evaluate the cost function
   */
  template <typename T>
  bool operator()(const T* const body_position, const T* const body_orientation, const T* const calibration,
                  const T* const point, const T* const ext_position, const T* const ext_orientation, T* residual) const;

private:
  vesta_core::Matrix4d A_;
  vesta_core::Vector4d b_;
};

StereoReprojectionErrorWithExtrinsicCostFunctor::StereoReprojectionErrorWithExtrinsicCostFunctor(
    const vesta_core::Matrix4d& A, const vesta_core::Vector4d& b)
  : A_(A), b_(b)
{
}

template <typename T>
bool StereoReprojectionErrorWithExtrinsicCostFunctor::operator()(const T* const body_position,
                                                                 const T* const body_orientation,
                                                                 const T* const calibration, const T* const point,
                                                                 const T* const ext_position,
                                                                 const T* const ext_orientation, T* residual) const
{
  // Compute sensor-frame pose from body-frame pose + extrinsic
  T sensor_position[3];
  T sensor_orientation[4];
  computeSensorPose(body_position, body_orientation, ext_position, ext_orientation, sensor_position,
                    sensor_orientation);

  // World-frame convention: p_cam = R_ws^{-1} * (X_world - p_ws)
  T diff[3];
  diff[0] = point[0] - sensor_position[0];
  diff[1] = point[1] - sensor_position[1];
  diff[2] = point[2] - sensor_position[2];

  // Rotate by inverse of sensor orientation: q^{-1} = [w, -x, -y, -z]
  T q_inv[4];
  q_inv[0] = sensor_orientation[0];
  q_inv[1] = -sensor_orientation[1];
  q_inv[2] = -sensor_orientation[2];
  q_inv[3] = -sensor_orientation[3];

  T p[3];
  ceres::QuaternionRotatePoint(q_inv, diff, p);

  // Extract calibration parameters
  const T& fx = calibration[0];
  const T& fy = calibration[1];
  const T& cx = calibration[2];
  const T& cy = calibration[3];
  const T& baseline = calibration[4];

  // Project to left camera
  T inv_z = T(1.0) / p[2];
  T u_left = fx * p[0] * inv_z + cx;
  T v_left = fy * p[1] * inv_z + cy;

  // Project to right camera
  T u_right = fx * (p[0] - baseline) * inv_z + cx;
  T v_right = fy * p[1] * inv_z + cy;

  // Compute residuals
  residual[0] = u_left - b_[0];
  residual[1] = v_left - b_[1];
  residual[2] = u_right - b_[2];
  residual[3] = v_right - b_[3];

  // Weight residuals
  Eigen::Map<Eigen::Matrix<T, 4, 1>> residual_map(residual);
  residual_map.applyOnTheLeft(A_.template cast<T>());

  return true;
}

}  // namespace vesta_constraints
