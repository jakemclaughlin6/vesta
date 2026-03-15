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

#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/util.h>

#include <ceres/rotation.h>
#include <Eigen/Core>

namespace vesta_constraints
{

/**
 * @brief Stereo reprojection error cost function using world-frame pose
 * convention.
 *
 * Uses the world-frame variable convention:
 *   - position = world-frame position of the camera/body
 *   - orientation = world-from-camera rotation quaternion (R_wc)
 *
 * The projection model is:
 *
 *   p_cam = R_wc^{-1} * (X_world - p_world)
 *
 *   Left camera:  u_l = fx * p_cam[0] / p_cam[2] + cx
 *                 v_l = fy * p_cam[1] / p_cam[2] + cy
 *   Right camera: u_r = fx * (p_cam[0] - baseline) / p_cam[2] + cx
 *                 v_r = fy * p_cam[1] / p_cam[2] + cy
 *
 * This convention is compatible with IMU preintegration constraints which
 * also use world-frame position and orientation variables.
 */
class StereoReprojectionErrorCostFunctor
{
public:
  VESTA_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A The residual weighting matrix (4x4), most likely derived from
   * the square root information matrix in order (u_left, v_left, u_right,
   * v_right)
   * @param[in] b The 4D observation vector in order (u_left, v_left, u_right,
   * v_right)
   */
  StereoReprojectionErrorCostFunctor(const vesta_core::Matrix4d& A, const vesta_core::Vector4d& b);

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   *
   * @param[in] position     The camera position (3D vector: x, y, z)
   * @param[in] orientation  The camera orientation (quaternion: w, x, y, z)
   * @param[in] calibration  The stereo camera calibration (5D vector: fx, fy,
   * cx, cy, baseline)
   * @param[in] point        The 3D landmark point (3D vector: x, y, z)
   * @param[out] residual    The computed residuals (4D vector)
   */
  template <typename T>
  bool operator()(const T* const position, const T* const orientation, const T* const calibration, const T* const point,
                  T* residual) const;

private:
  vesta_core::Matrix4d A_;
  vesta_core::Vector4d b_;
};

inline StereoReprojectionErrorCostFunctor::StereoReprojectionErrorCostFunctor(const vesta_core::Matrix4d& A,
                                                                              const vesta_core::Vector4d& b)
  : A_(A), b_(b)
{
}

template <typename T>
bool StereoReprojectionErrorCostFunctor::operator()(const T* const position, const T* const orientation,
                                                    const T* const calibration, const T* const point, T* residual) const
{
  // World-frame convention: p_cam = R_wc^{-1} * (X_world - p_world)
  T diff[3];
  diff[0] = point[0] - position[0];
  diff[1] = point[1] - position[1];
  diff[2] = point[2] - position[2];

  // Rotate by inverse of orientation: q^{-1} = [w, -x, -y, -z]
  T q_inv[4];
  q_inv[0] = orientation[0];
  q_inv[1] = -orientation[1];
  q_inv[2] = -orientation[2];
  q_inv[3] = -orientation[3];

  T p[3];
  ceres::QuaternionRotatePoint(q_inv, diff, p);

  // Extract calibration parameters
  const T& fx = calibration[0];
  const T& fy = calibration[1];
  const T& cx = calibration[2];
  const T& cy = calibration[3];
  const T& baseline = calibration[4];

  // Project to left camera: u_l = fx * p[0] / p[2] + cx, v_l = fy * p[1] / p[2]
  // + cy
  T inv_z = T(1.0) / p[2];
  T u_left = fx * p[0] * inv_z + cx;
  T v_left = fy * p[1] * inv_z + cy;

  // Project to right camera: u_r = fx * (p[0] - baseline) / p[2] + cx, v_r = fy
  // * p[1] / p[2] + cy
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
