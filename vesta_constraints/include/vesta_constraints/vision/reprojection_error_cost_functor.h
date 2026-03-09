#pragma once

/*
 * Software License Agreement (BSD License)
 *
 *  Author: Oscar Mendez
 *  Created on Mon Dec 12 2023
 *
 *  Copyright (c) 2023, Locus Robotics
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

#include <Eigen/Core>
#include <ceres/rotation.h>

namespace vesta_constraints {

/**
 * @brief Reprojection error cost function using world-frame pose convention.
 *
 * Uses the world-frame variable convention:
 *   - position = world-frame position of the camera/body
 *   - orientation = world-from-camera rotation quaternion (R_wc)
 *
 * The projection model is:
 *
 *   p_cam = R_wc^{-1} * (X_world - p_world)
 *   u = fx * p_cam.x / p_cam.z + cx
 *   v = fy * p_cam.y / p_cam.z + cy
 *
 * This convention is compatible with IMU preintegration constraints which
 * also use world-frame position and orientation variables.
 *
 * The cost function is:
 *
 *   cost = || A * ([u, v] - obs) ||
 *
 * where A is the square root information matrix.
 */
class ReprojectionErrorCostFunctor {
public:
  VESTA_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A The residual weighting matrix, most likely derived from the
   *square root information matrix in order (u, v)
   * @param[in] b The 2D pose measurement or prior in order (u, v)
   *
   **/
  ReprojectionErrorCostFunctor(const vesta_core::Matrix2d &A,
                               const vesta_core::Vector2d &b);

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   */
  template <typename T>
  bool operator()(const T *const position, const T *const orientation,
                  const T *const calibration, const T *const point,
                  T *residual) const;

private:
  vesta_core::Matrix2d A_;
  vesta_core::Vector2d b_;
};

ReprojectionErrorCostFunctor::ReprojectionErrorCostFunctor(
    const vesta_core::Matrix2d &A, const vesta_core::Vector2d &b)
    : A_(A), b_(b) {}

template <typename T>
bool ReprojectionErrorCostFunctor::operator()(const T *const position,
                                              const T *const orientation,
                                              const T *const calibration,
                                              const T *const point,
                                              T *residual) const {
  // World-frame convention: p_cam = R_wc^{-1} * (X_world - p_world)
  // Compute difference in world frame
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

  // Project to camera ([u,v] = KX)
  // u = (x'fx + z'cx)/z'
  // v = (y'fy + z'cy)/z'
  T uv[2];
  uv[0] = (p[0] * calibration[0] + p[2] * calibration[2]) / p[2];
  uv[1] = (p[1] * calibration[1] + p[2] * calibration[3]) / p[2];

  // Get Residuals
  residual[0] = uv[0] - b_[0];
  residual[1] = uv[1] - b_[1];

  // Weight Residuals
  Eigen::Map<Eigen::Matrix<T, 2, 1>> residual_map(residual);
  residual_map.applyOnTheLeft(A_.template cast<T>());

  return true;
}

} // namespace vesta_constraints
