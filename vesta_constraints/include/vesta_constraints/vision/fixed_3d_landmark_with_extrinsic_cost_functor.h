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
#include <vesta_core/util.h>

#include <ceres/rotation.h>
#include <Eigen/Core>

namespace vesta_constraints
{

/**
 * @brief Fixed 3D landmark cost function with extrinsic calibration.
 *
 * Extends Fixed3DLandmarkCostFunctor to accept body-frame pose variables and an extrinsic
 * transform T_body_sensor. The extrinsic is composed with the body pose to compute the
 * sensor-frame (camera-frame) pose before constructing the camera matrix Ta = [R_q | p].
 *
 * Parameter blocks: body_position (3), body_orientation (4), calibration (4),
 *                   ext_position (3), ext_orientation (4)
 *
 * Residuals: DYNAMIC = 2 * N (where N is the number of 3D marker points)
 */
class Fixed3DLandmarkWithExtrinsicCostFunctor
{
public:
  VESTA_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A    The residual weighting matrix (6x6 sqrt information)
   * @param[in] b    The 3D pose measurement (x, y, z, qw, qx, qy, qz)
   * @param[in] obs  The 2D projections of marker points (Nx2)
   * @param[in] pts3d The 3D points in marker coordinate frame (Nx3)
   */
  Fixed3DLandmarkWithExtrinsicCostFunctor(const vesta_core::MatrixXd& A, const vesta_core::Vector7d& b,
                                          const vesta_core::MatrixXd& obs, const vesta_core::MatrixXd& pts3d);

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   */
  template <typename T>
  bool operator()(const T* const body_position, const T* const body_orientation, const T* const calibration,
                  const T* const ext_position, const T* const ext_orientation, T* residual) const;

private:
  vesta_core::MatrixXd A_;
  vesta_core::Vector7d b_;
  vesta_core::MatrixXd obs_;
  vesta_core::MatrixXd pts3d_;
};

Fixed3DLandmarkWithExtrinsicCostFunctor::Fixed3DLandmarkWithExtrinsicCostFunctor(const vesta_core::MatrixXd& A,
                                                                                  const vesta_core::Vector7d& b,
                                                                                  const vesta_core::MatrixXd& obs,
                                                                                  const vesta_core::MatrixXd& pts3d)
  : A_(A), b_(b), obs_(obs), pts3d_(pts3d.transpose())  // Transpose from Nx3 to 3xN to make math easier.
{
  assert(pts3d_.rows() == 3);  // Check if we have 3xN

  // Create Marker Transform Matrix (Tm)
  double qm[4] = { b_(3), b_(4), b_(5), b_(6) };
  double rm[9];
  ceres::QuaternionToRotation(qm, rm);

  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Tm;
  Tm << rm[0], rm[1], rm[2], b_(0),  // NOLINT
      rm[3], rm[4], rm[5], b_(1),    // NOLINT
      rm[6], rm[7], rm[8], b_(2),    // NOLINT
      0.0, 0.0, 0.0, 1.0;            // NOLINT

  // Transform points to marker pose and store for later use.
  pts3d_ = Tm * pts3d_.colwise().homogeneous();
}

template <typename T>
bool Fixed3DLandmarkWithExtrinsicCostFunctor::operator()(const T* const body_position,
                                                          const T* const body_orientation,
                                                          const T* const calibration, const T* const ext_position,
                                                          const T* const ext_orientation, T* residual) const
{
  // Compute sensor-frame pose from body-frame pose + extrinsic
  T sensor_position[3];
  T sensor_orientation[4];
  computeSensorPose(body_position, body_orientation, ext_position, ext_orientation, sensor_position,
                    sensor_orientation);

  // Create Calibration Matrix K
  Eigen::Matrix<T, 4, 4, Eigen::RowMajor> K;
  K << calibration[0], T(0.0), calibration[2], T(0.0),  // NOLINT
      T(0.0), calibration[1], calibration[3], T(0.0),   // NOLINT
      T(0.0), T(0.0), T(1.0), T(0.0),                   // NOLINT
      T(0.0), T(0.0), T(0.0), T(1.0);                   // NOLINT

  // Create Camera Translation Matrix from sensor-frame pose params.
  T q[4] = { sensor_orientation[0], sensor_orientation[1], sensor_orientation[2], sensor_orientation[3] };
  T r[9];
  ceres::QuaternionToRotation(q, r);

  Eigen::Matrix<T, 4, 4, Eigen::RowMajor> Ta;
  Ta << r[0], r[1], r[2], sensor_position[0],  // NOLINT
      r[3], r[4], r[5], sensor_position[1],    // NOLINT
      r[6], r[7], r[8], sensor_position[2],    // NOLINT
      T(0.0), T(0.0), T(0.0), T(1.0);          // NOLINT

  // Transform 3D Points to Marker Location
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xp;

  // Project to camera at Ta
  xp = (K * Ta * pts3d_.cast<T>());
  xp.transposeInPlace();
  xp = xp.array().colwise() / xp.col(2).array();

  auto d = (obs_.cast<T>() - xp.block(0, 0, xp.rows(), 2));

  T fx = calibration[0];
  T fy = calibration[1];
  for (uint i = 0; i < pts3d_.cols(); i++)
  {
    // Get the covariance weighting to point losses from a pose uncertainty
    // From https://arxiv.org/pdf/2103.15980.pdf , equation A.7:
    T gx = pts3d_.cast<T>().col(i)[0];
    T gy = pts3d_.cast<T>().col(i)[1];
    T gz = pts3d_.cast<T>().col(i)[2];
    T gz2 = gz * gz;
    T gxyz = (gx * gy) / gz2;
    Eigen::Matrix<T, 2, 6, Eigen::RowMajor> J;
    J << fx / gz, T(0), -fx * (gx / gz2), -fx * gxyz, fx * (T(1) + (gx * gx) / gz2), -fx * gy / gz, T(0), fy / gz,
        -fy * (gy / gz2), -fy * (T(1) + (gy * gy) / gz2), fy * gxyz, fy * gx / gz;
    Eigen::Matrix<T, 2, 2, Eigen::RowMajor> A = J * A_ * J.transpose();

    // Weight Residuals
    auto r = A * d.row(i).transpose();
    residual[i * 2] = r[0];
    residual[i * 2 + 1] = r[1];
  }

  return true;
}

}  // namespace vesta_constraints
