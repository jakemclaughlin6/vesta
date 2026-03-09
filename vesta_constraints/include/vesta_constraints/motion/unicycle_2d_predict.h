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

#include <ceres/jet.h>
#include <vesta_core/eigen.h>
#include <vesta_core/util.h>

#include <array>

namespace vesta_constraints
{

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] position1_x - First X position
 * @param[in] position1_y - First Y position
 * @param[in] yaw1 - First orientation
 * @param[in] vel_linear1_x - First X velocity
 * @param[in] vel_linear1_y - First Y velocity
 * @param[in] vel_yaw1 - First yaw velocity
 * @param[in] acc_linear1_x - First X acceleration
 * @param[in] acc_linear1_y - First Y acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[out] position2_x - Second X position
 * @param[out] position2_y - Second Y position
 * @param[out] yaw2 - Second orientation
 * @param[out] vel_linear2_x - Second X velocity
 * @param[out] vel_linear2_y - Second Y velocity
 * @param[out] vel_yaw2 - Second yaw velocity
 * @param[out] acc_linear2_x - Second X acceleration
 * @param[out] acc_linear2_y - Second Y acceleration
 */
template <typename T>
inline void predict(const T position1_x, const T position1_y, const T yaw1, const T vel_linear1_x,
                    const T vel_linear1_y, const T vel_yaw1, const T acc_linear1_x, const T acc_linear1_y, const T dt,
                    T& position2_x, T& position2_y, T& yaw2, T& vel_linear2_x, T& vel_linear2_y, T& vel_yaw2,
                    T& acc_linear2_x, T& acc_linear2_y)
{
  // There are better models for this projection, but this matches the one used
  // by r_l.
  T sy = ceres::sin(yaw1);  // Should probably be sin((yaw1 + yaw2) / 2), but r_l
                            // uses this model
  T cy = ceres::cos(yaw1);
  T delta_x = vel_linear1_x * dt + T(0.5) * acc_linear1_x * dt * dt;
  T delta_y = vel_linear1_y * dt + T(0.5) * acc_linear1_y * dt * dt;

  position2_x = position1_x + cy * delta_x - sy * delta_y;
  position2_y = position1_y + sy * delta_x + cy * delta_y;
  yaw2 = yaw1 + vel_yaw1 * dt;
  vel_linear2_x = vel_linear1_x + acc_linear1_x * dt;
  vel_linear2_y = vel_linear1_y + acc_linear1_y * dt;
  vel_yaw2 = vel_yaw1;
  acc_linear2_x = acc_linear1_x;
  acc_linear2_y = acc_linear1_y;

  vesta_core::wrapAngle2D(yaw2);
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] position1_x - First X position
 * @param[in] position1_y - First Y position
 * @param[in] yaw1 - First orientation
 * @param[in] vel_linear1_x - First X velocity
 * @param[in] vel_linear1_y - First Y velocity
 * @param[in] vel_yaw1 - First yaw velocity
 * @param[in] acc_linear1_x - First X acceleration
 * @param[in] acc_linear1_y - First Y acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[out] position2_x - Second X position
 * @param[out] position2_y - Second Y position
 * @param[out] yaw2 - Second orientation
 * @param[out] vel_linear2_x - Second X velocity
 * @param[out] vel_linear2_y - Second Y velocity
 * @param[out] vel_yaw2 - Second yaw velocity
 * @param[out] acc_linear2_x - Second X acceleration
 * @param[out] acc_linear2_y - Second Y acceleration
 * @param[out] jacobians - Jacobians wrt the state
 */
inline void predict(const double position1_x, const double position1_y, const double yaw1, const double vel_linear1_x,
                    const double vel_linear1_y, const double vel_yaw1, const double acc_linear1_x,
                    const double acc_linear1_y, const double dt, double& position2_x, double& position2_y, double& yaw2,
                    double& vel_linear2_x, double& vel_linear2_y, double& vel_yaw2, double& acc_linear2_x,
                    double& acc_linear2_y, double** jacobians)
{
  // There are better models for this projection, but this matches the one used
  // by r_l.
  const double sy = ceres::sin(yaw1);  // Should probably be sin((yaw1 + yaw2) /
                                       // 2), but r_l uses this model
  const double cy = ceres::cos(yaw1);

  const double half_dt2 = 0.5 * dt * dt;
  const double delta_x = vel_linear1_x * dt + acc_linear1_x * half_dt2;
  const double delta_y = vel_linear1_y * dt + acc_linear1_y * half_dt2;

  const double delta_x_rot = cy * delta_x - sy * delta_y;
  const double delta_y_rot = sy * delta_x + cy * delta_y;

  position2_x = position1_x + delta_x_rot;
  position2_y = position1_y + delta_y_rot;
  yaw2 = yaw1 + vel_yaw1 * dt;
  vel_linear2_x = vel_linear1_x + acc_linear1_x * dt;
  vel_linear2_y = vel_linear1_y + acc_linear1_y * dt;
  vel_yaw2 = vel_yaw1;
  acc_linear2_x = acc_linear1_x;
  acc_linear2_y = acc_linear1_y;

  vesta_core::wrapAngle2D(yaw2);

  if (jacobians)
  {
    // Jacobian wrt position1
    if (jacobians[0])
    {
      Eigen::Map<vesta_core::Matrix<double, 8, 2>> jacobian(jacobians[0]);
      jacobian << 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }

    // Jacobian wrt yaw1
    if (jacobians[1])
    {
      Eigen::Map<vesta_core::Vector8d> jacobian(jacobians[1]);
      jacobian << -delta_y_rot, delta_x_rot, 1, 0, 0, 0, 0, 0;
    }

    // Jacobian wrt vel_linear1
    if (jacobians[2])
    {
      const double cy_dt = cy * dt;
      const double sy_dt = sy * dt;

      Eigen::Map<vesta_core::Matrix<double, 8, 2>> jacobian(jacobians[2]);
      jacobian << cy_dt, -sy_dt, sy_dt, cy_dt, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0;
    }

    // Jacobian wrt vel_yaw1
    if (jacobians[3])
    {
      Eigen::Map<vesta_core::Vector8d> jacobian(jacobians[3]);
      jacobian << 0, 0, dt, 0, 0, 1, 0, 0;
    }

    // Jacobian wrt acc_linear1
    if (jacobians[4])
    {
      const double cy_half_dt2 = cy * half_dt2;
      const double sy_half_dt2 = sy * half_dt2;

      Eigen::Map<vesta_core::Matrix<double, 8, 2>> jacobian(jacobians[4]);
      jacobian << cy_half_dt2, -sy_half_dt2, sy_half_dt2, cy_half_dt2, 0, 0, dt, 0, 0, dt, 0, 0, 1, 0, 0, 1;
    }
  }
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] position1 - First position (array with x at index 0, y at index 1)
 * @param[in] yaw1 - First orientation
 * @param[in] vel_linear1 - First velocity (array with x at index 0, y at index
 * 1)
 * @param[in] vel_yaw1 - First yaw velocity
 * @param[in] acc_linear1 - First linear acceleration (array with x at index 0,
 * y at index 1)
 * @param[in] dt - The time delta across which to predict the state
 * @param[out] position2 - Second position (array with x at index 0, y at index
 * 1)
 * @param[out] yaw2 - Second orientation
 * @param[out] vel_linear2 - Second velocity (array with x at index 0, y at
 * index 1)
 * @param[out] vel_yaw2 - Second yaw velocity
 * @param[out] acc_linear2 - Second linear acceleration (array with x at index
 * 0, y at index 1)
 */
template <typename T>
inline void predict(const T* const position1, const T* const yaw1, const T* const vel_linear1, const T* const vel_yaw1,
                    const T* const acc_linear1, const T dt, T* const position2, T* const yaw2, T* const vel_linear2,
                    T* const vel_yaw2, T* const acc_linear2)
{
  predict(position1[0], position1[1], *yaw1, vel_linear1[0], vel_linear1[1], *vel_yaw1, acc_linear1[0], acc_linear1[1],
          dt, position2[0], position2[1], *yaw2, vel_linear2[0], vel_linear2[1], *vel_yaw2, acc_linear2[0],
          acc_linear2[1]);
}

}  // namespace vesta_constraints
