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

#include <ceres/rotation.h>

namespace vesta_constraints
{

/**
 * @brief Compute the sensor-frame pose in the world frame given a body pose and an extrinsic.
 *
 * The extrinsic T_body_sensor transforms points from the sensor frame to the body frame:
 *   p_body = q_bs * p_sensor + t_bs
 *
 * Given the body pose in the world frame (p_wb, q_wb), the sensor pose in the world frame is:
 *   q_ws = q_wb * q_bs
 *   p_ws = p_wb + R_wb * t_bs
 *
 * @param[in]  body_position      Body position in world frame (3 elements: x, y, z)
 * @param[in]  body_orientation   Body orientation quaternion in world frame (4 elements: w, x, y, z)
 * @param[in]  ext_position       Extrinsic translation body-to-sensor (3 elements: x, y, z)
 * @param[in]  ext_orientation    Extrinsic rotation quaternion body-to-sensor (4 elements: w, x, y, z)
 * @param[out] sensor_position    Sensor position in world frame (3 elements)
 * @param[out] sensor_orientation Sensor orientation quaternion in world frame (4 elements)
 */
template <typename T>
inline void computeSensorPose(const T* const body_position, const T* const body_orientation,
                              const T* const ext_position, const T* const ext_orientation, T* sensor_position,
                              T* sensor_orientation)
{
  // q_ws = q_wb * q_bs
  ceres::QuaternionProduct(body_orientation, ext_orientation, sensor_orientation);

  // p_ws = p_wb + R_wb * t_bs
  T rotated_ext_position[3];
  ceres::QuaternionRotatePoint(body_orientation, ext_position, rotated_ext_position);
  sensor_position[0] = body_position[0] + rotated_ext_position[0];
  sensor_position[1] = body_position[1] + rotated_ext_position[1];
  sensor_position[2] = body_position[2] + rotated_ext_position[2];
}

}  // namespace vesta_constraints
