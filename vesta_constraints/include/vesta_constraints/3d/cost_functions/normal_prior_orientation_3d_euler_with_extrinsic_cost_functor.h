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
#include <vesta_core/util.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>

#include <ceres/rotation.h>
#include <Eigen/Core>

#include <vector>

namespace vesta_constraints
{

/**
 * @brief Cost function for an absolute 3D orientation Euler prior with an extrinsic calibration.
 *
 * This extends NormalPriorOrientation3DEulerCostFunctor to accept a body-frame orientation variable
 * and an extrinsic transform. The sensor orientation is computed as q_sensor = q_body * q_ext,
 * and the Euler angle prior residual is computed on q_sensor.
 *
 * The ext_position parameter block is present for graph consistency but is unused in the math.
 *
 * Parameter blocks: body_orientation (4), ext_position (3), ext_orientation (4)
 *
 * Residuals: DYNAMIC (depends on number of axes)
 */
class NormalPriorOrientation3DEulerWithExtrinsicCostFunctor
{
public:
  using Euler = vesta_variables::Orientation3DStamped::Euler;

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A    The residual weighting matrix. Its order must match the values in \p axes.
   * @param[in] b    The orientation measurement or prior. Its order must match the values in \p axes.
   * @param[in] axes The Euler angle axes for which we want to compute errors.
   */
  NormalPriorOrientation3DEulerWithExtrinsicCostFunctor(const vesta_core::MatrixXd& A, const vesta_core::VectorXd& b,
                                                        const std::vector<Euler>& axes = { Euler::ROLL, Euler::PITCH,
                                                                                           Euler::YAW })
    :  // NOLINT
    A_(A)
    , b_(b)
    , axes_(axes)
  {
  }

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   */
  template <typename T>
  bool operator()(const T* const body_orientation, const T* const /* ext_position */, const T* const ext_orientation,
                  T* residuals) const
  {
    // Compute sensor orientation: q_sensor = q_body * q_ext
    T sensor_orientation[4];
    ceres::QuaternionProduct(body_orientation, ext_orientation, sensor_orientation);

    for (size_t i = 0; i < axes_.size(); ++i)
    {
      T angle;
      switch (axes_[i])
      {
        case Euler::ROLL:
        {
          angle = vesta_core::getRoll(sensor_orientation[0], sensor_orientation[1], sensor_orientation[2],
                                      sensor_orientation[3]);
          break;
        }
        case Euler::PITCH:
        {
          angle = vesta_core::getPitch(sensor_orientation[0], sensor_orientation[1], sensor_orientation[2],
                                       sensor_orientation[3]);
          break;
        }
        case Euler::YAW:
        {
          angle = vesta_core::getYaw(sensor_orientation[0], sensor_orientation[1], sensor_orientation[2],
                                     sensor_orientation[3]);
          break;
        }
        default:
        {
          throw std::runtime_error("The provided axis specified is unknown. "
                                   "I should probably be more informative here");
        }
      }
      residuals[i] = angle - T(b_[i]);
    }

    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> residuals_map(residuals, A_.rows());
    residuals_map.applyOnTheLeft(A_.template cast<T>());

    return true;
  }

private:
  vesta_core::MatrixXd A_;   //!< The residual weighting matrix
  vesta_core::VectorXd b_;   //!< The measured Euler angle values
  std::vector<Euler> axes_;  //!< The Euler angle axes that we're measuring
};

}  // namespace vesta_constraints
