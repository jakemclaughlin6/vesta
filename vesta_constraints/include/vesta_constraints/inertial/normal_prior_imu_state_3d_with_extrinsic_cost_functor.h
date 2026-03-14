#pragma once

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
 * @brief Prior cost function on the full 3D IMU state with extrinsic calibration.
 *
 * This extends NormalPriorImuState3DCostFunctor to accept body-frame pose variables
 * and an extrinsic transform T_body_sensor. The extrinsic is applied to compute
 * sensor-frame orientation and position before applying the prior. Velocity and
 * biases are not transformed.
 *
 * The cost function is of the form:
 *
 *   cost(x) = || A * [  AngleAxis(b(0:3)^-1 * q_sensor)  ] ||^2
 *             ||     [  p_sensor - b(4:6)                 ] ||
 *             ||     [  v - b(7:9)                        ] ||
 *             ||     [  bg - b(10:12)                     ] ||
 *             ||     [  ba - b(13:15)                     ] ||
 *
 * where q_sensor and p_sensor are computed from body-frame variables + extrinsic.
 *
 * Parameter blocks: orientation[4], position[3], velocity[3], gyro_bias[3],
 *                   accel_bias[3], ext_position[3], ext_orientation[4]
 *
 * Residuals: 15 (orientation(3), position(3), velocity(3), gyro_bias(3), accel_bias(3))
 */
class NormalPriorImuState3DWithExtrinsicCostFunctor
{
public:
  VESTA_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A The residual weighting matrix (15x15), most likely the square root information
   *              matrix in order (qx, qy, qz, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz)
   * @param[in] b The IMU state measurement or prior (16x1) in order
   *              (qw, qx, qy, qz, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz)
   */
  NormalPriorImuState3DWithExtrinsicCostFunctor(const Eigen::Matrix<double, 15, 15>& A,
                                                const Eigen::Matrix<double, 16, 1>& b);

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   *
   * @param[in] orientation Body-frame orientation quaternion (4D, Ceres ordering: w, x, y, z)
   * @param[in] position    Body-frame position (3D: x, y, z)
   * @param[in] velocity    Linear velocity (3D: x, y, z)
   * @param[in] gyro_bias   Gyroscope bias (3D: x, y, z)
   * @param[in] accel_bias  Accelerometer bias (3D: x, y, z)
   * @param[in] ext_position    Extrinsic translation body-to-sensor (3D: x, y, z)
   * @param[in] ext_orientation Extrinsic rotation quaternion body-to-sensor (4D: w, x, y, z)
   * @param[out] residual   Output residual vector (15D)
   */
  template <typename T>
  bool operator()(const T* const orientation, const T* const position, const T* const velocity,
                  const T* const gyro_bias, const T* const accel_bias, const T* const ext_position,
                  const T* const ext_orientation, T* residual) const;

private:
  Eigen::Matrix<double, 15, 15> A_;
  Eigen::Matrix<double, 16, 1> b_;

  NormalPriorOrientation3DCostFunctor orientation_functor_;
};

inline NormalPriorImuState3DWithExtrinsicCostFunctor::NormalPriorImuState3DWithExtrinsicCostFunctor(
    const Eigen::Matrix<double, 15, 15>& A, const Eigen::Matrix<double, 16, 1>& b)
  : A_(A), b_(b), orientation_functor_(vesta_core::Matrix3d::Identity(), b_.head<4>())
{
}

template <typename T>
bool NormalPriorImuState3DWithExtrinsicCostFunctor::operator()(
    const T* const orientation, const T* const position, const T* const velocity, const T* const gyro_bias,
    const T* const accel_bias, const T* const ext_position, const T* const ext_orientation, T* residual) const
{
  // Compute sensor-frame pose from body-frame pose + extrinsic
  T sensor_position[3];
  T sensor_orientation[4];
  computeSensorPose(position, orientation, ext_position, ext_orientation, sensor_position, sensor_orientation);

  // Compute the orientation error (3D angle-axis residual) using sensor-frame orientation
  orientation_functor_(sensor_orientation, &residual[0]);

  // Compute the position error using sensor-frame position
  residual[3] = sensor_position[0] - T(b_(4));
  residual[4] = sensor_position[1] - T(b_(5));
  residual[5] = sensor_position[2] - T(b_(6));

  // Compute the velocity error (body-frame, not transformed)
  residual[6] = velocity[0] - T(b_(7));
  residual[7] = velocity[1] - T(b_(8));
  residual[8] = velocity[2] - T(b_(9));

  // Compute the gyroscope bias error (intrinsic to sensor, not transformed)
  residual[9] = gyro_bias[0] - T(b_(10));
  residual[10] = gyro_bias[1] - T(b_(11));
  residual[11] = gyro_bias[2] - T(b_(12));

  // Compute the accelerometer bias error (intrinsic to sensor, not transformed)
  residual[12] = accel_bias[0] - T(b_(13));
  residual[13] = accel_bias[1] - T(b_(14));
  residual[14] = accel_bias[2] - T(b_(15));

  // Scale the residuals by the square root information matrix to account for
  // the measurement uncertainty.
  Eigen::Map<Eigen::Matrix<T, 15, 1>> residual_map(residual);
  residual_map.applyOnTheLeft(A_.template cast<T>());

  return true;
}

}  // namespace vesta_constraints
