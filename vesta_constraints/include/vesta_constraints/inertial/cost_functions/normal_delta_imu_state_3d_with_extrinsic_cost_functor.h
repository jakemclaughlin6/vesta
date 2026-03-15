#pragma once

#include <vesta_constraints/3d/extrinsic_pose_3d.h>
#include <vesta_constraints/inertial/so3_utils.h>
#include <vesta_core/fuse_macros.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace vesta_constraints
{

/**
 * @brief Cost function for the relative constraint between two 3D IMU states
 * with an extrinsic calibration.
 *
 * This extends NormalDeltaImuState3DCostFunctor to accept body-frame pose
 * variables and an extrinsic transform T_body_sensor. The extrinsic is applied
 * to compute sensor-frame poses (orientation and position) before computing the
 * preintegration residual. Velocity and biases are not transformed.
 *
 * Parameter blocks (12 blocks):
 *   orientation1[4], position1[3], velocity1[3], gyro_bias1[3], accel_bias1[3],
 *   orientation2[4], position2[3], velocity2[3], gyro_bias2[3], accel_bias2[3],
 *   ext_position[3], ext_orientation[4]
 *
 * Residuals: 15 (orientation(3), position(3), velocity(3), gyro_bias(3), accel_bias(3))
 */
class NormalDeltaImuState3DWithExtrinsicCostFunctor
{
public:
  VESTA_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] sqrt_information  Square root information matrix (15x15)
   * @param[in] delta_q           Preintegrated rotation (unit quaternion)
   * @param[in] delta_p           Preintegrated position (3D)
   * @param[in] delta_v           Preintegrated velocity (3D)
   * @param[in] dt                Integration time interval (seconds)
   * @param[in] gravity           Gravity vector in world frame (3D)
   * @param[in] linearization_bg  Gyroscope bias at preintegration linearization point (3D)
   * @param[in] linearization_ba  Accelerometer bias at preintegration linearization point (3D)
   * @param[in] dq_dbg            Jacobian of preintegrated rotation w.r.t. gyro bias (3x3)
   * @param[in] dp_dbg            Jacobian of preintegrated position w.r.t. gyro bias (3x3)
   * @param[in] dp_dba            Jacobian of preintegrated position w.r.t. accel bias (3x3)
   * @param[in] dv_dbg            Jacobian of preintegrated velocity w.r.t. gyro bias (3x3)
   * @param[in] dv_dba            Jacobian of preintegrated velocity w.r.t. accel bias (3x3)
   */
  NormalDeltaImuState3DWithExtrinsicCostFunctor(const Eigen::Matrix<double, 15, 15>& sqrt_information,
                                                const Eigen::Quaterniond& delta_q, const Eigen::Vector3d& delta_p,
                                                const Eigen::Vector3d& delta_v, double dt,
                                                const Eigen::Vector3d& gravity, const Eigen::Vector3d& linearization_bg,
                                                const Eigen::Vector3d& linearization_ba, const Eigen::Matrix3d& dq_dbg,
                                                const Eigen::Matrix3d& dp_dbg, const Eigen::Matrix3d& dp_dba,
                                                const Eigen::Matrix3d& dv_dbg, const Eigen::Matrix3d& dv_dba);

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   *
   * Parameter block ordering (12 blocks):
   *   orientation1[4], position1[3], velocity1[3], gyro_bias1[3], accel_bias1[3],
   *   orientation2[4], position2[3], velocity2[3], gyro_bias2[3], accel_bias2[3],
   *   ext_position[3], ext_orientation[4]
   *
   * @param[out] residual Output residual vector (15D)
   */
  template <typename T>
  bool operator()(const T* const orientation1, const T* const position1, const T* const velocity1,
                  const T* const gyro_bias1, const T* const accel_bias1, const T* const orientation2,
                  const T* const position2, const T* const velocity2, const T* const gyro_bias2,
                  const T* const accel_bias2, const T* const ext_position, const T* const ext_orientation,
                  T* residual) const;

private:
  Eigen::Matrix<double, 15, 15> sqrt_information_;
  Eigen::Quaterniond delta_q_;
  Eigen::Vector3d delta_p_;
  Eigen::Vector3d delta_v_;
  double dt_;
  Eigen::Vector3d gravity_;
  Eigen::Vector3d linearization_bg_;
  Eigen::Vector3d linearization_ba_;
  Eigen::Matrix3d dq_dbg_;
  Eigen::Matrix3d dp_dbg_;
  Eigen::Matrix3d dp_dba_;
  Eigen::Matrix3d dv_dbg_;
  Eigen::Matrix3d dv_dba_;
};

inline NormalDeltaImuState3DWithExtrinsicCostFunctor::NormalDeltaImuState3DWithExtrinsicCostFunctor(
    const Eigen::Matrix<double, 15, 15>& sqrt_information, const Eigen::Quaterniond& delta_q,
    const Eigen::Vector3d& delta_p, const Eigen::Vector3d& delta_v, const double dt, const Eigen::Vector3d& gravity,
    const Eigen::Vector3d& linearization_bg, const Eigen::Vector3d& linearization_ba, const Eigen::Matrix3d& dq_dbg,
    const Eigen::Matrix3d& dp_dbg, const Eigen::Matrix3d& dp_dba, const Eigen::Matrix3d& dv_dbg,
    const Eigen::Matrix3d& dv_dba)
  : sqrt_information_(sqrt_information)
  , delta_q_(delta_q)
  , delta_p_(delta_p)
  , delta_v_(delta_v)
  , dt_(dt)
  , gravity_(gravity)
  , linearization_bg_(linearization_bg)
  , linearization_ba_(linearization_ba)
  , dq_dbg_(dq_dbg)
  , dp_dbg_(dp_dbg)
  , dp_dba_(dp_dba)
  , dv_dbg_(dv_dbg)
  , dv_dba_(dv_dba)
{
}

template <typename T>
bool NormalDeltaImuState3DWithExtrinsicCostFunctor::operator()(const T* const orientation1, const T* const position1,
                                                               const T* const velocity1, const T* const gyro_bias1,
                                                               const T* const accel_bias1, const T* const orientation2,
                                                               const T* const position2, const T* const velocity2,
                                                               const T* const gyro_bias2, const T* const accel_bias2,
                                                               const T* const ext_position,
                                                               const T* const ext_orientation, T* residual) const
{
  // Compute sensor-frame poses from body-frame poses + extrinsic
  T sensor_position1[3];
  T sensor_orientation1[4];
  computeSensorPose(position1, orientation1, ext_position, ext_orientation, sensor_position1, sensor_orientation1);

  T sensor_position2[3];
  T sensor_orientation2[4];
  computeSensorPose(position2, orientation2, ext_position, ext_orientation, sensor_position2, sensor_orientation2);

  // Map sensor-frame poses to Eigen types
  const Eigen::Quaternion<T> q_i(sensor_orientation1[0], sensor_orientation1[1], sensor_orientation1[2],
                                 sensor_orientation1[3]);
  const Eigen::Matrix<T, 3, 1> p_i(sensor_position1[0], sensor_position1[1], sensor_position1[2]);
  const Eigen::Matrix<T, 3, 1> v_i(velocity1[0], velocity1[1], velocity1[2]);
  const Eigen::Matrix<T, 3, 1> bg_i(gyro_bias1[0], gyro_bias1[1], gyro_bias1[2]);
  const Eigen::Matrix<T, 3, 1> ba_i(accel_bias1[0], accel_bias1[1], accel_bias1[2]);

  const Eigen::Quaternion<T> q_j(sensor_orientation2[0], sensor_orientation2[1], sensor_orientation2[2],
                                 sensor_orientation2[3]);
  const Eigen::Matrix<T, 3, 1> p_j(sensor_position2[0], sensor_position2[1], sensor_position2[2]);
  const Eigen::Matrix<T, 3, 1> v_j(velocity2[0], velocity2[1], velocity2[2]);
  const Eigen::Matrix<T, 3, 1> bg_j(gyro_bias2[0], gyro_bias2[1], gyro_bias2[2]);
  const Eigen::Matrix<T, 3, 1> ba_j(accel_bias2[0], accel_bias2[1], accel_bias2[2]);

  // Cast preintegrated values to templated type
  const T dt = T(dt_);
  const Eigen::Quaternion<T> dq = delta_q_.cast<T>();
  const Eigen::Matrix<T, 3, 1> dp = delta_p_.cast<T>();
  const Eigen::Matrix<T, 3, 1> dv = delta_v_.cast<T>();
  const Eigen::Matrix<T, 3, 1> G = gravity_.cast<T>();

  // Compute bias deltas from linearization point
  const Eigen::Matrix<T, 3, 1> dbg = bg_i - linearization_bg_.cast<T>();
  const Eigen::Matrix<T, 3, 1> dba = ba_i - linearization_ba_.cast<T>();

  // Cast Jacobian matrices
  const Eigen::Matrix<T, 3, 3> J_dq_dbg = dq_dbg_.cast<T>();
  const Eigen::Matrix<T, 3, 3> J_dp_dbg = dp_dbg_.cast<T>();
  const Eigen::Matrix<T, 3, 3> J_dp_dba = dp_dba_.cast<T>();
  const Eigen::Matrix<T, 3, 3> J_dv_dbg = dv_dbg_.cast<T>();
  const Eigen::Matrix<T, 3, 3> J_dv_dba = dv_dba_.cast<T>();

  // Apply first-order bias correction to preintegrated measurements
  const Eigen::Matrix<T, 3, 1> q_correction = J_dq_dbg * dbg;
  const Eigen::Quaternion<T> dq_corrected = dq * deltaQ(q_correction);
  const Eigen::Matrix<T, 3, 1> dp_corrected = dp + J_dp_dbg * dbg + J_dp_dba * dba;
  const Eigen::Matrix<T, 3, 1> dv_corrected = dv + J_dv_dbg * dbg + J_dv_dba * dba;

  // Orientation residual (uses sensor-frame orientations)
  const Eigen::Matrix<T, 3, 1> res_q = T(2) * (dq_corrected.inverse() * (q_i.inverse() * q_j)).vec();

  // Position residual (uses sensor-frame positions, body-frame velocity)
  const Eigen::Matrix<T, 3, 1> res_p = q_i.conjugate() * (p_j - p_i - dt * v_i - T(0.5) * dt * dt * G) - dp_corrected;

  // Velocity residual (body-frame velocity, not transformed)
  const Eigen::Matrix<T, 3, 1> res_v = q_i.conjugate() * (v_j - v_i - dt * G) - dv_corrected;

  // Bias residuals (random walk model, intrinsic to sensor)
  const Eigen::Matrix<T, 3, 1> res_bg = bg_j - bg_i;
  const Eigen::Matrix<T, 3, 1> res_ba = ba_j - ba_i;

  // Pack residuals: [orientation, position, velocity, gyro_bias, accel_bias]
  residual[0] = res_q[0];
  residual[1] = res_q[1];
  residual[2] = res_q[2];
  residual[3] = res_p[0];
  residual[4] = res_p[1];
  residual[5] = res_p[2];
  residual[6] = res_v[0];
  residual[7] = res_v[1];
  residual[8] = res_v[2];
  residual[9] = res_bg[0];
  residual[10] = res_bg[1];
  residual[11] = res_bg[2];
  residual[12] = res_ba[0];
  residual[13] = res_ba[1];
  residual[14] = res_ba[2];

  // Scale the residuals by the square root information matrix to account for
  // the measurement uncertainty.
  Eigen::Map<Eigen::Matrix<T, 15, 1>> residual_map(residual);
  residual_map.applyOnTheLeft(sqrt_information_.template cast<T>());

  return true;
}

}  // namespace vesta_constraints
