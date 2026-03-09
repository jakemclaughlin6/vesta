#pragma once

#include <vesta_constraints/3d/normal_prior_orientation_3d_cost_functor.h>
#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/util.h>

#include <Eigen/Core>

namespace vesta_constraints
{

/**
 * @brief Create a prior cost function on the full 3D IMU state.
 *
 * The cost function is of the form:
 *
 *   cost(x) = || A * [  AngleAxis(b(0:3)^-1 * q) ] ||^2
 *             ||     [  p - b(4:6)                ] ||
 *             ||     [  v - b(7:9)                ] ||
 *             ||     [  bg - b(10:12)             ] ||
 *             ||     [  ba - b(13:15)             ] ||
 *
 * where the matrix A and the vector b are fixed, q is the orientation variable,
 * p is the position variable, v is the linear velocity variable, bg is the
 * gyroscope bias variable, and ba is the accelerometer bias variable. Note that
 * the covariance submatrix for the quaternion is 3x3, representing errors in
 * the orientation local parameterization tangent space.
 *
 * The residual ordering is: orientation(3), position(3), velocity(3),
 * gyro_bias(3), accel_bias(3), matching the error-state indices used by the IMU
 * preintegrator.
 *
 * In case the user is interested in implementing a cost function of the form
 *
 *   cost(X) = (X - mu)^T S^{-1} (X - mu)
 *
 * where mu is a vector and S is a covariance matrix, then A = S^{-1/2}, i.e.
 * the matrix A is the square root information matrix (the inverse of the
 * covariance).
 */
class NormalPriorImuState3DCostFunctor
{
public:
  VESTA_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A The residual weighting matrix (15x15), most likely the square
   * root information matrix in order (qx, qy, qz, px, py, pz, vx, vy, vz, bgx,
   * bgy, bgz, bax, bay, baz)
   * @param[in] b The IMU state measurement or prior (16x1) in order
   *              (qw, qx, qy, qz, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax,
   * bay, baz)
   */
  NormalPriorImuState3DCostFunctor(const Eigen::Matrix<double, 15, 15>& A, const Eigen::Matrix<double, 16, 1>& b);

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   *
   * @param[in] orientation Orientation quaternion (4D, Ceres ordering: w, x, y,
   * z)
   * @param[in] position    Position (3D: x, y, z)
   * @param[in] velocity    Linear velocity (3D: x, y, z)
   * @param[in] gyro_bias   Gyroscope bias (3D: x, y, z)
   * @param[in] accel_bias  Accelerometer bias (3D: x, y, z)
   * @param[out] residual   Output residual vector (15D)
   */
  template <typename T>
  bool operator()(const T* const orientation, const T* const position, const T* const velocity,
                  const T* const gyro_bias, const T* const accel_bias, T* residual) const;

private:
  Eigen::Matrix<double, 15, 15> A_;
  Eigen::Matrix<double, 16, 1> b_;

  NormalPriorOrientation3DCostFunctor orientation_functor_;
};

inline NormalPriorImuState3DCostFunctor::NormalPriorImuState3DCostFunctor(const Eigen::Matrix<double, 15, 15>& A,
                                                                          const Eigen::Matrix<double, 16, 1>& b)
  : A_(A), b_(b), orientation_functor_(vesta_core::Matrix3d::Identity(), b_.head<4>())
{
}

template <typename T>
bool NormalPriorImuState3DCostFunctor::operator()(const T* const orientation, const T* const position,
                                                  const T* const velocity, const T* const gyro_bias,
                                                  const T* const accel_bias, T* residual) const
{
  // Compute the orientation error (3D angle-axis residual)
  orientation_functor_(orientation, &residual[0]);

  // Compute the position error
  residual[3] = position[0] - T(b_(4));
  residual[4] = position[1] - T(b_(5));
  residual[5] = position[2] - T(b_(6));

  // Compute the velocity error
  residual[6] = velocity[0] - T(b_(7));
  residual[7] = velocity[1] - T(b_(8));
  residual[8] = velocity[2] - T(b_(9));

  // Compute the gyroscope bias error
  residual[9] = gyro_bias[0] - T(b_(10));
  residual[10] = gyro_bias[1] - T(b_(11));
  residual[11] = gyro_bias[2] - T(b_(12));

  // Compute the accelerometer bias error
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
