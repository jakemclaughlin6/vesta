#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace vesta_constraints
{

/// @brief Compute the 3x3 skew-symmetric matrix of a 3-vector.
/// @param v Input 3-vector.
/// @return The skew-symmetric matrix [v]_x.
template <typename T>
inline Eigen::Matrix<T, 3, 3> skewSymmetric(const Eigen::Matrix<T, 3, 1>& v)
{
  Eigen::Matrix<T, 3, 3> m;
  m << T(0), -v(2), v(1), v(2), T(0), -v(0), -v(1), v(0), T(0);
  return m;
}

/// @brief SO3 exponential map via the Rodrigues formula.
/// @details Maps a rotation vector (angle-axis) to a 3x3 rotation matrix.
///          Uses a first-order Taylor approximation for small angles.
/// @param theta Rotation vector (axis * angle).
/// @return The corresponding 3x3 rotation matrix.
template <typename T>
inline Eigen::Matrix<T, 3, 3> expMapSO3(const Eigen::Matrix<T, 3, 1>& theta)
{
  const T angle_sq = theta.squaredNorm();
  const T angle = sqrt(angle_sq);
  const Eigen::Matrix<T, 3, 3> skew = skewSymmetric(theta);

  Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity();

  if (angle < T(1e-10))
  {
    // First-order Taylor expansion: R ≈ I + [theta]_x
    R += skew;
  }
  else
  {
    // Rodrigues formula: R = I + sin(a)/a * [theta]_x + (1-cos(a))/a^2 *
    // [theta]_x^2
    R += (sin(angle) / angle) * skew + ((T(1) - cos(angle)) / angle_sq) * (skew * skew);
  }

  return R;
}

/// @brief SO3 logarithmic map.
/// @details Maps a 3x3 rotation matrix to the corresponding rotation vector.
/// @param R A 3x3 rotation matrix.
/// @return The rotation vector (axis * angle).
template <typename T>
inline Eigen::Matrix<T, 3, 1> logMapSO3(const Eigen::Matrix<T, 3, 3>& R)
{
  const T trace_val = R.trace();
  // Clamp argument to acos to [-1, 1] for numerical safety
  const T cos_angle = (trace_val - T(1)) / T(2);
  const T cos_clamped = (cos_angle > T(1)) ? T(1) : ((cos_angle < T(-1)) ? T(-1) : cos_angle);
  const T angle = acos(cos_clamped);

  Eigen::Matrix<T, 3, 1> omega;
  omega << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);

  if (angle < T(1e-10))
  {
    // Small-angle approximation: omega ≈ 0.5 * vee(R - R^T)
    return T(0.5) * omega;
  }

  return (angle / (T(2) * sin(angle))) * omega;
}

/// @brief Right Jacobian of SO3.
/// @details Used during preintegration for bias correction. Double-only since
/// it
///          is not used inside Ceres cost functors.
/// @param theta Rotation vector (axis * angle).
/// @return The 3x3 right Jacobian matrix.
inline Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d& theta)
{
  const double angle_sq = theta.squaredNorm();
  const double angle = std::sqrt(angle_sq);
  const Eigen::Matrix3d skew = skewSymmetric<double>(theta);

  if (angle < 1e-10)
  {
    // First-order approximation: Jr ≈ I - 0.5 * [theta]_x
    return Eigen::Matrix3d::Identity() - 0.5 * skew;
  }

  // Jr = I - (1-cos(a))/a^2 * [theta]_x + (a-sin(a))/a^3 * [theta]_x^2
  const Eigen::Matrix3d skew_sq = skew * skew;
  return Eigen::Matrix3d::Identity() - ((1.0 - std::cos(angle)) / angle_sq) * skew +
         ((angle - std::sin(angle)) / (angle_sq * angle)) * skew_sq;
}

/// @brief Small-angle quaternion from a rotation vector (first-order
/// approximation).
/// @details Constructs q = [1, theta/2] and normalizes. This is the
/// approximation
///          used for incremental rotation updates in preintegration.
/// @param theta Small rotation vector.
/// @return The corresponding unit quaternion.
template <typename T>
inline Eigen::Quaternion<T> deltaQ(const Eigen::Matrix<T, 3, 1>& theta)
{
  const Eigen::Matrix<T, 3, 1> half_theta = theta / T(2);
  Eigen::Quaternion<T> q(T(1), half_theta.x(), half_theta.y(), half_theta.z());
  q.normalize();
  return q;
}

}  // namespace vesta_constraints
