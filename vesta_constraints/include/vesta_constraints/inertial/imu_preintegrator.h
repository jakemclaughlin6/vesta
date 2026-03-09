#pragma once

#include <vesta_constraints/inertial/so3_utils.h>
#include <vesta_core/timestamp.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <map>

namespace vesta_constraints {

// Gravity constant
inline constexpr double kGravityNominal = 9.80665;
inline const Eigen::Vector3d kGravityWorld{0.0, 0.0, -kGravityNominal};

// Error state locations in the 15x15 covariance
enum ErrorStateIndex : int {
  ES_Q = 0,   // orientation (3)
  ES_P = 3,   // position (3)
  ES_V = 6,   // velocity (3)
  ES_BG = 9,  // gyro bias (3)
  ES_BA = 12, // accel bias (3)
  ES_SIZE = 15
};

struct ImuData {
  ImuData() = default;
  ImuData(const vesta_core::Timestamp &stamp,
          const Eigen::Vector3d &angular_velocity,
          const Eigen::Vector3d &linear_acceleration)
      : stamp(stamp), angular_velocity(angular_velocity),
        linear_acceleration(linear_acceleration) {}

  vesta_core::Timestamp stamp;
  Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};    // rad/s
  Eigen::Vector3d linear_acceleration{Eigen::Vector3d::Zero()}; // m/s^2
};

struct PreintegrationDelta {
  double dt{0.0};
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Vector3d v{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, ES_SIZE, ES_SIZE> covariance{
      Eigen::Matrix<double, ES_SIZE, ES_SIZE>::Zero()};
  Eigen::Matrix<double, ES_SIZE, ES_SIZE> sqrt_information{
      Eigen::Matrix<double, ES_SIZE, ES_SIZE>::Zero()};
};

struct PreintegrationJacobian {
  Eigen::Matrix3d dq_dbg{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d dp_dbg{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d dp_dba{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d dv_dbg{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d dv_dba{Eigen::Matrix3d::Zero()};
};

class ImuPreintegrator {
public:
  ImuPreintegrator() = default;

  void reset();
  void clearBefore(const vesta_core::Timestamp &t);

  void increment(double dt, const ImuData &data, const Eigen::Vector3d &bg,
                 const Eigen::Vector3d &ba, bool compute_jacobian,
                 bool compute_covariance);

  bool integrate(const vesta_core::Timestamp &t, const Eigen::Vector3d &bg,
                 const Eigen::Vector3d &ba, bool compute_jacobian,
                 bool compute_covariance, bool compute_information);

  void computeSqrtInformation();

  // Noise covariances (continuous-time)
  Eigen::Matrix3d cov_gyro{Eigen::Matrix3d::Identity() * 1e-4};
  Eigen::Matrix3d cov_accel{Eigen::Matrix3d::Identity() * 1e-3};
  Eigen::Matrix3d cov_gyro_bias{Eigen::Matrix3d::Identity() * 1e-6};
  Eigen::Matrix3d cov_accel_bias{Eigen::Matrix3d::Identity() * 1e-4};

  PreintegrationDelta delta;
  PreintegrationJacobian jacobian;
  std::map<vesta_core::Timestamp, ImuData> data;

  double cov_tol{1e-5};
  double bias_cov_tol{1e-9};
  double invalid_sqrt_info_weight{1e-4};
};

} // namespace vesta_constraints
