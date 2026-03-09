#pragma once

#include <vesta_constraints/inertial/imu_preintegrator.h>
#include <vesta_core/constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/acceleration_bias_3d_stamped.h>
#include <vesta_variables/3d/gyroscope_bias_3d_stamped.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/3d/velocity_linear_3d_stamped.h>

#include <Eigen/Dense>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include <ostream>
#include <string>
#include <vector>

namespace vesta_constraints {

/**
 * @brief A constraint that represents the relative change between two 3D IMU
 * states using preintegrated IMU measurements.
 *
 * This constraint connects two full IMU states (orientation, position,
 * velocity, gyroscope bias, and accelerometer bias) using preintegrated IMU
 * measurements from the Forster et al. formulation. The preintegrated
 * measurements encode the relative motion between two keyframes while
 * accounting for bias corrections via first-order Jacobians.
 *
 * Variables order: ori1, pos1, vel1, bg1, ba1, ori2, pos2, vel2, bg2, ba2 (10
 * variables).
 */
class RelativeImuState3DStampedConstraint : public vesta_core::Constraint {
public:
  VESTA_CONSTRAINT_DEFINITIONS_WITH_EIGEN(RelativeImuState3DStampedConstraint);

  /**
   * @brief Default constructor
   */
  RelativeImuState3DStampedConstraint() = default;

  /**
   * @brief Create a constraint from preintegrated IMU measurements between two
   * IMU states
   *
   * @param[in] source          The name of the sensor or motion model that
   * generated this constraint
   * @param[in] orientation1    The orientation variable at the first state
   * @param[in] position1       The position variable at the first state
   * @param[in] velocity1       The linear velocity variable at the first state
   * @param[in] gyro_bias1      The gyroscope bias variable at the first state
   * @param[in] accel_bias1     The accelerometer bias variable at the first
   * state
   * @param[in] orientation2    The orientation variable at the second state
   * @param[in] position2       The position variable at the second state
   * @param[in] velocity2       The linear velocity variable at the second state
   * @param[in] gyro_bias2      The gyroscope bias variable at the second state
   * @param[in] accel_bias2     The accelerometer bias variable at the second
   * state
   * @param[in] preintegrator   The IMU preintegrator containing preintegrated
   * measurements
   * @param[in] linearization_bg The gyroscope bias at the preintegration
   * linearization point
   * @param[in] linearization_ba The accelerometer bias at the preintegration
   * linearization point
   * @param[in] gravity         The gravity vector in world frame (default: [0,
   * 0, -9.80665])
   * @param[in] info_weight     Scaling factor for the information matrix
   * (default: 1.0)
   */
  RelativeImuState3DStampedConstraint(
      const std::string &source,
      const vesta_variables::Orientation3DStamped &orientation1,
      const vesta_variables::Position3DStamped &position1,
      const vesta_variables::VelocityLinear3DStamped &velocity1,
      const vesta_variables::GyroscopeBias3DStamped &gyro_bias1,
      const vesta_variables::AccelerationBias3DStamped &accel_bias1,
      const vesta_variables::Orientation3DStamped &orientation2,
      const vesta_variables::Position3DStamped &position2,
      const vesta_variables::VelocityLinear3DStamped &velocity2,
      const vesta_variables::GyroscopeBias3DStamped &gyro_bias2,
      const vesta_variables::AccelerationBias3DStamped &accel_bias2,
      const ImuPreintegrator &preintegrator,
      const Eigen::Vector3d &linearization_bg,
      const Eigen::Vector3d &linearization_ba,
      const Eigen::Vector3d &gravity = kGravityWorld, double info_weight = 1.0);

  /**
   * @brief Destructor
   */
  virtual ~RelativeImuState3DStampedConstraint() = default;

  /**
   * @brief Read-only access to the preintegrated rotation.
   */
  const Eigen::Quaterniond &deltaQ() const { return delta_q_; }

  /**
   * @brief Read-only access to the preintegrated position.
   */
  const Eigen::Vector3d &deltaP() const { return delta_p_; }

  /**
   * @brief Read-only access to the preintegrated velocity.
   */
  const Eigen::Vector3d &deltaV() const { return delta_v_; }

  /**
   * @brief Read-only access to the integration time interval.
   */
  double dt() const { return dt_; }

  /**
   * @brief Read-only access to the square root information matrix.
   */
  const Eigen::Matrix<double, 15, 15> &sqrtInformation() const {
    return sqrt_information_;
  }

  /**
   * @brief Read-only access to the gravity vector.
   */
  const Eigen::Vector3d &gravity() const { return gravity_; }

  /**
   * @brief Print a human-readable description of the constraint to the provided
   * stream.
   *
   * @param[out] stream The stream to write to. Defaults to stdout.
   */
  void print(std::ostream &stream = std::cout) const override;

  /**
   * @brief Construct an instance of this constraint's cost function
   *
   * The function caller will own the new cost function instance. It is the
   * responsibility of the caller to delete the cost function object when it is
   * no longer needed. If the pointer is provided to a Ceres::Problem object,
   * the Ceres::Problem object will takes ownership of the pointer and delete it
   * during destruction.
   *
   * @return A base pointer to an instance of a derived CostFunction.
   */
  ceres::CostFunction *costFunction() const override;

protected:
  Eigen::Quaterniond delta_q_{
      Eigen::Quaterniond::Identity()};               //!< Preintegrated rotation
  Eigen::Vector3d delta_p_{Eigen::Vector3d::Zero()}; //!< Preintegrated position
  Eigen::Vector3d delta_v_{Eigen::Vector3d::Zero()}; //!< Preintegrated velocity
  double dt_{0.0}; //!< Integration time interval

  Eigen::Matrix<double, 15, 15> sqrt_information_{
      Eigen::Matrix<double, 15,
                    15>::Zero()}; //!< Square root information matrix

  Eigen::Vector3d linearization_bg_{
      Eigen::Vector3d::Zero()}; //!< Gyro bias linearization point
  Eigen::Vector3d linearization_ba_{
      Eigen::Vector3d::Zero()};            //!< Accel bias linearization point
  Eigen::Vector3d gravity_{kGravityWorld}; //!< Gravity vector in world frame

  Eigen::Matrix3d dq_dbg_{
      Eigen::Matrix3d::Zero()}; //!< Jacobian of preintegrated rotation w.r.t.
                                //!< gyro bias
  Eigen::Matrix3d dp_dbg_{
      Eigen::Matrix3d::Zero()}; //!< Jacobian of preintegrated position w.r.t.
                                //!< gyro bias
  Eigen::Matrix3d dp_dba_{
      Eigen::Matrix3d::Zero()}; //!< Jacobian of preintegrated position w.r.t.
                                //!< accel bias
  Eigen::Matrix3d dv_dbg_{
      Eigen::Matrix3d::Zero()}; //!< Jacobian of preintegrated velocity w.r.t.
                                //!< gyro bias
  Eigen::Matrix3d dv_dba_{
      Eigen::Matrix3d::Zero()}; //!< Jacobian of preintegrated velocity w.r.t.
                                //!< accel bias

private:
  // Allow Boost Serialization access to private methods
  friend class boost::serialization::access;

  /**
   * @brief The Boost Serialize method that serializes all of the data members
   * in to/out of the archive
   *
   * @param[in/out] archive - The archive object that holds the serialized class
   * members
   * @param[in] version - The version of the archive being read/written.
   * Generally unused.
   */
  template <class Archive>
  void serialize(Archive &archive, const unsigned int /* version */) {
    archive &boost::serialization::base_object<vesta_core::Constraint>(*this);
    // Serialize quaternion components individually (Eigen internal order: x, y,
    // z, w)
    archive & delta_q_.x();
    archive & delta_q_.y();
    archive & delta_q_.z();
    archive & delta_q_.w();
    archive & delta_p_;
    archive & delta_v_;
    archive & dt_;
    archive & sqrt_information_;
    archive & linearization_bg_;
    archive & linearization_ba_;
    archive & gravity_;
    archive & dq_dbg_;
    archive & dp_dbg_;
    archive & dp_dba_;
    archive & dv_dbg_;
    archive & dv_dba_;
  }
};

} // namespace vesta_constraints

BOOST_CLASS_EXPORT_KEY(vesta_constraints::RelativeImuState3DStampedConstraint);
