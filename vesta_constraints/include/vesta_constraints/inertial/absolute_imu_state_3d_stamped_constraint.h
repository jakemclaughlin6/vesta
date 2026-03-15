#pragma once

#include <vesta_core/constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/acceleration_bias_3d_stamped.h>
#include <vesta_variables/3d/extrinsic_3d_orientation.h>
#include <vesta_variables/3d/extrinsic_3d_position.h>
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

namespace vesta_constraints
{

/**
 * @brief A constraint that represents prior information about the full 3D IMU
 * state.
 *
 * The IMU state consists of orientation (quaternion), position, linear
 * velocity, gyroscope bias, and accelerometer bias. This constraint applies an
 * absolute prior on all five variables simultaneously, using a 15x15 covariance
 * in error-state ordering (Q, P, V, BG, BA).
 *
 * Mean vector (16x1): [qw, qx, qy, qz, px, py, pz, vx, vy, vz, bgx, bgy, bgz,
 * bax, bay, baz]
 */
class AbsoluteImuState3DStampedConstraint : public vesta_core::Constraint
{
public:
  VESTA_CONSTRAINT_DEFINITIONS_WITH_EIGEN(AbsoluteImuState3DStampedConstraint);

  /**
   * @brief Default constructor
   */
  AbsoluteImuState3DStampedConstraint() = default;

  /**
   * @brief Create a constraint using a measurement/prior of the full 3D IMU
   * state
   *
   * @param[in] source      The name of the sensor or motion model that
   * generated this constraint
   * @param[in] orientation The variable representing the orientation component
   * @param[in] position    The variable representing the position component
   * @param[in] velocity    The variable representing the linear velocity
   * component
   * @param[in] gyro_bias   The variable representing the gyroscope bias
   * component
   * @param[in] accel_bias  The variable representing the accelerometer bias
   * component
   * @param[in] mean        The measured/prior IMU state (16x1: qw, qx, qy, qz,
   * px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz)
   * @param[in] covariance  The measurement/prior covariance (15x15 in
   * error-state order: Q, P, V, BG, BA)
   */
  AbsoluteImuState3DStampedConstraint(const std::string& source,
                                      const vesta_variables::Orientation3DStamped& orientation,
                                      const vesta_variables::Position3DStamped& position,
                                      const vesta_variables::VelocityLinear3DStamped& velocity,
                                      const vesta_variables::GyroscopeBias3DStamped& gyro_bias,
                                      const vesta_variables::AccelerationBias3DStamped& accel_bias,
                                      const Eigen::Matrix<double, 16, 1>& mean,
                                      const Eigen::Matrix<double, 15, 15>& covariance);

  /**
   * @brief Create a constraint with extrinsic calibration using a measurement/prior of the full 3D IMU state
   *
   * This constructor accepts an extrinsic transform T_body_sensor that maps points from the IMU
   * sensor frame to the body frame. The orientation and position variables represent the body frame,
   * and the extrinsic is applied internally to compute sensor-frame pose for the prior. The mean
   * vector is expected in the sensor frame. Velocity and biases are not transformed.
   *
   * @param[in] source          The name of the sensor or motion model that generated this constraint
   * @param[in] orientation     The variable representing the body-frame orientation component
   * @param[in] position        The variable representing the body-frame position component
   * @param[in] velocity        The variable representing the linear velocity component
   * @param[in] gyro_bias       The variable representing the gyroscope bias component
   * @param[in] accel_bias      The variable representing the accelerometer bias component
   * @param[in] ext_position    The extrinsic translation (body-to-sensor)
   * @param[in] ext_orientation The extrinsic rotation (body-to-sensor)
   * @param[in] mean            The measured/prior IMU state (16x1: qw, qx, qy, qz, px, py, pz,
   *                            vx, vy, vz, bgx, bgy, bgz, bax, bay, baz)
   * @param[in] covariance      The measurement/prior covariance (15x15 in error-state order)
   */
  AbsoluteImuState3DStampedConstraint(const std::string& source,
                                      const vesta_variables::Orientation3DStamped& orientation,
                                      const vesta_variables::Position3DStamped& position,
                                      const vesta_variables::VelocityLinear3DStamped& velocity,
                                      const vesta_variables::GyroscopeBias3DStamped& gyro_bias,
                                      const vesta_variables::AccelerationBias3DStamped& accel_bias,
                                      const vesta_variables::Extrinsic3DPosition& ext_position,
                                      const vesta_variables::Extrinsic3DOrientation& ext_orientation,
                                      const Eigen::Matrix<double, 16, 1>& mean,
                                      const Eigen::Matrix<double, 15, 15>& covariance);

  /**
   * @brief Returns whether this constraint uses an extrinsic calibration.
   */
  bool hasExtrinsic() const
  {
    return has_extrinsic_;
  }

  /**
   * @brief Destructor
   */
  virtual ~AbsoluteImuState3DStampedConstraint() = default;

  /**
   * @brief Read-only access to the measured/prior vector of mean values.
   *
   * Order is (qw, qx, qy, qz, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay,
   * baz)
   */
  const Eigen::Matrix<double, 16, 1>& mean() const
  {
    return mean_;
  }

  /**
   * @brief Read-only access to the square root information matrix.
   *
   * Order is (qx, qy, qz, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz)
   */
  const Eigen::Matrix<double, 15, 15>& sqrtInformation() const
  {
    return sqrt_information_;
  }

  /**
   * @brief Compute the measurement covariance matrix.
   *
   * Order is (qx, qy, qz, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz)
   */
  Eigen::Matrix<double, 15, 15> covariance() const
  {
    return (sqrt_information_.transpose() * sqrt_information_).inverse();
  }

  /**
   * @brief Print a human-readable description of the constraint to the provided
   * stream.
   *
   * @param[out] stream The stream to write to. Defaults to stdout.
   */
  void print(std::ostream& stream = std::cout) const override;

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
  ceres::CostFunction* costFunction() const override;

protected:
  Eigen::Matrix<double, 16, 1> mean_;               //!< The measured/prior mean vector for this variable
  Eigen::Matrix<double, 15, 15> sqrt_information_;  //!< The square root information matrix
  bool has_extrinsic_{ false };                     //!< Whether this constraint uses an extrinsic calibration

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
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Constraint>(*this);
    archive & mean_;
    archive & sqrt_information_;
    archive & has_extrinsic_;
  }
};

}  // namespace vesta_constraints

BOOST_CLASS_EXPORT_KEY(vesta_constraints::AbsoluteImuState3DStampedConstraint);
