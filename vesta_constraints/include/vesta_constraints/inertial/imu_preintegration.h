#pragma once

#include <vesta_constraints/inertial/absolute_imu_state_3d_stamped_constraint.h>
#include <vesta_constraints/inertial/imu_preintegrator.h>
#include <vesta_constraints/inertial/relative_imu_state_3d_stamped_constraint.h>
#include <vesta_core/timestamp.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/acceleration_bias_3d_stamped.h>
#include <vesta_variables/3d/gyroscope_bias_3d_stamped.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/3d/velocity_linear_3d_stamped.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>
#include <string>

namespace vesta_constraints
{

struct ImuPreintegrationParams
{
  double cov_prior_noise{ 1e-9 };
  Eigen::Matrix3d cov_gyro_noise{ Eigen::Matrix3d::Identity() * 1e-4 };
  Eigen::Matrix3d cov_accel_noise{ Eigen::Matrix3d::Identity() * 1e-3 };
  Eigen::Matrix3d cov_gyro_bias{ Eigen::Matrix3d::Identity() * 1e-6 };
  Eigen::Matrix3d cov_accel_bias{ Eigen::Matrix3d::Identity() * 1e-4 };
  Eigen::Vector3d gravity{ kGravityWorld };
  std::string source{ "ImuPreintegration" };
};

/**
 * @brief Lightweight struct to hold an IMU state (values only, no graph
 * variables)
 */
struct ImuState
{
  vesta_core::Timestamp stamp;
  Eigen::Quaterniond orientation{ Eigen::Quaterniond::Identity() };
  Eigen::Vector3d position{ Eigen::Vector3d::Zero() };
  Eigen::Vector3d velocity{ Eigen::Vector3d::Zero() };
  Eigen::Vector3d gyro_bias{ Eigen::Vector3d::Zero() };
  Eigen::Vector3d accel_bias{ Eigen::Vector3d::Zero() };
};

/**
 * @brief High-level IMU preintegration manager.
 *
 * Manages IMU preintegration state, creates preintegrated factors between
 * keyframes, and provides state prediction. Adapted from beam_slam's
 * ImuPreintegration for vesta's library pattern (no ROS, no graph dependency,
 * no internal threading).
 *
 * Usage pattern:
 *   1. Construct with parameters
 *   2. Call addImuData() as measurements arrive
 *   3. Call setStart() to initialize the first keyframe
 *   4. Call createPreintegratedFactor() to create factors between keyframes
 *   5. After graph optimization, call updateState() with optimized values
 */
class ImuPreintegration
{
public:
  /**
   * @brief Construct an ImuPreintegration manager.
   *
   * @param[in] params  Noise and gravity parameters
   * @param[in] info_weight  Scaling factor for the information matrix
   * (default: 1.0)
   */
  explicit ImuPreintegration(const ImuPreintegrationParams& params = ImuPreintegrationParams{},
                             double info_weight = 1.0);

  /**
   * @brief Add an IMU measurement to the buffer.
   *
   * @param[in] data  The IMU measurement (timestamp, angular velocity, linear
   * acceleration)
   */
  void addImuData(const ImuData& data);

  /**
   * @brief Set the initial state (must be called before creating constraints).
   *
   * Clears data before the start time and initializes the current keyframe
   * variables.
   *
   * @param[in] stamp       The timestamp of the initial state
   * @param[in] orientation The initial orientation (default: identity)
   * @param[in] position    The initial position (default: zero)
   * @param[in] velocity    The initial velocity (default: zero)
   * @param[in] device_id   The device id for variable construction (default:
   * NIL)
   */
  void setStart(const vesta_core::Timestamp& stamp,
                const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity(),
                const Eigen::Vector3d& position = Eigen::Vector3d::Zero(),
                const Eigen::Vector3d& velocity = Eigen::Vector3d::Zero(),
                const vesta_core::UUID& device_id = vesta_core::uuid::NIL);

  /**
   * @brief Set/update bias estimates.
   *
   * @param[in] gyro_bias  The gyroscope bias vector
   * @param[in] accel_bias The accelerometer bias vector
   */
  void setBiases(const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& accel_bias);

  /**
   * @brief Predict state at a given time using IMU preintegration from the
   * current keyframe.
   *
   * This performs a read-only integration and state prediction without
   * modifying internal state.
   *
   * @param[in] stamp  The timestamp to predict state at
   * @return The predicted IMU state
   */
  ImuState predictState(const vesta_core::Timestamp& stamp) const;

  /**
   * @brief Result of creating a preintegrated factor.
   */
  struct PreintegrationResult
  {
    ImuState predicted_state;
    RelativeImuState3DStampedConstraint::SharedPtr relative_constraint;
    AbsoluteImuState3DStampedConstraint::SharedPtr prior_constraint;  // only on first window if requested

    // The variables at the NEW keyframe (caller adds these to their graph)
    std::shared_ptr<vesta_variables::Orientation3DStamped> orientation;
    std::shared_ptr<vesta_variables::Position3DStamped> position;
    std::shared_ptr<vesta_variables::VelocityLinear3DStamped> velocity;
    std::shared_ptr<vesta_variables::GyroscopeBias3DStamped> gyro_bias;
    std::shared_ptr<vesta_variables::AccelerationBias3DStamped> accel_bias;
  };

  /**
   * @brief Create a preintegrated factor between the current keyframe and a new
   * time.
   *
   * Integrates IMU data between the current keyframe and the given timestamp,
   * predicts the state at that time, creates new variables and a relative
   * constraint. If this is the first window and add_prior_on_first_window is
   * true, also creates a prior constraint on the first keyframe.
   *
   * After this call, the internal state advances: the new keyframe becomes the
   * predicted state at stamp.
   *
   * @param[in] stamp                       The timestamp for the new keyframe
   * @param[in] device_id                   The device id for variable
   * construction
   * @param[in] add_prior_on_first_window   Whether to add a prior on the first
   * window
   * @return The preintegration result containing constraints and variables
   */
  PreintegrationResult createPreintegratedFactor(const vesta_core::Timestamp& stamp,
                                                 const vesta_core::UUID& device_id = vesta_core::uuid::NIL,
                                                 bool add_prior_on_first_window = true);

  /**
   * @brief Update the current keyframe state (e.g., after graph optimization).
   *
   * Updates both the internal state values and the variable data arrays.
   *
   * @param[in] state  The updated IMU state
   */
  void updateState(const ImuState& state);

  /**
   * @brief Get the current keyframe state.
   */
  const ImuState& currentState() const
  {
    return current_state_;
  }

  /**
   * @brief Get the orientation variable at the current keyframe.
   */
  const vesta_variables::Orientation3DStamped& currentOrientation() const
  {
    return orientation_i_;
  }

  /**
   * @brief Get the position variable at the current keyframe.
   */
  const vesta_variables::Position3DStamped& currentPosition() const
  {
    return position_i_;
  }

  /**
   * @brief Get the velocity variable at the current keyframe.
   */
  const vesta_variables::VelocityLinear3DStamped& currentVelocity() const
  {
    return velocity_i_;
  }

  /**
   * @brief Get the gyroscope bias variable at the current keyframe.
   */
  const vesta_variables::GyroscopeBias3DStamped& currentGyroBias() const
  {
    return gyro_bias_i_;
  }

  /**
   * @brief Get the accelerometer bias variable at the current keyframe.
   */
  const vesta_variables::AccelerationBias3DStamped& currentAccelBias() const
  {
    return accel_bias_i_;
  }

  /**
   * @brief Clear IMU data before a given timestamp.
   *
   * @param[in] stamp  The timestamp before which to clear data
   */
  void clearDataBefore(const vesta_core::Timestamp& stamp);

  /**
   * @brief Reset to initial state — clears preintegrator and resets flags.
   */
  void reset();

  /**
   * @brief Get the number of IMU measurements in the buffer.
   */
  size_t bufferSize() const
  {
    return preintegrator_.data.size();
  }

private:
  /**
   * @brief Configure the preintegrator noise parameters from params_.
   */
  void setupPreintegrator();

  ImuPreintegrationParams params_;
  double info_weight_{ 1.0 };
  bool first_window_{ true };
  bool started_{ false };

  ImuState current_state_;

  // Variables at current keyframe
  vesta_variables::Orientation3DStamped orientation_i_;
  vesta_variables::Position3DStamped position_i_;
  vesta_variables::VelocityLinear3DStamped velocity_i_;
  vesta_variables::GyroscopeBias3DStamped gyro_bias_i_;
  vesta_variables::AccelerationBias3DStamped accel_bias_i_;

  ImuPreintegrator preintegrator_;
};

}  // namespace vesta_constraints
