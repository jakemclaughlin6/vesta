#include <vesta_constraints/inertial/imu_preintegration.h>

#include <glog/logging.h>

#include <stdexcept>

namespace vesta_constraints
{

ImuPreintegration::ImuPreintegration(const ImuPreintegrationParams& params, double info_weight)
  : params_(params), info_weight_(info_weight)
{
  if (params_.cov_prior_noise <= 0.0)
  {
    LOG(ERROR) << "Prior noise on IMU state must be greater than zero, value: " << params_.cov_prior_noise;
    throw std::invalid_argument("ImuPreintegrationParams::cov_prior_noise must be > 0");
  }
  setupPreintegrator();
}

void ImuPreintegration::setupPreintegrator()
{
  preintegrator_.cov_gyro = params_.cov_gyro_noise;
  preintegrator_.cov_accel = params_.cov_accel_noise;
  preintegrator_.cov_gyro_bias = params_.cov_gyro_bias;
  preintegrator_.cov_accel_bias = params_.cov_accel_bias;
}

void ImuPreintegration::addImuData(const ImuData& data)
{
  preintegrator_.data.emplace(data.stamp, data);
}

void ImuPreintegration::setStart(
    const vesta_core::Timestamp& stamp,
    const Eigen::Quaterniond& orientation,
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity,
    const vesta_core::UUID& device_id)
{
  // Clear data before the start time and reset preintegrator
  preintegrator_.clearBefore(stamp);
  preintegrator_.reset();

  // Set internal state
  current_state_.stamp = stamp;
  current_state_.orientation = orientation.normalized();
  current_state_.position = position;
  current_state_.velocity = velocity;
  // Keep existing biases (set via setBiases or default zero)

  // Create variables at the start keyframe
  orientation_i_ = vesta_variables::Orientation3DStamped(stamp, device_id);
  // Orientation3DStamped uses [W, X, Y, Z] ordering (Ceres convention)
  orientation_i_.w() = current_state_.orientation.w();
  orientation_i_.x() = current_state_.orientation.x();
  orientation_i_.y() = current_state_.orientation.y();
  orientation_i_.z() = current_state_.orientation.z();

  position_i_ = vesta_variables::Position3DStamped(stamp, device_id);
  position_i_.x() = current_state_.position.x();
  position_i_.y() = current_state_.position.y();
  position_i_.z() = current_state_.position.z();

  velocity_i_ = vesta_variables::VelocityLinear3DStamped(stamp, device_id);
  velocity_i_.x() = current_state_.velocity.x();
  velocity_i_.y() = current_state_.velocity.y();
  velocity_i_.z() = current_state_.velocity.z();

  gyro_bias_i_ = vesta_variables::GyroscopeBias3DStamped(stamp, device_id);
  gyro_bias_i_.x() = current_state_.gyro_bias.x();
  gyro_bias_i_.y() = current_state_.gyro_bias.y();
  gyro_bias_i_.z() = current_state_.gyro_bias.z();

  accel_bias_i_ = vesta_variables::AccelerationBias3DStamped(stamp, device_id);
  accel_bias_i_.x() = current_state_.accel_bias.x();
  accel_bias_i_.y() = current_state_.accel_bias.y();
  accel_bias_i_.z() = current_state_.accel_bias.z();

  started_ = true;
}

void ImuPreintegration::setBiases(const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& accel_bias)
{
  current_state_.gyro_bias = gyro_bias;
  current_state_.accel_bias = accel_bias;

  if (started_)
  {
    gyro_bias_i_.x() = gyro_bias.x();
    gyro_bias_i_.y() = gyro_bias.y();
    gyro_bias_i_.z() = gyro_bias.z();

    accel_bias_i_.x() = accel_bias.x();
    accel_bias_i_.y() = accel_bias.y();
    accel_bias_i_.z() = accel_bias.z();
  }
}

ImuState ImuPreintegration::predictState(const vesta_core::Timestamp& stamp) const
{
  if (!started_)
  {
    LOG(WARNING) << "ImuPreintegration::predictState called before setStart()";
    return current_state_;
  }

  // Make a temporary copy of the preintegrator for read-only prediction
  ImuPreintegrator temp_preintegrator = preintegrator_;
  temp_preintegrator.integrate(stamp, current_state_.gyro_bias, current_state_.accel_bias,
                               false, false, false);

  const double dt = temp_preintegrator.delta.dt;
  const Eigen::Matrix3d R_curr = current_state_.orientation.toRotationMatrix();

  // Predict new state using preintegrated measurements
  // q_new = q_curr * delta_q
  Eigen::Quaterniond q_new(R_curr * temp_preintegrator.delta.q.toRotationMatrix());
  q_new.normalize();

  // v_new = v_curr + g * dt + R_curr * delta_v
  Eigen::Vector3d v_new = current_state_.velocity + params_.gravity * dt
                          + R_curr * temp_preintegrator.delta.v;

  // p_new = p_curr + v_curr * dt + 0.5 * g * dt^2 + R_curr * delta_p
  Eigen::Vector3d p_new = current_state_.position + current_state_.velocity * dt
                          + 0.5 * params_.gravity * dt * dt
                          + R_curr * temp_preintegrator.delta.p;

  ImuState predicted;
  predicted.stamp = stamp;
  predicted.orientation = q_new;
  predicted.position = p_new;
  predicted.velocity = v_new;
  predicted.gyro_bias = current_state_.gyro_bias;
  predicted.accel_bias = current_state_.accel_bias;

  return predicted;
}

ImuPreintegration::PreintegrationResult ImuPreintegration::createPreintegratedFactor(
    const vesta_core::Timestamp& stamp,
    const vesta_core::UUID& device_id,
    bool add_prior_on_first_window)
{
  PreintegrationResult result;

  if (!started_)
  {
    LOG(ERROR) << "ImuPreintegration::createPreintegratedFactor called before setStart()";
    throw std::runtime_error("ImuPreintegration::createPreintegratedFactor called before setStart()");
  }

  if (preintegrator_.data.empty())
  {
    LOG(WARNING) << "Cannot create preintegrated factor, no IMU data is available.";
    throw std::runtime_error("Cannot create preintegrated factor, no IMU data is available.");
  }

  if (stamp < preintegrator_.data.begin()->first)
  {
    LOG(WARNING) << "Cannot create preintegrated factor, requested time is prior to the front of the window.";
    throw std::runtime_error("Requested time is prior to the front of the IMU data window.");
  }

  // Generate prior constraint on the first window if requested
  if (first_window_ && add_prior_on_first_window)
  {
    Eigen::Matrix<double, 15, 15> prior_covariance =
        params_.cov_prior_noise * Eigen::Matrix<double, 15, 15>::Identity();

    // Build the 16x1 mean vector: [qw, qx, qy, qz, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]
    Eigen::Matrix<double, 16, 1> mean;
    mean << current_state_.orientation.w(),
            current_state_.orientation.x(),
            current_state_.orientation.y(),
            current_state_.orientation.z(),
            current_state_.position,
            current_state_.velocity,
            current_state_.gyro_bias,
            current_state_.accel_bias;

    result.prior_constraint = std::make_shared<AbsoluteImuState3DStampedConstraint>(
        params_.source,
        orientation_i_, position_i_, velocity_i_, gyro_bias_i_, accel_bias_i_,
        mean, prior_covariance);

    first_window_ = false;
  }

  // Integrate between keyframes, computing jacobians, covariance, and sqrt information
  preintegrator_.integrate(stamp, current_state_.gyro_bias, current_state_.accel_bias,
                           true, true, true);

  // Predict state at end of window
  const double dt = preintegrator_.delta.dt;
  const Eigen::Matrix3d R_curr = current_state_.orientation.toRotationMatrix();

  Eigen::Quaterniond q_new(R_curr * preintegrator_.delta.q.toRotationMatrix());
  q_new.normalize();

  Eigen::Vector3d v_new = current_state_.velocity + params_.gravity * dt
                          + R_curr * preintegrator_.delta.v;

  Eigen::Vector3d p_new = current_state_.position + current_state_.velocity * dt
                          + 0.5 * params_.gravity * dt * dt
                          + R_curr * preintegrator_.delta.p;

  result.predicted_state.stamp = stamp;
  result.predicted_state.orientation = q_new;
  result.predicted_state.position = p_new;
  result.predicted_state.velocity = v_new;
  result.predicted_state.gyro_bias = current_state_.gyro_bias;
  result.predicted_state.accel_bias = current_state_.accel_bias;

  // Create new variables at the predicted state
  auto orientation_j = std::make_shared<vesta_variables::Orientation3DStamped>(stamp, device_id);
  orientation_j->w() = q_new.w();
  orientation_j->x() = q_new.x();
  orientation_j->y() = q_new.y();
  orientation_j->z() = q_new.z();

  auto position_j = std::make_shared<vesta_variables::Position3DStamped>(stamp, device_id);
  position_j->x() = p_new.x();
  position_j->y() = p_new.y();
  position_j->z() = p_new.z();

  auto velocity_j = std::make_shared<vesta_variables::VelocityLinear3DStamped>(stamp, device_id);
  velocity_j->x() = v_new.x();
  velocity_j->y() = v_new.y();
  velocity_j->z() = v_new.z();

  auto gyro_bias_j = std::make_shared<vesta_variables::GyroscopeBias3DStamped>(stamp, device_id);
  gyro_bias_j->x() = current_state_.gyro_bias.x();
  gyro_bias_j->y() = current_state_.gyro_bias.y();
  gyro_bias_j->z() = current_state_.gyro_bias.z();

  auto accel_bias_j = std::make_shared<vesta_variables::AccelerationBias3DStamped>(stamp, device_id);
  accel_bias_j->x() = current_state_.accel_bias.x();
  accel_bias_j->y() = current_state_.accel_bias.y();
  accel_bias_j->z() = current_state_.accel_bias.z();

  // Create relative constraint between keyframes i and j
  result.relative_constraint = std::make_shared<RelativeImuState3DStampedConstraint>(
      params_.source,
      orientation_i_, position_i_, velocity_i_, gyro_bias_i_, accel_bias_i_,
      *orientation_j, *position_j, *velocity_j, *gyro_bias_j, *accel_bias_j,
      preintegrator_,
      current_state_.gyro_bias, current_state_.accel_bias,
      params_.gravity, info_weight_);

  // Store new variables in result
  result.orientation = orientation_j;
  result.position = position_j;
  result.velocity = velocity_j;
  result.gyro_bias = gyro_bias_j;
  result.accel_bias = accel_bias_j;

  // Advance internal state: current keyframe = predicted state at stamp
  current_state_ = result.predicted_state;

  orientation_i_ = *orientation_j;
  position_i_ = *position_j;
  velocity_i_ = *velocity_j;
  gyro_bias_i_ = *gyro_bias_j;
  accel_bias_i_ = *accel_bias_j;

  // Clear and reset preintegrator for the next window
  preintegrator_.clearBefore(stamp);
  preintegrator_.reset();

  return result;
}

void ImuPreintegration::updateState(const ImuState& state)
{
  current_state_ = state;

  // Update variable data arrays to match new state
  orientation_i_.w() = state.orientation.w();
  orientation_i_.x() = state.orientation.x();
  orientation_i_.y() = state.orientation.y();
  orientation_i_.z() = state.orientation.z();

  position_i_.x() = state.position.x();
  position_i_.y() = state.position.y();
  position_i_.z() = state.position.z();

  velocity_i_.x() = state.velocity.x();
  velocity_i_.y() = state.velocity.y();
  velocity_i_.z() = state.velocity.z();

  gyro_bias_i_.x() = state.gyro_bias.x();
  gyro_bias_i_.y() = state.gyro_bias.y();
  gyro_bias_i_.z() = state.gyro_bias.z();

  accel_bias_i_.x() = state.accel_bias.x();
  accel_bias_i_.y() = state.accel_bias.y();
  accel_bias_i_.z() = state.accel_bias.z();
}

void ImuPreintegration::clearDataBefore(const vesta_core::Timestamp& stamp)
{
  preintegrator_.clearBefore(stamp);
}

void ImuPreintegration::reset()
{
  preintegrator_.data.clear();
  preintegrator_.reset();
  first_window_ = true;
  started_ = false;
  current_state_ = ImuState{};
}

}  // namespace vesta_constraints
