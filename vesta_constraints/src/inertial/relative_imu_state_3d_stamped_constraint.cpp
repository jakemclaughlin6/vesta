#include <vesta_constraints/inertial/relative_imu_state_3d_stamped_constraint.h>

#include <vesta_constraints/inertial/normal_delta_imu_state_3d_cost_functor.h>

#include <Eigen/Dense>
#include <boost/serialization/export.hpp>
#include <ceres/autodiff_cost_function.h>

#include <string>

namespace vesta_constraints {

RelativeImuState3DStampedConstraint::RelativeImuState3DStampedConstraint(
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
    const Eigen::Vector3d &linearization_ba, const Eigen::Vector3d &gravity,
    double info_weight)
    : vesta_core::Constraint(
          source, {orientation1.uuid(), position1.uuid(),
                   velocity1.uuid(), // NOLINT(whitespace/braces)
                   gyro_bias1.uuid(), accel_bias1.uuid(), orientation2.uuid(),
                   position2.uuid(), velocity2.uuid(), gyro_bias2.uuid(),
                   accel_bias2.uuid()}),
      delta_q_(preintegrator.delta.q), delta_p_(preintegrator.delta.p),
      delta_v_(preintegrator.delta.v), dt_(preintegrator.delta.dt),
      sqrt_information_(info_weight * preintegrator.delta.sqrt_information),
      linearization_bg_(linearization_bg), linearization_ba_(linearization_ba),
      gravity_(gravity), dq_dbg_(preintegrator.jacobian.dq_dbg),
      dp_dbg_(preintegrator.jacobian.dp_dbg),
      dp_dba_(preintegrator.jacobian.dp_dba),
      dv_dbg_(preintegrator.jacobian.dv_dbg),
      dv_dba_(preintegrator.jacobian.dv_dba) {}

void RelativeImuState3DStampedConstraint::print(std::ostream &stream) const {
  stream << type() << "\n"
         << "  source: " << source() << "\n"
         << "  uuid: " << uuid() << "\n"
         << "  orientation1 variable: " << variables().at(0) << "\n"
         << "  position1 variable: " << variables().at(1) << "\n"
         << "  velocity1 variable: " << variables().at(2) << "\n"
         << "  gyro_bias1 variable: " << variables().at(3) << "\n"
         << "  accel_bias1 variable: " << variables().at(4) << "\n"
         << "  orientation2 variable: " << variables().at(5) << "\n"
         << "  position2 variable: " << variables().at(6) << "\n"
         << "  velocity2 variable: " << variables().at(7) << "\n"
         << "  gyro_bias2 variable: " << variables().at(8) << "\n"
         << "  accel_bias2 variable: " << variables().at(9) << "\n"
         << "  dt: " << dt_ << "\n"
         << "  delta_q: [" << delta_q_.w() << ", " << delta_q_.x() << ", "
         << delta_q_.y() << ", " << delta_q_.z() << "]\n"
         << "  delta_p: " << delta_p_.transpose() << "\n"
         << "  delta_v: " << delta_v_.transpose() << "\n"
         << "  sqrt_info: " << sqrtInformation() << "\n";

  if (loss()) {
    stream << "  loss: ";
    loss()->print(stream);
  }
}

ceres::CostFunction *RelativeImuState3DStampedConstraint::costFunction() const {
  return new ceres::AutoDiffCostFunction<NormalDeltaImuState3DCostFunctor, 15,
                                         4, 3, 3, 3, 3, 4, 3, 3, 3, 3>(
      new NormalDeltaImuState3DCostFunctor(
          sqrt_information_, delta_q_, delta_p_, delta_v_, dt_, gravity_,
          linearization_bg_, linearization_ba_, dq_dbg_, dp_dbg_, dp_dba_,
          dv_dbg_, dv_dba_));
}

} // namespace vesta_constraints

BOOST_CLASS_EXPORT_IMPLEMENT(
    vesta_constraints::RelativeImuState3DStampedConstraint);
