#include <vesta_constraints/inertial/absolute_imu_state_3d_stamped_constraint.h>

#include <vesta_constraints/inertial/normal_prior_imu_state_3d_cost_functor.h>

#include <boost/serialization/export.hpp>
#include <ceres/autodiff_cost_function.h>
#include <Eigen/Dense>

#include <string>


namespace vesta_constraints
{

AbsoluteImuState3DStampedConstraint::AbsoluteImuState3DStampedConstraint(
  const std::string& source,
  const vesta_variables::Orientation3DStamped& orientation,
  const vesta_variables::Position3DStamped& position,
  const vesta_variables::VelocityLinear3DStamped& velocity,
  const vesta_variables::GyroscopeBias3DStamped& gyro_bias,
  const vesta_variables::AccelerationBias3DStamped& accel_bias,
  const Eigen::Matrix<double, 16, 1>& mean,
  const Eigen::Matrix<double, 15, 15>& covariance) :
    vesta_core::Constraint(
      source,
      {orientation.uuid(), position.uuid(), velocity.uuid(),  // NOLINT(whitespace/braces)
       gyro_bias.uuid(), accel_bias.uuid()}),
    mean_(mean),
    sqrt_information_(covariance.inverse().llt().matrixU())
{
}

void AbsoluteImuState3DStampedConstraint::print(std::ostream& stream) const
{
  stream << type() << "\n"
         << "  source: " << source() << "\n"
         << "  uuid: " << uuid() << "\n"
         << "  orientation variable: " << variables().at(0) << "\n"
         << "  position variable: " << variables().at(1) << "\n"
         << "  velocity variable: " << variables().at(2) << "\n"
         << "  gyro_bias variable: " << variables().at(3) << "\n"
         << "  accel_bias variable: " << variables().at(4) << "\n"
         << "  mean: " << mean().transpose() << "\n"
         << "  sqrt_info: " << sqrtInformation() << "\n";

  if (loss())
  {
    stream << "  loss: ";
    loss()->print(stream);
  }
}

ceres::CostFunction* AbsoluteImuState3DStampedConstraint::costFunction() const
{
  return new ceres::AutoDiffCostFunction<NormalPriorImuState3DCostFunctor, 15, 4, 3, 3, 3, 3>(
    new NormalPriorImuState3DCostFunctor(sqrt_information_, mean_));
}

}  // namespace vesta_constraints

BOOST_CLASS_EXPORT_IMPLEMENT(vesta_constraints::AbsoluteImuState3DStampedConstraint);
