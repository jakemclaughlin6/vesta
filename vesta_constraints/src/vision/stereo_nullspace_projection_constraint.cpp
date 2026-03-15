#include <vesta_constraints/vision/stereo_nullspace_projection_constraint.h>
#include <vesta_constraints/vision/stereo_nullspace_projection_cost_function.h>

#include <boost/serialization/export.hpp>

#include <cassert>
#include <string>
#include <vector>

namespace
{

std::vector<vesta_core::UUID> buildVariableUuids(const std::vector<vesta_variables::Position3DStamped>& positions,
                                                 const std::vector<vesta_variables::Orientation3DStamped>& orientations)
{
  std::vector<vesta_core::UUID> uuids;
  uuids.reserve(2 * positions.size());
  for (size_t i = 0; i < positions.size(); ++i)
  {
    uuids.push_back(positions[i].uuid());
    uuids.push_back(orientations[i].uuid());
  }
  return uuids;
}

}  // namespace

namespace vesta_constraints
{

// Private delegating constructor
StereoNullspaceProjectionConstraint::StereoNullspaceProjectionConstraint(
    const std::string& source, std::vector<vesta_core::UUID> variable_uuids,
    const std::vector<Eigen::Vector4d>& observations, const vesta_core::Matrix4d& covariance,
    const Eigen::Matrix<double, 5, 1>& calibration)
  : vesta_core::Constraint(source, variable_uuids.begin(), variable_uuids.end())
  , observations_(observations)
  , sqrt_information_(covariance.inverse().llt().matrixU())
  , calibration_(calibration)
{
}

// Public constructor: builds UUIDs then delegates
StereoNullspaceProjectionConstraint::StereoNullspaceProjectionConstraint(
    const std::string& source, const std::vector<vesta_variables::Position3DStamped>& positions,
    const std::vector<vesta_variables::Orientation3DStamped>& orientations,
    const vesta_variables::StereoCamera& calibration, const std::vector<Eigen::Vector4d>& observations,
    const vesta_core::Matrix4d& covariance)
  : StereoNullspaceProjectionConstraint(source, buildVariableUuids(positions, orientations), observations, covariance,
                                        (Eigen::Matrix<double, 5, 1>() << calibration.data()[0], calibration.data()[1],
                                         calibration.data()[2], calibration.data()[3], calibration.data()[4])
                                            .finished())
{
  assert(positions.size() == orientations.size());
  assert(positions.size() == observations.size());
  assert(positions.size() >= 2);
}

// Public constructor with extrinsic calibration
StereoNullspaceProjectionConstraint::StereoNullspaceProjectionConstraint(
    const std::string& source, const std::vector<vesta_variables::Position3DStamped>& positions,
    const std::vector<vesta_variables::Orientation3DStamped>& orientations,
    const vesta_variables::StereoCamera& calibration, const std::vector<Eigen::Vector4d>& observations,
    const vesta_core::Matrix4d& covariance, const vesta_variables::Extrinsic3DPosition& ext_position,
    const vesta_variables::Extrinsic3DOrientation& ext_orientation)
  : StereoNullspaceProjectionConstraint(
        source,
        [&]() {
          auto uuids = buildVariableUuids(positions, orientations);
          uuids.push_back(ext_position.uuid());
          uuids.push_back(ext_orientation.uuid());
          return uuids;
        }(),
        observations, covariance,
        (Eigen::Matrix<double, 5, 1>() << calibration.data()[0], calibration.data()[1], calibration.data()[2],
         calibration.data()[3], calibration.data()[4])
            .finished())
{
  assert(positions.size() == orientations.size());
  assert(positions.size() == observations.size());
  assert(positions.size() >= 2);
  has_extrinsic_ = true;
}

void StereoNullspaceProjectionConstraint::print(std::ostream& stream) const
{
  stream << type() << "\n"
         << "  source: " << source() << "\n"
         << "  uuid: " << uuid() << "\n"
         << "  observations: " << numObservations() << "\n"
         << "  calibration: [" << calibration_[0] << ", " << calibration_[1] << ", " << calibration_[2] << ", "
         << calibration_[3] << ", " << calibration_[4] << "]\n"
         << "  sqrt_info:\n"
         << sqrt_information_ << "\n";

  for (size_t i = 0; i < numObservations(); ++i)
  {
    stream << "  pose " << i << ": pos=" << variables().at(2 * i) << " ori=" << variables().at(2 * i + 1) << "\n";
    stream << "    observation: " << observations_[i].transpose() << "\n";
  }

  if (has_extrinsic_)
  {
    const size_t ext_base = 2 * numObservations();
    stream << "  ext_position variable: " << variables().at(ext_base) << "\n"
           << "  ext_orientation variable: " << variables().at(ext_base + 1) << "\n";
  }

  if (loss())
  {
    stream << "  loss: ";
    loss()->print(stream);
  }
}

ceres::CostFunction* StereoNullspaceProjectionConstraint::costFunction() const
{
  return new StereoNullspaceProjectionCostFunction(observations_, sqrt_information_, calibration_, has_extrinsic_);
}

}  // namespace vesta_constraints

BOOST_CLASS_EXPORT_IMPLEMENT(vesta_constraints::StereoNullspaceProjectionConstraint);
