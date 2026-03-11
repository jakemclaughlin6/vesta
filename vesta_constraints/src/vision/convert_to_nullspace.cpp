#include <vesta_constraints/vision/convert_to_nullspace.h>
#include <vesta_constraints/vision/nullspace_projection_constraint.h>
#include <vesta_constraints/vision/reprojection_error_constraint.h>
#include <vesta_constraints/vision/stereo_nullspace_projection_constraint.h>
#include <vesta_constraints/vision/stereo_reprojection_error_constraint.h>

#include <vesta_core/constraint.h>
#include <vesta_core/graph.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/pinhole_camera.h>
#include <vesta_variables/vision/stereo_camera.h>

#include <Eigen/Dense>

#include <stdexcept>
#include <string>
#include <vector>

namespace vesta_constraints
{

vesta_core::Transaction convertToNullspaceConstraints(const std::string& source,
                                                      const std::vector<vesta_core::UUID>& landmark_uuids,
                                                      const vesta_core::Graph& graph)
{
  vesta_core::Transaction transaction;

  for (const auto& landmark_uuid : landmark_uuids)
  {
    // Find all constraints connected to this landmark
    auto connected = graph.getConnectedConstraints(landmark_uuid);

    std::vector<vesta_variables::Position3DStamped> positions;
    std::vector<vesta_variables::Orientation3DStamped> orientations;
    std::vector<Eigen::Vector2d> observations;
    std::vector<vesta_core::UUID> constraints_to_remove;
    vesta_core::Matrix2d covariance;
    const vesta_variables::PinholeCamera* calibration_ptr = nullptr;

    for (const auto& constraint : connected)
    {
      const auto* reproj = dynamic_cast<const ReprojectionErrorConstraint*>(&constraint);
      if (!reproj)
      {
        continue;
      }

      // ReprojectionErrorConstraint variable order: [position, orientation, calibration, point]
      const auto& var_uuids = reproj->variables();

      // Extract the position variable
      const auto& pos_var = graph.getVariable(var_uuids[0]);
      const auto* pos = dynamic_cast<const vesta_variables::Position3DStamped*>(&pos_var);
      if (!pos)
      {
        throw std::runtime_error("Expected Position3DStamped variable");
      }

      // Extract the orientation variable
      const auto& ori_var = graph.getVariable(var_uuids[1]);
      const auto* ori = dynamic_cast<const vesta_variables::Orientation3DStamped*>(&ori_var);
      if (!ori)
      {
        throw std::runtime_error("Expected Orientation3DStamped variable");
      }

      // Extract calibration (use from first constraint)
      if (!calibration_ptr)
      {
        const auto& cal_var = graph.getVariable(var_uuids[2]);
        calibration_ptr = dynamic_cast<const vesta_variables::PinholeCamera*>(&cal_var);
        if (!calibration_ptr)
        {
          throw std::runtime_error("Expected PinholeCamera variable");
        }
        covariance = reproj->covariance();
      }

      positions.push_back(*pos);
      orientations.push_back(*ori);
      observations.push_back(reproj->mean());
      constraints_to_remove.push_back(reproj->uuid());
    }

    // Need at least 2 observations for triangulation / nullspace projection
    if (positions.size() < 2)
    {
      continue;
    }

    // Create the nullspace projection constraint
    auto nullspace_constraint = NullspaceProjectionConstraint::make_shared(source, positions, orientations,
                                                                          *calibration_ptr, observations, covariance);

    transaction.addConstraint(nullspace_constraint);

    // Remove original reprojection constraints
    for (const auto& uuid : constraints_to_remove)
    {
      transaction.removeConstraint(uuid);
    }

    // Remove the landmark variable
    transaction.removeVariable(landmark_uuid);
  }

  return transaction;
}

vesta_core::Transaction convertToStereoNullspaceConstraints(const std::string& source,
                                                            const std::vector<vesta_core::UUID>& landmark_uuids,
                                                            const vesta_core::Graph& graph)
{
  vesta_core::Transaction transaction;

  for (const auto& landmark_uuid : landmark_uuids)
  {
    auto connected = graph.getConnectedConstraints(landmark_uuid);

    std::vector<vesta_variables::Position3DStamped> positions;
    std::vector<vesta_variables::Orientation3DStamped> orientations;
    std::vector<Eigen::Vector4d> observations;
    std::vector<vesta_core::UUID> constraints_to_remove;
    vesta_core::Matrix4d covariance;
    const vesta_variables::StereoCamera* calibration_ptr = nullptr;

    for (const auto& constraint : connected)
    {
      const auto* stereo_reproj = dynamic_cast<const StereoReprojectionErrorConstraint*>(&constraint);
      if (!stereo_reproj)
      {
        continue;
      }

      // StereoReprojectionErrorConstraint variable order: [position, orientation, calibration, point]
      const auto& var_uuids = stereo_reproj->variables();

      const auto& pos_var = graph.getVariable(var_uuids[0]);
      const auto* pos = dynamic_cast<const vesta_variables::Position3DStamped*>(&pos_var);
      if (!pos)
      {
        throw std::runtime_error("Expected Position3DStamped variable");
      }

      const auto& ori_var = graph.getVariable(var_uuids[1]);
      const auto* ori = dynamic_cast<const vesta_variables::Orientation3DStamped*>(&ori_var);
      if (!ori)
      {
        throw std::runtime_error("Expected Orientation3DStamped variable");
      }

      if (!calibration_ptr)
      {
        const auto& cal_var = graph.getVariable(var_uuids[2]);
        calibration_ptr = dynamic_cast<const vesta_variables::StereoCamera*>(&cal_var);
        if (!calibration_ptr)
        {
          throw std::runtime_error("Expected StereoCamera variable");
        }
        covariance = stereo_reproj->covariance();
      }

      positions.push_back(*pos);
      orientations.push_back(*ori);
      observations.push_back(stereo_reproj->mean());
      constraints_to_remove.push_back(stereo_reproj->uuid());
    }

    if (positions.size() < 2)
    {
      continue;
    }

    auto nullspace_constraint = StereoNullspaceProjectionConstraint::make_shared(
        source, positions, orientations, *calibration_ptr, observations, covariance);

    transaction.addConstraint(nullspace_constraint);

    for (const auto& uuid : constraints_to_remove)
    {
      transaction.removeConstraint(uuid);
    }

    transaction.removeVariable(landmark_uuid);
  }

  return transaction;
}

}  // namespace vesta_constraints
