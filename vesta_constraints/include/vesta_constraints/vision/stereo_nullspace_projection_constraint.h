#pragma once

#include <vesta_core/constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/stereo_camera.h>

#include <Eigen/Dense>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>

#include <ostream>
#include <string>
#include <vector>

namespace vesta_constraints
{

/**
 * @brief A structureless stereo vision constraint that eliminates landmark variables via nullspace projection.
 *
 * Given N >= 2 stereo observations of a single 3D landmark from N camera poses, this constraint
 * analytically marginalizes the landmark by projecting the stereo reprojection errors through
 * the left nullspace of the landmark Jacobian. The result is a (4N-3)-dimensional residual
 * that constrains only the N camera poses -- no landmark variable is needed.
 *
 * Each stereo observation provides 4 measurements (u_left, v_left, u_right, v_right).
 * Unlike the mono variant, stereo provides scale observability through the known baseline.
 *
 * Variable ordering: [pos_0, ori_0, pos_1, ori_1, ..., pos_{N-1}, ori_{N-1}]
 * Calibration (fx, fy, cx, cy, baseline) is stored internally (fixed).
 */
class StereoNullspaceProjectionConstraint : public vesta_core::Constraint
{
public:
  VESTA_CONSTRAINT_DEFINITIONS(StereoNullspaceProjectionConstraint);

  /**
   * @brief Default constructor
   */
  StereoNullspaceProjectionConstraint() = default;

  /**
   * @brief Create a stereo nullspace projection constraint.
   *
   * @param[in] source        The name of the sensor or motion model that generated this constraint
   * @param[in] positions     The position variables for each observing camera pose (N >= 2)
   * @param[in] orientations  The orientation variables for each observing camera pose
   * @param[in] calibration   The stereo camera calibration (stored internally, not optimized)
   * @param[in] observations  The stereo pixel observations (u_l, v_l, u_r, v_r) for each camera pose
   * @param[in] covariance    The observation noise covariance (4x4, shared across all observations)
   */
  StereoNullspaceProjectionConstraint(const std::string& source,
                                      const std::vector<vesta_variables::Position3DStamped>& positions,
                                      const std::vector<vesta_variables::Orientation3DStamped>& orientations,
                                      const vesta_variables::StereoCamera& calibration,
                                      const std::vector<Eigen::Vector4d>& observations,
                                      const vesta_core::Matrix4d& covariance);

  ~StereoNullspaceProjectionConstraint() override = default;

  /**
   * @brief Read-only access to the stored observations.
   */
  const std::vector<Eigen::Vector4d>& observations() const
  {
    return observations_;
  }

  /**
   * @brief Read-only access to the square root information matrix.
   */
  const vesta_core::Matrix4d& sqrtInformation() const
  {
    return sqrt_information_;
  }

  /**
   * @brief Read-only access to the stored calibration [fx, fy, cx, cy, baseline].
   */
  const Eigen::Matrix<double, 5, 1>& calibration() const
  {
    return calibration_;
  }

  /**
   * @brief The number of camera observations.
   */
  size_t numObservations() const
  {
    return observations_.size();
  }

  /**
   * @brief Compute the measurement covariance matrix.
   */
  vesta_core::Matrix4d covariance() const
  {
    return (sqrt_information_.transpose() * sqrt_information_).inverse();
  }

  void print(std::ostream& stream = std::cout) const override;

  ceres::CostFunction* costFunction() const override;

protected:
  std::vector<Eigen::Vector4d> observations_;
  vesta_core::Matrix4d sqrt_information_;
  Eigen::Matrix<double, 5, 1> calibration_;

private:
  // Delegating constructor that takes a pre-built UUID vector
  StereoNullspaceProjectionConstraint(const std::string& source, std::vector<vesta_core::UUID> variable_uuids,
                                      const std::vector<Eigen::Vector4d>& observations,
                                      const vesta_core::Matrix4d& covariance,
                                      const Eigen::Matrix<double, 5, 1>& calibration);

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Constraint>(*this);
    archive& observations_;
    archive& sqrt_information_;
    archive& calibration_;
  }
};

}  // namespace vesta_constraints

BOOST_CLASS_EXPORT_KEY(vesta_constraints::StereoNullspaceProjectionConstraint);
