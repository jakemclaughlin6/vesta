#pragma once

#include <vesta_core/constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/fuse_macros.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_variables/3d/extrinsic_3d_orientation.h>
#include <vesta_variables/3d/extrinsic_3d_position.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/pinhole_camera.h>

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
 * @brief A structureless vision constraint that eliminates landmark variables via nullspace projection.
 *
 * Given N >= 2 observations of a single 3D landmark from N camera poses, this constraint
 * analytically marginalizes the landmark by projecting the reprojection errors through
 * the left nullspace of the landmark Jacobian. The result is a (2N-3)-dimensional residual
 * that constrains only the N camera poses — no landmark variable is needed.
 *
 * This is equivalent to the "smart factor" in GTSAM or the MSCKF measurement update.
 *
 * Variable ordering: [pos_0, ori_0, pos_1, ori_1, ..., pos_{N-1}, ori_{N-1}]
 * Calibration is stored internally (fixed).
 */
class NullspaceProjectionConstraint : public vesta_core::Constraint
{
public:
  VESTA_CONSTRAINT_DEFINITIONS(NullspaceProjectionConstraint);

  /**
   * @brief Default constructor
   */
  NullspaceProjectionConstraint() = default;

  /**
   * @brief Create a nullspace projection constraint.
   *
   * @param[in] source        The name of the sensor or motion model that generated this constraint
   * @param[in] positions     The position variables for each observing camera pose (N >= 2)
   * @param[in] orientations  The orientation variables for each observing camera pose
   * @param[in] calibration   The pinhole camera calibration (stored internally, not optimized)
   * @param[in] observations  The 2D pixel observations (u, v) for each camera pose
   * @param[in] covariance    The observation noise covariance (shared across all observations)
   */
  NullspaceProjectionConstraint(const std::string& source,
                                const std::vector<vesta_variables::Position3DStamped>& positions,
                                const std::vector<vesta_variables::Orientation3DStamped>& orientations,
                                const vesta_variables::PinholeCamera& calibration,
                                const std::vector<Eigen::Vector2d>& observations,
                                const vesta_core::Matrix2d& covariance);

  /**
   * @brief Create a nullspace projection constraint with extrinsic calibration.
   *
   * This constructor accepts an extrinsic transform T_body_sensor that maps points from sensor
   * frame to body frame. The position and orientation variables represent the body frame, and
   * the extrinsic is applied internally to compute the camera-frame pose.
   *
   * @param[in] source          The name of the sensor or motion model that generated this constraint
   * @param[in] positions       The position variables for each observing body pose (N >= 2)
   * @param[in] orientations    The orientation variables for each observing body pose
   * @param[in] calibration     The pinhole camera calibration (stored internally, not optimized)
   * @param[in] observations    The 2D pixel observations (u, v) for each camera pose
   * @param[in] covariance      The observation noise covariance (shared across all observations)
   * @param[in] ext_position    The extrinsic translation (body-to-sensor)
   * @param[in] ext_orientation The extrinsic rotation (body-to-sensor)
   */
  NullspaceProjectionConstraint(const std::string& source,
                                const std::vector<vesta_variables::Position3DStamped>& positions,
                                const std::vector<vesta_variables::Orientation3DStamped>& orientations,
                                const vesta_variables::PinholeCamera& calibration,
                                const std::vector<Eigen::Vector2d>& observations,
                                const vesta_core::Matrix2d& covariance,
                                const vesta_variables::Extrinsic3DPosition& ext_position,
                                const vesta_variables::Extrinsic3DOrientation& ext_orientation);

  /**
   * @brief Returns whether this constraint uses an extrinsic calibration.
   */
  bool hasExtrinsic() const
  {
    return has_extrinsic_;
  }

  ~NullspaceProjectionConstraint() override = default;

  /**
   * @brief Read-only access to the stored observations.
   */
  const std::vector<Eigen::Vector2d>& observations() const
  {
    return observations_;
  }

  /**
   * @brief Read-only access to the square root information matrix.
   */
  const vesta_core::Matrix2d& sqrtInformation() const
  {
    return sqrt_information_;
  }

  /**
   * @brief Read-only access to the stored calibration [fx, fy, cx, cy].
   */
  const Eigen::Vector4d& calibration() const
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
  vesta_core::Matrix2d covariance() const
  {
    return (sqrt_information_.transpose() * sqrt_information_).inverse();
  }

  void print(std::ostream& stream = std::cout) const override;

  ceres::CostFunction* costFunction() const override;

protected:
  std::vector<Eigen::Vector2d> observations_;
  vesta_core::Matrix2d sqrt_information_;
  Eigen::Vector4d calibration_;
  bool has_extrinsic_{ false };  //!< Whether this constraint uses an extrinsic calibration

private:
  // Delegating constructor that takes a pre-built UUID vector
  NullspaceProjectionConstraint(const std::string& source, std::vector<vesta_core::UUID> variable_uuids,
                                const std::vector<Eigen::Vector2d>& observations,
                                const vesta_core::Matrix2d& covariance, const Eigen::Vector4d& calibration);

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& archive, const unsigned int /* version */)
  {
    archive& boost::serialization::base_object<vesta_core::Constraint>(*this);
    archive & observations_;
    archive & sqrt_information_;
    archive & calibration_;
    archive & has_extrinsic_;
  }
};

}  // namespace vesta_constraints

BOOST_CLASS_EXPORT_KEY(vesta_constraints::NullspaceProjectionConstraint);
