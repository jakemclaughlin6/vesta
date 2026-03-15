/*
 * Software License Agreement (BSD License)
 *
 *  Author: Oscar Mendez
 *  Created on Mon Nov 12 2023
 *
 *  Copyright (c) 2023, Locus Robotics
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
#include <vesta_constraints/vision/fixed_3d_landmark_constraint.h>

#include <vesta_constraints/vision/fixed_3d_landmark_cost_functor.h>
#include <vesta_constraints/vision/fixed_3d_landmark_with_extrinsic_cost_functor.h>

#include <ceres/autodiff_cost_function.h>
#include <Eigen/Dense>
#include <boost/serialization/export.hpp>

#include <string>

namespace vesta_constraints
{

Fixed3DLandmarkConstraint::Fixed3DLandmarkConstraint(
    const std::string& source, const vesta_variables::Position3DStamped& position,
    const vesta_variables::Orientation3DStamped& orientation, const vesta_variables::PinholeCamera& calibration,
    const vesta_core::MatrixXd& pts3d, const vesta_core::MatrixXd& observations, const vesta_core::Vector7d& mean,
    const vesta_core::Matrix6d& covariance)
  : vesta_core::Constraint(source, { position.uuid(), orientation.uuid(), calibration.uuid() })
  , pts3d_(pts3d)
  , observations_(observations)
  , mean_(mean)
  , sqrt_information_(covariance.inverse().llt().matrixU())
{
  assert(pts3d_.cols() == 3);
  assert(observations_.cols() == 2);
  assert(pts3d_.rows() == observations_.rows());
}

Fixed3DLandmarkConstraint::Fixed3DLandmarkConstraint(
    const std::string& source, const vesta_variables::Position3DStamped& position,
    const vesta_variables::Orientation3DStamped& orientation, const vesta_variables::PinholeCamera& calibration,
    const double& marker_size, const vesta_core::MatrixXd& observations, const vesta_core::Vector7d& mean,
    const vesta_core::Matrix6d& covariance)
  : vesta_core::Constraint(source, { position.uuid(), orientation.uuid(), calibration.uuid() })
  , pts3d_(4, 3)
  , observations_(observations)
  , mean_(mean)
  , sqrt_information_(covariance.inverse().llt().matrixU())
{
  // Define 3D Homogeneous 3D Points at origin, assume z-up
  pts3d_ << -1.0, -1.0, 0.0,  // NOLINT
      -1.0, 1.0, 0.0,         // NOLINT
      1.0, -1.0, 0.0,         // NOLINT
      1.0, 1.0, 0.0;          // NOLINT
  pts3d_ *= marker_size;      // Scalar Multiplication

  assert(pts3d_.cols() == 3);
  assert(observations_.cols() == 2);
  assert(pts3d_.rows() == observations_.rows());
}

Fixed3DLandmarkConstraint::Fixed3DLandmarkConstraint(
    const std::string& source, const vesta_variables::Position3DStamped& position,
    const vesta_variables::Orientation3DStamped& orientation, const vesta_variables::PinholeCamera& calibration,
    const vesta_core::MatrixXd& pts3d, const vesta_core::MatrixXd& observations,
    const vesta_variables::Extrinsic3DPosition& ext_position,
    const vesta_variables::Extrinsic3DOrientation& ext_orientation, const vesta_core::Vector7d& mean,
    const vesta_core::Matrix6d& covariance)
  : vesta_core::Constraint(source, { position.uuid(), orientation.uuid(), calibration.uuid(), ext_position.uuid(),
                                     ext_orientation.uuid() })
  , pts3d_(pts3d)
  , observations_(observations)
  , mean_(mean)
  , sqrt_information_(covariance.inverse().llt().matrixU())
  , has_extrinsic_(true)
{
  assert(pts3d_.cols() == 3);
  assert(observations_.cols() == 2);
  assert(pts3d_.rows() == observations_.rows());
}

void Fixed3DLandmarkConstraint::print(std::ostream& stream) const
{
  stream << type() << "\n"
         << "  source: " << source() << "\n"
         << "  uuid: " << uuid() << "\n"
         << "  position variable: " << variables().at(0) << "\n"
         << "  orientation variable: " << variables().at(1) << "\n"
         << "  calibration variable: " << variables().at(2) << "\n";

  if (has_extrinsic_)
  {
    stream << "  ext_position variable: " << variables().at(3) << "\n"
           << "  ext_orientation variable: " << variables().at(4) << "\n";
  }

  stream << "  mean: " << mean().transpose() << "\n"
         << "  sqrt_info: " << sqrtInformation() << "\n"
         << "  observations: " << observations() << "\n";

  if (loss())
  {
    stream << "  loss: ";
    loss()->print(stream);
  }
}

ceres::CostFunction* Fixed3DLandmarkConstraint::costFunction() const
{
  if (has_extrinsic_)
  {
    // 2 Residuals Per 3D point
    return new ceres::AutoDiffCostFunction<Fixed3DLandmarkWithExtrinsicCostFunctor, ceres::DYNAMIC, 3, 4, 4, 3, 4>(
        new Fixed3DLandmarkWithExtrinsicCostFunctor(sqrt_information_, mean_, observations_, pts3d_),
        2 * pts3d_.rows());
  }

  // 2 Residuals Per 3D point
  return new ceres::AutoDiffCostFunction<Fixed3DLandmarkCostFunctor, ceres::DYNAMIC, 3, 4, 4>(
      new Fixed3DLandmarkCostFunctor(sqrt_information_, mean_, observations_, pts3d_), 2 * pts3d_.rows());
}

}  // namespace vesta_constraints

BOOST_CLASS_EXPORT_IMPLEMENT(vesta_constraints::Fixed3DLandmarkConstraint);
