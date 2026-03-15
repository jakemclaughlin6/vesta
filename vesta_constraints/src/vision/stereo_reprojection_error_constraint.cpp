/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2024, Locus Robotics
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
#include <vesta_constraints/vision/stereo_reprojection_error_constraint.h>
#include <vesta_constraints/vision/cost_functions/stereo_reprojection_error_cost_functor.h>
#include <vesta_constraints/vision/cost_functions/stereo_reprojection_error_with_extrinsic_cost_functor.h>

#include <ceres/autodiff_cost_function.h>
#include <Eigen/Dense>
#include <boost/serialization/export.hpp>

#include <string>

namespace vesta_constraints
{

StereoReprojectionErrorConstraint::StereoReprojectionErrorConstraint(
    const std::string& source, const vesta_variables::Position3DStamped& position,
    const vesta_variables::Orientation3DStamped& orientation, const vesta_variables::StereoCamera& calibration,
    const vesta_variables::Point3DLandmark& point, const vesta_core::Vector4d& mean,
    const vesta_core::Matrix4d& covariance)
  : vesta_core::Constraint(source, { position.uuid(), orientation.uuid(), calibration.uuid(), point.uuid() })
  , mean_(mean)
  , sqrt_information_(covariance.inverse().llt().matrixU())
{
}

StereoReprojectionErrorConstraint::StereoReprojectionErrorConstraint(
    const std::string& source, const vesta_variables::Position3DStamped& position,
    const vesta_variables::Orientation3DStamped& orientation, const vesta_variables::StereoCamera& calibration,
    const vesta_variables::Point3DLandmark& point, const vesta_variables::Extrinsic3DPosition& ext_position,
    const vesta_variables::Extrinsic3DOrientation& ext_orientation, const vesta_core::Vector4d& mean,
    const vesta_core::Matrix4d& covariance)
  : vesta_core::Constraint(source, { position.uuid(), orientation.uuid(), calibration.uuid(), point.uuid(),
                                     ext_position.uuid(), ext_orientation.uuid() })
  , mean_(mean)
  , sqrt_information_(covariance.inverse().llt().matrixU())
  , has_extrinsic_(true)
{
}

void StereoReprojectionErrorConstraint::print(std::ostream& stream) const
{
  stream << type() << "\n"
         << "  source: " << source() << "\n"
         << "  uuid: " << uuid() << "\n"
         << "  position variable: " << variables().at(0) << "\n"
         << "  orientation variable: " << variables().at(1) << "\n"
         << "  calibration variable: " << variables().at(2) << "\n"
         << "  point variable: " << variables().at(3) << "\n";

  if (has_extrinsic_)
  {
    stream << "  ext_position variable: " << variables().at(4) << "\n"
           << "  ext_orientation variable: " << variables().at(5) << "\n";
  }

  stream << "  mean: " << mean().transpose() << "\n"
         << "  sqrt_info:\n"
         << sqrtInformation() << "\n";

  if (loss())
  {
    stream << "  loss: ";
    loss()->print(stream);
  }
}

ceres::CostFunction* StereoReprojectionErrorConstraint::costFunction() const
{
  if (has_extrinsic_)
  {
    return new ceres::AutoDiffCostFunction<StereoReprojectionErrorWithExtrinsicCostFunctor, 4, 3, 4, 5, 3, 3, 4>(
        new StereoReprojectionErrorWithExtrinsicCostFunctor(sqrt_information_, mean_));
  }

  return new ceres::AutoDiffCostFunction<StereoReprojectionErrorCostFunctor, 4, 3, 4, 5, 3>(
      new StereoReprojectionErrorCostFunctor(sqrt_information_, mean_));
}

}  // namespace vesta_constraints

BOOST_CLASS_EXPORT_IMPLEMENT(vesta_constraints::StereoReprojectionErrorConstraint);
