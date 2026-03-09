/*
 * Software License Agreement (BSD License)
 *
 *  Author: Oscar Mendez
 *  Created on Mon Dec 12 2023
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
#include <vesta_constraints/vision/reprojection_error_constraint.h>
#include <vesta_constraints/vision/reprojection_error_cost_functor.h>

#include <Eigen/Dense>
#include <boost/serialization/export.hpp>
#include <ceres/autodiff_cost_function.h>

#include <string>

namespace vesta_constraints {

ReprojectionErrorConstraint::ReprojectionErrorConstraint(
    const std::string &source,
    const vesta_variables::Position3DStamped &position,
    const vesta_variables::Orientation3DStamped &orientation,
    const vesta_variables::PinholeCamera &calibration,
    const vesta_variables::Point3DLandmark &point,
    const vesta_core::Vector2d &mean, const vesta_core::Matrix2d &covariance)
    : vesta_core::Constraint(source, {position.uuid(), orientation.uuid(),
                                      calibration.uuid(), point.uuid()}),
      mean_(mean), sqrt_information_(covariance.inverse().llt().matrixU()) {}

void ReprojectionErrorConstraint::print(std::ostream &stream) const {
  stream << type() << "\n"
         << "  source: " << source() << "\n"
         << "  uuid: " << uuid() << "\n"
         << "  position variable: " << variables().at(0) << "\n"
         << "  orientation variable: " << variables().at(1) << "\n"
         << "  calibration variable: " << variables().at(2) << "\n"
         << "  point variable: " << variables().at(3) << "\n"
         << "  mean: " << mean().transpose() << "\n"
         << "  sqrt_info: " << sqrtInformation() << "\n";

  if (loss()) {
    stream << "  loss: ";
    loss()->print(stream);
  }
}

ceres::CostFunction *ReprojectionErrorConstraint::costFunction() const {
  return new ceres::AutoDiffCostFunction<ReprojectionErrorCostFunctor, 2, 3, 4,
                                         4, 3>(
      new ReprojectionErrorCostFunctor(sqrt_information_, mean_));
}

} // namespace vesta_constraints

BOOST_CLASS_EXPORT_IMPLEMENT(vesta_constraints::ReprojectionErrorConstraint);
