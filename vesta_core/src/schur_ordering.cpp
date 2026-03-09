/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026, Locus Robotics
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
#include <vesta_core/schur_ordering.h>

#include <glog/logging.h>

#include <memory>

namespace vesta_core {

std::shared_ptr<ceres::ParameterBlockOrdering>
buildSchurOrdering(const Graph &graph) {
  bool has_group_zero = false;
  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();

  for (const auto &variable : graph.getVariables()) {
    const int group = variable.schurGroup();
    if (group < -1 || group > 1) {
      LOG(WARNING) << "Variable '" << variable.type()
                   << "' returned unexpected schurGroup() value " << group
                   << ". Treating as group 1 (kept in reduced system).";
    }
    // The const_cast is safe: Graph::getVariables() returns const references to
    // the same Variable objects stored in the graph. The data() pointers match
    // those used by the persistent ceres::Problem, which is what the ordering
    // must reference.
    if (group == 0) {
      has_group_zero = true;
      ordering->AddElementToGroup(const_cast<double *>(variable.data()), 0);
    } else {
      ordering->AddElementToGroup(const_cast<double *>(variable.data()), 1);
    }
  }

  if (!has_group_zero) {
    return nullptr;
  }
  return ordering;
}

} // namespace vesta_core
