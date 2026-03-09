#pragma once

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

#include <vesta_core/graph.h>

#include <ceres/ordered_groups.h>

#include <memory>

namespace vesta_core {

/**
 * @brief Build a ceres::ParameterBlockOrdering from a graph for Schur
 * complement-based solvers.
 *
 * Iterates all variables in the graph and assigns them to Schur elimination
 * groups based on Variable::schurGroup(). Variables returning 0 are placed in
 * group 0 (eliminated first, e.g. landmarks in bundle adjustment). Variables
 * returning 1 or -1 (unclassified) are placed in group 1 (kept in the reduced
 * system, e.g. camera poses).
 *
 * The returned ordering is suitable for use with
 * ceres::Solver::Options::linear_solver_ordering when using DENSE_SCHUR,
 * SPARSE_SCHUR, or ITERATIVE_SCHUR linear solver types.
 *
 * @param[in] graph The graph whose variables define the ordering
 * @return A shared_ptr to the ordering, or nullptr if no variables have
 * schurGroup() == 0
 */
std::shared_ptr<ceres::ParameterBlockOrdering>
buildSchurOrdering(const Graph &graph);

} // namespace vesta_core
