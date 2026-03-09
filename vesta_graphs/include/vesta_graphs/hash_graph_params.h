#pragma once

/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2019 Clearpath Robotics
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

#include <vesta_core/jacobian_relinearization.h>

#include <ceres/problem.h>


namespace vesta_graphs
{

/**
 * @brief Defines the set of parameters required by the vesta_graphs::HashGraph class
 */
struct HashGraphParams
{
public:
  /**
   * @brief Ceres Problem::Options object that controls various aspects of the optimization problem.
   *
   * See https://ceres-solver.googlesource.com/ceres-solver/+/master/include/ceres/problem.h#123
   */
  ceres::Problem::Options problem_options;

  /**
   * @brief Controls how often Jacobians are recomputed during optimization.
   *
   * - kDefault: Standard Ceres behavior, recompute every iteration (zero overhead)
   * - kEveryN: Recompute every N iterations, use cached Jacobians otherwise
   * - kFirstEstimate: Full First Estimate Jacobian (FEJ) — freeze after first linearization
   */
  vesta_core::JacobianPolicy jacobian_policy = vesta_core::JacobianPolicy::kDefault;

  /**
   * @brief Relinearization period for JacobianPolicy::kEveryN.
   *
   * Only used when jacobian_policy is kEveryN. Jacobians are recomputed on iterations
   * 1, 1+period, 1+2*period, etc. Ignored for other policies.
   */
  int jacobian_relinearization_period = 1;

  /**
   * @brief Change threshold for JacobianPolicy::kAdaptive.
   *
   * Only used when jacobian_policy is kAdaptive. A cost function's Jacobians are recomputed
   * when any connected parameter block's L2 norm of change (since last linearization) exceeds
   * this value. Smaller values relinearize more often (more accurate, slower). Ignored for
   * other policies.
   */
  double jacobian_relinearization_threshold = 0.01;
};

}  // namespace vesta_graphs

