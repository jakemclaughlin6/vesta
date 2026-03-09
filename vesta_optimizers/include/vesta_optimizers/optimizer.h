#pragma once

/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Locus Robotics
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

#include <ceres/solver.h>
#include <memory>
#include <string>
#include <vesta_core/graph.h>
#include <vesta_core/transaction.h>

namespace vesta_optimizers {

/**
 * @brief A simple abstract base class for vesta optimizers
 *
 * An optimizer implements the basic vesta information flow contract:
 *  - Clients push information into the optimizer using addTransaction()
 *  - The optimizer computes the optimal variable values via optimize()
 *  - The optimizer provides access to the optimal variable values via graph()
 *
 * This is a pure library interface with no ROS dependencies, no internal
 * threads, and no plugin loading. The client is responsible for calling
 * optimize() when desired.
 */
class Optimizer {
public:
  virtual ~Optimizer() = default;

  /**
   * @brief Add a transaction to the optimizer
   *
   * @param[in] sensor_name The name of the sensor that produced the Transaction
   * @param[in] transaction The populated Transaction object
   */
  virtual void
  addTransaction(const std::string &sensor_name,
                 vesta_core::Transaction::SharedPtr transaction) = 0;

  /**
   * @brief Run the optimization
   *
   * Processes any pending transactions, applies them to the graph, and runs the
   * Ceres solver.
   *
   * @return The Ceres solver summary
   */
  virtual ceres::Solver::Summary optimize() = 0;

  /**
   * @brief Reset the optimizer to its initial state
   */
  virtual void reset() = 0;

  /**
   * @brief Read-only access to the current graph
   *
   * @return A const reference to the graph
   */
  virtual const vesta_core::Graph &graph() const = 0;
};

} // namespace vesta_optimizers
