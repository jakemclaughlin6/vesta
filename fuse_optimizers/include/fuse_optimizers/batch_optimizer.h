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
#ifndef FUSE_OPTIMIZERS_BATCH_OPTIMIZER_H
#define FUSE_OPTIMIZERS_BATCH_OPTIMIZER_H

#include <fuse_core/graph.h>
#include <fuse_core/timestamp.h>
#include <fuse_core/transaction.h>
#include <fuse_optimizers/batch_optimizer_params.h>
#include <fuse_optimizers/optimizer.h>

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace fuse_optimizers
{

/**
 * @brief A simple optimizer implementation that uses batch optimization
 *
 * Received sensor transactions are queued via addTransaction(). When optimize() is called,
 * all pending transactions are merged and applied to the graph, then Ceres optimization is run.
 *
 * Usage:
 * @code
 *   auto graph = std::make_unique<fuse_graphs::HashGraph>();
 *   BatchOptimizerParams params;
 *   BatchOptimizer optimizer(params, std::move(graph));
 *
 *   optimizer.addTransaction("sensor1", transaction1);
 *   optimizer.addTransaction("sensor2", transaction2);
 *   auto summary = optimizer.optimize();
 * @endcode
 */
class BatchOptimizer : public Optimizer
{
public:
  using ParameterType = BatchOptimizerParams;

  /**
   * @brief Constructor
   *
   * @param[in] params Configuration settings
   * @param[in] graph  The graph object (takes ownership)
   */
  BatchOptimizer(const BatchOptimizerParams& params, fuse_core::Graph::UniquePtr graph);

  /**
   * @brief Destructor
   */
  ~BatchOptimizer() override = default;

  /**
   * @brief Add a transaction to the pending queue
   *
   * Transactions are queued and not applied until optimize() is called.
   *
   * @param[in] sensor_name The name of the sensor that produced the Transaction
   * @param[in] transaction The populated Transaction object
   */
  void addTransaction(const std::string& sensor_name,
                      fuse_core::Transaction::SharedPtr transaction) override;

  /**
   * @brief Process pending transactions and run Ceres optimization
   *
   * All pending transactions are merged into a single combined transaction, applied to the graph,
   * and then the graph is optimized using the configured Ceres solver options.
   *
   * @return The Ceres solver summary
   */
  ceres::Solver::Summary optimize() override;

  /**
   * @brief Reset the optimizer to its initial state
   *
   * Clears the graph and all pending transactions.
   */
  void reset() override;

  /**
   * @brief Read-only access to the current graph
   */
  const fuse_core::Graph& graph() const override;

private:
  /**
   * Structure containing the information required to process a transaction after it was received.
   */
  struct TransactionQueueElement
  {
    std::string sensor_name;
    fuse_core::Transaction::SharedPtr transaction;

    TransactionQueueElement(
      const std::string& sensor_name,
      fuse_core::Transaction::SharedPtr transaction) :
        sensor_name(sensor_name),
        transaction(std::move(transaction)) {}
  };

  /**
   * @brief Queue of Transaction objects, sorted by timestamp.
   */
  using TransactionQueue = std::multimap<fuse_core::Timestamp, TransactionQueueElement>;

  ParameterType params_;  //!< Configuration settings for this optimizer
  fuse_core::Graph::UniquePtr graph_;  //!< The graph object that holds all variables and constraints
  fuse_core::Transaction::SharedPtr combined_transaction_;  //!< Aggregated transaction from multiple sensors
  TransactionQueue pending_transactions_;  //!< Pending transactions not yet applied to the graph
  bool started_;  //!< Flag indicating the optimizer has received at least one transaction
};

}  // namespace fuse_optimizers

#endif  // FUSE_OPTIMIZERS_BATCH_OPTIMIZER_H
