/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2019, Locus Robotics
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
#ifndef FUSE_OPTIMIZERS_FIXED_LAG_SMOOTHER_H
#define FUSE_OPTIMIZERS_FIXED_LAG_SMOOTHER_H

#include <fuse_core/graph.h>
#include <fuse_core/timestamp.h>
#include <fuse_core/transaction.h>
#include <fuse_optimizers/fixed_lag_smoother_params.h>
#include <fuse_optimizers/optimizer.h>
#include <fuse_optimizers/variable_stamp_index.h>

#include <memory>
#include <string>
#include <vector>

namespace fuse_optimizers
{

/**
 * @brief A fixed-lag smoother implementation that marginalizes out variables older than a defined lag time
 *
 * This implementation assumes that all added variable types are either derived from the fuse_variables::Stamped class,
 * or are directly connected to at least one fuse_variables::Stamped variable via a constraint. The current time of
 * the fixed-lag smoother is determined by the newest stamp of all added fuse_variables::Stamped variables.
 *
 * During optimization:
 *  (1) pending transactions are processed and applied to the graph
 *  (2) the lag expiration time is computed
 *  (3) variables to marginalize are identified
 *  (4) a marginal transaction is computed and applied
 *  (5) the augmented graph is optimized via Ceres
 *  (6) the solver summary is returned
 *
 * Usage:
 * @code
 *   auto graph = std::make_unique<fuse_graphs::HashGraph>();
 *   FixedLagSmootherParams params;
 *   params.lag_duration = fuse_core::Duration::fromSec(5.0);
 *   FixedLagSmoother smoother(params, std::move(graph));
 *
 *   smoother.addTransaction("sensor1", transaction1);
 *   auto summary = smoother.optimize();
 * @endcode
 */
class FixedLagSmoother : public Optimizer
{
public:
  using ParameterType = FixedLagSmootherParams;

  /**
   * @brief Constructor
   *
   * @param[in] params Configuration settings
   * @param[in] graph  The graph object (takes ownership)
   */
  FixedLagSmoother(const FixedLagSmootherParams& params, fuse_core::Graph::UniquePtr graph);

  /**
   * @brief Destructor
   */
  ~FixedLagSmoother() override = default;

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
   * @brief Process pending transactions, marginalize old variables, and run Ceres optimization
   *
   * This method:
   *  1. Processes the pending transaction queue
   *  2. Applies pending transactions to the graph
   *  3. Computes the lag expiration time
   *  4. Computes variables to marginalize
   *  5. Creates and applies the marginal transaction
   *  6. Runs Ceres optimization
   *  7. Returns the solver summary
   *
   * @return The Ceres solver summary
   */
  ceres::Solver::Summary optimize() override;

  /**
   * @brief Reset the optimizer to its initial state
   *
   * Clears the graph, all pending transactions, and all marginalization state.
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

    const fuse_core::Timestamp& stamp() const { return transaction->stamp(); }
    const fuse_core::Timestamp& minStamp() const { return transaction->minStamp(); }
    const fuse_core::Timestamp& maxStamp() const { return transaction->maxStamp(); }
  };

  /**
   * @brief Queue of Transaction objects, sorted by timestamp.
   *
   * Sorted with smallest stamp last for efficient pop_back().
   */
  using TransactionQueue = std::vector<TransactionQueueElement>;

  /**
   * @brief Perform any required preprocessing steps before computeVariablesToMarginalize() is called
   *
   * @param[in] new_transaction All new, non-marginal-related transactions that will be applied to the graph
   */
  void preprocessMarginalization(const fuse_core::Transaction& new_transaction);

  /**
   * @brief Compute the oldest timestamp that is part of the configured lag window
   */
  fuse_core::Timestamp computeLagExpirationTime() const;

  /**
   * @brief Compute the set of variables that should be marginalized from the graph
   *
   * @param[in] lag_expiration The oldest timestamp that should remain in the graph
   * @return A container with the set of variables to marginalize out
   */
  std::vector<fuse_core::UUID> computeVariablesToMarginalize(const fuse_core::Timestamp& lag_expiration);

  /**
   * @brief Perform any required post-marginalization bookkeeping
   *
   * @param[in] marginal_transaction The actual changes to the graph caused by marginalizing out the requested variables
   */
  void postprocessMarginalization(const fuse_core::Transaction& marginal_transaction);

  /**
   * @brief Process pending transactions into a combined transaction
   *
   * Transactions are processed sequentially based on timestamp. Expired transactions are removed.
   *
   * @param[out] transaction The transaction object to be augmented with pending transactions
   * @param[in]  lag_expiration The oldest timestamp that should remain in the graph
   */
  void processQueue(fuse_core::Transaction& transaction, const fuse_core::Timestamp& lag_expiration);

  ParameterType params_;  //!< Configuration settings for this fixed-lag smoother
  fuse_core::Graph::UniquePtr graph_;  //!< The graph object that holds all variables and constraints
  TransactionQueue pending_transactions_;  //!< Pending transactions not yet applied
  fuse_core::Timestamp lag_expiration_;  //!< The oldest stamp inside the fixed-lag smoother window
  fuse_core::Transaction marginal_transaction_;  //!< Marginals to add during the next optimization cycle
  VariableStampIndex timestamp_tracking_;  //!< Tracks timestamps associated with each variable
  fuse_core::Timestamp start_time_;  //!< The timestamp of the first transaction
  bool started_;  //!< Flag indicating the optimizer has received at least one transaction
};

}  // namespace fuse_optimizers

#endif  // FUSE_OPTIMIZERS_FIXED_LAG_SMOOTHER_H
