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
#include <fuse_core/transaction.h>
#include <fuse_optimizers/batch_optimizer.h>
#include <glog/logging.h>

#include <string>
#include <utility>

namespace fuse_optimizers
{

BatchOptimizer::BatchOptimizer(
  const BatchOptimizerParams& params,
  fuse_core::Graph::UniquePtr graph) :
    params_(params),
    graph_(std::move(graph)),
    combined_transaction_(fuse_core::Transaction::make_shared()),
    started_(false)
{
}

void BatchOptimizer::addTransaction(
  const std::string& sensor_name,
  fuse_core::Transaction::SharedPtr transaction)
{
  fuse_core::Timestamp transaction_time = transaction->stamp();
  pending_transactions_.emplace(transaction_time, TransactionQueueElement(sensor_name, std::move(transaction)));
  if (!started_)
  {
    started_ = true;
  }
}

ceres::Solver::Summary BatchOptimizer::optimize()
{
  // Process pending transactions into the combined transaction
  // Use the most recent transaction time as the current time
  fuse_core::Timestamp current_time(0);
  if (!pending_transactions_.empty())
  {
    current_time = pending_transactions_.rbegin()->first;
  }

  // Attempt to process each pending transaction
  auto iter = pending_transactions_.begin();
  while (iter != pending_transactions_.end())
  {
    auto& element = iter->second;
    // Check if this transaction has timed out
    if (element.transaction->stamp() + params_.transaction_timeout < current_time)
    {
      LOG(ERROR) << "The queued transaction with timestamp " << element.transaction->stamp()
                 << " could not be processed after " << (current_time - element.transaction->stamp())
                 << " seconds, which is greater than the 'transaction_timeout' value of "
                 << params_.transaction_timeout << ". Ignoring this transaction.";
      iter = pending_transactions_.erase(iter);
      continue;
    }
    // Merge the transaction into the combined transaction
    combined_transaction_->merge(*element.transaction, true);
    iter = pending_transactions_.erase(iter);
  }

  // Apply the combined transaction to the graph
  graph_->update(*combined_transaction_);

  // Reset the combined transaction for the next cycle
  combined_transaction_ = fuse_core::Transaction::make_shared();

  // Optimize the entire graph
  return graph_->optimize(params_.solver_options);
}

void BatchOptimizer::reset()
{
  pending_transactions_.clear();
  combined_transaction_ = fuse_core::Transaction::make_shared();
  graph_->clear();
  started_ = false;
}

const fuse_core::Graph& BatchOptimizer::graph() const
{
  return *graph_;
}

}  // namespace fuse_optimizers
