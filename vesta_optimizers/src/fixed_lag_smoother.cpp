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
#include <vesta_optimizers/fixed_lag_smoother.h>

#include <glog/logging.h>
#include <vesta_constraints/common/marginalize_variables.h>
#include <vesta_constraints/common/qr_marginalizer.h>
#include <vesta_core/graph.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace
{
/**
 * @brief Delete an element from the vector using a reverse iterator
 *
 * @param[in] container The container to delete from
 * @param[in] position  A reverse iterator that accesses the element to be
 * erased
 * @return A reverse iterator pointing to the element after the erased element
 */
template <typename T>
typename std::vector<T>::reverse_iterator erase(std::vector<T>& container,
                                                typename std::vector<T>::reverse_iterator position)
{
  // Reverse iterators are weird
  // https://stackoverflow.com/questions/1830158/how-to-call-erase-with-a-reverse-iterator
  std::advance(position, 1);
  container.erase(position.base());
  return position;
}
}  // namespace

namespace vesta_optimizers
{

FixedLagSmoother::FixedLagSmoother(const FixedLagSmootherParams& params, vesta_core::Graph::UniquePtr graph)
  : params_(params)
  , graph_(std::move(graph))
  , started_(false)
  , marginalizer_(std::make_unique<vesta_constraints::QRMarginalizer>())
{
}

void FixedLagSmoother::setMarginalizer(std::unique_ptr<vesta_constraints::Marginalizer> marginalizer)
{
  marginalizer_ = std::move(marginalizer);
}

void FixedLagSmoother::addTransaction(const std::string& sensor_name, vesta_core::Transaction::SharedPtr transaction)
{
  // If this transaction occurs before the start time, just ignore it
  const auto max_time = transaction->maxStamp();
  if (started_ && max_time < start_time_)
  {
    LOG(INFO) << "Received a transaction before the start time from sensor '" << sensor_name << "'."
              << " start_time: " << start_time_ << ", maximum involved stamp: " << max_time
              << ", difference: " << (start_time_ - max_time) << "s";
    return;
  }

  // Add the new transaction to the pending set
  // The pending set is arranged "smallest stamp last" for efficient pop_back()
  auto comparator = [](const vesta_core::Timestamp& value, const TransactionQueueElement& element) {
    return value >= element.stamp();
  };
  auto position =
      std::upper_bound(pending_transactions_.begin(), pending_transactions_.end(), transaction->stamp(), comparator);
  pending_transactions_.insert(position, { sensor_name, std::move(transaction) });

  // If we haven't "started" yet, auto-start on the first transaction
  if (!started_)
  {
    started_ = true;
    start_time_ = pending_transactions_.back().minStamp();
  }
}

ceres::Solver::Summary FixedLagSmoother::optimize()
{
  // Process pending transactions into a combined transaction
  auto new_transaction = vesta_core::Transaction::make_shared();
  processQueue(*new_transaction, lag_expiration_);

  // If the transaction is empty, return a default summary
  if (new_transaction->empty())
  {
    return ceres::Solver::Summary();
  }

  // Prepare for selecting the marginal variables
  preprocessMarginalization(*new_transaction);

  // Combine the new transactions with any marginal transaction from the end of
  // the last cycle
  new_transaction->merge(marginal_transaction_);

  // Update the graph
  try
  {
    graph_->update(*new_transaction);
  }
  catch (const std::exception& ex)
  {
    std::ostringstream oss;
    oss << "Graph:\n";
    graph_->print(oss);
    oss << "\nTransaction:\n";
    new_transaction->print(oss);

    LOG(FATAL) << "Failed to update graph with transaction: " << ex.what() << "\n" << oss.str();
  }

  // Optimize the entire graph
  auto summary = graph_->optimize(params_.solver_options);

  // Abort if optimization failed. Not converging is not a failure because the
  // solution found is usable.
  if (!summary.IsSolutionUsable())
  {
    std::ostringstream oss;
    oss << "Graph:\n";
    graph_->print(oss);
    oss << "\nTransaction:\n";
    new_transaction->print(oss);

    LOG(ERROR) << "Optimization failed after updating the graph with the "
                  "transaction with timestamp "
               << new_transaction->stamp() << ".\n"
               << oss.str();
    LOG(INFO) << summary.FullReport();
  }

  // Compute a transaction that marginalizes out old variables
  lag_expiration_ = computeLagExpirationTime();
  marginal_transaction_ =
      marginalizer_->marginalize("FixedLagSmoother", computeVariablesToMarginalize(lag_expiration_), *graph_);

  // Perform any post-marginal cleanup
  postprocessMarginalization(marginal_transaction_);
  // Note: The marginal transaction will not be applied until the next
  // optimization iteration

  return summary;
}

void FixedLagSmoother::reset()
{
  pending_transactions_.clear();
  graph_->clear();
  marginal_transaction_ = vesta_core::Transaction();
  timestamp_tracking_.clear();
  lag_expiration_ = vesta_core::Timestamp(0);
  start_time_ = vesta_core::Timestamp(0);
  started_ = false;
}

const vesta_core::Graph& FixedLagSmoother::graph() const
{
  return *graph_;
}

void FixedLagSmoother::preprocessMarginalization(const vesta_core::Transaction& new_transaction)
{
  timestamp_tracking_.addNewTransaction(new_transaction);
}

vesta_core::Timestamp FixedLagSmoother::computeLagExpirationTime() const
{
  // Find the most recent variable timestamp
  auto now = timestamp_tracking_.currentStamp();
  // Then carefully subtract the lag duration. Timestamp objects do not handle
  // negative values.
  return (start_time_ + params_.lag_duration < now) ? now - params_.lag_duration : start_time_;
}

std::vector<vesta_core::UUID>
FixedLagSmoother::computeVariablesToMarginalize(const vesta_core::Timestamp& lag_expiration)
{
  auto marginalize_variable_uuids = std::vector<vesta_core::UUID>();
  timestamp_tracking_.query(lag_expiration, std::back_inserter(marginalize_variable_uuids));
  return marginalize_variable_uuids;
}

void FixedLagSmoother::postprocessMarginalization(const vesta_core::Transaction& marginal_transaction)
{
  timestamp_tracking_.addMarginalTransaction(marginal_transaction);
}

void FixedLagSmoother::processQueue(vesta_core::Transaction& transaction, const vesta_core::Timestamp& lag_expiration)
{
  if (pending_transactions_.empty())
  {
    return;
  }

  // Use the most recent transaction time as the current time
  const auto current_time = pending_transactions_.front().stamp();

  // Attempt to process each pending transaction
  auto transaction_riter = pending_transactions_.rbegin();
  while (transaction_riter != pending_transactions_.rend())
  {
    auto& element = *transaction_riter;
    const auto& min_stamp = element.minStamp();
    if (min_stamp < lag_expiration)
    {
      LOG(INFO) << "The current lag expiration time is " << lag_expiration << ". The queued transaction with "
                << "timestamp " << element.stamp() << " from sensor " << element.sensor_name << " has a minimum "
                << "involved timestamp of " << min_stamp << ", which is " << (lag_expiration - min_stamp)
                << " seconds too old. Ignoring this transaction.";
      transaction_riter = erase(pending_transactions_, transaction_riter);
    }
    else
    {
      // Check the transaction timeout to determine if it should be removed or
      // kept
      const auto& max_stamp = element.maxStamp();
      if (max_stamp + params_.transaction_timeout < current_time)
      {
        // Warn that this transaction has expired, then skip it.
        LOG(ERROR) << "The queued transaction with timestamp " << element.stamp() << " and maximum "
                   << "involved stamp of " << max_stamp << " from sensor " << element.sensor_name
                   << " could not be processed after " << (current_time - max_stamp) << " seconds, "
                   << "which is greater than the 'transaction_timeout' value of " << params_.transaction_timeout
                   << ". Ignoring this transaction.";
        transaction_riter = erase(pending_transactions_, transaction_riter);
      }
      else
      {
        // Processing was successful. Add the results to the final transaction,
        // delete this one, and move to the next.
        transaction.merge(*element.transaction, true);
        transaction_riter = erase(pending_transactions_, transaction_riter);
      }
    }
  }
}

}  // namespace vesta_optimizers
