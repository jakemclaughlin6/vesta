# vesta_optimizers

High-level optimizer implementations for running factor graph optimization. This package provides the top-level entry point for using vesta in applications.

## Design

Vesta optimizers are **pure library interfaces** with no ROS dependencies, no internal threads, and no plugin loading. The client controls all execution through a simple synchronous contract:

```
addTransaction() --> optimize() --> graph()
```

## Classes

### Optimizer (Abstract Base)

Defines the optimization contract that all optimizers implement:

- `addTransaction(sensor_name, transaction)` -- queue a transaction
- `optimize()` -- apply pending transactions and run the Ceres solver
- `graph()` -- read-only access to the optimized graph
- `reset()` -- clear all state

### BatchOptimizer

Full-problem optimizer. All pending transactions are merged and applied to the graph, then the entire problem is solved from scratch.

Best for offline/batch processing or problems where the full history matters.

```cpp
#include <vesta_graphs/hash_graph.h>
#include <vesta_optimizers/batch_optimizer.h>

auto graph = std::make_unique<vesta_graphs::HashGraph>();
vesta_optimizers::BatchOptimizerParams params;
vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

optimizer.addTransaction("lidar", lidar_transaction);
optimizer.addTransaction("odom", odom_transaction);
auto summary = optimizer.optimize();

// Access optimized state
const auto& optimized_graph = optimizer.graph();
```

### FixedLagSmoother

Incremental sliding-window optimizer. Variables older than the configured lag duration are automatically marginalized out, keeping the problem size bounded for real-time operation.

The marginalization process:
1. Pending transactions are applied to the graph
2. The lag expiration time is computed from the newest variable timestamp
3. Variables older than the expiration are identified
4. Marginal constraints are computed and applied (preserving information)
5. Old variables and their constraints are removed
6. The reduced problem is optimized via Ceres

```cpp
#include <vesta_graphs/hash_graph.h>
#include <vesta_optimizers/fixed_lag_smoother.h>

auto graph = std::make_unique<vesta_graphs::HashGraph>();
vesta_optimizers::FixedLagSmootherParams params;
params.lag_duration = vesta_core::Duration::fromSec(5.0);

vesta_optimizers::FixedLagSmoother smoother(params, std::move(graph));

// Real-time loop
smoother.addTransaction("sensor", transaction);
auto summary = smoother.optimize();
```

### VariableStampIndex

Utility class that indexes variables by their timestamp, enabling efficient lookup of variables within time ranges. Used internally by `FixedLagSmoother` for marginalization decisions.

## Contents

| Header | Description |
|--------|-------------|
| `optimizer.h` | Abstract base class for all optimizers |
| `batch_optimizer.h` | Full-problem batch optimizer |
| `batch_optimizer_params.h` | Configuration for `BatchOptimizer` |
| `fixed_lag_smoother.h` | Sliding-window optimizer with marginalization |
| `fixed_lag_smoother_params.h` | Configuration for `FixedLagSmoother` |
| `variable_stamp_index.h` | Timestamp-based variable index |

## Include Convention

```cpp
#include <vesta_optimizers/batch_optimizer.h>
#include <vesta_optimizers/fixed_lag_smoother.h>
```
