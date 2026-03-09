# vesta_graphs

Concrete graph storage implementations for the vesta factor graph library. This package provides implementations of the `vesta_core::Graph` interface.

## HashGraph

The primary graph implementation, using hash maps (`std::unordered_map`) for O(1) variable and constraint lookup by UUID.

Key features:

- **Persistent Ceres Problem**: Maintains a live `ceres::Problem` instance with `enable_fast_removal=true`, so variables and constraints can be incrementally added and removed without rebuilding the problem from scratch.
- **Incremental Updates**: Apply transactions (batches of additions/removals) efficiently via the `update()` method.
- **Schur Ordering**: Supports Schur complement elimination ordering for efficient solving of problems with landmark variables.
- **Jacobian Relinearization**: Tracks which constraints need Jacobian recomputation, enabling selective relinearization policies.
- **Serialization**: Full Boost serialization support for graph persistence and transmission.

### Usage

```cpp
#include <vesta_graphs/hash_graph.h>

auto graph = std::make_unique<vesta_graphs::HashGraph>();

// Apply a transaction
vesta_core::Transaction transaction;
transaction.addVariable(my_variable);
transaction.addConstraint(my_constraint);
graph->update(transaction);

// Query the graph
bool has_var = graph->variableExists(variable_uuid);
const auto& var = graph->getVariable(variable_uuid);

// Iterate
for (const auto& variable : graph->getVariables()) { /* ... */ }
for (const auto& constraint : graph->getConstraints()) { /* ... */ }
```

### HashGraphParams

Configuration structure for `HashGraph`:

```cpp
#include <vesta_graphs/hash_graph_params.h>
```

## Include Convention

```cpp
#include <vesta_graphs/hash_graph.h>
#include <vesta_graphs/hash_graph_params.h>
```

## Contents

| Header | Description |
|--------|-------------|
| `hash_graph.h` | Hash map-based graph with persistent `ceres::Problem` |
| `hash_graph_params.h` | Configuration parameters for `HashGraph` |
