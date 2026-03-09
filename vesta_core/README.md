# vesta_core

Core abstractions and interface definitions for the vesta factor graph library. This package defines the base classes and common types that all other vesta packages depend on. Concrete implementations of these interfaces are provided by sibling packages such as `vesta_variables`, `vesta_constraints`, `vesta_graphs`, `vesta_loss`, and `vesta_optimizers`.

## Key Concepts

### Variable

A Variable represents a semantically meaningful group of one or more scalar values that are treated as a single block by the optimizer. Examples include a 2D point (x, y), a 3D point (x, y, z), or camera intrinsics (fx, fy, cx, cy). Variables must store their data in contiguous memory for integration with the Ceres solver. They support optional manifold parameterizations for over-parameterized or nonlinear state representations (e.g., quaternions), Schur elimination group assignment for structured problems, and variable bounds.

Each variable is identified by a deterministic UUID generated from its metadata, ensuring that variables representing the same quantity always receive the same identifier.

### Constraint

A Constraint defines a cost function connecting one or more variables. The cost function must be a valid `ceres::CostFunction` and may optionally include a loss function for robustness against outliers. Constraints are identified by a randomly generated UUID and carry a source string indicating which sensor or motion model produced them.

### Transaction

A Transaction groups variable and constraint additions and removals into an atomic unit. This ensures that graph modifications such as replacing one constraint with another are applied together rather than independently. Transactions also track involved timestamps for motion model coordination.

### Graph

The Graph interface defines the factor graph structure: a collection of variables and constraints with methods for adding, removing, querying, optimizing, and evaluating the problem. Derived classes (in `vesta_graphs`) provide concrete storage and Ceres integration.

### Loss

The Loss interface wraps `ceres::LossFunction` with Boost serialization support. Loss functions define the penalty curve applied to constraint residuals, enabling robust estimation in the presence of outliers. Concrete loss functions are provided in `vesta_loss`.

### Manifold

The Manifold class extends `ceres::Manifold` with Boost serialization support. Manifolds define the Plus/Minus operations and their Jacobians for variables that live on non-Euclidean spaces (e.g., rotations).

### UUID

A type alias for `boost::uuids::uuid` with deterministic generation utilities. UUIDs can be generated from strings, raw data buffers, timestamps, or namespace/data pairs to produce repeatable identifiers for variables and other objects.

### Timestamp and Duration

Lightweight time types representing nanoseconds since epoch and time deltas, respectively. They replace ROS time types with a standalone implementation supporting `std::chrono` interoperability and Boost serialization.

## Public Headers

### Core Interfaces

| Header | Description |
|--------|-------------|
| `variable.h` | Abstract `Variable` base class and convenience macros (`VESTA_VARIABLE_DEFINITIONS`, etc.) |
| `constraint.h` | Abstract `Constraint` base class and convenience macros (`VESTA_CONSTRAINT_DEFINITIONS`, etc.) |
| `graph.h` | Abstract `Graph` interface for factor graph storage, optimization, evaluation, and covariance computation |
| `transaction.h` | `Transaction` class for atomic batches of variable/constraint additions and removals |
| `loss.h` | Abstract `Loss` interface wrapping `ceres::LossFunction` with serialization |
| `manifold.h` | `Manifold` base class extending `ceres::Manifold` with Boost serialization |
| `motion_model.h` | Abstract `MotionModel` interface for generating inter-timestamp constraints |
| `publisher.h` | Abstract `Publisher` interface for consuming optimized graph state |

### Identity and Time

| Header | Description |
|--------|-------------|
| `uuid.h` | `UUID` type alias and `vesta_core::uuid::generate()` family of deterministic UUID generators |
| `timestamp.h` | `Timestamp` and `Duration` types with nanosecond precision and `std::chrono` support |

### Serialization

| Header | Description |
|--------|-------------|
| `serialization.h` | Archive type aliases (`BinaryInputArchive`, `TextOutputArchive`, etc.), stream source/sink classes for byte-buffer I/O, and Eigen matrix serialization |
| `graph_deserializer.h` | `serializeGraph()` and `deserializeGraph()` free functions for binary graph round-tripping |
| `transaction_deserializer.h` | `serializeTransaction()` and `deserializeTransaction()` free functions for binary transaction round-tripping |

### Ceres Integration

| Header | Description |
|--------|-------------|
| `autodiff_manifold.h` | `AutoDiffManifold` template for creating manifolds with automatically differentiated Jacobians |
| `ceres_macros.h` | `CERES_VERSION_AT_LEAST()` version-checking macro |
| `ceres_options.h` | `ToString()` and `FromString()` helpers for Ceres solver option enums |
| `schur_ordering.h` | `buildSchurOrdering()` utility for constructing `ceres::ParameterBlockOrdering` from variable Schur groups |
| `jacobian_relinearization.h` | `JacobianRelinearizationController`, `CachedJacobianCostFunction`, and `JacobianEvaluationCallback` for controlling Jacobian recomputation policies (every-N, first-estimate-Jacobian, adaptive) |

### Utilities

| Header | Description |
|--------|-------------|
| `eigen.h` | Row-major Eigen typedefs (`MatrixXd`, `Vector3d`, etc.) and covariance validation helpers |
| `eigen_gtest.h` | `EXPECT_MATRIX_EQ` / `EXPECT_MATRIX_NEAR` GTest macros for Eigen matrix comparisons |
| `fuse_macros.h` | Smart pointer aliases and factory macros (`VESTA_SMART_PTR_DEFINITIONS`, `VESTA_SMART_PTR_DEFINITIONS_WITH_EIGEN`) |
| `macros.h` | Additional smart pointer and utility macros |
| `type_name.h` | `typeName<T>()` template returning the demangled C++ type name |
| `util.h` | Quaternion-to-Euler conversion utilities compatible with `ceres::Jet` automatic differentiation |
| `message_buffer.h` | `MessageBuffer<T>` template for maintaining a time-indexed history of messages |
| `message_buffer_impl.h` | Implementation details for `MessageBuffer` |
| `timestamp_manager.h` | `TimestampManager` utility for tracking motion model timestamp chains and computing required constraint splits |

## Dependencies

- Ceres Solver 2.2+
- Eigen3
- Boost (serialization, uuid, iostreams, range)
- glog
