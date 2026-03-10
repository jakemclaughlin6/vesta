# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Vesta?

Vesta is a pure C++ library for factor graph-based sensor fusion (SLAM / Visual SLAM), built on top of Ceres Solver. It is a ROS-agnostic fork of [fuse](https://github.com/locusrobotics/fuse) by Locus Robotics. It uses C++20, modern CMake, and the Ceres 2.2+ Manifold API.

## Build Commands

```bash
# Configure and build (from repo root)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run all tests
ctest --test-dir build --output-on-failure

# Run a single test by name
ctest --test-dir build -R test_hash_graph --output-on-failure

# Build without tests
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

### Docker

```bash
docker build -t vesta .
docker run -v $(pwd):/workspace -it vesta
```

### Formatting and Linting

```bash
# Format (uses .clang-format at repo root, Google-based, 120 char limit)
clang-format -i <file>

# Lint (uses .clang-tidy at repo root)
clang-tidy <file> -- -std=c++20 -I<include_paths>
```

## Package Architecture

Six packages, built in dependency order:

```
vesta_core  (base: Variable, Constraint, Graph, Transaction, Loss, Manifold, UUID, Timestamp)
  ├── vesta_variables  (concrete variable types: 2D/3D poses, landmarks, cameras, IMU biases)
  ├── vesta_loss       (robust loss wrappers: Huber, Cauchy, Tukey, DCS, etc.)
  └── vesta_graphs     (HashGraph: O(1) lookup, persistent ceres::Problem)
        └── vesta_constraints  (cost functions: odometry, reprojection, IMU preintegration, marginalization)
              └── vesta_optimizers  (BatchOptimizer, FixedLagSmoother)
```

Each package follows the same layout:
- `include/vesta_<pkg>/` — public headers (some have subdirs: `2d/`, `3d/`, `vision/`, `inertial/`)
- `src/` — implementation files
- `test/` — GoogleTest files named `test_*.cpp`

## Naming Conventions (enforced by .clang-tidy)

- Classes/Enums/Unions: `CamelCase`
- Functions/Methods: `camelBack`
- Variables: `lower_case`
- Protected/Private members: `lower_case_` (trailing underscore)
- Static variables: `s_lower_case` (prefix `s_`)
- Constants: `UPPER_CASE`

## Key Design Patterns

- **Synchronous, caller-driven**: No internal threads or event loops. You call `optimize()` explicitly.
- **Transaction-based**: All graph mutations go through `Transaction` objects applied atomically.
- **Boost serialization**: All variables, constraints, and graphs are serializable. New variable/constraint types must register with `BOOST_CLASS_EXPORT`.
- **UUID identity**: Each Variable and Constraint instance has a UUID. Variables combine a type hash + device UUID (+ optional timestamp). Constraints generate a random UUID.
- **Schur ordering**: `Point3DLandmark::schurGroup()` returns 0 to place landmarks in the first elimination group. Use `buildSchurOrdering()` to construct `ceres::ParameterBlockOrdering` from the graph.
- **Manifold API**: Variables define their tangent space via `ceres::Manifold` (not the deprecated LocalParameterization). `Orientation3DStamped` uses `ceres::EigenQuaternionManifold`.

## Test Helpers

Test directories contain shared fixtures/helpers:
- `vesta_core/test/`: `example_variable.h`, `example_constraint.h` — minimal concrete implementations for testing abstract interfaces
- `vesta_constraints/test/`: BAL problem data files used by `test_bal_problem`

## Dependencies

Ceres Solver (>=2.2), Eigen3, Boost (serialization), glog, SuiteSparse (CCOLAMD), GoogleTest (optional).
