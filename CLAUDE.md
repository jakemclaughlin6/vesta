# Vesta

## What is Vesta?

Vesta is a pure C++ library for factor graph-based sensor fusion (SLAM / Visual SLAM), built on top of Ceres Solver. It is a ROS-agnostic fork of [fuse](https://github.com/locusrobotics/fuse) by Locus Robotics. It uses C++20, modern CMake, and the Ceres 2.2+ Manifold API.

## Build Commands

**All builds and tests must run inside the Docker container** — cmake and dependencies are not installed on the host.

```bash
# Build the Docker image (one-time)
docker build -t vesta .

# Configure and build
docker run --rm -v $(pwd):$(pwd) -w $(pwd) vesta bash -c "cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build -j\$(nproc)"

# Run all tests
docker run --rm -v $(pwd):$(pwd) -w $(pwd) vesta bash -c "ctest --test-dir build --output-on-failure"

# Run a single test by name
docker run --rm -v $(pwd):$(pwd) -w $(pwd) vesta bash -c "ctest --test-dir build -R test_hash_graph --output-on-failure"

# Interactive shell
docker run -v $(pwd):$(pwd) -w $(pwd) -it vesta
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

## Token Optimization with rtk

`rtk` is installed at `~/.local/bin/rtk`. Use it to wrap shell commands whenever possible to reduce token consumption. Prefix commands with `rtk` — it filters and compresses output automatically.

**Use `rtk` for these commands:**

```bash
# Git
rtk git status
rtk git diff
rtk git log
rtk git add <files>
rtk git commit -m "msg"
rtk git push
rtk git pull

# File browsing
rtk ls .
rtk read <file>              # instead of cat/head/tail
rtk grep "pattern" <path>    # instead of grep/rg
rtk find "*.cpp" .           # instead of find

# Docker (builds and tests run inside Docker)
rtk docker ps
rtk docker images
rtk docker logs <container>

# GitHub CLI
rtk gh pr list
rtk gh pr view <number>
rtk gh issue list
```

**Do NOT use `rtk` for:**
- Commands inside `docker run ... bash -c "..."` (rtk is on the host, not in the container)
- Heredocs or piped commands — pass those through directly
- Commands that are already prefixed with `rtk`
