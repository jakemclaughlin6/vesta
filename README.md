# Vesta

Vesta is a standalone C++ library for factor graph-based sensor fusion, built as an extension to [Google Ceres Solver](http://ceres-solver.org). It is designed specifically for SLAM (Simultaneous Localization and Mapping) and Visual SLAM applications.

Vesta is a fork of [fuse](https://github.com/locusrobotics/fuse) (by Locus Robotics), re-engineered to be **fully ROS-agnostic**. Where fuse was tightly coupled to the ROS ecosystem (plugins via pluginlib, ROS parameters, nodelets, topics), Vesta is a pure C++ library with no framework dependencies beyond its core math libraries. This makes it suitable for embedding in any C++ application -- robotics or otherwise.

## Key Differences from Fuse

- **No ROS dependency** -- pure C++ library with CMake build
- **No pluginlib** -- polymorphic serialization via Boost instead
- **No internal threads or event loops** -- synchronous, caller-driven API
- **C++20** with modern CMake
- **Ceres 2.2+ Manifold API** -- uses the modern manifold interface instead of the deprecated local parameterization
- **Visual SLAM support** -- camera models, landmark variables, reprojection error constraints, and IMU preintegration
- **Boost serialization** -- all variables, constraints, and graphs are fully serializable

## Packages

Vesta is organized into six packages, each with a focused responsibility:

| Package | Description |
|---------|-------------|
| [**vesta_core**](vesta_core/) | Abstract base classes and interfaces (`Variable`, `Constraint`, `Graph`, `Transaction`, `Loss`, `Manifold`), UUID utilities, timestamps, and Boost serialization support |
| [**vesta_variables**](vesta_variables/) | Concrete variable types: 2D/3D poses, velocities, accelerations, IMU biases, camera intrinsics, and 2D/3D landmarks |
| [**vesta_constraints**](vesta_constraints/) | Concrete constraint types: absolute/relative pose constraints, reprojection errors, stereo vision, IMU preintegration, motion models, and marginalization utilities |
| [**vesta_graphs**](vesta_graphs/) | Graph storage implementations. `HashGraph` provides O(1) lookup with a persistent `ceres::Problem` for incremental solving |
| [**vesta_loss**](vesta_loss/) | Robust loss functions (Huber, Cauchy, Tukey, DCS, etc.) with Boost serialization, wrapping `ceres::LossFunction` |
| [**vesta_optimizers**](vesta_optimizers/) | High-level optimizers: `BatchOptimizer` for full solves and `FixedLagSmoother` for real-time sliding-window SLAM with automatic marginalization |

## Dependencies

- **CMake** >= 3.20
- **C++20** compiler (GCC 10+, Clang 13+)
- **Ceres Solver** >= 2.2
- **Eigen3**
- **Boost** (serialization component)
- **glog**
- **SuiteSparse** (CCOLAMD)
- **Google Test** (for tests, optional)

## Building

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt install libceres-dev libeigen3-dev libboost-serialization-dev \
                 libgoogle-glog-dev libsuitesparse-dev libgtest-dev

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

To disable tests:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

## Usage

Vesta follows a simple, synchronous pattern:

1. **Create variables** representing your state (poses, landmarks, etc.)
2. **Create constraints** representing measurements (odometry, reprojection errors, GPS, etc.)
3. **Pack them into transactions** and submit to an optimizer
4. **Call optimize()** to run the Ceres solver
5. **Read results** from the optimized graph

### Example: 2D Pose Graph SLAM

```cpp
#include <vesta_variables/2d/position_2d_stamped.h>
#include <vesta_variables/2d/orientation_2d_stamped.h>
#include <vesta_constraints/2d/relative_pose_2d_stamped_constraint.h>
#include <vesta_constraints/2d/absolute_pose_2d_stamped_constraint.h>
#include <vesta_graphs/hash_graph.h>
#include <vesta_optimizers/batch_optimizer.h>
#include <vesta_core/transaction.h>

// Create an optimizer with a hash graph
auto graph = std::make_unique<vesta_graphs::HashGraph>();
vesta_optimizers::BatchOptimizerParams params;
vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

// Create a transaction with variables and constraints
auto transaction = std::make_shared<vesta_core::Transaction>();

// Add pose variables at two timestamps
vesta_core::Timestamp t1(0, 0);
vesta_core::Timestamp t2(1, 0);

auto pos1 = vesta_variables::Position2DStamped(t1);
auto ori1 = vesta_variables::Orientation2DStamped(t1);
auto pos2 = vesta_variables::Position2DStamped(t2);
auto ori2 = vesta_variables::Orientation2DStamped(t2);

transaction->addVariable(std::make_shared<vesta_variables::Position2DStamped>(pos1));
transaction->addVariable(std::make_shared<vesta_variables::Orientation2DStamped>(ori1));
transaction->addVariable(std::make_shared<vesta_variables::Position2DStamped>(pos2));
transaction->addVariable(std::make_shared<vesta_variables::Orientation2DStamped>(ori2));

// Add an absolute prior on the first pose
Eigen::Vector3d mean_abs;
mean_abs << 0.0, 0.0, 0.0;  // x, y, yaw
Eigen::Matrix3d cov_abs = Eigen::Matrix3d::Identity() * 0.01;
transaction->addConstraint(
    std::make_shared<vesta_constraints::AbsolutePose2DStampedConstraint>(
        "gps", pos1, ori1, mean_abs, cov_abs));

// Add a relative odometry constraint between poses
Eigen::Vector3d delta;
delta << 1.0, 0.0, 0.1;  // dx, dy, dyaw
Eigen::Matrix3d cov_rel = Eigen::Matrix3d::Identity() * 0.1;
transaction->addConstraint(
    std::make_shared<vesta_constraints::RelativePose2DStampedConstraint>(
        "odometry", pos1, ori1, pos2, ori2, delta, cov_rel));

// Submit and optimize
optimizer.addTransaction("sensors", transaction);
ceres::Solver::Summary summary = optimizer.optimize();

// Access optimized values
const auto& optimized_graph = optimizer.graph();
```

### Example: Fixed-Lag Smoother for Real-Time SLAM

```cpp
#include <vesta_graphs/hash_graph.h>
#include <vesta_optimizers/fixed_lag_smoother.h>

auto graph = std::make_unique<vesta_graphs::HashGraph>();
vesta_optimizers::FixedLagSmootherParams params;
params.lag_duration = vesta_core::Duration::fromSec(5.0);  // 5-second window

vesta_optimizers::FixedLagSmoother smoother(params, std::move(graph));

// In your sensor callback loop:
// smoother.addTransaction("lidar", lidar_transaction);
// smoother.addTransaction("imu", imu_transaction);
// auto summary = smoother.optimize();
// Variables older than 5 seconds are automatically marginalized
```

### Example: Bundle Adjustment with Schur Complement Solver

```cpp
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/vision/pinhole_camera_fixed.h>
#include <vesta_variables/vision/point_3d_landmark.h>
#include <vesta_constraints/vision/reprojection_error_constraint.h>
#include <vesta_graphs/hash_graph.h>
#include <vesta_optimizers/batch_optimizer.h>
#include <vesta_core/schur_ordering.h>
#include <vesta_loss/huber_loss.h>

// -- Set up the optimizer with a Schur complement linear solver --
auto graph = std::make_unique<vesta_graphs::HashGraph>();
vesta_optimizers::BatchOptimizerParams params;
params.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
params.solver_options.max_num_iterations = 50;

vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));

// -- Create camera intrinsics (fixed, not optimized) --
// Camera 0 with fx=525, fy=525, cx=320, cy=240
auto camera = std::make_shared<vesta_variables::PinholeCameraFixed>(
    /*camera_id=*/0, /*fx=*/525.0, /*fy=*/525.0, /*cx=*/320.0, /*cy=*/240.0);

// -- Build the bundle adjustment problem --
auto transaction = std::make_shared<vesta_core::Transaction>();
transaction->addVariable(camera);

// Add camera poses (one per keyframe)
const int num_keyframes = 10;
std::vector<vesta_variables::Position3DStamped> positions;
std::vector<vesta_variables::Orientation3DStamped> orientations;

for (int i = 0; i < num_keyframes; ++i) {
  vesta_core::Timestamp t(i, 0);
  auto device = vesta_core::uuid::generate("camera0");

  auto pos = std::make_shared<vesta_variables::Position3DStamped>(t, device);
  auto ori = std::make_shared<vesta_variables::Orientation3DStamped>(t, device);

  // Initialize with your visual odometry estimate
  pos->x() = initial_positions[i].x();
  pos->y() = initial_positions[i].y();
  pos->z() = initial_positions[i].z();
  ori->w() = initial_orientations[i].w();
  ori->x() = initial_orientations[i].x();
  ori->y() = initial_orientations[i].y();
  ori->z() = initial_orientations[i].z();

  transaction->addVariable(pos);
  transaction->addVariable(ori);
  positions.push_back(*pos);
  orientations.push_back(*ori);
}

// Add 3D landmarks
// Point3DLandmark returns schurGroup() == 0, so it is automatically
// placed in the first elimination group for the Schur complement solver
const int num_landmarks = 500;
std::vector<vesta_variables::Point3DLandmark> landmarks;

for (int j = 0; j < num_landmarks; ++j) {
  auto lm = std::make_shared<vesta_variables::Point3DLandmark>(/*landmark_id=*/j);
  lm->x() = initial_points[j].x();
  lm->y() = initial_points[j].y();
  lm->z() = initial_points[j].z();

  transaction->addVariable(lm);
  landmarks.push_back(*lm);
}

// Add reprojection error constraints for each observation
auto loss = std::make_shared<vesta_loss::HuberLoss>(1.0);  // Robust to outliers

for (const auto& obs : observations) {
  vesta_core::Vector2d pixel;
  pixel << obs.u, obs.v;

  vesta_core::Matrix2d cov = vesta_core::Matrix2d::Identity();  // 1px std dev

  auto constraint =
      std::make_shared<vesta_constraints::ReprojectionErrorConstraint>(
          "visual_frontend",
          positions[obs.keyframe_idx],
          orientations[obs.keyframe_idx],
          *camera,
          landmarks[obs.landmark_idx],
          pixel, cov);
  constraint->loss(loss);

  transaction->addConstraint(constraint);
}

// -- Submit and optimize --
optimizer.addTransaction("ba", transaction);

// Build Schur ordering from the graph (landmarks in group 0, poses in group 1)
// This is done automatically when using SPARSE_SCHUR with Point3DLandmark
// since Point3DLandmark::schurGroup() returns 0.
// To explicitly set it:
auto ordering = vesta_core::buildSchurOrdering(optimizer.graph());
if (ordering) {
  params.solver_options.linear_solver_ordering = ordering;
}

ceres::Solver::Summary summary = optimizer.optimize();

// -- Read optimized results --
const auto& optimized_graph = optimizer.graph();
// Iterate variables to extract optimized camera poses and landmark positions
```

Key points for bundle adjustment performance:
- `Point3DLandmark::schurGroup()` returns `0`, placing landmarks in the first elimination group. Camera poses default to group `1`.
- Use `SPARSE_SCHUR` for large problems (many landmarks) or `DENSE_SCHUR` for small-to-medium problems.
- `buildSchurOrdering()` reads each variable's `schurGroup()` and builds the `ceres::ParameterBlockOrdering` automatically.
- Use `PinholeCameraFixed` to hold intrinsics constant, or `PinholeCamera` to jointly optimize intrinsics.
- Apply a robust loss like `HuberLoss` to handle feature matching outliers.

## Architecture

Vesta is a **library**, not a framework. It does not manage threads, event loops, or sensor drivers. You control the execution:

```
Sensor Data --> Your Code --> Transaction --> Optimizer --> Optimized Graph --> Your Code
```

The core optimization loop:

```
addTransaction("sensor_a", transaction_a)
addTransaction("sensor_b", transaction_b)
summary = optimize()          // Runs Ceres solver
graph = optimizer.graph()     // Read optimized state
```

### Core Abstractions

- **Variable**: A block of related scalar values (e.g., a 3D position is 3 scalars). Each variable instance has a UUID and optionally a timestamp.
- **Constraint**: A cost function connecting one or more variables. Wraps a `ceres::CostFunction` with an optional loss function.
- **Graph**: Stores variables and constraints. The `HashGraph` implementation maintains a live `ceres::Problem`.
- **Transaction**: A batch of variable/constraint additions and removals, applied atomically.
- **Optimizer**: Accepts transactions, maintains a graph, and runs the Ceres solver.

## API Documentation

- [Variables](doc/Variables.md) -- Design principles and custom variable creation
- [Constraints](doc/Constraints.md) -- Cost functions, loss functions, and custom constraint creation

## License

BSD License. See [LICENSE](LICENSE) for details.

Originally developed by Locus Robotics. Modified and maintained as Vesta.
