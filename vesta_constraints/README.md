# vesta_constraints

This package provides commonly used constraint types for SLAM and visual SLAM applications. Constraints represent measurements or prior information that relate one or more state variables, forming the factors in a factor graph. All constraint classes derive from `vesta_core::Constraint` and produce Ceres cost functions via their `costFunction()` method.

## Include Path Convention

Headers are included using the subfolder structure:

```cpp
#include <vesta_constraints/2d/absolute_pose_2d_stamped_constraint.h>
#include <vesta_constraints/3d/relative_pose_3d_stamped_constraint.h>
#include <vesta_constraints/vision/reprojection_error_constraint.h>
#include <vesta_constraints/inertial/imu_preintegration.h>
#include <vesta_constraints/motion/cost_functions/unicycle_2d_state_cost_functor.h>
#include <vesta_constraints/common/marginal_constraint.h>
```

All public types reside in the `vesta_constraints` namespace.

## 2D Constraints (`2d/`)

Constraints for 2D position and orientation variables (x, y, yaw).

### Constraint Classes

| Header | Class | Description |
|--------|-------|-------------|
| `absolute_pose_2d_stamped_constraint.h` | `AbsolutePose2DStampedConstraint` | Prior or direct measurement of a 2D pose (position + orientation). Supports partial measurements via index subsets. |
| `relative_pose_2d_stamped_constraint.h` | `RelativePose2DStampedConstraint` | Measurement of the difference between two 2D poses. Used for odometry, scan matching, and other incremental measurements. Supports partial measurements. |

### Cost Functions

| Header | Class | Description |
|--------|-------|-------------|
| `normal_prior_pose_2d.h` | `NormalPriorPose2D` | Prior cost function on both position (2D) and orientation (1D) variables simultaneously. Derives from `ceres::SizedCostFunction`. |
| `normal_prior_orientation_2d.h` | `NormalPriorOrientation2D` | Prior cost function on a 2D orientation variable with proper 2*pi wrap-around handling. |
| `normal_delta_pose_2d.h` | `NormalDeltaPose2D` | Relative cost function between two 2D poses using rigid-body transformation math: `delta = [R1|t1]^-1 * [R2|t2]`. Derives from `ceres::SizedCostFunction`. |
| `normal_delta_orientation_2d.h` | `NormalDeltaOrientation2D` | Relative cost function between two 2D orientation variables with proper 2*pi wrap-around handling. |
| `normal_prior_pose_2d_cost_functor.h` | `NormalPriorPose2DCostFunctor` | Ceres-compatible auto-diff functor for 2D pose priors. |
| `normal_delta_pose_2d_cost_functor.h` | `NormalDeltaPose2DCostFunctor` | Ceres-compatible auto-diff functor for 2D relative pose differences. |

## 3D Constraints (`3d/`)

Constraints for 3D position and orientation variables. Orientations are represented as quaternions (w, x, y, z ordering). Covariance matrices for orientation use the 3D tangent-space parameterization (qx, qy, qz).

### Constraint Classes

| Header | Class | Description |
|--------|-------|-------------|
| `absolute_pose_3d_stamped_constraint.h` | `AbsolutePose3DStampedConstraint` | Prior or direct measurement of a 3D pose (position + orientation). Mean is 7D (x, y, z, qw, qx, qy, qz), covariance is 6x6. |
| `absolute_orientation_3d_stamped_constraint.h` | `AbsoluteOrientation3DStampedConstraint` | Prior or direct measurement of a 3D orientation as a quaternion. Mean is 4D (w, x, y, z), covariance is 3x3. |
| `absolute_orientation_3d_stamped_euler_constraint.h` | `AbsoluteOrientation3DStampedEulerConstraint` | Prior or direct measurement of a 3D orientation as roll-pitch-yaw Euler angles. Supports measurement of a subset of Euler axes. |
| `relative_pose_3d_stamped_constraint.h` | `RelativePose3DStampedConstraint` | Measurement of the difference between two 3D poses. Delta is 7D, covariance is 6x6. Used for visual odometry, lidar odometry, etc. |
| `relative_orientation_3d_stamped_constraint.h` | `RelativeOrientation3DStampedConstraint` | Measurement of the difference between two 3D orientations as quaternions. Delta is 4D (w, x, y, z), covariance is 3x3. |

### Cost Functors

| Header | Class | Description |
|--------|-------|-------------|
| `normal_prior_pose_3d_cost_functor.h` | `NormalPriorPose3DCostFunctor` | Auto-diff functor for 3D pose priors. Operates on position (3D) and orientation (quaternion) variable blocks. |
| `normal_prior_orientation_3d_cost_functor.h` | `NormalPriorOrientation3DCostFunctor` | Auto-diff functor for 3D orientation priors using angle-axis error representation. |
| `normal_prior_orientation_3d_euler_cost_functor.h` | `NormalPriorOrientation3DEulerCostFunctor` | Auto-diff functor for 3D orientation priors using Euler angle error. Supports per-axis subset selection. |
| `normal_delta_pose_3d_cost_functor.h` | `NormalDeltaPose3DCostFunctor` | Auto-diff functor for relative 3D pose differences using rigid-body transformation math. |
| `normal_delta_orientation_3d_cost_functor.h` | `NormalDeltaOrientation3DCostFunctor` | Auto-diff functor for relative 3D orientation differences using angle-axis error representation. |

## Vision Constraints (`vision/`)

Constraints for visual SLAM, including monocular and stereo reprojection error and fixed landmark observations.

### Constraint Classes

| Header | Class | Description |
|--------|-------|-------------|
| `reprojection_error_constraint.h` | `ReprojectionErrorConstraint` | Observation of a 3D point through a pinhole camera model (fx, fy, cx, cy). Constrains camera pose, calibration, and 3D landmark position. |
| `stereo_reprojection_error_constraint.h` | `StereoReprojectionErrorConstraint` | Stereo observation of a 3D point. Uses a stereo camera model (fx, fy, cx, cy, baseline) with a 4D observation (u_left, v_left, u_right, v_right). |
| `fixed_3d_landmark_constraint.h` | `Fixed3DLandmarkConstraint` | Observation of a known 3D fiducial marker (e.g., ARTag). Constrains camera pose and calibration using reprojection of fixed 3D marker points. |
| `fixed_3d_landmark_simple_covariance_constraint.h` | `Fixed3DLandmarkSimpleCovarianceConstraint` | Variant of `Fixed3DLandmarkConstraint` with a simplified covariance model for fiducial marker observations. |

### Cost Functors

| Header | Class | Description |
|--------|-------|-------------|
| `reprojection_error_cost_functor.h` | `ReprojectionErrorCostFunctor` | Auto-diff functor for pinhole reprojection error minimization. |
| `stereo_reprojection_error_cost_functor.h` | `StereoReprojectionErrorCostFunctor` | Auto-diff functor for stereo reprojection error. Computes left and right image projections. |
| `fixed_3d_landmark_cost_functor.h` | `Fixed3DLandmarkCostFunctor` | Auto-diff functor for fixed 3D landmark reprojection error. |
| `fixed_3d_landmark_simple_covariance_cost_functor.h` | `Fixed3DLandmarkSimpleCovarianceCostFunctor` | Auto-diff functor for fixed 3D landmark reprojection error with simplified covariance. |

## Inertial / IMU Constraints (`inertial/`)

Constraints for IMU preintegration and inertial navigation. Based on the on-manifold IMU preintegration formulation from Forster et al. (RSS 2015). The IMU state comprises orientation (quaternion), position, velocity, gyroscope bias, and accelerometer bias (15D error state, 16D state vector).

### Constraint Classes

| Header | Class | Description |
|--------|-------|-------------|
| `absolute_imu_state_3d_stamped_constraint.h` | `AbsoluteImuState3DStampedConstraint` | Prior or direct measurement of the full 3D IMU state. Mean is 16D, covariance is 15x15 in error-state order (Q, P, V, BG, BA). |
| `relative_imu_state_3d_stamped_constraint.h` | `RelativeImuState3DStampedConstraint` | Preintegrated IMU factor between two IMU states (10 variable blocks). Includes bias-correction Jacobians for first-order bias updates without re-integration. |

### Preintegration

| Header | Class / Struct | Description |
|--------|----------------|-------------|
| `imu_preintegration.h` | `ImuPreintegration` | High-level IMU preintegration manager. Buffers IMU data, creates preintegrated factors between keyframes, and provides state prediction. |
| `imu_preintegration.h` | `ImuPreintegrationParams` | Configuration struct for IMU noise covariances and gravity vector. |
| `imu_preintegration.h` | `ImuState` | Lightweight struct holding a full IMU state (orientation, position, velocity, biases). |
| `imu_preintegrator.h` | `ImuPreintegrator` | Low-level preintegration engine. Performs midpoint integration, accumulates preintegrated deltas (rotation, position, velocity), propagates covariance, and computes bias-correction Jacobians. |
| `imu_preintegrator.h` | `ImuData` | Timestamped IMU measurement (angular velocity, linear acceleration). |
| `imu_preintegrator.h` | `PreintegrationDelta` | Accumulated preintegrated measurements (dt, delta q/p/v, covariance, sqrt information). |
| `imu_preintegrator.h` | `PreintegrationJacobian` | First-order Jacobians of preintegrated measurements with respect to gyroscope and accelerometer biases. |

### Cost Functors

| Header | Class | Description |
|--------|-------|-------------|
| `normal_delta_imu_state_3d_cost_functor.h` | `NormalDeltaImuState3DCostFunctor` | Auto-diff functor for the relative IMU preintegration residual (15D). Implements bias-corrected preintegration error with gravity compensation. |
| `normal_prior_imu_state_3d_cost_functor.h` | `NormalPriorImuState3DCostFunctor` | Auto-diff functor for the absolute IMU state prior (15D). Applies prior on all five state components simultaneously. |

### Utilities

| Header | Function | Description |
|--------|----------|-------------|
| `so3_utils.h` | `skewSymmetric()` | Computes the 3x3 skew-symmetric matrix of a 3-vector. |
| `so3_utils.h` | `expMapSO3()` | SO(3) exponential map via the Rodrigues formula (rotation vector to rotation matrix). |
| `so3_utils.h` | `logMapSO3()` | SO(3) logarithmic map (rotation matrix to rotation vector). |

## Motion Model Constraints (`motion/`)

Kinematic motion models for state prediction.

| Header | Class / Function | Description |
|--------|------------------|-------------|
| `unicycle_2d_predict.h` | `predict()` | Free function template that predicts a new 2D state (position, orientation, velocity, acceleration) given a time delta using a unicycle motion model. |
| `unicycle_2d_state_cost_functor.h` | `Unicycle2DStateCostFunctor` | Auto-diff functor that computes the cost between a predicted and observed 2D state vector (8D: x, y, yaw, x_vel, y_vel, yaw_vel, x_acc, y_acc). |

## Common / Generic Constraints (`common/`)

Type-generic constraints and marginalization utilities that work with any variable type.

### Generic Constraint Templates

| Header | Class | Description |
|--------|-------|-------------|
| `absolute_constraint.h` | `AbsoluteConstraint<Variable>` | Templated prior/measurement constraint on a single variable of any type. Uses element-wise comparison via `ceres::NormalPrior`. |
| `absolute_constraint_impl.h` | (implementation) | Template implementation details for `AbsoluteConstraint`. |
| `relative_constraint.h` | `RelativeConstraint<Variable>` | Templated constraint on the element-wise difference between two variables of the same type. |
| `relative_constraint_impl.h` | (implementation) | Template implementation details for `RelativeConstraint`. |

### Cost Functions

| Header | Class | Description |
|--------|-------|-------------|
| `normal_delta.h` | `NormalDelta` | Generic cost function modeling `cost(x) = \|\|A * ((x1 - x0) - b)\|\|^2` for element-wise variable differences. Supports rectangular A matrices for partial measurements. Derives from `ceres::CostFunction`. |

### Marginalization

| Header | Class / Function | Description |
|--------|------------------|-------------|
| `marginal_constraint.h` | `MarginalConstraint` | Constraint encoding remaining marginal information after variable elimination. Cost form: `A1*(x1-x1_bar) + A2*(x2-x2_bar) + ... + b`. |
| `marginal_cost_function.h` | `MarginalCostFunction` | Ceres cost function for precomputed marginal distributions. Supports multiple variable blocks with manifold-aware minus operators. |
| `marginalize_variables.h` | `computeEliminationOrder()` | Computes an efficient variable elimination order using CCOLAMD, ensuring marginalized variables are eliminated first. |
| `marginalize_variables.h` | `marginalizeVariables()` | Generates a transaction that marginalizes out specified variables, producing linear `MarginalConstraint` factors on the remaining connected variables. |

### Utilities

| Header | Class | Description |
|--------|-------|-------------|
| `uuid_ordering.h` | `UuidOrdering` | Bidirectional mapping between UUIDs and sequential indices. Used for variable elimination ordering in marginalization. |
| `variable_constraints.h` | `VariableConstraints` | Adjacency structure mapping variable indices to their connected constraint indices. Used internally for marginalization bookkeeping. |

## Architecture Notes

- All constraint classes derive from `vesta_core::Constraint` and implement `costFunction()` to return a `ceres::CostFunction*`.
- Constraints use the square root information matrix internally for numerical stability. Covariance matrices provided at construction are decomposed via Cholesky factorization.
- Polymorphic serialization is handled via `BOOST_CLASS_EXPORT` macros, enabling constraints to be serialized and deserialized through base class pointers.
- 3D cost functors are designed for Ceres automatic differentiation (`operator()` templates) while 2D cost functions typically provide analytical Jacobians via `ceres::SizedCostFunction::Evaluate()`.
