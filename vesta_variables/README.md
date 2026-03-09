# vesta_variables

Commonly used variable types for state estimation in factor graph optimization. All variables derive from `vesta_core::Variable` and support Boost serialization for persistence and polymorphic deserialization.

## Architecture

Variables are organized into four subdirectories by domain:

- **`2d/`** -- Planar state variables (position, orientation, velocity, acceleration)
- **`3d/`** -- Spatial state variables (position, orientation, velocity, acceleration, IMU biases)
- **`vision/`** -- Camera intrinsic models and visual landmarks
- **`common/`** -- Base classes and mixins shared across variable types

Most variables inherit from two base classes via multiple inheritance:

- `FixedSizeVariable<N>` -- Provides a compile-time-sized `std::array<double, N>` for the variable's parameter block
- `Stamped` -- Adds a timestamp and device ID for time-varying quantities

Vision variables (cameras and landmarks) are not stamped, as they represent quantities that persist across time.

## Include Convention

```cpp
#include <vesta_variables/2d/position_2d_stamped.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/vision/pinhole_camera.h>
#include <vesta_variables/common/fixed_size_variable.h>
```

## Variable Reference

### Common (`common/`)

| Class | Header | Description |
|---|---|---|
| `FixedSizeVariable<N>` | `common/fixed_size_variable.h` | Template base class providing a statically-sized `std::array<double, N>` parameter block. Implements `Variable::data()` and `Variable::size()`. |
| `Stamped` | `common/stamped.h` | Mixin base class that associates a timestamp and device UUID with a variable. Used via multiple inheritance for time-varying quantities. |

### 2D Variables (`2d/`)

All 2D variables inherit from both `FixedSizeVariable<N>` and `Stamped`.

| Class | Header | Size | Parameters | Description |
|---|---|---|---|---|
| `Position2DStamped` | `2d/position_2d_stamped.h` | 2 | x, y | 2D position in a plane |
| `Orientation2DStamped` | `2d/orientation_2d_stamped.h` | 1 | yaw | 2D heading angle with custom manifold for angle wrapping |
| `VelocityLinear2DStamped` | `2d/velocity_linear_2d_stamped.h` | 2 | vx, vy | 2D linear velocity |
| `VelocityAngular2DStamped` | `2d/velocity_angular_2d_stamped.h` | 1 | vyaw | 2D angular velocity (yaw rate) |
| `AccelerationLinear2DStamped` | `2d/acceleration_linear_2d_stamped.h` | 2 | ax, ay | 2D linear acceleration |
| `AccelerationAngular2DStamped` | `2d/acceleration_angular_2d_stamped.h` | 1 | ayaw | 2D angular acceleration |

The `Orientation2DStamped` variable provides a custom `Orientation2DManifold` that handles the 2*pi angle wrapping discontinuity.

### 3D Variables (`3d/`)

All 3D variables inherit from both `FixedSizeVariable<N>` and `Stamped`.

| Class | Header | Size | Parameters | Description |
|---|---|---|---|---|
| `Position3DStamped` | `3d/position_3d_stamped.h` | 3 | x, y, z | 3D position |
| `Orientation3DStamped` | `3d/orientation_3d_stamped.h` | 4 | w, x, y, z | 3D orientation as a quaternion (tangent size 3) with custom manifold |
| `VelocityLinear3DStamped` | `3d/velocity_linear_3d_stamped.h` | 3 | vx, vy, vz | 3D linear velocity |
| `VelocityAngular3DStamped` | `3d/velocity_angular_3d_stamped.h` | 3 | vroll, vpitch, vyaw | 3D angular velocity |
| `AccelerationLinear3DStamped` | `3d/acceleration_linear_3d_stamped.h` | 3 | ax, ay, az | 3D linear acceleration |
| `AccelerationAngular3DStamped` | `3d/acceleration_angular_3d_stamped.h` | 3 | aroll, apitch, ayaw | 3D angular acceleration |
| `AccelerationBias3DStamped` | `3d/acceleration_bias_3d_stamped.h` | 3 | bx, by, bz | Accelerometer bias for IMU modeling |
| `GyroscopeBias3DStamped` | `3d/gyroscope_bias_3d_stamped.h` | 3 | bx, by, bz | Gyroscope bias for IMU modeling |

The `Orientation3DStamped` variable uses a quaternion representation (w, x, y, z) with an `Orientation3DManifold` that applies angle-axis updates via Rodrigues formula. The ambient size is 4 but the tangent size is 3, reflecting the true degrees of freedom.

### Vision Variables (`vision/`)

#### Camera Models

Camera variables inherit from `BaseCamera<N>` and are identified by a camera ID rather than a timestamp. Fixed variants override `holdConstant()` to prevent optimization of intrinsic parameters.

| Class | Header | Size | Parameters | Description |
|---|---|---|---|---|
| `BaseCamera<N>` | `vision/base_camera.h` | N | (template) | Template base class for camera intrinsic variables, identified by camera ID |
| `PinholeCamera` | `vision/pinhole_camera.h` | 4 | fx, fy, cx, cy | Standard pinhole camera intrinsics (optimizable) |
| `PinholeCameraFixed` | `vision/pinhole_camera_fixed.h` | 4 | fx, fy, cx, cy | Pinhole camera intrinsics held constant during optimization |
| `PinholeCameraRadial` | `vision/pinhole_camera_radial.h` | 3 | f, r1, r2 | Pinhole camera with shared focal length and two radial distortion coefficients |
| `StereoCamera` | `vision/stereo_camera.h` | 5 | fx, fy, cx, cy, baseline | Stereo camera intrinsics including baseline distance (optimizable) |
| `StereoCameraFixed` | `vision/stereo_camera_fixed.h` | 5 | fx, fy, cx, cy, baseline | Stereo camera intrinsics held constant during optimization |

#### Landmark Variables

Landmark variables represent persistent spatial features identified by a landmark ID. They are not stamped since landmarks exist across time. Fixed variants hold the landmark position constant during optimization. Landmarks return Schur group 0 for efficient elimination in Schur complement-based solvers.

| Class | Header | Size | Parameters | Description |
|---|---|---|---|---|
| `Point2DLandmark` | `vision/point_2d_landmark.h` | 2 | x, y | 2D point landmark (optimizable) |
| `Point2DFixedLandmark` | `vision/point_2d_fixed_landmark.h` | 2 | x, y | 2D point landmark held constant during optimization |
| `Point3DLandmark` | `vision/point_3d_landmark.h` | 3 | x, y, z | 3D point landmark (optimizable) |
| `Point3DFixedLandmark` | `vision/point_3d_fixed_landmark.h` | 3 | x, y, z | 3D point landmark held constant during optimization |
