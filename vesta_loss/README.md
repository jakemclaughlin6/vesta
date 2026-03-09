# vesta_loss

Robust loss function implementations for outlier rejection in nonlinear least squares optimization.

## Overview

In factor graph optimization, outlier measurements can severely distort the solution. Loss functions (also called robust kernels) reduce the influence of residuals with large errors, making the optimization robust to outliers.

Each loss function wraps a `ceres::LossFunction` and adds Boost serialization support for graph persistence. All loss functions derive from `vesta_core::Loss`.

## Loss Functions

| Header | Class | Description |
|--------|-------|-------------|
| `trivial_loss.h` | `TrivialLoss` | Identity (no-op) -- equivalent to standard least squares |
| `huber_loss.h` | `HuberLoss` | Quadratic for small residuals, linear for large. Classic robust choice |
| `cauchy_loss.h` | `CauchyLoss` | Heavy-tailed, highly robust to large outliers |
| `arctan_loss.h` | `ArctanLoss` | Bounded influence function using arctan |
| `fair_loss.h` | `FairLoss` | Smooth approximation between L1 and L2 |
| `geman_mcclure_loss.h` | `GemanMcClureLoss` | Redescending influence -- large outliers are fully suppressed |
| `welsch_loss.h` | `WelschLoss` | Exponential-based redescending loss |
| `tukey_loss.h` | `TukeyLoss` | Tukey biweight -- zero influence beyond a threshold |
| `dcs_loss.h` | `DCSLoss` | Dynamic Covariance Scaling -- adapts to the residual distribution |
| `softlone_loss.h` | `SoftLOneLoss` | Smooth approximation to the L1 norm |
| `tolerant_loss.h` | `TolerantLoss` | Ignores residuals below a tolerance, penalizes above |
| `scaled_loss.h` | `ScaledLoss` | Wraps another loss function with a scale factor |
| `composed_loss.h` | `ComposedLoss` | Composes two loss functions: `f(g(x))` |

### Optional

| Header | Description |
|--------|-------------|
| `qwt_loss_plot.h` | Qt/Qwt-based visualization of loss function curves (requires Qt5 and Qwt) |

## Include Convention

```cpp
#include <vesta_loss/huber_loss.h>
#include <vesta_loss/tukey_loss.h>
#include <vesta_loss/scaled_loss.h>
```

## Usage

```cpp
#include <vesta_loss/huber_loss.h>

// Create a Huber loss with scale parameter 1.0
auto loss = std::make_shared<vesta_loss::HuberLoss>(1.0);

// Attach to a constraint via Transaction
transaction->addConstraint(my_constraint, loss);
```
