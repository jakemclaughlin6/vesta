#pragma once

#include <vesta_constraints/vision/triangulation.h>
#include <vesta_core/eigen.h>

#include <ceres/cost_function.h>
#include <Eigen/Dense>
#include <Eigen/QR>

#include <cassert>
#include <vector>

namespace vesta_constraints
{

/**
 * @brief Cost function that implements nullspace projection (structureless vision factor).
 *
 * Given N >= 2 observations of a 3D landmark from N camera poses, this cost function
 * analytically eliminates the landmark by projecting the stacked reprojection errors
 * through the left nullspace of the landmark Jacobian. The result is a (2N-3)-dimensional
 * residual that depends only on the camera poses, not the landmark position.
 *
 * Algorithm:
 *   1. Triangulate the landmark from current poses + stored measurements (DLT)
 *   2. Compute per-observation reprojection errors and Jacobians
 *   3. Stack the landmark Jacobian E (2N x 3), compute SVD
 *   4. Project residuals and pose Jacobians through E's left nullspace
 *
 * Parameter blocks: [pos_0(3), ori_0(4), pos_1(3), ori_1(4), ..., pos_{N-1}(3), ori_{N-1}(4)]
 * Residual dimension: 2N - 3
 *
 * Camera intrinsics are stored internally (fixed).
 */
class NullspaceProjectionCostFunction : public ceres::CostFunction
{
public:
  /**
   * @brief Construct the cost function.
   *
   * @param[in] observations     Per-observation 2D pixel measurements (u, v)
   * @param[in] sqrt_information Square root information matrix for the observation noise
   * @param[in] calibration      Camera intrinsics [fx, fy, cx, cy]
   */
  NullspaceProjectionCostFunction(std::vector<Eigen::Vector2d> observations,
                                  const vesta_core::Matrix2d& sqrt_information, const Eigen::Vector4d& calibration)
    : observations_(std::move(observations)), sqrt_information_(sqrt_information), calibration_(calibration)
  {
    const int n = static_cast<int>(observations_.size());
    assert(n >= 2);

    for (int i = 0; i < n; ++i)
    {
      mutable_parameter_block_sizes()->push_back(3);  // position
      mutable_parameter_block_sizes()->push_back(4);  // orientation (quaternion)
    }
    set_num_residuals(2 * n - 3);
  }

  ~NullspaceProjectionCostFunction() override = default;

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    const int n = static_cast<int>(observations_.size());
    const int residual_dim = 2 * n - 3;

    const double fx = calibration_[0];
    const double fy = calibration_[1];
    const double cx = calibration_[2];
    const double cy = calibration_[3];

    // Extract rotation matrices from quaternions
    std::vector<Eigen::Matrix3d> R_wc_list(n);
    std::vector<Eigen::Vector3d> p_world_list(n);

    for (int i = 0; i < n; ++i)
    {
      p_world_list[i] = Eigen::Map<const Eigen::Vector3d>(parameters[2 * i]);

      const double* q = parameters[2 * i + 1];
      const double w = q[0], x = q[1], y = q[2], z = q[3];

      // clang-format off
      R_wc_list[i] << 1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
                       2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
                       2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y);
      // clang-format on
    }

    // Triangulate the landmark
    Eigen::Vector3d landmark;
    try
    {
      landmark = triangulateDLT(R_wc_list, p_world_list, observations_, calibration_);
    }
    catch (...)
    {
      Eigen::Map<Eigen::VectorXd>(residuals, residual_dim).setZero();
      if (jacobians)
      {
        for (int i = 0; i < 2 * n; ++i)
        {
          if (jacobians[i])
          {
            Eigen::Map<vesta_core::MatrixXd>(jacobians[i], residual_dim, parameter_block_sizes()[i]).setZero();
          }
        }
      }
      return true;
    }

    // Compute per-observation reprojection errors and Jacobians
    Eigen::VectorXd b(2 * n);                                          // stacked weighted residuals
    Eigen::MatrixXd E(2 * n, 3);                                       // stacked landmark Jacobians
    std::vector<Eigen::Matrix<double, 2, 3>> F_pos(n);                 // pose position Jacobians
    std::vector<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> F_ori(n);  // pose orientation Jacobians (ambient)

    for (int i = 0; i < n; ++i)
    {
      const Eigen::Matrix3d R_cw = R_wc_list[i].transpose();
      const Eigen::Vector3d d = landmark - p_world_list[i];
      const Eigen::Vector3d p_cam = R_cw * d;

      const double X = p_cam[0], Y = p_cam[1], Z = p_cam[2];

      // Projection
      const double u_proj = fx * X / Z + cx;
      const double v_proj = fy * Y / Z + cy;

      // Weighted reprojection error
      Eigen::Vector2d raw_error(u_proj - observations_[i][0], v_proj - observations_[i][1]);
      b.segment<2>(2 * i) = sqrt_information_ * raw_error;

      // Projection Jacobian w.r.t. p_cam (2x3)
      Eigen::Matrix<double, 2, 3> J_proj;
      // clang-format off
      J_proj << fx / Z, 0.0,    -fx * X / (Z * Z),
                0.0,    fy / Z, -fy * Y / (Z * Z);
      // clang-format on

      // Weighted projection Jacobian
      const Eigen::Matrix<double, 2, 3> AJ = sqrt_information_ * J_proj;

      // Jacobians via chain rule: dp_cam/dp_w = -R_cw, dp_cam/dX = R_cw
      F_pos[i] = AJ * (-R_cw);
      E.block<2, 3>(2 * i, 0) = AJ * R_cw;

      // Quaternion Jacobian: dp_cam/dq where p_cam = R(q^{-1}) * d
      // Uses the analytical derivative of quaternion rotation.
      const double* q = parameters[2 * i + 1];
      const double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
      const Eigen::Vector3d qu(qx, qy, qz);
      const Eigen::Vector3d cross_u_d = qu.cross(d);

      // dp_cam/dw = -2*(u x d)
      const Eigen::Vector3d dp_dw = -2.0 * cross_u_d;

      // dp_cam/d[x,y,z] = 2*(w*skew(d) - skew(u x d) - skew(u)*skew(d))
      const Eigen::Matrix3d skew_d = skewSymmetric(d);
      const Eigen::Matrix3d skew_c = skewSymmetric(cross_u_d);
      const Eigen::Matrix3d skew_u = skewSymmetric(qu);
      const Eigen::Matrix3d dp_du = 2.0 * (qw * skew_d - skew_c - skew_u * skew_d);

      Eigen::Matrix<double, 3, 4> dp_dq;
      dp_dq.col(0) = dp_dw;
      dp_dq.rightCols<3>() = dp_du;

      F_ori[i] = AJ * dp_dq;
    }

    // Compute left nullspace of E via QR decomposition (deterministic, no sign ambiguity)
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(E);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2 * n, 2 * n);
    Q = qr.householderQ() * Q;
    const Eigen::MatrixXd E_null = Q.rightCols(residual_dim);  // (2N x 2N-3)

    // Projected residuals
    Eigen::Map<Eigen::VectorXd> r(residuals, residual_dim);
    r = E_null.transpose() * b;

    // Projected Jacobians
    if (jacobians)
    {
      for (int i = 0; i < n; ++i)
      {
        // Rows of E_null corresponding to observation i
        const Eigen::MatrixXd E_null_i_T = E_null.middleRows(2 * i, 2).transpose();  // (2N-3) x 2

        if (jacobians[2 * i])
        {
          Eigen::Map<vesta_core::MatrixXd> J(jacobians[2 * i], residual_dim, 3);
          J = E_null_i_T * F_pos[i];
        }
        if (jacobians[2 * i + 1])
        {
          Eigen::Map<vesta_core::MatrixXd> J(jacobians[2 * i + 1], residual_dim, 4);
          J = E_null_i_T * F_ori[i];
        }
      }
    }

    return true;
  }

private:
  static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v)
  {
    Eigen::Matrix3d S;
    // clang-format off
    S <<     0, -v[2],  v[1],
          v[2],     0, -v[0],
         -v[1],  v[0],     0;
    // clang-format on
    return S;
  }

  std::vector<Eigen::Vector2d> observations_;
  vesta_core::Matrix2d sqrt_information_;
  Eigen::Vector4d calibration_;
};

}  // namespace vesta_constraints
