#pragma once

#include <vesta_constraints/3d/extrinsic_pose_3d.h>
#include <vesta_constraints/vision/triangulation.h>
#include <vesta_core/eigen.h>

#include <ceres/cost_function.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Eigen/QR>

#include <cassert>
#include <cstring>
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
 * Parameter blocks (without extrinsic):
 *   [pos_0(3), ori_0(4), pos_1(3), ori_1(4), ..., pos_{N-1}(3), ori_{N-1}(4)]
 *
 * Parameter blocks (with extrinsic):
 *   [pos_0(3), ori_0(4), ..., pos_{N-1}(3), ori_{N-1}(4), ext_pos(3), ext_ori(4)]
 *
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
   * @param[in] has_extrinsic    Whether extrinsic parameter blocks are appended
   */
  NullspaceProjectionCostFunction(std::vector<Eigen::Vector2d> observations,
                                  const vesta_core::Matrix2d& sqrt_information, const Eigen::Vector4d& calibration,
                                  bool has_extrinsic = false)
    : observations_(std::move(observations))
    , sqrt_information_(sqrt_information)
    , calibration_(calibration)
    , has_extrinsic_(has_extrinsic)
  {
    const int n = static_cast<int>(observations_.size());
    assert(n >= 2);

    for (int i = 0; i < n; ++i)
    {
      mutable_parameter_block_sizes()->push_back(3);  // position
      mutable_parameter_block_sizes()->push_back(4);  // orientation (quaternion)
    }
    if (has_extrinsic_)
    {
      mutable_parameter_block_sizes()->push_back(3);  // ext_position
      mutable_parameter_block_sizes()->push_back(4);  // ext_orientation
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

    // Compute sensor-frame poses (either directly from parameters or via extrinsic transform)
    std::vector<Eigen::Matrix3d> R_wc_list(n);
    std::vector<Eigen::Vector3d> p_world_list(n);
    std::vector<Eigen::Vector4d> q_sensor_list(n);  // sensor quaternions (w,x,y,z)

    if (has_extrinsic_)
    {
      const double* ext_pos = parameters[2 * n];
      const double* ext_ori = parameters[2 * n + 1];

      for (int i = 0; i < n; ++i)
      {
        const double* body_pos = parameters[2 * i];
        const double* body_ori = parameters[2 * i + 1];

        double sensor_pos[3];
        double sensor_ori[4];
        computeSensorPose(body_pos, body_ori, ext_pos, ext_ori, sensor_pos, sensor_ori);

        p_world_list[i] = Eigen::Map<const Eigen::Vector3d>(sensor_pos);
        q_sensor_list[i] = Eigen::Map<const Eigen::Vector4d>(sensor_ori);

        const double w = sensor_ori[0], x = sensor_ori[1], y = sensor_ori[2], z = sensor_ori[3];
        // clang-format off
        R_wc_list[i] << 1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
                         2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
                         2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y);
        // clang-format on
      }
    }
    else
    {
      for (int i = 0; i < n; ++i)
      {
        p_world_list[i] = Eigen::Map<const Eigen::Vector3d>(parameters[2 * i]);

        const double* q = parameters[2 * i + 1];
        q_sensor_list[i] = Eigen::Map<const Eigen::Vector4d>(q);
        const double w = q[0], x = q[1], y = q[2], z = q[3];

        // clang-format off
        R_wc_list[i] << 1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
                         2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
                         2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y);
        // clang-format on
      }
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
        const int total_blocks = static_cast<int>(parameter_block_sizes().size());
        for (int i = 0; i < total_blocks; ++i)
        {
          if (jacobians[i])
          {
            Eigen::Map<vesta_core::MatrixXd>(jacobians[i], residual_dim, parameter_block_sizes()[i]).setZero();
          }
        }
      }
      return true;
    }

    // Compute per-observation reprojection errors and Jacobians w.r.t. sensor-frame poses
    Eigen::VectorXd b(2 * n);                                            // stacked weighted residuals
    Eigen::MatrixXd E(2 * n, 3);                                         // stacked landmark Jacobians
    std::vector<Eigen::Matrix<double, 2, 3>> F_pos(n);                   // sensor position Jacobians
    std::vector<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> F_ori(n);  // sensor orientation Jacobians (ambient)

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
      const double qw = q_sensor_list[i][0], qx = q_sensor_list[i][1];
      const double qy = q_sensor_list[i][2], qz = q_sensor_list[i][3];
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
      if (!has_extrinsic_)
      {
        // No extrinsic: body == sensor, use analytical Jacobians directly
        for (int i = 0; i < n; ++i)
        {
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
      else
      {
        // With extrinsic: use chain rule for body position (trivial: identity) and
        // numeric differentiation for body orientation and extrinsic parameters.
        //
        // d(residual)/d(p_body_i) = d(residual)/d(p_sensor_i) since d(p_sensor)/d(p_body) = I
        // d(residual)/d(q_body_i), d(residual)/d(ext_pos), d(residual)/d(ext_ori) are computed numerically.

        // Body position Jacobians: analytical (same as sensor position Jacobians)
        for (int i = 0; i < n; ++i)
        {
          if (jacobians[2 * i])
          {
            const Eigen::MatrixXd E_null_i_T = E_null.middleRows(2 * i, 2).transpose();
            Eigen::Map<vesta_core::MatrixXd> J(jacobians[2 * i], residual_dim, 3);
            J = E_null_i_T * F_pos[i];
          }
        }

        // Body orientation Jacobians: numeric differentiation
        for (int i = 0; i < n; ++i)
        {
          if (jacobians[2 * i + 1])
          {
            numericJacobian(parameters, 2 * i + 1, 4, residual_dim, residuals, jacobians[2 * i + 1]);
          }
        }

        // Extrinsic position Jacobian
        if (jacobians[2 * n])
        {
          numericJacobian(parameters, 2 * n, 3, residual_dim, residuals, jacobians[2 * n]);
        }

        // Extrinsic orientation Jacobian
        if (jacobians[2 * n + 1])
        {
          numericJacobian(parameters, 2 * n + 1, 4, residual_dim, residuals, jacobians[2 * n + 1]);
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

  /**
   * @brief Compute Jacobian of residuals w.r.t. one parameter block via central differences.
   *
   * @param[in]  parameters    Array of parameter block pointers
   * @param[in]  block_idx     Index of the parameter block to differentiate
   * @param[in]  block_size    Size of the parameter block
   * @param[in]  residual_dim  Number of residuals
   * @param[in]  residuals_0   Pre-computed residuals at the current point
   * @param[out] jacobian      Output Jacobian (residual_dim x block_size, row-major)
   */
  void numericJacobian(double const* const* parameters, int block_idx, int block_size, int residual_dim,
                       const double* residuals_0, double* jacobian) const
  {
    constexpr double kEps = 1e-8;
    const int total_blocks = static_cast<int>(parameter_block_sizes().size());

    // Create mutable copy of the parameter pointers and the block we're differentiating
    std::vector<const double*> params_plus(parameters, parameters + total_blocks);
    std::vector<double> block_copy(parameters[block_idx], parameters[block_idx] + block_size);
    params_plus[block_idx] = block_copy.data();

    std::vector<double> r_plus(residual_dim);
    std::vector<double> r_minus(residual_dim);

    Eigen::Map<vesta_core::MatrixXd> J(jacobian, residual_dim, block_size);

    for (int j = 0; j < block_size; ++j)
    {
      const double orig = parameters[block_idx][j];

      // Forward
      block_copy[j] = orig + kEps;
      Evaluate(params_plus.data(), r_plus.data(), nullptr);

      // Backward
      block_copy[j] = orig - kEps;
      Evaluate(params_plus.data(), r_minus.data(), nullptr);

      // Central difference
      for (int r = 0; r < residual_dim; ++r)
      {
        J(r, j) = (r_plus[r] - r_minus[r]) / (2.0 * kEps);
      }

      // Restore
      block_copy[j] = orig;
    }
  }

  std::vector<Eigen::Vector2d> observations_;
  vesta_core::Matrix2d sqrt_information_;
  Eigen::Vector4d calibration_;
  bool has_extrinsic_{ false };
};

}  // namespace vesta_constraints
