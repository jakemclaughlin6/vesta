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
 * @brief Cost function that implements stereo nullspace projection (structureless stereo vision factor).
 *
 * Given N >= 2 stereo observations of a 3D landmark from N camera poses, this cost function
 * analytically eliminates the landmark by projecting the stacked stereo reprojection errors
 * through the left nullspace of the landmark Jacobian. The result is a (4N-3)-dimensional
 * residual that depends only on the camera poses, not the landmark position.
 *
 * Each stereo observation provides 4 measurements (u_left, v_left, u_right, v_right).
 * The right camera is offset from the left by the baseline along the camera x-axis.
 *
 * Algorithm:
 *   1. Triangulate the landmark from current poses + stored measurements (DLT with left+right)
 *   2. Compute per-observation stereo reprojection errors and Jacobians
 *   3. Stack the landmark Jacobian E (4N x 3), compute QR
 *   4. Project residuals and pose Jacobians through E's left nullspace
 *
 * Parameter blocks (without extrinsic):
 *   [pos_0(3), ori_0(4), pos_1(3), ori_1(4), ..., pos_{N-1}(3), ori_{N-1}(4)]
 *
 * Parameter blocks (with extrinsic):
 *   [pos_0(3), ori_0(4), ..., pos_{N-1}(3), ori_{N-1}(4), ext_pos(3), ext_ori(4)]
 *
 * Residual dimension: 4N - 3
 *
 * Camera intrinsics and baseline are stored internally (fixed).
 */
class StereoNullspaceProjectionCostFunction : public ceres::CostFunction
{
public:
  /**
   * @brief Construct the cost function.
   *
   * @param[in] observations     Per-observation stereo measurements (u_l, v_l, u_r, v_r)
   * @param[in] sqrt_information Square root information matrix for the observation noise (4x4)
   * @param[in] calibration      Stereo camera intrinsics [fx, fy, cx, cy, baseline]
   * @param[in] has_extrinsic    Whether extrinsic parameter blocks are appended
   */
  StereoNullspaceProjectionCostFunction(std::vector<Eigen::Vector4d> observations,
                                        const vesta_core::Matrix4d& sqrt_information,
                                        const Eigen::Matrix<double, 5, 1>& calibration, bool has_extrinsic = false)
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
    set_num_residuals(4 * n - 3);
  }

  ~StereoNullspaceProjectionCostFunction() override = default;

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    const int n = static_cast<int>(observations_.size());
    const int residual_dim = 4 * n - 3;

    const double fx = calibration_[0];
    const double fy = calibration_[1];
    const double cx = calibration_[2];
    const double cy = calibration_[3];
    const double baseline = calibration_[4];

    // Compute sensor-frame poses (either directly from parameters or via extrinsic transform)
    std::vector<Eigen::Matrix3d> R_wc_list(n);
    std::vector<Eigen::Vector3d> p_world_list(n);
    std::vector<Eigen::Vector4d> q_sensor_list(n);

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

    // Triangulate using both left and right observations (2N total)
    std::vector<Eigen::Matrix3d> tri_R_wc(2 * n);
    std::vector<Eigen::Vector3d> tri_p_world(2 * n);
    std::vector<Eigen::Vector2d> tri_obs(2 * n);
    const Eigen::Vector4d mono_cal(fx, fy, cx, cy);

    for (int i = 0; i < n; ++i)
    {
      // Left camera
      tri_R_wc[2 * i] = R_wc_list[i];
      tri_p_world[2 * i] = p_world_list[i];
      tri_obs[2 * i] = observations_[i].head<2>();

      // Right camera: shifted by baseline along camera x-axis in world frame
      tri_R_wc[2 * i + 1] = R_wc_list[i];
      tri_p_world[2 * i + 1] = p_world_list[i] + R_wc_list[i].col(0) * baseline;
      tri_obs[2 * i + 1] = observations_[i].tail<2>();
    }

    Eigen::Vector3d landmark;
    try
    {
      landmark = triangulateDLT(tri_R_wc, tri_p_world, tri_obs, mono_cal);
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

    // Compute per-observation stereo reprojection errors and Jacobians w.r.t. sensor-frame poses
    Eigen::VectorXd b(4 * n);                                            // stacked weighted residuals
    Eigen::MatrixXd E(4 * n, 3);                                         // stacked landmark Jacobians
    std::vector<Eigen::Matrix<double, 4, 3>> F_pos(n);                   // sensor position Jacobians
    std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> F_ori(n);  // sensor orientation Jacobians

    for (int i = 0; i < n; ++i)
    {
      const Eigen::Matrix3d R_cw = R_wc_list[i].transpose();
      const Eigen::Vector3d d = landmark - p_world_list[i];
      const Eigen::Vector3d p_cam = R_cw * d;

      const double X = p_cam[0], Y = p_cam[1], Z = p_cam[2];
      const double inv_Z = 1.0 / Z;
      const double inv_Z2 = inv_Z * inv_Z;

      // Left projection
      const double u_l = fx * X * inv_Z + cx;
      const double v_l = fy * Y * inv_Z + cy;
      // Right projection
      const double u_r = fx * (X - baseline) * inv_Z + cx;
      const double v_r = fy * Y * inv_Z + cy;

      // Raw 4D error
      Eigen::Vector4d raw_error(u_l - observations_[i][0], v_l - observations_[i][1], u_r - observations_[i][2],
                                v_r - observations_[i][3]);
      b.segment<4>(4 * i) = sqrt_information_ * raw_error;

      // Left projection Jacobian w.r.t. p_cam (2x3)
      Eigen::Matrix<double, 2, 3> J_proj_left;
      // clang-format off
      J_proj_left << fx * inv_Z, 0.0,        -fx * X * inv_Z2,
                     0.0,        fy * inv_Z,  -fy * Y * inv_Z2;
      // clang-format on

      // Right projection Jacobian w.r.t. p_cam (2x3)
      Eigen::Matrix<double, 2, 3> J_proj_right;
      // clang-format off
      J_proj_right << fx * inv_Z, 0.0,        -fx * (X - baseline) * inv_Z2,
                      0.0,        fy * inv_Z,  -fy * Y * inv_Z2;
      // clang-format on

      // Stack into 4x3 raw Jacobian
      Eigen::Matrix<double, 4, 3> J_proj;
      J_proj.topRows<2>() = J_proj_left;
      J_proj.bottomRows<2>() = J_proj_right;

      // Weighted projection Jacobian
      const Eigen::Matrix<double, 4, 3> AJ = sqrt_information_ * J_proj;

      // Chain rule: dp_cam/dp_w = -R_cw, dp_cam/dlandmark = R_cw
      F_pos[i] = AJ * (-R_cw);
      E.block<4, 3>(4 * i, 0) = AJ * R_cw;

      // Quaternion Jacobian: dp_cam/dq where p_cam = R(q^{-1}) * d
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
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4 * n, 4 * n);
    Q = qr.householderQ() * Q;
    const Eigen::MatrixXd E_null = Q.rightCols(residual_dim);  // (4N x 4N-3)

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
          const Eigen::MatrixXd E_null_i_T = E_null.middleRows(4 * i, 4).transpose();  // (4N-3) x 4

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

        // Body position Jacobians: analytical (same as sensor position Jacobians)
        for (int i = 0; i < n; ++i)
        {
          if (jacobians[2 * i])
          {
            const Eigen::MatrixXd E_null_i_T = E_null.middleRows(4 * i, 4).transpose();
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

  std::vector<Eigen::Vector4d> observations_;
  vesta_core::Matrix4d sqrt_information_;
  Eigen::Matrix<double, 5, 1> calibration_;
  bool has_extrinsic_{ false };
};

}  // namespace vesta_constraints
