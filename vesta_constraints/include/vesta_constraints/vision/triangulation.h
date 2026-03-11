#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <cassert>
#include <stdexcept>
#include <vector>

namespace vesta_constraints
{

/**
 * @brief Triangulate a 3D point from N calibrated camera observations using DLT.
 *
 * Uses the Direct Linear Transform (Hartley & Zisserman, 2nd Ed., p.312).
 * Requires at least 2 observations from distinct viewpoints.
 *
 * @param[in] R_wc          Per-observation world-from-camera rotation matrices
 * @param[in] p_world       Per-observation world-frame camera positions
 * @param[in] observations  Per-observation 2D pixel measurements (u, v)
 * @param[in] calibration   Camera intrinsics [fx, fy, cx, cy]
 * @return The triangulated 3D point in world coordinates
 * @throws std::runtime_error if the triangulation is degenerate (point at infinity)
 */
inline Eigen::Vector3d triangulateDLT(const std::vector<Eigen::Matrix3d>& R_wc,
                                      const std::vector<Eigen::Vector3d>& p_world,
                                      const std::vector<Eigen::Vector2d>& observations,
                                      const Eigen::Vector4d& calibration)
{
  const auto n = static_cast<int>(observations.size());
  assert(n >= 2);
  assert(static_cast<int>(R_wc.size()) == n);
  assert(static_cast<int>(p_world.size()) == n);

  const double fx = calibration[0];
  const double fy = calibration[1];
  const double cx = calibration[2];
  const double cy = calibration[3];

  Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
  K(0, 0) = fx;
  K(0, 2) = cx;
  K(1, 1) = fy;
  K(1, 2) = cy;
  K(2, 2) = 1.0;

  // Build the DLT matrix A (2N x 4)
  Eigen::MatrixXd A(2 * n, 4);
  for (int i = 0; i < n; ++i)
  {
    // Camera-from-world transform
    Eigen::Matrix3d R_cw = R_wc[i].transpose();
    Eigen::Vector3d t_cw = -R_cw * p_world[i];

    // 3x4 projection matrix P = K * [R_cw | t_cw]
    Eigen::Matrix<double, 3, 4> P;
    P.leftCols<3>() = R_cw;
    P.col(3) = t_cw;
    P = K * P;

    const double u = observations[i][0];
    const double v = observations[i][1];

    // DLT equations: u * P[2,:] - P[0,:] and v * P[2,:] - P[1,:]
    A.row(2 * i) = u * P.row(2) - P.row(0);
    A.row(2 * i + 1) = v * P.row(2) - P.row(1);
  }

  // SVD of A — solution is last column of V
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  Eigen::Vector4d X_h = svd.matrixV().col(3);

  if (std::abs(X_h[3]) < 1e-10)
  {
    throw std::runtime_error("Degenerate triangulation: point at infinity");
  }

  Eigen::Vector3d point = X_h.head<3>() / X_h[3];

  // Gauss-Newton refinement to minimize reprojection error (makes E^T * r ≈ 0,
  // which is required for nullspace projection Jacobians to be consistent)
  for (int iter = 0; iter < 5; ++iter)
  {
    Eigen::MatrixXd J(2 * n, 3);
    Eigen::VectorXd r(2 * n);

    for (int i = 0; i < n; ++i)
    {
      Eigen::Matrix3d R_cw = R_wc[i].transpose();
      Eigen::Vector3d p_cam = R_cw * (point - p_world[i]);
      double X = p_cam[0], Y = p_cam[1], Z = p_cam[2];

      double u_proj = fx * X / Z + cx;
      double v_proj = fy * Y / Z + cy;

      r[2 * i] = u_proj - observations[i][0];
      r[2 * i + 1] = v_proj - observations[i][1];

      // Jacobian of projection w.r.t. world point = J_proj * R_cw
      Eigen::Matrix<double, 2, 3> J_proj;
      J_proj << fx / Z, 0.0, -fx * X / (Z * Z), 0.0, fy / Z, -fy * Y / (Z * Z);
      J.block<2, 3>(2 * i, 0) = J_proj * R_cw;
    }

    // Gauss-Newton step: delta = -(J^T J)^{-1} J^T r
    Eigen::Vector3d delta = -(J.transpose() * J).ldlt().solve(J.transpose() * r);
    point += delta;

    if (delta.norm() < 1e-12)
    {
      break;
    }
  }

  return point;
}

}  // namespace vesta_constraints
