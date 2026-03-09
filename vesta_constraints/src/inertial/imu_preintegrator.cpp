#include <vesta_constraints/inertial/imu_preintegrator.h>

#include <Eigen/Cholesky>

#include <algorithm>

namespace vesta_constraints
{

void ImuPreintegrator::reset()
{
  delta.dt = 0.0;
  delta.q.setIdentity();
  delta.p.setZero();
  delta.v.setZero();
  delta.covariance.setZero();
  delta.sqrt_information.setZero();

  jacobian.dq_dbg.setZero();
  jacobian.dp_dbg.setZero();
  jacobian.dp_dba.setZero();
  jacobian.dv_dbg.setZero();
  jacobian.dv_dba.setZero();
}

void ImuPreintegrator::clearBefore(const vesta_core::Timestamp& t)
{
  data.erase(data.begin(), data.lower_bound(t));
}

void ImuPreintegrator::increment(double dt, const ImuData& imu,
                                 const Eigen::Vector3d& bg,
                                 const Eigen::Vector3d& ba,
                                 bool compute_jacobian,
                                 bool compute_covariance)
{
  Eigen::Vector3d w = imu.angular_velocity - bg;
  Eigen::Vector3d a = imu.linear_acceleration - ba;

  Eigen::Quaterniond q_full(expMapSO3<double>(w * dt));
  Eigen::Quaterniond q_half(expMapSO3<double>(0.5 * w * dt));

  if (compute_covariance)
  {
    Eigen::Matrix<double, 9, 9> A;
    A.setIdentity();

    A.block<3, 3>(ES_Q, ES_Q) = q_full.conjugate().matrix();
    A.block<3, 3>(ES_V, ES_Q) =
        -dt * delta.q.matrix() * skewSymmetric<double>(a);
    A.block<3, 3>(ES_P, ES_Q) =
        -0.5 * dt * dt * delta.q.matrix() * skewSymmetric<double>(a);
    A.block<3, 3>(ES_P, ES_V) = dt * Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 9, 6> B;
    B.setZero();
    B.block<3, 3>(ES_Q, ES_BG - ES_BG) =
        dt * rightJacobianSO3(w * dt);
    B.block<3, 3>(ES_V, ES_BA - ES_BG) = dt * delta.q.matrix();
    B.block<3, 3>(ES_P, ES_BA - ES_BG) = 0.5 * dt * dt * delta.q.matrix();

    Eigen::Matrix<double, 6, 6> white_noise_cov;
    double inv_dt = 1.0 / std::max(dt, 1.0e-7);
    white_noise_cov.setZero();
    white_noise_cov.block<3, 3>(ES_BG - ES_BG, ES_BG - ES_BG) = cov_gyro * inv_dt;
    white_noise_cov.block<3, 3>(ES_BA - ES_BG, ES_BA - ES_BG) = cov_accel * inv_dt;

    delta.covariance.block<9, 9>(ES_Q, ES_Q) =
        A * delta.covariance.block<9, 9>(0, 0) * A.transpose() +
        B * white_noise_cov * B.transpose();
    delta.covariance.block<3, 3>(ES_BG, ES_BG) += cov_gyro_bias * dt;
    delta.covariance.block<3, 3>(ES_BA, ES_BA) += cov_accel_bias * dt;
  }

  if (compute_jacobian)
  {
    jacobian.dp_dbg +=
        dt * jacobian.dv_dbg - 0.5 * dt * dt * delta.q.matrix() *
                                   skewSymmetric<double>(a) * jacobian.dq_dbg;
    jacobian.dp_dba +=
        dt * jacobian.dv_dba - 0.5 * dt * dt * delta.q.matrix();
    jacobian.dv_dbg -=
        dt * delta.q.matrix() * skewSymmetric<double>(a) * jacobian.dq_dbg;
    jacobian.dv_dba -= dt * delta.q.matrix();
    jacobian.dq_dbg = q_full.conjugate().matrix() * jacobian.dq_dbg -
                      dt * rightJacobianSO3(w * dt);
  }

  Eigen::Quaterniond q_mid = delta.q * q_half;
  Eigen::Vector3d a_mid = q_mid * a;

  delta.dt += dt;
  delta.p = delta.p + dt * delta.v + 0.5 * dt * dt * a_mid;
  delta.v = delta.v + dt * a_mid;
  delta.q = (delta.q * q_full).normalized();
}

bool ImuPreintegrator::integrate(const vesta_core::Timestamp& t,
                                 const Eigen::Vector3d& bg,
                                 const Eigen::Vector3d& ba,
                                 bool compute_jacobian,
                                 bool compute_covariance,
                                 bool compute_information)
{
  if (data.empty())
  {
    return false;
  }

  reset();

  // Increment over window such that it is less or equal to the requested time
  // Track the last processed iterator for the final increment
  auto last_iter = data.begin();
  for (auto iter = data.begin(); std::next(iter) != data.end(); ++iter)
  {
    const auto& next = std::next(iter);
    if (next->first > t)
    {
      break;
    }
    const double dt_sec =
        static_cast<double>(next->first.nanoseconds - iter->first.nanoseconds) * 1e-9;
    increment(dt_sec, iter->second, bg, ba, compute_jacobian, compute_covariance);
    last_iter = next;
  }

  // Final increment from last processed measurement to requested time
  const double dt_final =
      static_cast<double>(t.nanoseconds - last_iter->first.nanoseconds) * 1e-9;
  if (dt_final > 0.0)
  {
    increment(dt_final, last_iter->second, bg, ba, compute_jacobian,
              compute_covariance);
  }

  if (compute_information)
  {
    computeSqrtInformation();
  }

  return true;
}

void ImuPreintegrator::computeSqrtInformation()
{
  // Ensure covariance non-zero (within pre-defined tolerance) to avoid
  // ill-conditioned matrix during optimization

  // Upper left 9x9 block (Q, P, V)
  const auto norm1 = delta.covariance.block<9, 9>(ES_Q, ES_Q).norm();
  if (norm1 < cov_tol)
  {
    delta.covariance.block<9, 9>(ES_Q, ES_Q).setIdentity();
    delta.covariance.block<9, 9>(ES_Q, ES_Q) *= cov_tol;
  }

  // Lower right 6x6 block (BG, BA)
  const auto norm2 = delta.covariance.block<6, 6>(ES_BG, ES_BG).norm();
  if (norm2 < bias_cov_tol)
  {
    delta.covariance.block<6, 6>(ES_BG, ES_BG).setIdentity();
    delta.covariance.block<6, 6>(ES_BG, ES_BG) *= bias_cov_tol;
  }

  delta.sqrt_information =
      Eigen::LLT<Eigen::Matrix<double, ES_SIZE, ES_SIZE>>(
          delta.covariance.inverse())
          .matrixL()
          .transpose();

  if (!delta.sqrt_information.allFinite())
  {
    // Fall back to low-weight diagonal
    delta.sqrt_information =
        Eigen::Matrix<double, ES_SIZE, ES_SIZE>::Identity() * invalid_sqrt_info_weight;
  }
}

}  // namespace vesta_constraints
