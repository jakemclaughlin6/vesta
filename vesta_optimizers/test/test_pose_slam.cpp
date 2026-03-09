#include <vesta_constraints/2d/absolute_pose_2d_stamped_constraint.h>
#include <vesta_constraints/2d/relative_pose_2d_stamped_constraint.h>
#include <vesta_constraints/3d/absolute_pose_3d_stamped_constraint.h>
#include <vesta_constraints/3d/relative_pose_3d_stamped_constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/timestamp.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>
#include <vesta_graphs/hash_graph.h>
#include <vesta_optimizers/batch_optimizer.h>
#include <vesta_optimizers/batch_optimizer_params.h>
#include <vesta_optimizers/fixed_lag_smoother.h>
#include <vesta_optimizers/fixed_lag_smoother_params.h>
#include <vesta_variables/2d/orientation_2d_stamped.h>
#include <vesta_variables/2d/position_2d_stamped.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>

#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

// Ground truth 2D poses: square trajectory
// Poses: (0,0,0), (1,0,0), (2,0,0), (2,1,pi/2), (2,2,pi/2),
//        (1,2,pi), (0,2,pi), (0,1,-pi/2), (0,0,-pi/2), (1,0,0)
struct Pose2D {
  double x, y, yaw;
};

static std::vector<Pose2D> generateSquareTrajectory2D() {
  return {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {2.0, 0.0, M_PI / 2.0},
      {2.0, 1.0, M_PI / 2.0},
      {2.0, 2.0, M_PI},
      {1.0, 2.0, M_PI},
      {0.0, 2.0, -M_PI / 2.0},
      {0.0, 1.0, -M_PI / 2.0},
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
  };
}

static double normalizeAngle(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

// Ground truth 3D poses: linear trajectory along x-axis with identity
// orientation
struct Pose3D {
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
};

static std::vector<Pose3D> generateLinearTrajectory3D() {
  std::vector<Pose3D> poses;
  for (int i = 0; i < 10; ++i) {
    poses.push_back(
        {Eigen::Vector3d(static_cast<double>(i), 0.0, 0.0),
         Eigen::Quaterniond::Identity()});
  }
  return poses;
}

// =============================================================================
// Test 1: 2D Pose SLAM with BatchOptimizer
// =============================================================================
TEST(PoseSlam, Batch2D) {
  auto gt = generateSquareTrajectory2D();
  const int n = static_cast<int>(gt.size());
  const auto device_id = vesta_core::uuid::generate("robot");

  std::mt19937 rng(42);
  std::normal_distribution<double> noise_pos(0.0, 0.01);
  std::normal_distribution<double> noise_yaw(0.0, 0.005);
  std::normal_distribution<double> init_noise(0.0, 0.1);

  // Create variables
  std::vector<std::shared_ptr<vesta_variables::Position2DStamped>> positions;
  std::vector<std::shared_ptr<vesta_variables::Orientation2DStamped>>
      orientations;
  for (int i = 0; i < n; ++i) {
    auto stamp = vesta_core::Timestamp(static_cast<int64_t>(i) * 1000000000LL);
    auto pos =
        std::make_shared<vesta_variables::Position2DStamped>(stamp, device_id);
    auto ori = std::make_shared<vesta_variables::Orientation2DStamped>(
        stamp, device_id);
    // Initialize with perturbed ground truth
    pos->x() = gt[i].x + init_noise(rng);
    pos->y() = gt[i].y + init_noise(rng);
    ori->yaw() = gt[i].yaw + init_noise(rng) * 0.1;
    positions.push_back(pos);
    orientations.push_back(ori);
  }

  // Build transaction
  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(
      vesta_core::Timestamp(static_cast<int64_t>(n - 1) * 1000000000LL));

  for (int i = 0; i < n; ++i) {
    txn->addVariable(positions[i]);
    txn->addVariable(orientations[i]);
  }

  // Prior on first pose
  {
    vesta_core::VectorXd mean(3);
    mean << gt[0].x, gt[0].y, gt[0].yaw;
    vesta_core::MatrixXd cov = vesta_core::MatrixXd::Identity(3, 3) * 1e-6;
    auto prior = std::make_shared<
        vesta_constraints::AbsolutePose2DStampedConstraint>(
        "prior", *positions[0], *orientations[0], mean, cov);
    txn->addConstraint(prior);
  }

  // Relative pose constraints between consecutive poses
  for (int i = 0; i < n - 1; ++i) {
    double dx = gt[i + 1].x - gt[i].x;
    double dy = gt[i + 1].y - gt[i].y;
    double dyaw = normalizeAngle(gt[i + 1].yaw - gt[i].yaw);
    // Rotate delta into local frame
    double c = std::cos(gt[i].yaw);
    double s = std::sin(gt[i].yaw);
    double local_dx = c * dx + s * dy;
    double local_dy = -s * dx + c * dy;

    vesta_core::VectorXd delta(3);
    delta << local_dx + noise_pos(rng), local_dy + noise_pos(rng),
        dyaw + noise_yaw(rng);
    vesta_core::MatrixXd cov = vesta_core::MatrixXd::Identity(3, 3) * 0.01;
    auto rel = std::make_shared<
        vesta_constraints::RelativePose2DStampedConstraint>(
        "odom", *positions[i], *orientations[i], *positions[i + 1],
        *orientations[i + 1], delta, cov);
    txn->addConstraint(rel);
  }

  // Optimize
  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));
  optimizer.addTransaction("odom", txn);
  auto summary = optimizer.optimize();

  EXPECT_TRUE(summary.IsSolutionUsable());

  // Check convergence
  const auto &g = optimizer.graph();
  for (int i = 0; i < n; ++i) {
    const auto &pos = dynamic_cast<const vesta_variables::Position2DStamped &>(
        g.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), gt[i].x, 0.15) << "Pose " << i << " x";
    EXPECT_NEAR(pos.y(), gt[i].y, 0.15) << "Pose " << i << " y";
  }
}

// =============================================================================
// Test 2: 3D Pose SLAM with BatchOptimizer
// =============================================================================
TEST(PoseSlam, Batch3D) {
  auto gt = generateLinearTrajectory3D();
  const int n = static_cast<int>(gt.size());
  const auto device_id = vesta_core::uuid::generate("robot");

  std::mt19937 rng(42);
  std::normal_distribution<double> noise_pos(0.0, 0.01);
  std::normal_distribution<double> init_noise(0.0, 0.1);

  std::vector<std::shared_ptr<vesta_variables::Position3DStamped>> positions;
  std::vector<std::shared_ptr<vesta_variables::Orientation3DStamped>>
      orientations;
  for (int i = 0; i < n; ++i) {
    auto stamp = vesta_core::Timestamp(static_cast<int64_t>(i) * 1000000000LL);
    auto pos =
        std::make_shared<vesta_variables::Position3DStamped>(stamp, device_id);
    auto ori = std::make_shared<vesta_variables::Orientation3DStamped>(
        stamp, device_id);
    pos->x() = gt[i].position.x() + init_noise(rng);
    pos->y() = gt[i].position.y() + init_noise(rng);
    pos->z() = gt[i].position.z() + init_noise(rng);
    ori->w() = gt[i].orientation.w();
    ori->x() = gt[i].orientation.x();
    ori->y() = gt[i].orientation.y();
    ori->z() = gt[i].orientation.z();
    positions.push_back(pos);
    orientations.push_back(ori);
  }

  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(
      vesta_core::Timestamp(static_cast<int64_t>(n - 1) * 1000000000LL));

  for (int i = 0; i < n; ++i) {
    txn->addVariable(positions[i]);
    txn->addVariable(orientations[i]);
  }

  // Prior on first pose
  {
    vesta_core::Vector7d mean;
    mean << gt[0].position.x(), gt[0].position.y(), gt[0].position.z(),
        gt[0].orientation.w(), gt[0].orientation.x(), gt[0].orientation.y(),
        gt[0].orientation.z();
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 1e-6;
    auto prior = std::make_shared<
        vesta_constraints::AbsolutePose3DStampedConstraint>(
        "prior", *positions[0], *orientations[0], mean, cov);
    txn->addConstraint(prior);
  }

  // Relative pose constraints
  for (int i = 0; i < n - 1; ++i) {
    Eigen::Vector3d dp =
        gt[i].orientation.inverse() * (gt[i + 1].position - gt[i].position);
    Eigen::Quaterniond dq =
        gt[i].orientation.inverse() * gt[i + 1].orientation;

    vesta_core::Vector7d delta;
    delta << dp.x() + noise_pos(rng), dp.y() + noise_pos(rng),
        dp.z() + noise_pos(rng), dq.w(), dq.x(), dq.y(), dq.z();
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 0.01;
    auto rel = std::make_shared<
        vesta_constraints::RelativePose3DStampedConstraint>(
        "odom", *positions[i], *orientations[i], *positions[i + 1],
        *orientations[i + 1], delta, cov);
    txn->addConstraint(rel);
  }

  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));
  optimizer.addTransaction("odom", txn);
  auto summary = optimizer.optimize();

  EXPECT_TRUE(summary.IsSolutionUsable());

  const auto &g = optimizer.graph();
  for (int i = 0; i < n; ++i) {
    const auto &pos = dynamic_cast<const vesta_variables::Position3DStamped &>(
        g.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), gt[i].position.x(), 0.15) << "Pose " << i;
    EXPECT_NEAR(pos.y(), gt[i].position.y(), 0.15) << "Pose " << i;
    EXPECT_NEAR(pos.z(), gt[i].position.z(), 0.15) << "Pose " << i;
  }
}

// =============================================================================
// Test 3: 2D Pose SLAM with FixedLagSmoother
// =============================================================================
TEST(PoseSlam, FixedLag2D) {
  auto gt = generateSquareTrajectory2D();
  const int n = static_cast<int>(gt.size());
  const auto device_id = vesta_core::uuid::generate("robot");

  std::mt19937 rng(42);
  std::normal_distribution<double> noise_pos(0.0, 0.01);
  std::normal_distribution<double> noise_yaw(0.0, 0.005);
  std::normal_distribution<double> init_noise(0.0, 0.1);

  // Create variables
  std::vector<std::shared_ptr<vesta_variables::Position2DStamped>> positions;
  std::vector<std::shared_ptr<vesta_variables::Orientation2DStamped>>
      orientations;
  for (int i = 0; i < n; ++i) {
    auto stamp = vesta_core::Timestamp(static_cast<int64_t>(i) * 1000000000LL);
    auto pos =
        std::make_shared<vesta_variables::Position2DStamped>(stamp, device_id);
    auto ori = std::make_shared<vesta_variables::Orientation2DStamped>(
        stamp, device_id);
    pos->x() = gt[i].x + init_noise(rng);
    pos->y() = gt[i].y + init_noise(rng);
    ori->yaw() = gt[i].yaw + init_noise(rng) * 0.1;
    positions.push_back(pos);
    orientations.push_back(ori);
  }

  // FixedLagSmoother with lag larger than trajectory (all poses stay in window)
  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::FixedLagSmootherParams params;
  params.lag_duration = vesta_core::Duration(static_cast<int64_t>(20) * 1000000000LL);
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::FixedLagSmoother smoother(params, std::move(graph));

  // Add first pose with prior
  {
    auto txn = std::make_shared<vesta_core::Transaction>();
    auto stamp = vesta_core::Timestamp(0);
    txn->stamp(stamp);
    txn->addInvolvedStamp(stamp);
    txn->addVariable(positions[0]);
    txn->addVariable(orientations[0]);

    vesta_core::VectorXd mean(3);
    mean << gt[0].x, gt[0].y, gt[0].yaw;
    vesta_core::MatrixXd cov = vesta_core::MatrixXd::Identity(3, 3) * 1e-6;
    auto prior = std::make_shared<
        vesta_constraints::AbsolutePose2DStampedConstraint>(
        "prior", *positions[0], *orientations[0], mean, cov);
    txn->addConstraint(prior);
    smoother.addTransaction("prior", txn);
    smoother.optimize();
  }

  // Add odometry constraints incrementally
  for (int i = 0; i < n - 1; ++i) {
    auto txn = std::make_shared<vesta_core::Transaction>();
    auto stamp =
        vesta_core::Timestamp(static_cast<int64_t>(i + 1) * 1000000000LL);
    txn->stamp(stamp);
    txn->addInvolvedStamp(stamp);
    txn->addInvolvedStamp(
        vesta_core::Timestamp(static_cast<int64_t>(i) * 1000000000LL));

    txn->addVariable(positions[i + 1]);
    txn->addVariable(orientations[i + 1]);

    double dx = gt[i + 1].x - gt[i].x;
    double dy = gt[i + 1].y - gt[i].y;
    double dyaw = normalizeAngle(gt[i + 1].yaw - gt[i].yaw);
    double c = std::cos(gt[i].yaw);
    double s = std::sin(gt[i].yaw);
    double local_dx = c * dx + s * dy;
    double local_dy = -s * dx + c * dy;

    vesta_core::VectorXd delta(3);
    delta << local_dx + noise_pos(rng), local_dy + noise_pos(rng),
        dyaw + noise_yaw(rng);
    vesta_core::MatrixXd cov = vesta_core::MatrixXd::Identity(3, 3) * 0.01;
    auto rel = std::make_shared<
        vesta_constraints::RelativePose2DStampedConstraint>(
        "odom", *positions[i], *orientations[i], *positions[i + 1],
        *orientations[i + 1], delta, cov);
    txn->addConstraint(rel);
    smoother.addTransaction("odom", txn);
    auto summary = smoother.optimize();
    ASSERT_TRUE(summary.IsSolutionUsable())
        << "FLS optimize failed at step " << i;
  }

  // Check that recent poses are close to ground truth
  const auto &g = smoother.graph();
  for (int i = n - 3; i < n; ++i) {
    if (!g.variableExists(positions[i]->uuid())) continue;
    const auto &pos = dynamic_cast<const vesta_variables::Position2DStamped &>(
        g.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), gt[i].x, 0.3) << "Pose " << i << " x";
    EXPECT_NEAR(pos.y(), gt[i].y, 0.3) << "Pose " << i << " y";
  }
}

// =============================================================================
// Test 4: 3D Pose SLAM with FixedLagSmoother
// =============================================================================
TEST(PoseSlam, FixedLag3D) {
  auto gt = generateLinearTrajectory3D();
  const int n = static_cast<int>(gt.size());
  const auto device_id = vesta_core::uuid::generate("robot");

  std::mt19937 rng(42);
  std::normal_distribution<double> noise_pos(0.0, 0.01);
  std::normal_distribution<double> init_noise(0.0, 0.1);

  std::vector<std::shared_ptr<vesta_variables::Position3DStamped>> positions;
  std::vector<std::shared_ptr<vesta_variables::Orientation3DStamped>>
      orientations;
  for (int i = 0; i < n; ++i) {
    auto stamp = vesta_core::Timestamp(static_cast<int64_t>(i) * 1000000000LL);
    auto pos =
        std::make_shared<vesta_variables::Position3DStamped>(stamp, device_id);
    auto ori = std::make_shared<vesta_variables::Orientation3DStamped>(
        stamp, device_id);
    pos->x() = gt[i].position.x() + init_noise(rng);
    pos->y() = gt[i].position.y() + init_noise(rng);
    pos->z() = gt[i].position.z() + init_noise(rng);
    ori->w() = gt[i].orientation.w();
    ori->x() = gt[i].orientation.x();
    ori->y() = gt[i].orientation.y();
    ori->z() = gt[i].orientation.z();
    positions.push_back(pos);
    orientations.push_back(ori);
  }

  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::FixedLagSmootherParams params;
  params.lag_duration = vesta_core::Duration(static_cast<int64_t>(20) * 1000000000LL);
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = ceres::DENSE_QR;
  vesta_optimizers::FixedLagSmoother smoother(params, std::move(graph));

  // First pose with prior
  {
    auto txn = std::make_shared<vesta_core::Transaction>();
    auto stamp = vesta_core::Timestamp(0);
    txn->stamp(stamp);
    txn->addInvolvedStamp(stamp);
    txn->addVariable(positions[0]);
    txn->addVariable(orientations[0]);

    vesta_core::Vector7d mean;
    mean << gt[0].position.x(), gt[0].position.y(), gt[0].position.z(),
        gt[0].orientation.w(), gt[0].orientation.x(), gt[0].orientation.y(),
        gt[0].orientation.z();
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 1e-6;
    auto prior = std::make_shared<
        vesta_constraints::AbsolutePose3DStampedConstraint>(
        "prior", *positions[0], *orientations[0], mean, cov);
    txn->addConstraint(prior);
    smoother.addTransaction("prior", txn);
    smoother.optimize();
  }

  // Add odometry incrementally
  for (int i = 0; i < n - 1; ++i) {
    auto txn = std::make_shared<vesta_core::Transaction>();
    auto stamp =
        vesta_core::Timestamp(static_cast<int64_t>(i + 1) * 1000000000LL);
    txn->stamp(stamp);
    txn->addInvolvedStamp(stamp);
    txn->addInvolvedStamp(
        vesta_core::Timestamp(static_cast<int64_t>(i) * 1000000000LL));

    txn->addVariable(positions[i + 1]);
    txn->addVariable(orientations[i + 1]);

    Eigen::Vector3d dp =
        gt[i].orientation.inverse() * (gt[i + 1].position - gt[i].position);
    Eigen::Quaterniond dq =
        gt[i].orientation.inverse() * gt[i + 1].orientation;

    vesta_core::Vector7d delta;
    delta << dp.x() + noise_pos(rng), dp.y() + noise_pos(rng),
        dp.z() + noise_pos(rng), dq.w(), dq.x(), dq.y(), dq.z();
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 0.01;
    auto rel = std::make_shared<
        vesta_constraints::RelativePose3DStampedConstraint>(
        "odom", *positions[i], *orientations[i], *positions[i + 1],
        *orientations[i + 1], delta, cov);
    txn->addConstraint(rel);
    smoother.addTransaction("odom", txn);
    auto summary = smoother.optimize();
    ASSERT_TRUE(summary.IsSolutionUsable())
        << "FLS optimize failed at step " << i;
  }

  const auto &g = smoother.graph();
  for (int i = n - 3; i < n; ++i) {
    if (!g.variableExists(positions[i]->uuid())) continue;
    const auto &pos = dynamic_cast<const vesta_variables::Position3DStamped &>(
        g.getVariable(positions[i]->uuid()));
    EXPECT_NEAR(pos.x(), gt[i].position.x(), 0.3) << "Pose " << i;
    EXPECT_NEAR(pos.y(), gt[i].position.y(), 0.3) << "Pose " << i;
    EXPECT_NEAR(pos.z(), gt[i].position.z(), 0.3) << "Pose " << i;
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
