#include <vesta_constraints/3d/absolute_pose_3d_stamped_constraint.h>
#include <vesta_constraints/3d/relative_pose_3d_stamped_constraint.h>
#include <vesta_constraints/vision/reprojection_error_constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/timestamp.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>
#include <vesta_graphs/hash_graph.h>
#include <vesta_optimizers/batch_optimizer.h>
#include <vesta_optimizers/batch_optimizer_params.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/pinhole_camera.h>
#include <vesta_variables/vision/point_3d_landmark.h>

#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

// Camera intrinsics
static constexpr double kFx = 500.0;
static constexpr double kFy = 500.0;
static constexpr double kCx = 320.0;
static constexpr double kCy = 240.0;

struct SchurTestResult
{
  bool solution_usable;
  double final_cost;
  std::vector<Eigen::Vector3d> cam_positions;  // world-frame positions
  std::vector<Eigen::Vector3d> landmark_positions;
};

// Build and solve a visual SLAM problem with the given linear solver type.
// The reprojection cost functor uses world-frame position and orientation
// (R_wc). For identity orientation: p_cam = landmark - position.
static SchurTestResult runVisualSlamWithSolver(ceres::LinearSolverType solver_type)
{
  const int num_cams = 5;
  const auto device_id = vesta_core::uuid::generate("cam");

  // Ground truth camera WORLD positions along x-axis
  std::vector<Eigen::Vector3d> gt_cam_world;
  for (int i = 0; i < num_cams; ++i)
  {
    gt_cam_world.emplace_back(static_cast<double>(i), 0.0, 0.0);
  }

  // Ground truth landmarks
  std::vector<Eigen::Vector3d> gt_landmarks = {
    { 0.5, 0.5, 5.0 },  { 1.5, -0.5, 6.0 }, { 2.5, 0.3, 7.0 }, { 3.5, -0.3, 5.5 }, { 0.0, 1.0, 8.0 },
    { 1.0, -1.0, 6.5 }, { 2.0, 0.0, 9.0 },  { 3.0, 0.5, 7.5 }, { 4.0, -0.5, 5.0 }, { 2.0, 1.0, 10.0 },
  };
  const int num_lms = static_cast<int>(gt_landmarks.size());

  std::mt19937 rng(42);
  std::normal_distribution<double> pixel_noise(0.0, 0.5);
  std::normal_distribution<double> pos_init_noise(0.0, 0.05);
  std::normal_distribution<double> lm_init_noise(0.0, 0.1);

  // Camera intrinsics
  auto cam_k = std::make_shared<vesta_variables::PinholeCamera>(0);
  cam_k->fx() = kFx;
  cam_k->fy() = kFy;
  cam_k->cx() = kCx;
  cam_k->cy() = kCy;

  // Camera pose variables (position = world-frame position)
  std::vector<std::shared_ptr<vesta_variables::Position3DStamped>> cam_positions;
  std::vector<std::shared_ptr<vesta_variables::Orientation3DStamped>> cam_orientations;
  for (int i = 0; i < num_cams; ++i)
  {
    auto stamp = vesta_core::Timestamp(static_cast<int64_t>(i) * 1000000000LL);
    auto pos = std::make_shared<vesta_variables::Position3DStamped>(stamp, device_id);
    auto ori = std::make_shared<vesta_variables::Orientation3DStamped>(stamp, device_id);
    pos->x() = gt_cam_world[i].x() + pos_init_noise(rng);
    pos->y() = gt_cam_world[i].y() + pos_init_noise(rng);
    pos->z() = gt_cam_world[i].z() + pos_init_noise(rng);
    ori->w() = 1.0;
    ori->x() = 0.0;
    ori->y() = 0.0;
    ori->z() = 0.0;
    cam_positions.push_back(pos);
    cam_orientations.push_back(ori);
  }

  // Landmark variables (schurGroup() = 0 automatically)
  std::vector<std::shared_ptr<vesta_variables::Point3DLandmark>> landmarks;
  for (int j = 0; j < num_lms; ++j)
  {
    auto lm = std::make_shared<vesta_variables::Point3DLandmark>(static_cast<uint64_t>(j));
    lm->x() = gt_landmarks[j].x() + lm_init_noise(rng);
    lm->y() = gt_landmarks[j].y() + lm_init_noise(rng);
    lm->z() = gt_landmarks[j].z() + lm_init_noise(rng);
    landmarks.push_back(lm);
  }

  // Build transaction
  auto txn = std::make_shared<vesta_core::Transaction>();
  txn->stamp(vesta_core::Timestamp(static_cast<int64_t>(num_cams - 1) * 1000000000LL));

  txn->addVariable(cam_k);
  for (int i = 0; i < num_cams; ++i)
  {
    txn->addVariable(cam_positions[i]);
    txn->addVariable(cam_orientations[i]);
  }
  for (int j = 0; j < num_lms; ++j)
  {
    txn->addVariable(landmarks[j]);
  }

  // Prior on first pose (world position = 0 for camera at origin)
  {
    vesta_core::Vector7d mean;
    mean << gt_cam_world[0].x(), gt_cam_world[0].y(), gt_cam_world[0].z(), 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 1e-6;
    auto prior = std::make_shared<vesta_constraints::AbsolutePose3DStampedConstraint>("prior", *cam_positions[0],
                                                                                      *cam_orientations[0], mean, cov);
    txn->addConstraint(prior);
  }

  // Relative pose constraints between consecutive cameras
  // delta = q1^{-1} * (p2 - p1), for identity q: delta = p2 - p1 = 1,0,0
  for (int i = 0; i < num_cams - 1; ++i)
  {
    Eigen::Vector3d dt = gt_cam_world[i + 1] - gt_cam_world[i];  // (1,0,0)
    vesta_core::Vector7d delta;
    delta << dt.x(), dt.y(), dt.z(), 1.0, 0.0, 0.0, 0.0;
    vesta_core::Matrix6d cov = vesta_core::Matrix6d::Identity() * 0.01;
    auto rel = std::make_shared<vesta_constraints::RelativePose3DStampedConstraint>(
        "odom", *cam_positions[i], *cam_orientations[i], *cam_positions[i + 1], *cam_orientations[i + 1], delta, cov);
    txn->addConstraint(rel);
  }

  // Reprojection constraints
  // Observation: p_cam = R_wc^T * (X - position) = X - position for identity R
  // pixel: u = fx * pc.x/pc.z + cx, v = fy * pc.y/pc.z + cy
  for (int i = 0; i < num_cams; ++i)
  {
    for (int j = 0; j < num_lms; ++j)
    {
      // Camera frame point = landmark - world_position for identity R
      Eigen::Vector3d pc = gt_landmarks[j] - gt_cam_world[i];
      if (pc.z() <= 0.1)
        continue;

      double u = kFx * pc.x() / pc.z() + kCx;
      double v = kFy * pc.y() / pc.z() + kCy;
      if (u < 0 || u > 640 || v < 0 || v > 480)
        continue;

      vesta_core::Vector2d obs;
      obs << u + pixel_noise(rng), v + pixel_noise(rng);
      vesta_core::Matrix2d cov = vesta_core::Matrix2d::Identity() * 1.0;

      auto c = std::make_shared<vesta_constraints::ReprojectionErrorConstraint>(
          "camera", *cam_positions[i], *cam_orientations[i], *cam_k, *landmarks[j], obs, cov);
      txn->addConstraint(c);
    }
  }

  // Optimize with specified solver
  auto graph = std::make_unique<vesta_graphs::HashGraph>();
  vesta_optimizers::BatchOptimizerParams params;
  params.solver_options.max_num_iterations = 100;
  params.solver_options.linear_solver_type = solver_type;
  vesta_optimizers::BatchOptimizer optimizer(params, std::move(graph));
  optimizer.addTransaction("slam", txn);
  auto summary = optimizer.optimize();

  // Extract results
  SchurTestResult result;
  result.solution_usable = summary.IsSolutionUsable();
  result.final_cost = summary.final_cost;

  const auto& g = optimizer.graph();
  for (int i = 0; i < num_cams; ++i)
  {
    const auto& pos = dynamic_cast<const vesta_variables::Position3DStamped&>(g.getVariable(cam_positions[i]->uuid()));
    result.cam_positions.emplace_back(pos.x(), pos.y(), pos.z());
  }
  for (int j = 0; j < num_lms; ++j)
  {
    const auto& lm = dynamic_cast<const vesta_variables::Point3DLandmark&>(g.getVariable(landmarks[j]->uuid()));
    result.landmark_positions.emplace_back(lm.x(), lm.y(), lm.z());
  }

  return result;
}

// =============================================================================
// Test 1: Visual SLAM with DENSE_SCHUR
// =============================================================================
TEST(SchurSlam, DenseSchur)
{
  auto result = runVisualSlamWithSolver(ceres::DENSE_SCHUR);
  EXPECT_TRUE(result.solution_usable);

  // Check camera world positions
  for (int i = 0; i < 5; ++i)
  {
    EXPECT_NEAR(result.cam_positions[i].x(), static_cast<double>(i), 0.2) << "Cam " << i;
    EXPECT_NEAR(result.cam_positions[i].y(), 0.0, 0.2) << "Cam " << i;
    EXPECT_NEAR(result.cam_positions[i].z(), 0.0, 0.2) << "Cam " << i;
  }
}

// =============================================================================
// Test 2: Visual SLAM with SPARSE_SCHUR
// =============================================================================
TEST(SchurSlam, SparseSchur)
{
  auto result = runVisualSlamWithSolver(ceres::SPARSE_SCHUR);
  EXPECT_TRUE(result.solution_usable);

  for (int i = 0; i < 5; ++i)
  {
    EXPECT_NEAR(result.cam_positions[i].x(), static_cast<double>(i), 0.2) << "Cam " << i;
    EXPECT_NEAR(result.cam_positions[i].y(), 0.0, 0.2) << "Cam " << i;
    EXPECT_NEAR(result.cam_positions[i].z(), 0.0, 0.2) << "Cam " << i;
  }
}

// =============================================================================
// Test 3: Visual SLAM with ITERATIVE_SCHUR
// =============================================================================
TEST(SchurSlam, IterativeSchur)
{
  auto result = runVisualSlamWithSolver(ceres::ITERATIVE_SCHUR);
  EXPECT_TRUE(result.solution_usable);

  for (int i = 0; i < 5; ++i)
  {
    EXPECT_NEAR(result.cam_positions[i].x(), static_cast<double>(i), 0.3) << "Cam " << i;
    EXPECT_NEAR(result.cam_positions[i].y(), 0.0, 0.3) << "Cam " << i;
    EXPECT_NEAR(result.cam_positions[i].z(), 0.0, 0.3) << "Cam " << i;
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
