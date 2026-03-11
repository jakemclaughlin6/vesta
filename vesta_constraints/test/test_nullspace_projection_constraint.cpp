#include <vesta_constraints/vision/convert_to_nullspace.h>
#include <vesta_constraints/vision/nullspace_projection_constraint.h>
#include <vesta_constraints/vision/reprojection_error_constraint.h>
#include <vesta_core/eigen.h>
#include <vesta_core/eigen_gtest.h>
#include <vesta_core/serialization.h>
#include <vesta_core/uuid.h>
#include <vesta_graphs/hash_graph.h>
#include <vesta_variables/3d/orientation_3d_stamped.h>
#include <vesta_variables/3d/position_3d_stamped.h>
#include <vesta_variables/vision/pinhole_camera_fixed.h>
#include <vesta_variables/vision/point_3d_landmark.h>

#include <ceres/problem.h>
#include <ceres/solver.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using vesta_constraints::NullspaceProjectionConstraint;
using vesta_constraints::ReprojectionErrorConstraint;
using vesta_variables::Orientation3DStamped;
using vesta_variables::PinholeCameraFixed;
using vesta_variables::Point3DLandmark;
using vesta_variables::Position3DStamped;

namespace
{

// Helper: project a world point through a camera
Eigen::Vector2d project(const Eigen::Vector3d& p_world_cam, const Eigen::Quaterniond& q_wc,
                        const Eigen::Vector3d& point, double fx, double fy, double cx, double cy)
{
  // p_cam = R_wc^{-1} * (X - p_w)
  Eigen::Vector3d p_cam = q_wc.inverse() * (point - p_world_cam);
  return { fx * p_cam.x() / p_cam.z() + cx, fy * p_cam.y() / p_cam.z() + cy };
}

struct TestScene
{
  // Camera intrinsics
  double fx = 500.0;
  double fy = 500.0;
  double cx = 320.0;
  double cy = 240.0;

  // 6 landmarks at varying depths for strong geometric constraints
  std::vector<Eigen::Vector3d> landmarks = { { -1.0, -1.0, 8.0 },  { 1.0, -1.0, 10.0 }, { 1.0, 1.0, 12.0 },
                                             { -1.0, 1.0, 9.0 },   { 0.0, 0.0, 15.0 },  { 0.5, -0.5, 6.0 } };

  // 3 camera poses with both translation and rotation variety
  struct Pose
  {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;  // world-from-camera
  };

  // Pose 0: at origin, looking forward
  // Pose 1: shifted right and slightly rotated about Y
  // Pose 2: shifted up-right with rotation about Y and X
  std::vector<Pose> poses = {
    { { 0.0, 0.0, 0.0 }, Eigen::Quaterniond::Identity() },
    { { 1.5, 0.0, 0.2 },
      Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY())).normalized() },
    { { 0.5, 1.0, -0.3 },
      Eigen::Quaterniond(Eigen::AngleAxisd(-0.08, Eigen::Vector3d::UnitY()) *
                         Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()))
          .normalized() }
  };

  // Generate observations for all landmarks from all poses
  std::vector<std::vector<Eigen::Vector2d>> generateObservations() const
  {
    std::vector<std::vector<Eigen::Vector2d>> obs(landmarks.size());
    for (size_t j = 0; j < landmarks.size(); ++j)
    {
      for (const auto& pose : poses)
      {
        obs[j].push_back(project(pose.position, pose.orientation, landmarks[j], fx, fy, cx, cy));
      }
    }
    return obs;
  }
};

}  // namespace

TEST(NullspaceProjectionConstraint, Constructor)
{
  TestScene scene;
  auto obs = scene.generateObservations();

  std::vector<Position3DStamped> positions;
  std::vector<Orientation3DStamped> orientations;
  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    positions.emplace_back(vesta_core::Timestamp(static_cast<int32_t>(i), 0), vesta_core::uuid::generate("cam"));
    orientations.emplace_back(vesta_core::Timestamp(static_cast<int32_t>(i), 0), vesta_core::uuid::generate("cam"));
  }

  PinholeCameraFixed calibration(0);
  calibration.fx() = scene.fx;
  calibration.fy() = scene.fy;
  calibration.cx() = scene.cx;
  calibration.cy() = scene.cy;

  vesta_core::Matrix2d cov;
  cov << 1.0, 0.0, 0.0, 1.0;

  // Construct for landmark 0 (observed from 3 poses)
  EXPECT_NO_THROW(
      NullspaceProjectionConstraint constraint("test", positions, orientations, calibration, obs[0], cov));

  NullspaceProjectionConstraint constraint("test", positions, orientations, calibration, obs[0], cov);
  EXPECT_EQ(3u, constraint.numObservations());
  EXPECT_EQ(6u, constraint.variables().size());  // 3 positions + 3 orientations
}

TEST(NullspaceProjectionConstraint, ResidualAtGroundTruth)
{
  // At ground truth poses, the residual should be near zero
  TestScene scene;
  auto obs = scene.generateObservations();

  std::vector<Position3DStamped> positions;
  std::vector<Orientation3DStamped> orientations;
  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    positions.emplace_back(vesta_core::Timestamp(static_cast<int32_t>(i), 0), vesta_core::uuid::generate("cam"));
    orientations.emplace_back(vesta_core::Timestamp(static_cast<int32_t>(i), 0), vesta_core::uuid::generate("cam"));
  }

  // Set ground truth values
  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    positions[i].x() = scene.poses[i].position.x();
    positions[i].y() = scene.poses[i].position.y();
    positions[i].z() = scene.poses[i].position.z();
    orientations[i].w() = scene.poses[i].orientation.w();
    orientations[i].x() = scene.poses[i].orientation.x();
    orientations[i].y() = scene.poses[i].orientation.y();
    orientations[i].z() = scene.poses[i].orientation.z();
  }

  PinholeCameraFixed calibration(0);
  calibration.fx() = scene.fx;
  calibration.fy() = scene.fy;
  calibration.cx() = scene.cx;
  calibration.cy() = scene.cy;

  vesta_core::Matrix2d cov;
  cov << 1.0, 0.0, 0.0, 1.0;

  NullspaceProjectionConstraint constraint("test", positions, orientations, calibration, obs[0], cov);

  // Evaluate the cost function at ground truth
  auto cost_fn = std::unique_ptr<ceres::CostFunction>(constraint.costFunction());

  // Build parameter block pointers
  std::vector<const double*> params;
  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    params.push_back(positions[i].data());
    params.push_back(orientations[i].data());
  }

  const int residual_dim = 2 * static_cast<int>(scene.poses.size()) - 3;
  std::vector<double> residuals(residual_dim);
  EXPECT_TRUE(cost_fn->Evaluate(params.data(), residuals.data(), nullptr));

  // All residuals should be near zero at ground truth
  for (int i = 0; i < residual_dim; ++i)
  {
    EXPECT_NEAR(0.0, residuals[i], 1e-8) << "Residual " << i << " is non-zero at ground truth";
  }
}

TEST(NullspaceProjectionConstraint, JacobianCheck)
{
  // Numerically verify analytical Jacobians via central finite differences
  TestScene scene;
  auto obs = scene.generateObservations();

  std::vector<Position3DStamped> positions;
  std::vector<Orientation3DStamped> orientations;
  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    positions.emplace_back(vesta_core::Timestamp(static_cast<int32_t>(i), 0), vesta_core::uuid::generate("cam"));
    orientations.emplace_back(vesta_core::Timestamp(static_cast<int32_t>(i), 0), vesta_core::uuid::generate("cam"));
  }

  // Set ground truth values
  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    positions[i].x() = scene.poses[i].position.x();
    positions[i].y() = scene.poses[i].position.y();
    positions[i].z() = scene.poses[i].position.z();
    orientations[i].w() = scene.poses[i].orientation.w();
    orientations[i].x() = scene.poses[i].orientation.x();
    orientations[i].y() = scene.poses[i].orientation.y();
    orientations[i].z() = scene.poses[i].orientation.z();
  }

  PinholeCameraFixed calibration(0);
  calibration.fx() = scene.fx;
  calibration.fy() = scene.fy;
  calibration.cx() = scene.cx;
  calibration.cy() = scene.cy;

  vesta_core::Matrix2d cov;
  cov << 1.0, 0.0, 0.0, 1.0;

  NullspaceProjectionConstraint constraint("test", positions, orientations, calibration, obs[0], cov);

  auto cost_fn = std::unique_ptr<ceres::CostFunction>(constraint.costFunction());
  const int num_blocks = static_cast<int>(cost_fn->parameter_block_sizes().size());
  const int residual_dim = cost_fn->num_residuals();

  // Collect mutable parameter data
  std::vector<std::vector<double>> param_storage(num_blocks);
  for (int i = 0; i < num_blocks; ++i)
  {
    const int block_size = cost_fn->parameter_block_sizes()[i];
    const double* src = (i % 2 == 0) ? positions[i / 2].data() : orientations[i / 2].data();
    param_storage[i].assign(src, src + block_size);
  }

  // Build parameter pointers
  std::vector<double*> params_mut(num_blocks);
  for (int i = 0; i < num_blocks; ++i)
  {
    params_mut[i] = param_storage[i].data();
  }
  std::vector<const double*> params_const(params_mut.begin(), params_mut.end());

  // Evaluate analytical Jacobians
  std::vector<double> residuals(residual_dim);
  std::vector<std::vector<double>> jacobian_storage(num_blocks);
  std::vector<double*> jacobian_ptrs(num_blocks);
  for (int i = 0; i < num_blocks; ++i)
  {
    const int block_size = cost_fn->parameter_block_sizes()[i];
    jacobian_storage[i].resize(residual_dim * block_size);
    jacobian_ptrs[i] = jacobian_storage[i].data();
  }

  ASSERT_TRUE(cost_fn->Evaluate(params_const.data(), residuals.data(), jacobian_ptrs.data()));

  // Finite-difference Jacobian check for each parameter block
  const double delta = 1e-6;
  for (int b = 0; b < num_blocks; ++b)
  {
    const int block_size = cost_fn->parameter_block_sizes()[b];
    double max_error = 0.0;

    for (int j = 0; j < block_size; ++j)
    {
      const double original = params_mut[b][j];

      // +delta
      params_mut[b][j] = original + delta;
      std::vector<double> r_plus(residual_dim);
      ASSERT_TRUE(cost_fn->Evaluate(params_const.data(), r_plus.data(), nullptr));

      // -delta
      params_mut[b][j] = original - delta;
      std::vector<double> r_minus(residual_dim);
      ASSERT_TRUE(cost_fn->Evaluate(params_const.data(), r_minus.data(), nullptr));

      // Restore
      params_mut[b][j] = original;

      // Central difference
      for (int r = 0; r < residual_dim; ++r)
      {
        const double numerical = (r_plus[r] - r_minus[r]) / (2.0 * delta);
        // Analytical Jacobian is stored row-major: J[r * block_size + j]
        const double analytical = jacobian_storage[b][r * block_size + j];
        const double error = std::abs(numerical - analytical);
        max_error = std::max(max_error, error);
      }
    }

    std::cout << "Block " << b << " (size " << block_size << "): max Jacobian error = " << max_error << std::endl;
    ASSERT_LT(max_error, 1e-4) << "Jacobian error too large for parameter block " << b;
  }
}

TEST(NullspaceProjectionConstraint, Optimization)
{
  // Perturb poses and verify that optimization recovers them
  TestScene scene;
  auto all_obs = scene.generateObservations();

  auto make_pos = [](int i)
  { return Position3DStamped::make_shared(vesta_core::Timestamp(i, 0), vesta_core::uuid::generate("cam")); };
  auto make_ori = [](int i)
  { return Orientation3DStamped::make_shared(vesta_core::Timestamp(i, 0), vesta_core::uuid::generate("cam")); };

  std::vector<Position3DStamped::SharedPtr> positions;
  std::vector<Orientation3DStamped::SharedPtr> orientations;
  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    positions.push_back(make_pos(static_cast<int>(i)));
    orientations.push_back(make_ori(static_cast<int>(i)));
  }

  // Set perturbed values (pose 0 is fixed as anchor)
  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    positions[i]->x() = scene.poses[i].position.x();
    positions[i]->y() = scene.poses[i].position.y();
    positions[i]->z() = scene.poses[i].position.z();
    orientations[i]->w() = scene.poses[i].orientation.w();
    orientations[i]->x() = scene.poses[i].orientation.x();
    orientations[i]->y() = scene.poses[i].orientation.y();
    orientations[i]->z() = scene.poses[i].orientation.z();
  }

  // Perturb poses 1 and 2
  positions[1]->x() += 0.1;
  positions[1]->y() += -0.05;
  positions[1]->z() += 0.02;
  positions[2]->x() += -0.08;
  positions[2]->y() += 0.06;

  auto calibration = PinholeCameraFixed::make_shared(0);
  calibration->fx() = scene.fx;
  calibration->fy() = scene.fy;
  calibration->cx() = scene.cx;
  calibration->cy() = scene.cy;

  vesta_core::Matrix2d cov;
  cov << 0.25, 0.0, 0.0, 0.25;

  // Build Ceres problem
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = vesta_core::Loss::Ownership;
  ceres::Problem problem(problem_options);

  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    problem.AddParameterBlock(positions[i]->data(), positions[i]->size(), positions[i]->manifold());
    problem.AddParameterBlock(orientations[i]->data(), orientations[i]->size(), orientations[i]->manifold());
  }

  // Fix pose 0 as anchor
  problem.SetParameterBlockConstant(positions[0]->data());
  problem.SetParameterBlockConstant(orientations[0]->data());

  // Add a nullspace constraint for each landmark
  for (size_t j = 0; j < scene.landmarks.size(); ++j)
  {
    std::vector<Position3DStamped> pos_vec;
    std::vector<Orientation3DStamped> ori_vec;
    for (size_t i = 0; i < scene.poses.size(); ++i)
    {
      pos_vec.push_back(*positions[i]);
      ori_vec.push_back(*orientations[i]);
    }

    auto constraint =
        NullspaceProjectionConstraint::make_shared("test", pos_vec, ori_vec, *calibration, all_obs[j], cov);

    std::vector<double*> param_blocks;
    for (size_t i = 0; i < scene.poses.size(); ++i)
    {
      param_blocks.push_back(positions[i]->data());
      param_blocks.push_back(orientations[i]->data());
    }

    problem.AddResidualBlock(constraint->costFunction(), constraint->lossFunction(), param_blocks);
  }

  // Solve
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 100;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  EXPECT_TRUE(summary.IsSolutionUsable()) << summary.FullReport();

  // Nullspace projection is scale-invariant (epipolar geometry only constrains direction,
  // not distance). Verify the recovered geometry modulo a global scale factor.
  // Compute scale from the baseline between pose 0 and pose 1.
  Eigen::Vector3d gt_baseline = scene.poses[1].position - scene.poses[0].position;
  Eigen::Vector3d opt_baseline =
      Eigen::Vector3d(positions[1]->x(), positions[1]->y(), positions[1]->z()) - scene.poses[0].position;
  double scale = gt_baseline.norm() / opt_baseline.norm();

  for (size_t i = 1; i < scene.poses.size(); ++i)
  {
    Eigen::Vector3d opt_pos(positions[i]->x(), positions[i]->y(), positions[i]->z());
    Eigen::Vector3d scaled = scene.poses[0].position + scale * (opt_pos - scene.poses[0].position);
    EXPECT_NEAR(scene.poses[i].position.x(), scaled.x(), 1e-3) << "Pose " << i << " position x";
    EXPECT_NEAR(scene.poses[i].position.y(), scaled.y(), 1e-3) << "Pose " << i << " position y";
    EXPECT_NEAR(scene.poses[i].position.z(), scaled.z(), 1e-3) << "Pose " << i << " position z";
  }
}

TEST(NullspaceProjectionConstraint, ConvertFromReprojection)
{
  TestScene scene;
  auto all_obs = scene.generateObservations();

  // Build a graph with reprojection constraints
  vesta_graphs::HashGraph graph;

  std::vector<Position3DStamped::SharedPtr> positions;
  std::vector<Orientation3DStamped::SharedPtr> orientations;
  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    auto pos =
        Position3DStamped::make_shared(vesta_core::Timestamp(static_cast<int32_t>(i), 0), vesta_core::uuid::generate("cam"));
    pos->x() = scene.poses[i].position.x();
    pos->y() = scene.poses[i].position.y();
    pos->z() = scene.poses[i].position.z();
    positions.push_back(pos);

    auto ori =
        Orientation3DStamped::make_shared(vesta_core::Timestamp(static_cast<int32_t>(i), 0), vesta_core::uuid::generate("cam"));
    ori->w() = scene.poses[i].orientation.w();
    ori->x() = scene.poses[i].orientation.x();
    ori->y() = scene.poses[i].orientation.y();
    ori->z() = scene.poses[i].orientation.z();
    orientations.push_back(ori);

    graph.addVariable(pos);
    graph.addVariable(ori);
  }

  auto calibration = PinholeCameraFixed::make_shared(0);
  calibration->fx() = scene.fx;
  calibration->fy() = scene.fy;
  calibration->cx() = scene.cx;
  calibration->cy() = scene.cy;
  graph.addVariable(calibration);

  vesta_core::Matrix2d cov;
  cov << 1.0, 0.0, 0.0, 1.0;

  // Add landmarks and reprojection constraints
  std::vector<vesta_core::UUID> landmark_uuids;
  for (size_t j = 0; j < scene.landmarks.size(); ++j)
  {
    auto landmark = Point3DLandmark::make_shared(j);
    landmark->x() = scene.landmarks[j].x();
    landmark->y() = scene.landmarks[j].y();
    landmark->z() = scene.landmarks[j].z();
    graph.addVariable(landmark);
    landmark_uuids.push_back(landmark->uuid());

    for (size_t i = 0; i < scene.poses.size(); ++i)
    {
      auto constraint = ReprojectionErrorConstraint::make_shared("test", *positions[i], *orientations[i], *calibration,
                                                                 *landmark, all_obs[j][i], cov);
      graph.addConstraint(constraint);
    }
  }

  // Convert to nullspace constraints
  auto transaction = vesta_constraints::convertToNullspaceConstraints("test", landmark_uuids, graph);

  // Apply the transaction
  graph.update(transaction);

  // Verify: landmarks should be removed
  for (const auto& uuid : landmark_uuids)
  {
    EXPECT_FALSE(graph.variableExists(uuid));
  }

  // Verify: should have one nullspace constraint per landmark
  int nullspace_count = 0;
  for (const auto& constraint : graph.getConstraints())
  {
    if (dynamic_cast<const NullspaceProjectionConstraint*>(&constraint))
    {
      ++nullspace_count;
    }
  }
  EXPECT_EQ(static_cast<int>(scene.landmarks.size()), nullspace_count);

  // Verify: no reprojection constraints remain
  for (const auto& constraint : graph.getConstraints())
  {
    EXPECT_EQ(nullptr, dynamic_cast<const ReprojectionErrorConstraint*>(&constraint));
  }
}

TEST(NullspaceProjectionConstraint, Serialization)
{
  TestScene scene;
  auto obs = scene.generateObservations();

  std::vector<Position3DStamped> positions;
  std::vector<Orientation3DStamped> orientations;
  for (size_t i = 0; i < scene.poses.size(); ++i)
  {
    positions.emplace_back(vesta_core::Timestamp(static_cast<int32_t>(i), 0), vesta_core::uuid::generate("cam"));
    orientations.emplace_back(vesta_core::Timestamp(static_cast<int32_t>(i), 0), vesta_core::uuid::generate("cam"));
  }

  PinholeCameraFixed calibration(0);
  calibration.fx() = scene.fx;
  calibration.fy() = scene.fy;
  calibration.cx() = scene.cx;
  calibration.cy() = scene.cy;

  vesta_core::Matrix2d cov;
  cov << 1.0, 0.0, 0.0, 1.0;

  NullspaceProjectionConstraint expected("test", positions, orientations, calibration, obs[0], cov);

  // Serialize
  std::stringstream stream;
  {
    vesta_core::TextOutputArchive archive(stream);
    expected.serialize(archive);
  }

  // Deserialize
  NullspaceProjectionConstraint actual;
  {
    vesta_core::TextInputArchive archive(stream);
    actual.deserialize(archive);
  }

  // Compare
  EXPECT_EQ(expected.uuid(), actual.uuid());
  EXPECT_EQ(expected.variables(), actual.variables());
  EXPECT_EQ(expected.numObservations(), actual.numObservations());
  EXPECT_MATRIX_NEAR(expected.sqrtInformation(), actual.sqrtInformation(), 1e-9);

  for (size_t i = 0; i < expected.numObservations(); ++i)
  {
    EXPECT_NEAR(expected.observations()[i][0], actual.observations()[i][0], 1e-9);
    EXPECT_NEAR(expected.observations()[i][1], actual.observations()[i][1], 1e-9);
  }

  EXPECT_NEAR(expected.calibration()[0], actual.calibration()[0], 1e-9);
  EXPECT_NEAR(expected.calibration()[1], actual.calibration()[1], 1e-9);
  EXPECT_NEAR(expected.calibration()[2], actual.calibration()[2], 1e-9);
  EXPECT_NEAR(expected.calibration()[3], actual.calibration()[3], 1e-9);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
