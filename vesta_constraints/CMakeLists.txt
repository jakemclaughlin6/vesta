## vesta_constraints library
add_library(vesta_constraints
  # Common
  src/common/absolute_constraint.cpp
  src/common/marginal_constraint.cpp
  src/common/marginal_cost_function.cpp
  src/common/marginalize_variables.cpp
  src/common/normal_delta.cpp
  src/common/relative_constraint.cpp
  src/common/uuid_ordering.cpp
  src/common/variable_constraints.cpp
  # 2D
  src/2d/absolute_pose_2d_stamped_constraint.cpp
  src/2d/normal_delta_orientation_2d.cpp
  src/2d/normal_delta_pose_2d.cpp
  src/2d/normal_prior_orientation_2d.cpp
  src/2d/normal_prior_pose_2d.cpp
  src/2d/relative_pose_2d_stamped_constraint.cpp
  # 3D
  src/3d/absolute_orientation_3d_stamped_constraint.cpp
  src/3d/absolute_orientation_3d_stamped_euler_constraint.cpp
  src/3d/absolute_pose_3d_stamped_constraint.cpp
  src/3d/relative_orientation_3d_stamped_constraint.cpp
  src/3d/relative_pose_3d_stamped_constraint.cpp
  # Vision
  src/vision/fixed_3d_landmark_constraint.cpp
  src/vision/fixed_3d_landmark_simple_covariance_constraint.cpp
  src/vision/reprojection_error_constraint.cpp
  src/vision/reprojection_error_snavelly_constraint.cpp
  src/vision/stereo_reprojection_error_constraint.cpp
)
target_include_directories(vesta_constraints PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(vesta_constraints PUBLIC
  vesta_core
  vesta_variables
  vesta_graphs
  Ceres::ceres
  Eigen3::Eigen
  ${SUITESPARSE_LIBRARIES}
)
target_include_directories(vesta_constraints PUBLIC
  ${SUITESPARSE_INCLUDE_DIRS}
)
set_target_properties(vesta_constraints PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)

## Tests
if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  function(vesta_constraints_add_test TEST_NAME)
    add_executable(${TEST_NAME} test/${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE vesta_constraints GTest::GTest GTest::Main)
    target_include_directories(${TEST_NAME} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(${TEST_NAME} PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  endfunction()

  vesta_constraints_add_test(test_absolute_constraint)
  vesta_constraints_add_test(test_absolute_orientation_2d_stamped_constraint)
  vesta_constraints_add_test(test_absolute_orientation_3d_stamped_constraint)
  vesta_constraints_add_test(test_absolute_orientation_3d_stamped_euler_constraint)
  vesta_constraints_add_test(test_absolute_pose_2d_stamped_constraint)
  vesta_constraints_add_test(test_absolute_pose_3d_stamped_constraint)
  vesta_constraints_add_test(test_fixed_3d_landmark_constraint)
  vesta_constraints_add_test(test_reprojection_error_constraint)
  vesta_constraints_add_test(test_reprojection_error_snavelly_constraint)
  vesta_constraints_add_test(test_stereo_reprojection_error_constraint)
  vesta_constraints_add_test(test_marginal_constraint)
  vesta_constraints_add_test(test_marginalize_variables)
  vesta_constraints_add_test(test_normal_delta_pose_2d)
  vesta_constraints_add_test(test_normal_prior_pose_2d)
  vesta_constraints_add_test(test_relative_constraint)
  vesta_constraints_add_test(test_relative_pose_2d_stamped_constraint)
  vesta_constraints_add_test(test_relative_pose_3d_stamped_constraint)
  vesta_constraints_add_test(test_uuid_ordering)
  vesta_constraints_add_test(test_variable_constraints)

  # BAL Problem test (needs WORKING_DIRECTORY set)
  add_executable(test_bal_problem test/test_bal_problem.cpp)
  target_link_libraries(test_bal_problem PRIVATE vesta_constraints GTest::GTest GTest::Main)
  target_include_directories(test_bal_problem PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_target_properties(test_bal_problem PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
  )
  add_test(NAME test_bal_problem COMMAND test_bal_problem
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test
  )

  # Benchmarks
  find_package(benchmark QUIET)
  if(benchmark_FOUND)
    add_executable(benchmark_normal_delta_pose_2d
      benchmark/benchmark_normal_delta_pose_2d.cpp
    )
    target_link_libraries(benchmark_normal_delta_pose_2d PRIVATE
      benchmark::benchmark
      vesta_constraints
      Ceres::ceres
    )
    set_target_properties(benchmark_normal_delta_pose_2d PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
    )

    add_executable(benchmark_normal_prior_pose_2d
      benchmark/benchmark_normal_prior_pose_2d.cpp
    )
    target_link_libraries(benchmark_normal_prior_pose_2d PRIVATE
      benchmark::benchmark
      vesta_constraints
      Ceres::ceres
    )
    set_target_properties(benchmark_normal_prior_pose_2d PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
    )
  endif()
endif()
