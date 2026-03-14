## vesta_optimizers library
add_library(vesta_optimizers
  src/batch_optimizer.cpp
  src/fixed_lag_smoother.cpp
  src/optimizer.cpp
  src/variable_stamp_index.cpp
)
target_include_directories(vesta_optimizers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(vesta_optimizers PUBLIC
  vesta_core
  vesta_constraints
  vesta_graphs
  vesta_variables
)
set_target_properties(vesta_optimizers PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)

## Tests
if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  add_executable(test_variable_stamp_index test/test_variable_stamp_index.cpp)
  target_link_libraries(test_variable_stamp_index PRIVATE vesta_optimizers GTest::GTest GTest::Main)
  set_target_properties(test_variable_stamp_index PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
  )
  add_test(NAME test_variable_stamp_index COMMAND test_variable_stamp_index)

  add_executable(test_pose_slam test/test_pose_slam.cpp)
  target_link_libraries(test_pose_slam PRIVATE vesta_optimizers GTest::GTest GTest::Main)
  set_target_properties(test_pose_slam PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
  add_test(NAME test_pose_slam COMMAND test_pose_slam)

  add_executable(test_visual_slam test/test_visual_slam.cpp)
  target_link_libraries(test_visual_slam PRIVATE vesta_optimizers GTest::GTest GTest::Main)
  set_target_properties(test_visual_slam PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
  add_test(NAME test_visual_slam COMMAND test_visual_slam)

  add_executable(test_schur_slam test/test_schur_slam.cpp)
  target_link_libraries(test_schur_slam PRIVATE vesta_optimizers GTest::GTest GTest::Main)
  set_target_properties(test_schur_slam PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
  add_test(NAME test_schur_slam COMMAND test_schur_slam)

  add_executable(test_visual_inertial_slam test/test_visual_inertial_slam.cpp)
  target_link_libraries(test_visual_inertial_slam PRIVATE vesta_optimizers GTest::GTest GTest::Main)
  set_target_properties(test_visual_inertial_slam PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
  add_test(NAME test_visual_inertial_slam COMMAND test_visual_inertial_slam)

  add_executable(test_marginalizer_comparison test/test_marginalizer_comparison.cpp)
  target_link_libraries(test_marginalizer_comparison PRIVATE vesta_optimizers GTest::GTest GTest::Main)
  set_target_properties(test_marginalizer_comparison PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
  add_test(NAME test_marginalizer_comparison COMMAND test_marginalizer_comparison)

  add_executable(test_nullspace_visual_slam test/test_nullspace_visual_slam.cpp)
  target_link_libraries(test_nullspace_visual_slam PRIVATE vesta_optimizers GTest::GTest GTest::Main)
  set_target_properties(test_nullspace_visual_slam PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
  add_test(NAME test_nullspace_visual_slam COMMAND test_nullspace_visual_slam)
endif()
