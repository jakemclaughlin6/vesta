## fuse_variables library
add_library(fuse_variables
  # 2D variables
  src/2d/acceleration_angular_2d_stamped.cpp
  src/2d/acceleration_linear_2d_stamped.cpp
  src/2d/orientation_2d_stamped.cpp
  src/2d/position_2d_stamped.cpp
  src/2d/velocity_angular_2d_stamped.cpp
  src/2d/velocity_linear_2d_stamped.cpp
  # 3D variables
  src/3d/acceleration_angular_3d_stamped.cpp
  src/3d/acceleration_linear_3d_stamped.cpp
  src/3d/orientation_3d_stamped.cpp
  src/3d/position_3d_stamped.cpp
  src/3d/velocity_angular_3d_stamped.cpp
  src/3d/velocity_linear_3d_stamped.cpp
  # Vision variables
  src/vision/pinhole_camera.cpp
  src/vision/pinhole_camera_fixed.cpp
  src/vision/pinhole_camera_radial.cpp
  src/vision/point_2d_fixed_landmark.cpp
  src/vision/point_2d_landmark.cpp
  src/vision/point_3d_fixed_landmark.cpp
  src/vision/point_3d_landmark.cpp
  # Common
  src/common/stamped.cpp
)
target_include_directories(fuse_variables PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(fuse_variables PUBLIC
  fuse_core
  Ceres::ceres
)
set_target_properties(fuse_variables PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)

## Tests
if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  function(fuse_variables_add_test TEST_NAME)
    add_executable(${TEST_NAME} test/${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE fuse_variables GTest::GTest GTest::Main)
    set_target_properties(${TEST_NAME} PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  endfunction()

  fuse_variables_add_test(test_acceleration_angular_2d_stamped)
  fuse_variables_add_test(test_acceleration_angular_3d_stamped)
  fuse_variables_add_test(test_acceleration_linear_2d_stamped)
  fuse_variables_add_test(test_acceleration_linear_3d_stamped)
  fuse_variables_add_test(test_pinhole_camera)
  fuse_variables_add_test(test_pinhole_camera_fixed)
  fuse_variables_add_test(test_pinhole_camera_radial)
  fuse_variables_add_test(test_fixed_size_variable)
  fuse_variables_add_test(test_orientation_2d_stamped)
  fuse_variables_add_test(test_orientation_3d_stamped)
  fuse_variables_add_test(test_point_2d_fixed_landmark)
  fuse_variables_add_test(test_point_2d_landmark)
  fuse_variables_add_test(test_point_3d_fixed_landmark)
  fuse_variables_add_test(test_point_3d_landmark)
  fuse_variables_add_test(test_position_2d_stamped)
  fuse_variables_add_test(test_position_3d_stamped)
  fuse_variables_add_test(test_velocity_angular_2d_stamped)
  fuse_variables_add_test(test_velocity_angular_3d_stamped)
  fuse_variables_add_test(test_velocity_linear_2d_stamped)
  fuse_variables_add_test(test_velocity_linear_3d_stamped)
endif()
