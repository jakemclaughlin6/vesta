## fuse_loss library
add_library(fuse_loss
  src/arctan_loss.cpp
  src/cauchy_loss.cpp
  src/composed_loss.cpp
  src/dcs_loss.cpp
  src/fair_loss.cpp
  src/geman_mcclure_loss.cpp
  src/huber_loss.cpp
  src/loss_function.cpp
  src/scaled_loss.cpp
  src/softlone_loss.cpp
  src/tolerant_loss.cpp
  src/trivial_loss.cpp
  src/tukey_loss.cpp
  src/welsch_loss.cpp
)
target_include_directories(fuse_loss PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(fuse_loss PUBLIC
  fuse_core
  Ceres::ceres
)
set_target_properties(fuse_loss PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)

## Tests
if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  function(fuse_loss_add_test TEST_NAME)
    add_executable(${TEST_NAME} test/${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE fuse_loss GTest::GTest GTest::Main)
    set_target_properties(${TEST_NAME} PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  endfunction()

  fuse_loss_add_test(test_loss_function)
  fuse_loss_add_test(test_composed_loss)
  fuse_loss_add_test(test_huber_loss)
  fuse_loss_add_test(test_tukey_loss)
  fuse_loss_add_test(test_scaled_loss)
endif()
