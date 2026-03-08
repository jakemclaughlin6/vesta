## fuse_optimizers library
add_library(fuse_optimizers
  src/batch_optimizer.cpp
  src/fixed_lag_smoother.cpp
  src/optimizer.cpp
  src/variable_stamp_index.cpp
)
target_include_directories(fuse_optimizers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(fuse_optimizers PUBLIC
  fuse_core
  fuse_constraints
  fuse_graphs
  fuse_variables
)
set_target_properties(fuse_optimizers PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)

## Tests
if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  add_executable(test_variable_stamp_index test/test_variable_stamp_index.cpp)
  target_link_libraries(test_variable_stamp_index PRIVATE fuse_optimizers GTest::GTest GTest::Main)
  set_target_properties(test_variable_stamp_index PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
  )
  add_test(NAME test_variable_stamp_index COMMAND test_variable_stamp_index)
endif()
