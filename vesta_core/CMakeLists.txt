## vesta_core library
add_library(vesta_core
  src/constraint.cpp
  src/graph.cpp
  src/graph_deserializer.cpp
  src/jacobian_relinearization.cpp
  src/loss.cpp
  src/schur_ordering.cpp
  src/serialization.cpp
  src/timestamp.cpp
  src/timestamp_manager.cpp
  src/transaction.cpp
  src/transaction_deserializer.cpp
  src/uuid.cpp
  src/variable.cpp
)
target_include_directories(vesta_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(vesta_core PUBLIC
  Ceres::ceres
  Eigen3::Eigen
  Boost::serialization
  ${GLOG_LIBRARIES}
)
target_include_directories(vesta_core PUBLIC
  ${GLOG_INCLUDE_DIRS}
)
set_target_properties(vesta_core PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)

## Tests
if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  # Helper function to reduce repetition
  function(vesta_core_add_test TEST_NAME)
    add_executable(${TEST_NAME} test/${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE vesta_core GTest::GTest GTest::Main)
    target_include_directories(${TEST_NAME} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(${TEST_NAME} PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  endfunction()

  vesta_core_add_test(test_constraint)
  vesta_core_add_test(test_eigen)
  vesta_core_add_test(test_manifold)
  vesta_core_add_test(test_loss)
  vesta_core_add_test(test_message_buffer)
  vesta_core_add_test(test_timestamp_manager)
  vesta_core_add_test(test_transaction)
  vesta_core_add_test(test_util)
  vesta_core_add_test(test_uuid)
  vesta_core_add_test(test_variable)
  # Schur ordering test needs vesta_graphs for HashGraph
  add_executable(test_schur_ordering test/test_schur_ordering.cpp)
  target_link_libraries(test_schur_ordering PRIVATE vesta_core vesta_graphs GTest::GTest GTest::Main)
  target_include_directories(test_schur_ordering PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(test_schur_ordering PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
  add_test(NAME test_schur_ordering COMMAND test_schur_ordering)
  # Jacobian relinearization test needs vesta_graphs for HashGraph
  add_executable(test_jacobian_relinearization test/test_jacobian_relinearization.cpp)
  target_link_libraries(test_jacobian_relinearization PRIVATE vesta_core vesta_graphs GTest::GTest GTest::Main)
  target_include_directories(test_jacobian_relinearization PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(test_jacobian_relinearization PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
  add_test(NAME test_jacobian_relinearization COMMAND test_jacobian_relinearization)
endif()
