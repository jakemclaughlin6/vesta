## vesta_graphs library
add_library(vesta_graphs
  src/hash_graph.cpp
)
target_include_directories(vesta_graphs PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(vesta_graphs PUBLIC
  vesta_core
  Ceres::ceres
  Boost::serialization
)
set_target_properties(vesta_graphs PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)

## Tests
if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  add_executable(test_hash_graph test/test_hash_graph.cpp)
  target_link_libraries(test_hash_graph PRIVATE vesta_graphs GTest::GTest GTest::Main)
  target_include_directories(test_hash_graph PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_target_properties(test_hash_graph PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
  )
  add_test(NAME test_hash_graph COMMAND test_hash_graph)

  # Benchmarks
  find_package(benchmark QUIET)
  if(benchmark_FOUND)
    add_executable(benchmark_create_problem
      benchmark/benchmark_create_problem.cpp
    )
    target_link_libraries(benchmark_create_problem PRIVATE
      benchmark::benchmark
      vesta_graphs
    )
    target_include_directories(benchmark_create_problem PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(benchmark_create_problem PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
    )
  endif()
endif()
