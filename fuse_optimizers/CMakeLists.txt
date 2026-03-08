cmake_minimum_required(VERSION 3.0.2)
project(fuse_optimizers)

set(build_depends
  fuse_constraints
  fuse_core
  fuse_graphs
  fuse_variables
)

find_package(catkin REQUIRED COMPONENTS
  ${build_depends}
)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    ${build_depends}
)

###########
## Build ##
###########
add_compile_options(-Wall -Werror)

## fuse_optimizers library
add_library(${PROJECT_NAME}
  src/batch_optimizer.cpp
  src/fixed_lag_smoother.cpp
  src/optimizer.cpp
  src/variable_stamp_index.cpp
)
add_dependencies(${PROJECT_NAME}
  ${catkin_EXPORTED_TARGETS}
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    include
    ${catkin_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)
set_target_properties(${PROJECT_NAME}
  PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)

#############
## Install ##
#############

install(
  TARGETS
    ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(roslint REQUIRED)

  # Lint tests
  set(ROSLINT_CPP_OPTS "--filter=-build/c++11,-runtime/references")
  roslint_cpp()
  roslint_add_test()

  # VariableStampIndex Tests
  catkin_add_gtest(test_variable_stamp_index
    test/test_variable_stamp_index.cpp
  )
  target_include_directories(test_variable_stamp_index
    PRIVATE
      include
      ${catkin_INCLUDE_DIRS}
  )
  target_link_libraries(test_variable_stamp_index
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
  )
  set_target_properties(test_variable_stamp_index
    PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
  )
endif()
