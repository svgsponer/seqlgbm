cmake_minimum_required(VERSION 3.8)
add_definitions(-Wall -Wextra -Wpedantic)

project("SEQL" CXX)

## Install the executable to bin fodler
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin") 

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${LIB_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIB_PATH})


find_package(Armadillo CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)


add_subdirectory (src)

OPTION(BUILD_TESTING "Build tests" OFF)

if(BUILD_TESTING)
  find_package(Catch2 REQUIRED)
  include(CTest)
  include(Catch)
  add_subdirectory(test)
endif(BUILD_TESTING)


