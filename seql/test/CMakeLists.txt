cmake_minimum_required(VERSION 3.8)

include(TestFiles.cmake)


set(TEST_NAME "test_seql")

enable_testing()

add_executable(${TEST_NAME} ${TEST_SRC})
target_link_libraries(${TEST_NAME} Catch2::Catch2 seql ${ARMADILLO_LIBRARIES})

set_target_properties(${TEST_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${TEST_NAME} PROPERTIES CXX_STANDARD 17)

catch_discover_tests(${TEST_NAME})
