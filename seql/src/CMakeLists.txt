cmake_minimum_required(VERSION 3.8)


add_library(seql seql.cpp evaluation.cpp search_node.cpp seql_learn.cpp linear_model.cpp seql_boost.cpp preprocessing.cpp)

target_link_libraries(seql PUBLIC nlohmann_json::nlohmann_json stdc++fs)
target_link_libraries(seql PRIVATE ${ARMADILLO_LIBRARIES})


set_target_properties(seql PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(seql PROPERTIES CXX_STANDARD 17)

find_package(Threads)
add_executable(seqlr seqlr.cpp)
# target_include_directories(seqlr )
target_link_libraries(seqlr seql nlohmann_json::nlohmann_json ${ARMADILLO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

set_target_properties(seqlr PROPERTIES CXX_STANDARD 17)
