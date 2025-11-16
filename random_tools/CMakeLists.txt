cmake_minimum_required(VERSION 3.1)

project(CorrectionMatrix VERSION 1.0
                  LANGUAGES CXX)

add_library(corr_Tree SHARED corr_Tree.cpp)
target_compile_features(corr_Tree PUBLIC cxx_std_14)
target_compile_options(corr_Tree PRIVATE -Wall -std=c++14)
set_target_properties(corr_Tree  PROPERTIES CXX_EXTENSIONS OFF)

find_package(ROOT 6.20 CONFIG REQUIRED)
include("${ROOT_USE_FILE}")

add_executable(corr_Tree.x corr_Tree.cpp)
target_link_libraries(corr_Tree.x PRIVATE corr_Tree ROOT::MathCore ROOT::Hist ROOT::Physics ROOT::Gpad ROOT::Tree)
target_compile_options(corr_Tree.x PRIVATE -Wall -std=c++14)
