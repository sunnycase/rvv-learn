cmake_minimum_required(VERSION 3.18)
project(rvv-learn LANGUAGES CXX ASM)
set (CMAKE_CXX_STANDARD 17)

set(SRCS main.cpp)
add_executable(rvv-learn ${SRCS})