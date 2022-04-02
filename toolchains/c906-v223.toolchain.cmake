set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(C906 True)

set(RISCV_ROOT_PATH "/opt/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.3")

set(CMAKE_C_COMPILER "${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++")

set(CMAKE_FIND_ROOT_PATH "${RISCV_ROOT_PATH}/riscv64-unknown-linux-gnu")

set(CMAKE_SYSROOT "${RISCV_ROOT_PATH}/sysroot")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_C_FLAGS "-march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c906 -DC906=1 -static")
set(CMAKE_CXX_FLAGS "-march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c906 -DC906=1 -static")

# replace vfredsum_vs* with vfredusum_vs*
add_definitions(-Dvfredsum_vs_f32m1_f32m1=vfredusum_vs_f32m1_f32m1)
add_definitions(-Dvfredsum_vs_f32m2_f32m1=vfredusum_vs_f32m2_f32m1)
add_definitions(-Dvfredsum_vs_f16m1_f16m1=vfredusum_vs_f16m1_f16m1)
add_definitions(-Dvfredsum_vs_f32m8_f32m1=vfredusum_vs_f32m8_f32m1)
add_definitions(-Dvfredsum_vs_f16m8_f16m1=vfredusum_vs_f16m8_f16m1)
add_definitions(-Dvfredsum_vs_f16m4_f16m1=vfredusum_vs_f16m4_f16m1)
