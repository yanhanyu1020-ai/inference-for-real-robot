cmake_minimum_required(VERSION 3.10)
project(go2_command_executor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(UNITREE_SDK2_ROOT "/home/unitree/unitree_sdk2-main" CACHE PATH
    "Path to the Unitree SDK2 source tree")

if(NOT EXISTS "${UNITREE_SDK2_ROOT}/include/unitree/robot/go2/sport/sport_client.hpp")
  message(FATAL_ERROR "UNITREE_SDK2_ROOT does not point to a Unitree SDK2 tree")
endif()

set(UNITREE_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
find_package(Threads REQUIRED)

function(add_go2_sdk_binary name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE
    "${UNITREE_SDK2_ROOT}/include"
    "${UNITREE_SDK2_ROOT}/thirdparty/include"
    "${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx")
  target_link_libraries(${name} PRIVATE
    "${UNITREE_SDK2_ROOT}/lib/${UNITREE_ARCH}/libunitree_sdk2.a"
    "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}/libddscxx.so"
    "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}/libddsc.so"
    Threads::Threads)
  set_target_properties(${name} PROPERTIES
    BUILD_RPATH "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}")
endfunction()

add_go2_sdk_binary(go2_command_executor go2_command_executor.cpp)
add_go2_sdk_binary(go2_velocity_bridge go2_velocity_bridge.cpp)
