cmake_minimum_required(VERSION 3.10)
project(go2_sdk_recorder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(UNITREE_SDK2_ROOT "/home/unitree/unitree_sdk2-main" CACHE PATH
    "Path to the Unitree SDK2 source tree")

if(NOT EXISTS "${UNITREE_SDK2_ROOT}/include/unitree/robot/channel/channel_subscriber.hpp")
  message(FATAL_ERROR "UNITREE_SDK2_ROOT does not point to a Unitree SDK2 tree")
endif()

set(UNITREE_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
find_package(Threads REQUIRED)

add_executable(go2_sdk_recorder go2_sdk_recorder.cpp)
target_include_directories(go2_sdk_recorder PRIVATE
  "${UNITREE_SDK2_ROOT}/include"
  "${UNITREE_SDK2_ROOT}/thirdparty/include"
  "${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx")
target_link_libraries(go2_sdk_recorder PRIVATE
  "${UNITREE_SDK2_ROOT}/lib/${UNITREE_ARCH}/libunitree_sdk2.a"
  "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}/libddscxx.so"
  "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}/libddsc.so"
  Threads::Threads)
set_target_properties(go2_sdk_recorder PROPERTIES
  BUILD_RPATH "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}")
