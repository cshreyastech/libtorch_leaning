cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(libtorch_learning VERSION 1.0.0 LANGUAGES CXX)

#include_directories( include )

find_package(Torch REQUIRED PATHS "/home/shreyas/pytorchcpp/libtorch/")

set(EXECUTABLE_NAME dcgan)
add_executable(${EXECUTABLE_NAME})

set (DCGAN_HDR_FILES
    include/${PROJECT_NAME}/dcgangeneratorimpl.h
    include/${PROJECT_NAME}/dcgandiscriminatorimpl.h)

set(DCGAN_SOURCE_FILES
    src/dcgangeneratorimpl.cpp
    src/dcgandiscriminatorimpl.cpp
    src/main.cpp)

#add_executable(${EXECUTABLE_NAME} ${DCGAN_HDR_FILES} ${DCGAN_SOURCE_FILES})

target_sources(${EXECUTABLE_NAME} PRIVATE ${DCGAN_HDR_FILES} ${DCGAN_SOURCE_FILES})

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

