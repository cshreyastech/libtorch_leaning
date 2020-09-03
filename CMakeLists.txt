cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(libtorch_learning VERSION 1.0.0 LANGUAGES CXX)

include_directories( include )

find_package(Torch REQUIRED PATHS "/home/shreyas/pytorchcpp/libtorch/")



set (DCGAN_HDR_FILES
    include/${PROJECT_NAME}/dcgangeneratorimpl.h
    include/${PROJECT_NAME}/dcgandiscriminatorimpl.h)

set(DCGAN_SOURCE_FILES
    src/dcgangeneratorimpl.cpp
    src/dcgandiscriminatorimpl.cpp
    src/main.cpp)

add_executable(dcgan ${DCGAN_HDR_FILES} ${DCGAN_SOURCE_FILES})
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)
