cmake_minimum_required(VERSION 3.30)
project(3DReconstruction)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

include_directories(
        ${OPENCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}

        "3rdparty/spdlog/include"
        "3rdparty/cmdLine"
)

add_executable(3DReconstruction
        src/main.cpp
)

target_link_libraries(3DReconstruction
        ${OpenCV_LIBS}
)
