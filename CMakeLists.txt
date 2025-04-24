cmake_minimum_required(VERSION 3.30)
project(3DReconstruction)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_HEADER_ONLY")

# OpenCV
find_package(spdlog REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
# OpenCV
find_package(OpenCV 4 REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
# Ceres
find_package(Ceres 2.2 REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
# PCL
find_package(PCL REQUIRED COMPONENTS common io visualization)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
# VTK
find_package(VTK 9.1 REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})

include_directories(
        3rdparty/spdlog/include
        3rdparty/stats/include
        3rdparty/OpenMVS/libs
)

add_executable(3DReconstruction
        src/main.cpp
        src/world/WorldStructure.h
        src/component/Image.h
        src/frame/LocalFrame.h
        src/actuator/SequentialActuator.h
        src/frame/GlobalFrame.h
        src/adjuster/SnavelyReprojectionError.h
        src/adjuster/BundleAdjuster.h
        src/world/WorldPoint.h
        src/component/Camera.h
        src/common/common.h
        src/denseBuilder/DenseBuilder.h
        src/denseBuilder/example.h
)

target_link_libraries(3DReconstruction
        ${OpenCV_LIBS}
        opencv_cvv
        ${VTK_LIBRARIES}
        ${PCL_LIBRARIES}
        ${CERES_LIBRARIES}
)
