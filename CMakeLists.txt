cmake_minimum_required(VERSION 3.30)
project(3DReconstruction)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_HEADER_ONLY")

# 设置 OpenMVG 和 OpenMVS 安装路径
set(OpenMVG_DIR "3rdparty/openMVG/install/lib/openMVG/cmake")

# 查找库
find_package(OpenMVG REQUIRED)

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

find_package(PCL REQUIRED COMPONENTS common io visualization)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

#add_subdirectory("3rdparty/openMVS/" EXCLUDE_FROM_ALL)
add_subdirectory("3rdparty/openMVG/src" EXCLUDE_FROM_ALL)

include_directories(
        3rdparty/spdlog/include
        3rdparty/stats/include

#        ${OpenMVS_INCLUDE_DIRS}

        3rdparty/cpp-httplib
        3rdparty/nlohmann/include

#        3rdparty/openMVS/install/include/OpenMVS

        3rdparty/openMVG/install/include/openMVG_dependencies/cereal/include
        3rdparty/openMVG/install/include/openMVG/third_party/lemon
        3rdparty/openMVG/install/include/openMVG/third_party
        3rdparty/openMVG/install/include/openMVG
        3rdparty/openMVG/install/include

        3rdparty/openMVG/install/include/openMVG_dependencies/osi_clp/Clp/src
        3rdparty/openMVG/install/include/openMVG_dependencies/osi_clp/CoinUtils/src
        3rdparty/openMVG/install/include/openMVG_dependencies/osi_clp/Osi/src/Osi

        src
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
        src/example/OpenMVGExample.cpp
        src/example/OpenMVGExample.h

        3rdparty/openMVG/src/dependencies/osi_clp/Clp/src/OsiClp/OsiClpSolverInterface.cpp

        src/sparseBuilder/sparseBuilder.cpp
        src/sparseBuilder/sparseBuilder.h
        src/utils/Result.h
        src/utils/Unit.h
        src/utils/RegionsProvider.h
        src/utils/FeaturesProvider.h
        src/utils/MatchesProvider.h
        src/utils/ConcurrentQueue.h
        src/utils/Event.h
        src/utils/MyProgress.h
)

file(GLOB OpenMVG_ALL_LIBS "3rdparty/openMVG/install/lib/*.a")

link_directories(3rdparty/openMVG/install/lib)

target_link_libraries(3DReconstruction
        ${OpenMVG_ALL_LIBS}

        ${OpenCV_LIBS}
        opencv_cvv
        ${VTK_LIBRARIES}
        ${PCL_LIBRARIES}
        ${CERES_LIBRARIES}

#        ${OpenMVS_LIBRARIES}
#        ${OpenMVG_LIBRARIES}

#        Boost::asio
#        Boost::beast
#        Boost::filesystem

        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/liblib_clp.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/liblib_CoinUtils.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/liblib_Osi.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/liblib_OsiClpSolver.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_easyexif.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_exif.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_fast.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_features.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_geometry.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_image.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_kvld.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_linearProgramming.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_lInftyComputerVision.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_matching.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_matching_image_collection.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_multiview.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_numeric.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_robust_estimation.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_sfm.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_stlplus.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libopenMVG_system.a
        /home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/install/lib/libvlsift.a
)
