#include <vector>
#include <fstream>
#include <exception>
#include <filesystem>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include "frame/LocalFrame.h"
#include "actuator/SequentialActuator.h"

// spdlog
#include <spdlog/spdlog.h>

#include<vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2)
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);

using namespace std::string_literals;

std::string base_path = "/home/rainbowx/Documents/Projects/3DRebuild/";

std::string images_path = base_path + "images2/";
std::string chessboards_path = base_path + "chessboard/";

int main() {
    namespace fs = std::filesystem;

    cv::Mat camera_matrix{}, dist_coeffs{};

    camera_matrix  = (cv::Mat_<double>(3,3)<<2905.88,0,1416,0,2905.88,1064,0,0,1);
    // {
    //     // 计算相机内参
    //     spdlog::info("Start to calibrate camera.");
    //     float square_size = 1.f; // 方格尺寸，单位：毫米
    //     cv::Size board_size(8, 11); // 棋盘格内角点数量
    //     std::vector<cv::Mat> chessboards_images;
    //
    //     for (const auto& entry : fs::directory_iterator(chessboards_path)) {
    //         if (entry.is_regular_file()) {
    //             chessboards_images.emplace_back(cv::imread(entry.path()));
    //         }
    //     }
    //
    //     [[maybe_unused]]auto rms = calibrateCameraFromImages(chessboards_images, board_size, square_size, camera_matrix, dist_coeffs);
    //
    //     spdlog::info("Calibrate camera done.");
    //     std::cout<< camera_matrix<<std::endl;
    //     std::cout<< dist_coeffs<<std::endl;
    // }

    std::vector<Image> images;
    {
        // 读取图像文件
        spdlog::info("Start to read images.");
        std::vector<std::string> image_paths;
        for (const auto& entry : fs::directory_iterator(images_path)) {
            if (entry.is_regular_file() and (entry.path().extension() != ".txt")) {
                image_paths.emplace_back(entry.path());
            }
        }
        std::ranges::sort(image_paths);
        auto camera = std::make_shared<Camera>(camera_matrix);
        for (const auto& image_path: image_paths) {
            cv::Mat img = cv::imread(image_path);
            images.emplace_back(img, camera);
        }
        spdlog::info("Read images done.");
    }

    SequentialActuator actuator;

    actuator.init(images[0], images[1]);
    actuator.bundleAdjustment();
    actuator.show(true);
    for (int i = 2;i< images.size();i++) {
        actuator.addSingleImage(images[i]);
        actuator.bundleAdjustment();
        actuator.show(true);
    }

    // auto world = actuator.getWorld();
    // world->writeToFile("test.pcd");
    actuator.show();

    return 0;
}
