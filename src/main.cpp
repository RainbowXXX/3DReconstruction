#include <vector>
#include <fstream>
#include <exception>
#include <filesystem>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

// // openmvg
// #include "openMVG/cameras/cameras.hpp"
// #include "openMVG/exif/exif_IO_EasyExif.hpp"
// #include "openMVG/exif/sensor_width_database/ParseDatabase.hpp"
// #include "openMVG/geodesy/geodesy.hpp"
// #include "openMVG/image/image_io.hpp"
// #include "openMVG/numeric/eigen_alias_definition.hpp"
// #include "openMVG/sfm/sfm_data.hpp"
// #include "openMVG/sfm/sfm_data_io.hpp"
// #include "openMVG/sfm/sfm_data_utils.hpp"
// #include "openMVG/sfm/sfm_view.hpp"
// #include "openMVG/sfm/sfm_view_priors.hpp"
// #include "openMVG/system/loggerprogress.hpp"
// #include "openMVG/types.hpp"

// spdlog
// #include <spdlog/spdlog.h>

#define MYCALL
#ifdef _DEBUG
#define Crash(...) (__debugbreak(),0)
#define Debug(...) (__debugbreak(),0)
#else
#define Crash(...) (exit(-1),0)
#define Debug(...) ((void)0)
#endif // _DEBUG

#define ensure(expression) (void)(!!(expression)||Crash())
#define static_ensure(expression) static_assert(expression)


using namespace std::string_literals;

std::string base_path = "/home/rainbowx/Documents/Projects/3DRebuild/";

std::string images_path = base_path + "images/";
std::string chessboards_path = base_path + "chessboard/";

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;

// 保证不溢出屏幕的展示图片
auto showImage(
        const std::string& winname,
        const cv::Mat& image,
        const cv::Size& fixed_size = cv::Size(800, 600)
    ) -> void
{
    // 创建或更新窗口属性
    try{
        if (cv::getWindowProperty(winname, cv::WND_PROP_VISIBLE) < 0) {
            cv::namedWindow(winname, cv::WINDOW_NORMAL);
        }
    }
    catch (const cv::Exception& e) {
        cv::namedWindow(winname, cv::WINDOW_NORMAL);
    }
    // 设置窗口尺寸
    cv::resizeWindow(winname, fixed_size.width, fixed_size.height);
    // 显示图像
    cv::imshow(winname, image);
}

// 计算相机内参
auto [[maybe_unused]] calibrateCameraFromImages(
        const std::vector<cv::Mat>& images,
        const cv::Size& board_size,
        float square_size,
        cv::Mat& camera_matrix,
        cv::Mat& dist_coeffs,
        [[maybe_unused]] bool debug = false
    ) -> double
{
    int board_width = board_size.width - 1;  // 棋盘格横向内角点数量
    int board_height = board_size.height - 1; // 棋盘格纵向内角点数量

    // 准备对象点（世界坐标系下的角点坐标）
    std::vector<cv::Point3f> obj_points;
    for (int i = 0; i < board_height; ++i) {
        for (int j = 0; j < board_width; ++j) {
            obj_points.emplace_back(static_cast<double>(j) * square_size, static_cast<double>(i) * square_size, 0);
        }
    }

    // 存储对象点和图像点
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;

    // 读取标定图像
    for (const auto& img : images) {
        cv::Mat gray;
        cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        // 寻找棋盘格角点
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(img, cv::Size(board_width, board_height), corners);
        if (found) {
            // 提取亚像素级角点
            cv::cornerSubPix(
                    gray,
                    corners,
                    cv::Size{11, 11},
                    cv::Size{-1, -1},
                    cv::TermCriteria{cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1}
                );
            object_points.push_back(obj_points);
            image_points.push_back(corners);

            if (debug) {
                // 在图像上绘制角点
                cv::drawChessboardCorners(img, cv::Size(board_width, board_height), corners, found);
                showImage("Chessboard", img);
                cv::waitKey(0);
            }
        }
    }

    // 相机标定
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(
        object_points,
        image_points,
        images[0].size(),
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs
    );

    return rms;
}

int main() {
    namespace fs = std::filesystem;

    cv::Mat camera_matrix, dist_coeffs;

    {
        // 计算相机内参
        // spdlog::info("Start to calibrate camera.");
        float square_size = 1.f; // 方格尺寸，单位：毫米
        cv::Size board_size(8, 11); // 棋盘格内角点数量
        std::vector<cv::Mat> chessboards_images;

        for (const auto& entry : fs::directory_iterator(chessboards_path)) {
            if (entry.is_regular_file()) {
                chessboards_images.emplace_back(cv::imread(entry.path()));
            }
        }

        [[maybe_unused]]auto rms = calibrateCameraFromImages(chessboards_images, board_size, square_size, camera_matrix, dist_coeffs);

        // spdlog::info("Calibrate camera done.");
        // spdlog::debug("Camera matrix K: {}", camera_matrix);

        std::cout<< camera_matrix<<std::endl;
    }

    // std::vector<cv::Mat> images;
    // {
    //     // 读取图像文件
    //     std::vector<std::string> image_paths;
    //     std::optional<cv::Size> image_size = std::nullopt;
    //     for (const auto& entry : fs::directory_iterator(images_path)) {
    //         if (entry.is_regular_file()) {
    //             image_paths.emplace_back(entry.path());
    //         }
    //     }
    //     std::ranges::sort(image_paths);
    //     for (const auto& image_path: image_paths) {
    //         images.emplace_back(cv::imread(image_path));
    //         cv::Size curSize = cv::Size{images.back().cols, images.back().rows};
    //         if (not image_size.has_value()) {
    //             *image_size = curSize;
    //             continue;
    //         }
    //         if (image_size.value() != curSize) {
    //             throw std::runtime_error("Size of images must be same.");
    //         }
    //     }
    // }


    return 0;
}
