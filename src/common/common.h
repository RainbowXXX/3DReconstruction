//
// Created by rainbowx on 25-4-20.
//

#ifndef COMMON_H
#define COMMON_H

#include <opencv4/opencv2/opencv.hpp>

#include "utils/Event.h"
#include "utils/ConcurrentQueue.h"

#define MYCALL
#if CMAKE_BUILD_TYPE == Debug
#define Crash(...) (__builtin_trap(),0)
#define Debug(...) (__builtin_trap(),0)
#else
#define Crash(...) (exit(-1),0)
#define Debug(...) ((void)0)
#endif // _DEBUG

#define ensure(expression) (void)(!!(expression)||Crash())
#define static_ensure(expression) static_assert(expression)

extern ConcurrentQueue<::Event> event_queue;

inline bool isNormal(const double& x) {
    if (std::isnan(x)) return false;
    if (std::isinf(x)) return false;
    return true;
}

inline bool isNormal(const float& x) {
    if (std::isnan(x)) return false;
    if (std::isinf(x)) return false;
    return true;
}

inline double calc3DPointDistance(const cv::Matx<double, 1, 3>& lhs, const cv::Matx<double, 1, 3>& rhs) {
    const auto dx = lhs(0, 0) - rhs(0, 0);
    const auto dy = lhs(0, 1) - rhs(0, 1);
    const auto dz = lhs(0, 2) - rhs(0, 2);

    const auto distance = dx*dx + dy*dy + dz*dz;

    ensure(isNormal(distance));

    return distance;
}

// 保证不溢出屏幕的展示图片
inline auto showImage(
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
inline auto [[maybe_unused]] calibrateCameraFromImages(
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

#endif //COMMON_H
