//
// Created by rainbowx on 25-4-20.
//

#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

class Camera {
public:
    using Ptr = std::shared_ptr<Camera>;

private:
    double fx, fy, cx, cy;

public:
    Camera(const double fx, const double fy, const double cx, const double cy) :
            fx(fx), fy(fy), cx(cx), cy(cy) {}

    explicit Camera(cv::Mat camera_matrix) {
        fx = camera_matrix.at<double>(0, 0);
        fy = camera_matrix.at<double>(1, 1);
        cx = camera_matrix.at<double>(0, 2);
        cy = camera_matrix.at<double>(1, 2);
    }

    template<typename T>
    void setIntrinsic(cv::Matx<T, 1, 4> intrinsic) {
        fx=intrinsic(0);
        fy=intrinsic(1);
        cx=intrinsic(2);
        cy=intrinsic(3);
    }

    //坐标转换

    Eigen::Vector3d world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3<double> &T_c_w) {
        return T_c_w * p_w;
    }

    Eigen::Vector3d camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3<double> &T_c_w) {
        return T_c_w.inverse() * p_c;
    }

    Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c) {
        return {
                fx * p_c(0, 0) / p_c(2, 0) + cx,
                fy * p_c(1, 0) / p_c(2, 0) + cy
        };
    }

    Eigen::Vector3d pixel2camera(const Eigen::Vector2d &p_p, double depth = 1) {
        return {
                (p_p(0, 0) - cx) * depth / fx,
                (p_p(1, 0) - cy) * depth / fy,
                depth
        };
    }

    Eigen::Vector3d pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3<double> &T_c_w, double depth = 1) {
        return camera2world(pixel2camera(p_p, depth), T_c_w);
    }

    Eigen::Vector2d world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3<double> &T_c_w) {
        return camera2pixel(world2camera(p_w, T_c_w));
    }

    [[nodiscard]] cv::Point2f pixel2normal(const cv::Point2d &p) const {
        return {
                        static_cast<float>((p.x - cx) / fx),
                        static_cast<float>((p.y - cy) / fy)
                };
    }

    //获取参数
    [[nodiscard]] double getFocalLength() const {
        return (fx + fy) / 2;
    }

    [[nodiscard]] cv::Point2d getPrincipalPoint() const {
        return {cx, cy};
    }

    [[nodiscard]] cv::Matx<double, 3, 3> getKMatxCV() const {
        return {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    }

    [[nodiscard]] cv::Mat getKMatCV() const {
        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        return K;
    }

    [[nodiscard]] auto getIntrinsic() const {
        return cv::Matx14d(fx, fy, cx, cy);
    }
};

#endif //CAMERA_H
