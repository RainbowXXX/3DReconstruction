//
// Created by rainbowx on 25-4-14.
//

#ifndef RECONSTRUCTION_IMAGE_H
#define RECONSTRUCTION_IMAGE_H

#include <utility>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "../world/WorldPoint.h"

class Image {
public:
    using Idx = std::size_t;
    using Ptr = std::shared_ptr<Image>;

private:
    static inline Idx cur_idx_ = 0;

    Idx idx_;
    Camera::Ptr camera_;
    cv::Mat descriptors_;
    cv::Mat image_, desc_mask_;
    std::vector<cv::KeyPoint> key_points_;

    Sophus::SE3<double> Tcw;      // transform from world to camera

    std::unordered_map<std::size_t, WorldPoint::Idx> kpt_wpt_idx_map_;
public:
    [[nodiscard]] explicit Image(cv::Mat image, Camera::Ptr camera)
        : image_(std::move(image)), camera_{std::move(camera)}, idx_{cur_idx_++} {
    }

    auto detectAndCompute(cv::Ptr<cv::Feature2D>& detector) {
        detector->detectAndCompute(image_, cv::Mat(), key_points_, descriptors_);
    }

    static auto getTotalImages() {
        return cur_idx_;
    }

    auto getCamera() {
        return camera_;
    }

    auto getIdx() const {
        return idx_;
    }

    auto getKeyPoints() const -> const std::vector<cv::KeyPoint>& {
        if (key_points_.empty()) {
            spdlog::warn("The key point list is empty, is Func detectAndCompute is called?");
        }
        return key_points_;
    }

    auto getDescriptors() const -> const cv::Mat&  {
        if (descriptors_.empty()) {
            spdlog::warn("The descriptor list is empty, is Func detectAndCompute is called?");
        }
        return descriptors_;
    }

    auto getWorldPointIdx(std::size_t key_point_idx) const {
        if (not kpt_wpt_idx_map_.contains(key_point_idx)) {
            return WorldPoint::InvalidIdx;
        }
        return kpt_wpt_idx_map_.at(key_point_idx);
    }

    auto setWorldPoint(const std::size_t kpt_idx, const WorldPoint::Idx wpt_idx) {
        if (kpt_wpt_idx_map_.contains(kpt_idx)) {
            spdlog::warn("Key point has already exists.");
        }
        kpt_wpt_idx_map_[kpt_idx] = wpt_idx;
    }

    template<typename T>
    cv::Matx<T, 1, 3> getAngleAxisWcMatxCV() {
        Eigen::AngleAxisd angleAxis(Tcw.so3().matrix());
        auto axis = angleAxis.angle() * angleAxis.axis();
        cv::Matx<T, 1, 3> angleAxisCV(axis[0],axis[1],axis[2]);
        return angleAxisCV;
    };

    cv::Mat getTcwMatCV(int rtype) const {
        cv::Mat TcwCV, TcwCVR;
        cv::eigen2cv(Tcw.matrix(), TcwCV);
        TcwCV.convertTo(TcwCVR, rtype);
        return TcwCVR;
    }

    cv::Mat getTcw34MatCV(int rtype) const {
        auto TcwCV = getTcwMatCV(rtype);
        cv::Mat Tcw34;
        TcwCV(cv::Range(0, 3), cv::Range(0, 4)).convertTo(Tcw34, rtype);
        return Tcw34;
    }

    [[nodiscard]] Sophus::SE3<double> getTcw() const {
        return Tcw;
    }

    void setTcw(const Sophus::SE3<double> &tcw) {
        Tcw = tcw;
    }

    template<typename T>
    void setTcw(cv::Matx<T,2,3> angleAxisAndTranslation){
        Eigen::Matrix3d R =
            (Eigen::AngleAxisd(angleAxisAndTranslation(0,0), Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(angleAxisAndTranslation(0,1), Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(angleAxisAndTranslation(0,2), Eigen::Vector3d::UnitX())).toRotationMatrix();

        Tcw.so3()=Sophus::SO3<double>(R);
        Tcw.translation()=Eigen::Vector3d(angleAxisAndTranslation(1,0),
                                     angleAxisAndTranslation(1,1),
                                     angleAxisAndTranslation(1,2));
    }

    auto type() {
        return image_.type();
    }

    template<typename Ty>
    auto at(cv::Point pt) {
        return image_.at<Ty>(pt);
    }

    auto getImage() {
        return image_;
    }
};

using ImageRef = std::shared_ptr<Image>;
#endif // RECONSTRUCTION_IMAGE_H
