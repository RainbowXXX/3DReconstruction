//
// Created by rainbowx on 25-4-15.
//

#ifndef RECONSTRUCTION_WORLDPOINT_H
#define RECONSTRUCTION_WORLDPOINT_H
#include <cstddef>

#include <opencv2/opencv.hpp>

class Image;

struct WorldPoint {
    using Ptr = std::shared_ptr<WorldPoint>;

    using Idx = std::size_t;

    static constexpr Idx InvalidIdx = static_cast<Idx>(-1);

    Idx idx_;
    Eigen::Vector3d world_pos_;
    cv::Vec3b point_color_;
    cv::Mat last_descriptor_;
    std::vector<std::pair<std::shared_ptr<Image>, cv::Point2d>> observed_frames_;

    auto operator <=> (const WorldPoint & other) const {
        return idx_ <=> other.idx_;
    }

    template <typename T>
    cv::Matx<T,1,3> getPosMatx13() const{
        return cv::Matx<T,1,3>(world_pos_(0, 0), world_pos_(1, 0), world_pos_(2, 0));
    }

    template <typename T>
    void setPos(cv::Matx<T,1,3> posMatx13){
        world_pos_(0)=posMatx13(0);
        world_pos_(1)=posMatx13(1);
        world_pos_(2)=posMatx13(2);
    }
};

using WorldPointRef = std::shared_ptr<WorldPoint>;

#endif // RECONSTRUCTION_WORLDPOINT_H
