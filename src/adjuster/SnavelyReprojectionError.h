//
// Created by rainbowx on 25-4-15.
//

#ifndef RECONSTRUCTION_SNAVELY_REPROJECTIONE_RROR_H
#define RECONSTRUCTION_SNAVELY_REPROJECTIONE_RROR_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/types.hpp>

struct SnavelyReprojectionError {
    [[nodiscard]] SnavelyReprojectionError(const cv::Point2d &observed_point, const double camera_focal_x,
        const double camera_focal_y, const double camera_center_x, const double camera_center_y)
        : observed_point_(observed_point),
          camera_focal_x_(camera_focal_x),
          camera_focal_y_(camera_focal_y),
          camera_center_x_(camera_center_x),
          camera_center_y_(camera_center_y) {
    }

    template <typename T>
    auto operator()(
        const T *const extrinsic,   // 6
        const T *const pos3d,       // 3
        T *residuals
        ) const
    {
        const T *r = extrinsic;
        const T *t = &extrinsic[4];

        T pos_proj[3];
        ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

        // Apply the camera translation
        pos_proj[0] += t[0];
        pos_proj[1] += t[1];
        pos_proj[2] += t[2];

        const T x = pos_proj[0] / pos_proj[2];
        const T y = pos_proj[1] / pos_proj[2];

        const auto fx = camera_focal_x_;
        const auto fy = camera_focal_y_;
        const auto cx = camera_center_x_;
        const auto cy = camera_center_y_;

        // Apply intrinsic
        const T u = fx * x + cx;
        const T v = fy * y + cy;

        residuals[0] = u - T(observed_point_.x);
        residuals[1] = v - T(observed_point_.y);

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static auto Create(
        const cv::Point2d &observed_point,
        const double camera_focal_x, const double camera_focal_y,
        const double camera_center_x, const double camera_center_y
        ) {
        return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(new SnavelyReprojectionError{
            observed_point,
            camera_focal_x, camera_focal_y,
            camera_center_x, camera_center_y
        });
    }

    cv::Point2d observed_point_;
    double camera_focal_x_, camera_focal_y_;
    double camera_center_x_, camera_center_y_;
};



#endif // RECONSTRUCTION_SNAVELY_REPROJECTIONE_RROR_H
