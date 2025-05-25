//
// Created by rainbowx on 25-4-15.
//

#ifndef RECONSTRUCTION_BUNDLE_ADJUSTER_H
#define RECONSTRUCTION_BUNDLE_ADJUSTER_H

#include <ranges>

#include <ceres/solver.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/types.hpp>

#include "../world/WorldStructure.h"

struct Node {
    WorldPoint::Idx idx;
    double distance;

    [[nodiscard]] Node(const WorldPoint::Idx idx, const double distance)
        : idx(idx),
          distance(distance) {
    }

    friend bool operator<(const Node &lhs, const Node &rhs) {
        return lhs.distance < rhs.distance;
    }
};

class BundleAdjuster {
    class ReprojectCost {
        cv::Point2d observation;
    public:
        explicit ReprojectCost(const cv::Point2d &observation) : observation(observation) {}

        template<typename T>
        bool
        operator()(const T *const intrinsic, const T *const extrinsic, const T *const pos3d, T *residuals) const {
            const T *r = extrinsic;
            const T *t = &extrinsic[3];

            T pos_proj[3];
            ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

            // Apply the camera translation
            pos_proj[0] += t[0];
            pos_proj[1] += t[1];
            pos_proj[2] += t[2];

            const T x = pos_proj[0] / pos_proj[2];
            const T y = pos_proj[1] / pos_proj[2];

            const T fx = intrinsic[0];
            const T fy = intrinsic[1];
            const T cx = intrinsic[2];
            const T cy = intrinsic[3];

            // Apply intrinsic
            const T u = fx * x + cx;
            const T v = fy * y + cy;

            residuals[0] = u - T(observation.x);
            residuals[1] = v - T(observation.y);

            return true;
        }
    };

    WorldStructure::Ptr structure_;
    ceres::Solver::Options ceres_config_options_;

    std::unordered_map<Image::Ptr, cv::Matx23d> image_extrinsic_;
    std::unordered_map<Camera::Ptr, cv::Matx14d> camera_intrinsics_;
    std::unordered_map<WorldPoint::Ptr, cv::Matx13d> world_points_pos_;

    void setWorld(const WorldStructure::Ptr &world) {
        structure_ = world;
    }

    void loadDataFromWorld() {
        //加载mapPointsPos
        for (auto &mapPoints: structure_->world_points_ | std::ranges::views::values) {
            world_points_pos_[mapPoints] = mapPoints->getPosMatx13<double>();
        }
        //加载frameExtrinsics和cameraIntrinsics
        for (const auto &frame: structure_->local_frames_) {
            auto frame_image = frame->getImage2();
            auto angleAxis = frame_image->getAngleAxisWcMatxCV<double>();
            auto t = frame_image->getTcw().translation();

            image_extrinsic_[frame_image] = cv::Matx23d(angleAxis(0), angleAxis(1), angleAxis(2),
                                             t[0], t[1], t[2]);
            if (not camera_intrinsics_.contains(frame_image->getCamera()))
                camera_intrinsics_[frame_image->getCamera()] = frame_image->getCamera()->getIntrinsic();
        }
    }

    bool bundleAdjustment() {
        ceres::Problem problem;
        for (auto &val: image_extrinsic_ | std::views::values)
            problem.AddParameterBlock(val.val, 6);

        problem.SetParameterBlockConstant(image_extrinsic_[structure_->local_frames_.front()->getImage2()].val);
        for (auto &val: camera_intrinsics_ | std::views::values)
            problem.AddParameterBlock(val.val, 4);

        ceres::LossFunction *lossFunction = new ceres::HuberLoss(4);
        for (auto &[world_point, point_pos]: world_points_pos_) {
            for (auto &[image, observed_pos]: world_point->observed_frames_) {
                ceres::CostFunction *costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(
                                new ReprojectCost(observed_pos));
                problem.AddResidualBlock(
                        costFunction,
                        lossFunction,
                        camera_intrinsics_[image->getCamera()].val,            // Intrinsic
                        image_extrinsic_[image].val,  // View Rotation and Translation
                        point_pos.val          // Point in 3D space
                );
            }
        }

        ceres::Solver::Summary summary;
        ceres::Solve(ceres_config_options_, &problem, &summary);

        if (!summary.IsSolutionUsable()) {
            std::cout << "Bundle Adjustment failed." << std::endl;
            return false;
        }

        // Display statistics about the minimization
        std::cout << "Bundle Adjustment statistics (approximated RMSE):\n"
                  << "    #views: " << image_extrinsic_.size() << "\n"
                  << "    #residuals: " << summary.num_residuals << "\n"
                  << "    Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
                  << "    Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
                  << "    Time (s): " << summary.total_time_in_seconds << "\n";
        return true;
    }

    void updateWorld() const {
        //写mapPointsPos
        for (auto &mapPointPos: world_points_pos_) {
            mapPointPos.first->setPos(mapPointPos.second);
        }
        //写frameExtrinsics
        for (auto &frameExtrinsic: image_extrinsic_) {
            frameExtrinsic.first->setIntrinsic(frameExtrinsic.second);
        }
        //写cameraIntrinsics
        for (auto &cameraIntrinsic: camera_intrinsics_) {
            cameraIntrinsic.first->setIntrinsic(cameraIntrinsic.second);
        }
    }

    void clear() {
        structure_ = nullptr;

        camera_intrinsics_.clear();
        image_extrinsic_.clear();
        world_points_pos_.clear();
    }

public:
    [[nodiscard]] BundleAdjuster() {
        ceres_config_options_.minimizer_progress_to_stdout = false;
        ceres_config_options_.logging_type = ceres::SILENT;
        ceres_config_options_.num_threads = 1;
        ceres_config_options_.preconditioner_type = ceres::JACOBI;
        ceres_config_options_.linear_solver_type = ceres::SPARSE_SCHUR;
        ceres_config_options_.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    }

    void operator()(WorldStructure::Ptr &world) {
        setWorld(world);
        loadDataFromWorld();
        bool success = bundleAdjustment();
        if (success) {
            // washPoint();
            // bundleAdjustment();
            updateWorld();
        }
        clear();
    }

};

#endif // RECONSTRUCTION_BUNDLE_ADJUSTER_H
