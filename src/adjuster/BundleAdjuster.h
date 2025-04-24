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

#include "SnavelyReprojectionError.h"
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

    WorldStructure::Ptr structure_;
    ceres::Solver::Options ceres_config_options_;

    std::unordered_map<Image::Ptr, cv::Matx23d> image_extrinsic_;
    std::unordered_map<Camera::Ptr, cv::Matx13d> camera_intrinsics_;
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
        auto frame_image1 = structure_->local_frames_.front()->getImage1();
        if (not camera_intrinsics_.contains(frame_image1->getCamera()))
            camera_intrinsics_[frame_image1->getCamera()] = frame_image1->getCamera()->getIntrinsic();
        image_extrinsic_[frame_image1] = frame_image1->getIntrinsic();
        for (const auto &frame: structure_->local_frames_) {
            auto frame_image = frame->getImage2();

            if (not camera_intrinsics_.contains(frame_image->getCamera()))
                camera_intrinsics_[frame_image->getCamera()] = frame_image->getCamera()->getIntrinsic();
            image_extrinsic_[frame_image1] = frame_image1->getIntrinsic();
        }
    }

    auto washPoint() {
        std::vector<Node> data;
        std::unordered_set<WorldPoint::Idx> invalid_point_idx;

        for (auto &world_point: structure_->adjust_points_ | std::ranges::views::values) {
            const auto cur_pos = world_points_pos_[world_point];
            const auto old_pos = world_point->getPosMatx13<double>();
            const auto distance = calc3DPointDistance(cur_pos, old_pos);

            if (not isNormal(distance)) {
                invalid_point_idx.insert(world_point->idx_);
                spdlog::warn("Fail to calc min and max distance of point {}.", world_point->idx_);
                continue;
            }

            data.emplace_back(world_point->idx_, distance);
        }

        std::sort(data.begin(), data.end());

        auto min_node = data.front(), max_node = data.back();
        if (not (isNormal(max_node.distance) and isNormal(min_node.distance))) {
            spdlog::error("Fail to calc min and max distance.");
            return;
        }

        // 清洗world_points
        for (auto idx: invalid_point_idx) {
            structure_->adjust_points_.erase(idx);
        }

        const auto q1_idx = data.size()/4, q3_idx = data.size() - (data.size()+3)/4;
        const auto q1 = data[q1_idx].distance, q3 = data[q3_idx].distance, irq = q3 - q1;

        for (auto node: data) {
            if (node.distance > q3 + 1.5 * irq) {
                structure_->adjust_points_.erase(node.idx);
                continue;
            }
            if (node.distance > q3) {
                structure_->fixed_points_[node.idx] = structure_->adjust_points_[node.idx];
                structure_->adjust_points_.erase(node.idx);
            }
        }

        // 重新加载数据
        loadDataFromWorld();
    }

    bool bundleAdjustment() {
        ceres::Problem problem;
        ceres::LossFunction *lossFunction = new ceres::HuberLoss(4);

        for (auto &[world_point, point_pos]: world_points_pos_) {
            for (auto &[image, observed_pos]: world_point->observed_frames_) {
                ceres::CostFunction *costFunction = SnavelyReprojectionError::Create(observed_pos.x, observed_pos.y);
                problem.AddResidualBlock(
                        costFunction,
                        lossFunction,
                        image_extrinsic_[image].val,
                        camera_intrinsics_[image->getCamera()].val,               // Intrinsic
                        point_pos.val                               // Point in 3D space
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
                  << "    #points: " << world_points_pos_.size() << "\n"
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
        //写cameraIntrinsics
        for (auto &cameraIntrinsic: camera_intrinsics_) {
            cameraIntrinsic.first->setIntrinsic(cameraIntrinsic.second);
        }

        //写cameraIntrinsics
        for (auto &cameraIntrinsic: image_extrinsic_) {
            cameraIntrinsic.first->setIntrinsic(cameraIntrinsic.second);
        }
    }

    void clear() {
        structure_ = nullptr;

        image_extrinsic_.clear();
        camera_intrinsics_.clear();
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
