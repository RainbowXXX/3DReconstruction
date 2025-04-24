//
// Created by rainbowx on 25-4-14.
//

#ifndef RECONSTRUCTION_WORLD_STRUCTURE_H
#define RECONSTRUCTION_WORLD_STRUCTURE_H

#include <ranges>
#include <unordered_map>

#include <opencv2/core/types.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include "../component/Image.h"
#include "../frame/LocalFrame.h"

#include "WorldPoint.h"
#include "../common/common.h"

class WorldStructure {
public:
    using Ptr = std::shared_ptr<WorldStructure>;
private:
    WorldPoint::Idx cur_idx_ = 0;

    std::vector<LocalFrame::Ptr> local_frames_;
    std::unordered_map<Image::Idx, Image::Ptr> images_{};
    std::unordered_map<WorldPoint::Idx, WorldPoint::Ptr> world_points_{};
    std::unordered_map<WorldPoint::Idx, WorldPoint::Ptr> adjust_points_{}, fixed_points_{};

    friend class GlobalFrame;
    friend class BundleAdjuster;
public:
    WorldStructure() = default;

    auto addImage(Image::Ptr& image) {
        images_[image->getIdx()] = image;
    }

    // 添加点到结构中
    auto addPoint(Eigen::Vector3d world_pos, cv::Vec3b color, cv::Mat descriptor) -> std::size_t {
        auto point_idx = cur_idx_++;
        auto point_ref = std::make_shared<WorldPoint>(point_idx, world_pos, color, descriptor);
        world_points_[point_idx] = point_ref;
        adjust_points_[point_idx] = point_ref;
        return point_idx;
    }

    auto addLocalFrame(LocalFrame::Ptr local_frame) {
        local_frames_.emplace_back(std::move(local_frame));
    }

    auto getLocalFrames() -> const std::vector<LocalFrame::Ptr>& {
        return local_frames_;
    }

    auto getPointFromIdx(const WorldPoint::Idx idx) const {
        return world_points_.at(idx);
    }

    auto writeToPCDFile(const std::string& file_path) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->is_dense = false;
        for (const auto& [idx, point]: world_points_) {
            pcl::PointXYZRGB pointXYZ{point->point_color_[0], point->point_color_[1], point->point_color_[2]};

            pointXYZ.x = static_cast<float>(point->world_pos_(0));
            pointXYZ.y = static_cast<float>(point->world_pos_(1));
            pointXYZ.z = static_cast<float>(point->world_pos_(2));

            ensure(isNormal(pointXYZ.x) and isNormal(pointXYZ.y) and isNormal(pointXYZ.z));

            cloud->push_back(pointXYZ);
        }
        pcl::io::savePCDFileASCII (file_path, *cloud);
    }

    auto writeToPLYFile(const std::string& file_path) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->is_dense = false;
        for (const auto& [idx, point]: world_points_) {
            pcl::PointXYZRGB pointXYZ{point->point_color_[0], point->point_color_[1], point->point_color_[2]};

            pointXYZ.x = static_cast<float>(point->world_pos_(0));
            pointXYZ.y = static_cast<float>(point->world_pos_(1));
            pointXYZ.z = static_cast<float>(point->world_pos_(2));

            ensure(isNormal(pointXYZ.x) and isNormal(pointXYZ.y) and isNormal(pointXYZ.z));

            cloud->push_back(pointXYZ);
        }
        pcl::io::savePLYFileASCII(file_path, *cloud);
    }

    auto getWorldPoints() const -> const std::unordered_map<WorldPoint::Idx, WorldPoint::Ptr>& {
        return world_points_;
    }

    auto getImages() const -> const std::unordered_map<Image::Idx, Image::Ptr>& {
        return images_;
    }

    void show(bool debug = false) {
        static std::unordered_map<WorldPoint::Idx, WorldPoint::Ptr> last_points_{};

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->is_dense = false;

        for (const auto& [idx, point]: world_points_) {
            // if ((idx&0x1) == 0) continue;
            pcl::PointXYZRGB pointXYZ;
            if (last_points_.contains(idx) or (not debug)) {
                pointXYZ = pcl::PointXYZRGB(point->point_color_[0], point->point_color_[1], point->point_color_[2]);
            } else {
                cv::Vec3b tp = {255,0,0};
                pointXYZ = pcl::PointXYZRGB(tp[0], tp[1], tp[2]);
            }
            pointXYZ.x = static_cast<float>(point->world_pos_(0));
            pointXYZ.y = static_cast<float>(point->world_pos_(1));
            pointXYZ.z = static_cast<float>(point->world_pos_(2));

            ensure(isNormal(pointXYZ.x) and isNormal(pointXYZ.y) and isNormal(pointXYZ.z));

            cloud->push_back(pointXYZ);
        }

        pcl::visualization::PCLVisualizer viewer("Viewer");
        viewer.setBackgroundColor(50, 50, 50);
        viewer.addPointCloud(cloud, "Triangulated Point Cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                3,
                                                "Triangulated Point Cloud");
        viewer.addCoordinateSystem(1.0);

        int indexFrame = 0;
        for (const auto& frame: local_frames_) {
            auto tcw = frame->getImage2()->getTcw();
            Eigen::Matrix4f camPose;
            auto T_c_w = tcw.inverse().matrix();
            for (int i = 0; i < camPose.rows(); ++i)
                for (int j = 0; j < camPose.cols(); ++j)
                    camPose(i, j) = static_cast<float>(T_c_w(i, j));
            viewer.addCoordinateSystem(1.0, Eigen::Affine3f(camPose), "cam" + std::to_string(indexFrame++));
        }
        viewer.initCameraParameters ();
        while (!viewer.wasStopped ()) {
            viewer.spin();
        }
        last_points_ = std::unordered_map{world_points_};
    }
};

#endif // RECONSTRUCTION_WORLD_STRUCTURE_H
