//
// Created by rainbowx on 25-4-14.
//

#ifndef RECONSTRUCTION_SEQUENTIAL_ACTUATOR_H
#define RECONSTRUCTION_SEQUENTIAL_ACTUATOR_H

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>

#include "../frame/LocalFrame.h"
#include "../frame/GlobalFrame.h"

#include "../adjuster/BundleAdjuster.h"

class SequentialActuator {
    Sophus::SE3<double> cur_se3_;

    cv::Ptr<cv::Feature2D> detector_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;

    LocalFrame::Ptr cur_frame_;
    std::shared_ptr<WorldStructure> world_structure_;

    auto savePointCloudToWorld(const cv::Mat &inlierMask, const cv::Mat &points4D, const std::vector<cv::DMatch> &matches) {
        world_structure_->addLocalFrame(cur_frame_);
        for (int i = 0; i < points4D.cols; ++i) {
            // 如果是outlier，跳过
            if (!inlierMask.empty() && !inlierMask.at<uint8_t>(i, 0))
                continue;

            // 如果是之前加入过的点, 则跳过
            cv::Mat descriptor = cur_frame_->getImage2()->getDescriptors().row(matches[i].trainIdx);
            auto world_point_idx = cur_frame_->getImage1()->getWorldPointIdx(matches[i].queryIdx);
            if (world_point_idx != WorldPoint::InvalidIdx) {
                auto key_point = cur_frame_->getImage2()->getKeyPoints()[matches[i].trainIdx];
                auto world_point = world_structure_->getPointFromIdx(world_point_idx);
                auto& observed_frame = world_point->observed_frames_;

                world_point->last_descriptor_ = descriptor;
                observed_frame.emplace_back(cur_frame_->getImage2(), key_point.pt);
                cur_frame_->getImage2()->setWorldPoint(matches[i].trainIdx, world_point_idx);
                continue;
            }

            // 如果是新的点
            cv::Vec3b rgb;
            cv::Mat x = points4D.col(i);
            if (cur_frame_->getImage2()->type() == CV_8UC3) {
                rgb = cur_frame_->getImage2()->at<cv::Vec3b>(cur_frame_->getImage2()->getKeyPoints()[matches[i].trainIdx].pt);
                std::swap(rgb[0], rgb[2]);
            } else if (cur_frame_->getImage2()->type() == CV_8UC1) {
                cvtColor(cur_frame_->getImage2()->at<uint8_t>(cur_frame_->getImage2()->getKeyPoints()[matches[i].trainIdx].pt),
                         rgb,
                         cv::COLOR_GRAY2RGB);
            }

            if (x.type() == CV_32FC1) {
                x /= x.at<float>(3, 0); // 归一化
                world_point_idx = world_structure_->addPoint(Eigen::Vector3d(x.at<float>(0, 0),x.at<float>(1, 0),x.at<float>(2, 0)), rgb, descriptor);
            } else if (x.type() == CV_64FC1) {
                x /= x.at<double>(3, 0);
                world_point_idx = world_structure_->addPoint(Eigen::Vector3d(x.at<double>(0, 0),x.at<double>(1, 0),x.at<double>(2, 0)), rgb, descriptor);
            }
            auto world_point = world_structure_->getPointFromIdx(world_point_idx);
            world_point->observed_frames_.emplace_back(cur_frame_->getImage1(), cur_frame_->getImage1()->getKeyPoints()[matches[i].queryIdx].pt);
            world_point->observed_frames_.emplace_back(cur_frame_->getImage2(), cur_frame_->getImage2()->getKeyPoints()[matches[i].trainIdx].pt);

            cur_frame_->getImage1()->setWorldPoint(matches[i].queryIdx, world_point_idx);
            cur_frame_->getImage2()->setWorldPoint(matches[i].trainIdx, world_point_idx);
        }
    }

public:
    [[nodiscard]] SequentialActuator() {
        detector_ = cv::SIFT::create();
        matcher_ = cv::BFMatcher::create(cv::NORM_L2, true);
        world_structure_ = std::make_shared<WorldStructure>();
    }

    auto getWorld() {
        return world_structure_;
    }

    auto init(const Image& image1, const Image& image2) {
        std::shared_ptr<Image> image_ptr1 = std::make_shared<Image>(image1);
        std::shared_ptr<Image> image_ptr2 = std::make_shared<Image>(image2);

        world_structure_->addImage(image_ptr1);
        world_structure_->addImage(image_ptr2);

        cur_frame_.reset();
        cur_frame_ = std::make_shared<LocalFrame>(image_ptr1, image_ptr2);

        image_ptr1->detectAndCompute(detector_);
        image_ptr2->detectAndCompute(detector_);

        cur_frame_->matchFeatureAndFilter(matcher_);

        //解对极约束并三角化
        std::vector<cv::Point2f> matchPoints1, matchPoints2;
        for (auto match: cur_frame_->getMatches()) {
            matchPoints1.push_back(image_ptr1->getKeyPoints()[match.queryIdx].pt);
            matchPoints2.push_back(image_ptr2->getKeyPoints()[match.trainIdx].pt);
        }

        cv::Mat essentialMatrix, inlierMask;
        essentialMatrix = cv::findEssentialMat(matchPoints1, matchPoints2,
                                           image_ptr1->getCamera()->getKMatCV(),
                                           cv::RANSAC, 0.999, 1.0, inlierMask);

        //解frame2的R、t并计算se3,三角化
        cv::Mat R, t, points4D;
        recoverPose(essentialMatrix,
            matchPoints1, matchPoints2,
                    image_ptr2->getCamera()->getKMatCV(),
                    R, t, 100, inlierMask,
                    points4D);

        // 以第一章图片的拍摄点为原点，计算旋转和平移
        Eigen::Matrix3d eigenR2;
        cv::cv2eigen(R, eigenR2);
        cur_se3_ = Sophus::SE3(eigenR2,
            Eigen::Vector3d(
                t.at<double>(0, 0),
                t.at<double>(1, 0),
                t.at<double>(2, 0)
            ));

        image_ptr1->setTcw({});
        image_ptr2->setTcw(cur_se3_);

        savePointCloudToWorld(inlierMask, points4D, cur_frame_->getMatches());

        cur_frame_.reset();
    }

    auto addSingleImage(const Image& next_image) {
        std::shared_ptr<Image> image_ptr1 = world_structure_->getLocalFrames().back()->getImage2();
        std::shared_ptr<Image> image_ptr2 = std::make_shared<Image>(next_image);

        world_structure_->addImage(image_ptr2);

        cur_frame_ = std::make_shared<LocalFrame>(image_ptr1, image_ptr2);

        image_ptr2->detectAndCompute(detector_);

        cur_frame_->matchFeatureAndFilter(matcher_);

        //解PnP得相机位姿
        std::vector<cv::Point2f> points2DPnP;
        std::vector<cv::Point3f> points3DPnP;
        GlobalFrame global_frame{world_structure_, image_ptr2};
        global_frame.matchFeatureAndFilter(matcher_);
        for (const auto match: global_frame.getMatches()) {
            auto point3D = global_frame.get_world_points()[match.queryIdx]->world_pos_;
            points2DPnP.push_back(cur_frame_->getImage2()->getKeyPoints()[match.trainIdx].pt);
            points3DPnP.emplace_back(point3D[0], point3D[1], point3D[2]);
        }

        cv::Mat r, t, indexInliers;
        solvePnPRansac(points3DPnP, points2DPnP, cur_frame_->getImage2()->getCamera()->getKMatCV(),
                       cv::noArray(), r, t, false, 100, 8.0, 0.99,
                       indexInliers);

        cv::Mat R;
        Eigen::Matrix3d R_eigen;
        cv::Rodrigues(r, R);
        cv::cv2eigen(R, R_eigen);
        cur_se3_ = Sophus::SE3<double>(
            Sophus::SO3<double>(R_eigen),
            Eigen::Vector3d(
                t.at<double>(0, 0),
                t.at<double>(1, 0),
                t.at<double>(2, 0)
            ));

        image_ptr2->setTcw(cur_se3_);

        if (indexInliers.rows < 30) {
            spdlog::warn("current frame has bad matched points, dropping.");
            return;
        }

        //通过findEssentialMat筛选异常点
        //解对极约束并三角化
        std::vector<cv::Point2f> matchPoints1, matchPoints2;
        for (auto match: cur_frame_->getMatches()) {
            matchPoints1.push_back(image_ptr1->getKeyPoints()[match.queryIdx].pt);
            matchPoints2.push_back(image_ptr2->getKeyPoints()[match.trainIdx].pt);
        }

        cv::Mat inlierMask;
        cv::findEssentialMat(matchPoints1, matchPoints2,
                                           image_ptr1->getCamera()->getKMatCV(),
                                           cv::RANSAC, 0.999, 1.0, inlierMask);

        //三角化
        std::vector<cv::Point2f> matchPointsNorm1, matchPointsNorm2;
        matchPointsNorm1.reserve(cur_frame_->getMatches().size());
        matchPointsNorm2.reserve(cur_frame_->getMatches().size());
        for (auto &match: cur_frame_->getMatches()) {
            matchPointsNorm1.push_back(image_ptr1->getCamera()->pixel2normal(image_ptr1->getKeyPoints()[match.queryIdx].pt));
            matchPointsNorm2.push_back(image_ptr2->getCamera()->pixel2normal(image_ptr2->getKeyPoints()[match.trainIdx].pt));
        }
        cv::Mat points4D;
        triangulatePoints(image_ptr1->getTcw34MatCV(CV_32F), image_ptr2->getTcw34MatCV(CV_32F),
                          matchPointsNorm1, matchPointsNorm2, points4D);

        savePointCloudToWorld(inlierMask, points4D, cur_frame_->getMatches());
    };

    auto bundleAdjustment() {
        BundleAdjuster adjuster{};
        adjuster(world_structure_);
    };

    auto show(bool debug = false) {
        world_structure_->show(debug);
    }
};

#endif // RECONSTRUCTION_SEQUENTIAL_ACTUATOR_H
