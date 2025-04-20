//
// Created by rainbowx on 25-4-14.
//

#ifndef RECONSTRUCTION_LOCAL_FRAME_H
#define RECONSTRUCTION_LOCAL_FRAME_H

#include <memory>

#include <Eigen/Eigen>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <spdlog/spdlog.h>

#include "../component/Image.h"

class LocalFrame {
public:
    using Ptr = std::shared_ptr<LocalFrame>;
private:
    std::vector<cv::DMatch> matches_;
    Image::Ptr image1_ptr_, image2_ptr_;
public:
    [[nodiscard]] LocalFrame(const std::shared_ptr<Image> &image1_ptr, const std::shared_ptr<Image> &image2_ptr)
        : image1_ptr_(image1_ptr),
          image2_ptr_(image2_ptr) {
    }

    auto matchFeature(cv::Ptr<cv::DescriptorMatcher>& matcher) {
        if (not matches_.empty()) {
            spdlog::warn("Rematch feature for already has match vector.");
            matches_.clear();
        }

        std::vector< std::vector<cv::DMatch> > vmatches;
        matcher->knnMatch(image1_ptr_->getDescriptors(), image2_ptr_->getDescriptors(), vmatches, 1);
        for (auto & vmatche : vmatches) {
            if (vmatche.empty()) {
                continue;
            }
            matches_.push_back(vmatche[0]);
        }

        return matches_.size();
    }

    auto filterMatches() {
        auto minDisIter = std::min_element(
        matches_.begin(), matches_.end(),
        [](const cv::DMatch &m1, const cv::DMatch &m2) {
            return m1.distance < m2.distance;
        });

        auto minDis = minDisIter->distance;

        std::vector<cv::DMatch> goodMatches;
        for (auto match: matches_) {
            if (match.distance <= 4 * minDis)
                goodMatches.push_back(match);
        }
        return goodMatches;
    }

    auto matchFeatureAndFilter(cv::Ptr<cv::DescriptorMatcher>& matcher) {
        [[maybe_unused]]auto match_cnt = matchFeature(matcher);
        matches_ =std::move(filterMatches());
        return matches_.size();
    }

    auto getImage1() {
        return image1_ptr_;
    }

    auto getImage2() const {
        return image2_ptr_;
    }

    [[nodiscard]] auto getMatches() const -> const std::vector<cv::DMatch> {
        return matches_;
    }
};

#endif // RECONSTRUCTION_LOCAL_FRAME_H
