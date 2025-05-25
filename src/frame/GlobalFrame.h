//
// Created by rainbowx on 25-4-15.
//

#ifndef RECONSTRUCTION_GLOBAL_FRAME_H
#define RECONSTRUCTION_GLOBAL_FRAME_H

#include "../world/WorldStructure.h"

class GlobalFrame {
    std::vector<cv::DMatch> matches_;
    std::shared_ptr<Image> image_ptr_;
    std::vector<WorldPoint::Ptr> world_points_;
public:
    [[nodiscard]] GlobalFrame(std::shared_ptr<WorldStructure> world, const std::shared_ptr<Image> &image_ptr)
        : image_ptr_(image_ptr) {
        for (auto [idx, point]: world->world_points_) {
            world_points_.emplace_back(point);
        }
    }

    auto matchFeature(cv::Ptr<cv::DescriptorMatcher>& matcher) {
        if (not matches_.empty()) {
            spdlog::warn("Rematch feature for already has match vector.");
            matches_.clear();
        }

        cv::Mat descriptors, img_descriptors;
        std::vector< std::vector<cv::DMatch> > vmatches;
        for (auto point: world_points_) {
            descriptors.push_back(point->last_descriptor_);
        }
        img_descriptors = image_ptr_->getDescriptors();
        matcher->knnMatch(descriptors, img_descriptors, vmatches, 1);
        for (auto & vmatche : vmatches) {
            if (vmatche.empty()) {
                continue;
            }
            matches_.push_back(vmatche[0]);
        }

        return matches_.size();
    }

    auto filterMatches() {
        auto minDisIter = std::ranges::min_element(matches_,
           [](const cv::DMatch &m1, const cv::DMatch &m2) {
               return m1.distance < m2.distance;
           });

        auto minDis = minDisIter->distance;

        std::vector<cv::DMatch> goodMatches;
        for (auto match: matches_) {
            if (match.distance > 3 * minDis) continue;
            // if (not adjust_points_.contains(world_points_[match.queryIdx]->idx_)) continue;
            goodMatches.push_back(match);
        }
        return goodMatches;
    }

    auto matchFeatureAndFilter(cv::Ptr<cv::DescriptorMatcher>& matcher) {
        [[maybe_unused]]auto match_cnt = matchFeature(matcher);
        matches_ =std::move(filterMatches());
        return matches_.size();
    }

    [[nodiscard]] auto get_world_points() const -> const std::vector<WorldPoint::Ptr>& {
        return world_points_;
    }

    [[nodiscard]] auto getMatches() const -> const std::vector<cv::DMatch>& {
        return matches_;
    }
};



#endif // RECONSTRUCTION_GLOBAL_FRAME_H
