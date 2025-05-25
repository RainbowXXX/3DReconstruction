//
// Created by rainbowx on 25-5-10.
//

#ifndef FEATURESPROVIDER_H
#define FEATURESPROVIDER_H

#include <memory>
#include <string>

#include "openMVG/features/feature_container.hpp"
#include "openMVG/features/regions.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/types.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"

using namespace openMVG;
using namespace openMVG::sfm;

struct FeaturesProvider: Features_Provider {
    void load(const SfM_Data &sfm_data, std::vector<std::shared_ptr<features::Regions>>& regions) {
        const auto view_cnt = sfm_data.GetViews().size();
        for (auto i = 0;i<view_cnt;i++) {
            const auto& view = sfm_data.GetViews().at(i);
            feats_per_view[view->id_view] = regions[i]->GetRegionsPositions();
        }
    }
};

#endif //FEATURESPROVIDER_H
