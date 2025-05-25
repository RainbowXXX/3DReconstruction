//
// Created by rainbowx on 25-5-9.
//

#ifndef REGIONSPROVIDER_H
#define REGIONSPROVIDER_H

#include <atomic>
#include <memory>
#include <string>

#include "../../3rdparty/openMVG/src/openMVG/sfm/pipelines/sfm_regions_provider.hpp"
#include "openMVG/features/image_describer.hpp"
#include "openMVG/features/regions_factory.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/types.hpp"
#include "system/progressinterface.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

struct RegionsProvider: openMVG::sfm::Regions_Provider
{
  bool load(
    SfM_Data& sfm_data,
    std::unique_ptr<features::Regions> region_type,
    const std::vector<std::shared_ptr<openMVG::features::Regions>>& regions
  ){
    region_type_ = std::move(region_type);
    for (const auto& [i, view]: sfm_data.GetViews()) {
      cache_[view->id_view] = regions[i];
    }
    return true;
  }
};

#endif //REGIONSPROVIDER_H
