//
// Created by rainbowx on 25-5-10.
//

#ifndef MATCHESPROVIDER_H
#define MATCHESPROVIDER_H

#include <string>
#include <utility>

#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching/indMatch_utils.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/types.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include "openMVG/sfm/pipelines/sfm_matches_provider.hpp"

using namespace openMVG;
using namespace openMVG::sfm;

struct MatchesProvider: Matches_Provider {
    void load(matching::PairWiseMatches map_geometricMatches) {
        pairWise_matches_ = std::move(map_geometricMatches);
    }
};

#endif //MATCHESPROVIDER_H
