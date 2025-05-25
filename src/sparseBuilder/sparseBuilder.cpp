//
// Created by rainbowx on 25-5-9.
//

#include "sparseBuilder.h"

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <locale>
#include <filesystem>

#include <cereal/archives/json.hpp>
#include <cereal/details/helpers.hpp>

#include "openMVG/cameras/cameras.hpp"
#include "openMVG/exif/exif_IO_EasyExif.hpp"
#include "openMVG/exif/sensor_width_database/ParseDatabase.hpp"
#include "openMVG/geodesy/geodesy.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_data_utils.hpp"
#include "openMVG/sfm/sfm_view.hpp"
#include "openMVG/sfm/sfm_view_priors.hpp"
#include "openMVG/types.hpp"
#include "openMVG/features/akaze/image_describer_akaze_io.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer_io.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/system/timer.hpp"
#include "openMVG/matching_image_collection/Pair_Builder.hpp"
#include "openMVG/graph/graph.hpp"
#include "openMVG/graph/graph_stats.hpp"
#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching/indMatch_utils.hpp"
#include "openMVG/matching/pairwiseAdjacencyDisplay.hpp"
#include "openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/Matcher_Regions.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_preemptive_regions_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider_cache.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/features/akaze/image_describer_akaze.hpp"
#include "openMVG/matching_image_collection/E_ACRobust.hpp"
#include "openMVG/matching_image_collection/E_ACRobust_Angular.hpp"
#include "openMVG/matching_image_collection/Eo_Robust.hpp"
#include "openMVG/matching_image_collection/F_ACRobust.hpp"
#include "openMVG/matching_image_collection/GeometricFilter.hpp"
#include "openMVG/matching_image_collection/H_ACRobust.hpp"
#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Cameras_Common_command_line_helper.hpp"
#include "openMVG/sfm/pipelines/sfm_matches_provider.hpp"
#include "openMVG/sfm/sfm_report.hpp"
#include "openMVG/sfm/pipelines/global/GlobalSfM_rotation_averaging.hpp"
#include "openMVG/sfm/pipelines/global/GlobalSfM_translation_averaging.hpp"
#include "openMVG/sfm/pipelines/global/sfm_global_engine_relative_motions.hpp"
#include "openMVG/sfm/pipelines/sequential/sequential_SfM.hpp"
#include "openMVG/sfm/pipelines/sequential/sequential_SfM2.hpp"
#include "openMVG/sfm/pipelines/sequential/SfmSceneInitializerMaxPair.hpp"
#include "openMVG/sfm/pipelines/sequential/SfmSceneInitializerStellar.hpp"
#include "openMVG/sfm/pipelines/stellar/sfm_stellar_engine.hpp"
#include "openMVG/sfm/sfm_data_colorization.hpp"

#include "nonFree/sift/SIFT_describer_io.hpp"

#include "../helpers/SfMPlyHelper.hpp"
#include "common/common.h"
#include "spdlog/spdlog.h"

using namespace openMVG;
using namespace openMVG::sfm;
using namespace openMVG::exif;
using namespace openMVG::image;
using namespace openMVG::robust;
using namespace openMVG::cameras;
using namespace openMVG::geodesy;
using namespace openMVG::matching;
using namespace openMVG::features;
using namespace openMVG::matching_image_collection;

namespace fs = std::filesystem;

bool checkIntrinsicStringValidity(const std::string &Kmatrix, double &focal, double &ppx, double &ppy) {
    std::vector<std::string> vec_str;
    stl::split(Kmatrix, ';', vec_str);
    if (vec_str.size() != 9) {
        spdlog::error("k矩阵格式不正确: 缺少`;`或缺少数据");
        return false;
    }
    // Check that all K matrix value are valid numbers
    for (size_t i = 0; i < vec_str.size(); ++i) {
        double readvalue = 0.0;
        std::stringstream ss;
        ss.str(vec_str[i]);
        if (!(ss >> readvalue)) {
            spdlog::error("k矩阵格式不正确: 矩阵中含有非法字符");
            return false;
        }
        if (i == 0) focal = readvalue;
        if (i == 2) ppx = readvalue;
        if (i == 5) ppy = readvalue;
    }
    return true;
}

bool getGPS(const std::string &filename, const int &GPS_to_XYZ_method, Vec3 &pose_center) {
    std::unique_ptr<Exif_IO> exifReader(new Exif_IO_EasyExif);
    if (exifReader) {
        // Try to parse EXIF metada & check existence of EXIF data
        if (exifReader->open(filename) && exifReader->doesHaveExifInfo()) {
            // Check existence of GPS coordinates
            double latitude, longitude, altitude;
            if (exifReader->GPSLatitude(&latitude) &&
                exifReader->GPSLongitude(&longitude) &&
                exifReader->GPSAltitude(&altitude)) {
                // Add ECEF or UTM XYZ position to the GPS position array
                switch (GPS_to_XYZ_method) {
                    case 1:
                        pose_center = lla_to_utm(latitude, longitude, altitude);
                        break;
                    case 0:
                    default:
                        pose_center = lla_to_ecef(latitude, longitude, altitude);
                        break;
                }
                return true;
            }
        }
    }
    return false;
}

features::EDESCRIBER_PRESET stringToEnum(const std::string &sPreset) {
    features::EDESCRIBER_PRESET preset;
    if (sPreset == "NORMAL")
        preset = features::NORMAL_PRESET;
    else if (sPreset == "HIGH")
        preset = features::HIGH_PRESET;
    else if (sPreset == "ULTRA")
        preset = features::ULTRA_PRESET;
    else
        preset = features::EDESCRIBER_PRESET(-1);
    return preset;
}

std::pair<bool, Vec3> checkPriorWeightsString(const std::string &sWeights) {
    std::pair<bool, Vec3> val(true, Vec3::Zero());
    std::vector<std::string> vec_str;
    stl::split(sWeights, ';', vec_str);
    if (vec_str.size() != 3) {
        spdlog::error("Missing ';' character in prior weights");
        val.first = false;
    }
    // Check that all weight values are valid numbers
    for (size_t i = 0; i < vec_str.size(); ++i) {
        double readvalue = 0.0;
        std::stringstream ss;
        ss.str(vec_str[i]);
        if (!(ss >> readvalue)) {
            spdlog::error("Used an invalid not a number character in local frame origin");
            val.first = false;
        }
        val.second[static_cast<long>(i)] = readvalue;
    }
    return val;
}

enum EPairMode {
    PAIR_EXHAUSTIVE = 0, // Build every combination of image pairs
    PAIR_CONTIGUOUS = 1 // Only consecutive image pairs (useful for video mode)
};

enum EGeometricModel {
    FUNDAMENTAL_MATRIX = 0,
    ESSENTIAL_MATRIX = 1,
    HOMOGRAPHY_MATRIX = 2,
    ESSENTIAL_MATRIX_ANGULAR = 3,
    ESSENTIAL_MATRIX_ORTHO = 4,
    ESSENTIAL_MATRIX_UPRIGHT = 5
};

enum class ESfMSceneInitializer {
    INITIALIZE_EXISTING_POSES,
    INITIALIZE_MAX_PAIR,
    INITIALIZE_AUTO_PAIR,
    INITIALIZE_STELLAR
};

enum class ESfMEngine {
    INCREMENTAL,
    INCREMENTALV2,
    GLOBAL,
    STELLAR
};

bool StringToEnum
(
    const std::string &str,
    ESfMEngine &sfm_engine
) {
    const std::map<std::string, ESfMEngine> string_to_enum_mapping =
    {
        {"INCREMENTAL", ESfMEngine::INCREMENTAL},
        {"INCREMENTALV2", ESfMEngine::INCREMENTALV2},
        {"GLOBAL", ESfMEngine::GLOBAL},
        {"STELLAR", ESfMEngine::STELLAR},
    };
    const auto it = string_to_enum_mapping.find(str);
    if (it == string_to_enum_mapping.end())
        return false;
    sfm_engine = it->second;
    return true;
}

bool StringToEnum
(
    const std::string &str,
    ESfMSceneInitializer &scene_initializer
) {
    const std::map<std::string, ESfMSceneInitializer> string_to_enum_mapping =
    {
        {"EXISTING_POSE", ESfMSceneInitializer::INITIALIZE_EXISTING_POSES},
        {"MAX_PAIR", ESfMSceneInitializer::INITIALIZE_MAX_PAIR},
        {"AUTO_PAIR", ESfMSceneInitializer::INITIALIZE_AUTO_PAIR},
        {"STELLAR", ESfMSceneInitializer::INITIALIZE_STELLAR},
    };
    const auto it = string_to_enum_mapping.find(str);
    if (it == string_to_enum_mapping.end())
        return false;
    scene_initializer = it->second;
    return true;
}

bool StringToEnum_EGraphSimplification
(
    const std::string &str,
    EGraphSimplification &graph_simplification
) {
    const std::map<std::string, EGraphSimplification> string_to_enum_mapping =
    {
        {"NONE", EGraphSimplification::NONE},
        {"MST_X", EGraphSimplification::MST_X},
        {"STAR_X", EGraphSimplification::STAR_X},
    };
    auto it = string_to_enum_mapping.find(str);
    if (it == string_to_enum_mapping.end())
        return false;
    graph_simplification = it->second;
    return true;
}

bool computeIndexFromImageNames(
    const SfM_Data &sfm_data,
    const std::pair<std::string, std::string> &initialPairName,
    Pair &initialPairIndex) {
    if (initialPairName.first == initialPairName.second) {
        spdlog::error("Invalid image names. You cannot use the same image to initialize a pair.");
        return false;
    }

    initialPairIndex = {UndefinedIndexT, UndefinedIndexT};

    /// List views filenames and find the one that correspond to the user ones:
    for (auto it = sfm_data.GetViews().cbegin(); it != sfm_data.GetViews().cend(); ++it) {
        const View *v = it->second.get();
        const std::string filename = fs::path(v->s_Img_path).filename();
        if (filename == initialPairName.first) {
            initialPairIndex.first = v->id_view;
        } else {
            if (filename == initialPairName.second) {
                initialPairIndex.second = v->id_view;
            }
        }
    }
    return (initialPairIndex.first != UndefinedIndexT &&
            initialPairIndex.second != UndefinedIndexT);
}

void GetCameraPositions(const SfM_Data &sfm_data, std::vector<Vec3> &vec_camPosition) {
    for (const auto &view: sfm_data.GetViews()) {
        if (sfm_data.IsPoseAndIntrinsicDefined(view.second.get())) {
            const geometry::Pose3 pose = sfm_data.GetPoseOrDie(view.second.get());
            vec_camPosition.push_back(pose.center());
        }
    }
}

std::vector<std::string> list_files(const fs::path& dir_path) {
    if (!fs::exists(dir_path)) {
        throw std::runtime_error("目录不存在: " + dir_path.string());
    }
    if (!fs::is_directory(dir_path)) {
        throw std::runtime_error("路径不是目录: " + dir_path.string());
    }

    std::vector<std::string> file_names;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (fs::is_regular_file(entry.status())) {
            file_names.push_back(entry.path().filename().string());
        }
    }

    return file_names;
}

namespace sparse {
    void sparseBuilder::readImagesCluster(double f) const {
        std::string sImageDir = input_dir, sfileDatabase = camera_file_params, sOutputDir = matches_dir, sKmatrix;

        std::string sPriorWeights = "1.0;1.0;1.0";
        std::pair prior_w_info(false, Vec3{});

        int i_User_camera_model = PINHOLE_CAMERA_RADIAL3;

        bool b_Group_camera_model = true;

        int i_GPS_XYZ_method = 0;

        double focal_pixels = -1.0;

        const bool b_Use_pose_prior = false;

        // Expected properties for each image
        double width = -1, height = -1, focal = f, ppx = -1, ppy = -1;

        const auto e_User_camera_model = static_cast<EINTRINSIC>(i_User_camera_model);

        if (!std::filesystem::is_directory(sImageDir)) {
            OPENMVG_LOG_ERROR << "The input directory doesn't exist";
            return;
        }

        if (sOutputDir.empty()) {
            OPENMVG_LOG_ERROR << "Invalid output directory";
            return;
        }

        if (!std::filesystem::is_directory(sOutputDir)) {
            if (!std::filesystem::create_directories(sOutputDir)) {
                OPENMVG_LOG_ERROR << "Cannot create output directory";
                return;
            }
        }

        if (sKmatrix.size() > 0 &&
            !checkIntrinsicStringValidity(sKmatrix, focal, ppx, ppy)) {
            OPENMVG_LOG_ERROR << "Invalid K matrix input";
            return;
        }

        if (sKmatrix.size() > 0 && focal_pixels != -1.0) {
            OPENMVG_LOG_ERROR << "Cannot combine -f and -k options";
            return;
        }

        std::vector<Datasheet> vec_database;
        if (!sfileDatabase.empty()) {
            if (!parseDatabase(sfileDatabase, vec_database)) {
                OPENMVG_LOG_ERROR
       << "Invalid input database: " << sfileDatabase
       << ", please specify a valid file.";
                return;
            }
        }

        // Check if prior weights are given
        if (b_Use_pose_prior) {
            prior_w_info = checkPriorWeightsString(sPriorWeights);
        }

        std::vector<std::string> vec_image = list_files(sImageDir);
        std::sort(vec_image.begin(), vec_image.end());

        // Configure an empty scene with Views and their corresponding cameras
        SfM_Data sfm_data;
        sfm_data.s_root_path = sImageDir; // Setup main image root_path
        Views &views = sfm_data.views;
        Intrinsics &intrinsics = sfm_data.intrinsics;

        std::ostringstream error_report_stream;
        std::size_t image_cnt = vec_image.size(), cur_cnt = 1;
        for (std::vector<std::string>::const_iterator iter_image = vec_image.begin(); iter_image != vec_image.end(); ++iter_image, ++cur_cnt) {
            // Read meta data to fill camera parameter (w,h,focal,ppx,ppy) fields.
            width = height = ppx = ppy = focal = -1.0;

            const std::string sImageFilename = fs::path(sImageDir) / *iter_image;
            const std::string sImFilenamePart = fs::path(sImageFilename).filename();

            // Test if the image format is supported:
            if (openMVG::image::GetFormat(sImageFilename.c_str()) == openMVG::image::Unknown) {
                error_report_stream
                        << sImFilenamePart << ": Unkown image file format." << "\n";
                continue; // image cannot be opened
            }

            if (sImFilenamePart.find("mask.png") != std::string::npos
                || sImFilenamePart.find("_mask.png") != std::string::npos) {
                error_report_stream
                        << sImFilenamePart << " is a mask image" << "\n";
                continue;
            }

            ImageHeader imgHeader;
            if (!openMVG::image::ReadImageHeader(sImageFilename.c_str(), &imgHeader))
                continue; // image cannot be read

            width = imgHeader.width;
            height = imgHeader.height;
            ppx = width / 2.0;
            ppy = height / 2.0;


            // Consider the case where the focal is provided manually
            if (sKmatrix.size() > 0) // Known user calibration K matrix
            {
                if (!checkIntrinsicStringValidity(sKmatrix, focal, ppx, ppy))
                    focal = -1.0;
            } else // User provided focal length value
                if (focal_pixels != -1)
                    focal = focal_pixels;

            // If not manually provided or wrongly provided
            if (focal == -1) {
                std::unique_ptr<Exif_IO> exifReader(new Exif_IO_EasyExif);
                exifReader->open(sImageFilename);

                const bool bHaveValidExifMetadata =
                        exifReader->doesHaveExifInfo()
                        && !exifReader->getModel().empty()
                        && !exifReader->getBrand().empty();

                if (bHaveValidExifMetadata) // If image contains meta data
                {
                    // Handle case where focal length is equal to 0
                    if (exifReader->getFocal() == 0.0f) {
                        error_report_stream
                                << fs::path(sImageFilename).filename() << ": Focal length is missing." << "\n";
                        focal = -1.0;
                    } else
                    // Create the image entry in the list file
                    {
                        const std::string sCamModel = exifReader->getBrand() + " " + exifReader->getModel();

                        Datasheet datasheet;
                        if (getInfo(sCamModel, vec_database, datasheet)) {
                            // The camera model was found in the database so we can compute it's approximated focal length
                            const double ccdw = datasheet.sensorSize_;
                            focal = std::max(width, height) * exifReader->getFocal() / ccdw;
                        } else {
                            error_report_stream
                                    << fs::path(sImageFilename).filename()
                                    << "\" model \"" << sCamModel << "\" doesn't exist in the database" << "\n"
                                    << "Please consider add your camera model and sensor width in the database." <<
                                    "\n";
                        }
                    }
                }
            }
            // Build intrinsic parameter related to the view
            std::shared_ptr<IntrinsicBase> intrinsic;

            if (focal > 0 && ppx > 0 && ppy > 0 && width > 0 && height > 0) {
                // Create the desired camera type
                switch (e_User_camera_model) {
                    case PINHOLE_CAMERA:
                        intrinsic = std::make_shared<Pinhole_Intrinsic>
                                (width, height, focal, ppx, ppy);
                        break;
                    case PINHOLE_CAMERA_RADIAL1:
                        intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K1>
                                (width, height, focal, ppx, ppy, 0.0); // setup no distortion as initial guess
                        break;
                    case PINHOLE_CAMERA_RADIAL3:
                        intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K3>
                                (width, height, focal, ppx, ppy, 0.0, 0.0, 0.0); // setup no distortion as initial guess
                        break;
                    case PINHOLE_CAMERA_BROWN:
                        intrinsic = std::make_shared<Pinhole_Intrinsic_Brown_T2>
                                (width, height, focal, ppx, ppy, 0.0, 0.0, 0.0, 0.0, 0.0);
                    // setup no distortion as initial guess
                        break;
                    case PINHOLE_CAMERA_FISHEYE:
                        intrinsic = std::make_shared<Pinhole_Intrinsic_Fisheye>
                                (width, height, focal, ppx, ppy, 0.0, 0.0, 0.0, 0.0);
                    // setup no distortion as initial guess
                        break;
                    case CAMERA_SPHERICAL:
                        intrinsic = std::make_shared<Intrinsic_Spherical>
                                (width, height);
                        break;
                    default:
                        OPENMVG_LOG_ERROR << "Error: unknown camera model: " << (int) e_User_camera_model;
                        return;
                }
            }

            // Build the view corresponding to the image
            Vec3 pose_center;
            if (getGPS(sImageFilename, i_GPS_XYZ_method, pose_center) && b_Use_pose_prior) {
                ViewPriors v(*iter_image, views.size(), views.size(), views.size(), width, height);

                // Add intrinsic related to the image (if any)
                if (!intrinsic) {
                    //Since the view have invalid intrinsic data
                    // (export the view, with an invalid intrinsic field value)
                    v.id_intrinsic = UndefinedIndexT;
                } else {
                    // Add the defined intrinsic to the sfm_container
                    intrinsics[v.id_intrinsic] = intrinsic;
                }

                v.b_use_pose_center_ = true;
                v.pose_center_ = pose_center;
                // prior weights
                if (prior_w_info.first == true) {
                    v.center_weight_ = prior_w_info.second;
                }

                // Add the view to the sfm_container
                views[v.id_view] = std::make_shared<ViewPriors>(v);
            } else {
                View v(*iter_image, views.size(), views.size(), views.size(), width, height);

                // Add intrinsic related to the image (if any)
                if (!intrinsic) {
                    //Since the view have invalid intrinsic data
                    // (export the view, with an invalid intrinsic field value)
                    v.id_intrinsic = UndefinedIndexT;
                } else {
                    // Add the defined intrinsic to the sfm_container
                    intrinsics[v.id_intrinsic] = intrinsic;
                }

                // Add the view to the sfm_container
                views[v.id_view] = std::make_shared<View>(v);
            }
        }

        // Display saved warning & error messages if any.
        if (!error_report_stream.str().empty()) {
            OPENMVG_LOG_WARNING
      << "Warning & Error messages:\n"
      << error_report_stream.str();
        }

        // Group camera that share common properties if desired (leads to more faster & stable BA).
        if (b_Group_camera_model) {
            GroupSharedIntrinsics(sfm_data);
        }

        // Store SfM_Data views & intrinsic data
        if (!Save(
            sfm_data,
            (fs::path(sOutputDir) / "sfm_data.json").c_str(),
            ESfM_Data(VIEWS | INTRINSICS))) {
            return;
        }

        OPENMVG_LOG_INFO
    << "SfMInit_ImageListing report:\n"
    << "listed #File(s): " << vec_image.size() << "\n"
    << "usable #File(s) listed in sfm_data: " << sfm_data.GetViews().size() << "\n"
    << "usable #Intrinsic(s) listed in sfm_data: " << sfm_data.GetIntrinsics().size();

        return;
    }

    void sparseBuilder::detectFeature() {
        std::string sSfM_Data_Filename = matches_dir + "/sfm_data.json";
        std::string sOutDir = matches_dir;
        bool bUpRight = false;
        std::string sImage_Describer_Method = "SIFT";
        bool bForce = false;
        std::string sFeaturePreset = "";

        if (sOutDir.empty()) {
            OPENMVG_LOG_ERROR << "\nIt is an invalid output directory";
            return;
        }

        // Create output dir
        if (!std::filesystem::is_directory(sOutDir)) {
            if (!std::filesystem::create_directories(sOutDir)) {
                OPENMVG_LOG_ERROR << "Cannot create output directory";
                return;
            }
        }

        //---------------------------------------
        // a. Load input scene
        //---------------------------------------
        SfM_Data sfm_data;
        if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS | INTRINSICS))) {
            OPENMVG_LOG_ERROR
      << "The input file \"" << sSfM_Data_Filename << "\" cannot be read";
            return;
        }

        // b. Init the image_describer
        // - retrieve the used one in case of pre-computed features
        // - else create the desired one

        using namespace openMVG::features;
        std::unique_ptr<Image_describer> image_describer;

        const std::string sImage_describer = fs::path(sOutDir) / "image_describer.json";
        if (!bForce && fs::is_regular_file(sImage_describer)) {
            // Dynamically load the image_describer from the file (will restore old used settings)
            std::ifstream stream(sImage_describer.c_str());
            if (!stream)
                return;

            try {
                cereal::JSONInputArchive archive(stream);
                archive(cereal::make_nvp("image_describer", image_describer));
            } catch (const cereal::Exception &e) {
                OPENMVG_LOG_ERROR << e.what() << '\n'
        << "Cannot dynamically allocate the Image_describer interface.";
                return;
            }
        } else {
            // Create the desired Image_describer method.
            // Don't use a factory, perform direct allocation
            if (sImage_Describer_Method == "SIFT") {
                image_describer.reset(new SIFT_Image_describer(SIFT_Image_describer::Params(), !bUpRight));
            } else if (sImage_Describer_Method == "SIFT_ANATOMY") {
                image_describer.reset(
                    new SIFT_Anatomy_Image_describer(SIFT_Anatomy_Image_describer::Params()));
            } else if (sImage_Describer_Method == "AKAZE_FLOAT") {
                image_describer = AKAZE_Image_describer::create
                        (AKAZE_Image_describer::Params(AKAZE::Params(), AKAZE_MSURF), !bUpRight);
            } else if (sImage_Describer_Method == "AKAZE_MLDB") {
                image_describer = AKAZE_Image_describer::create
                        (AKAZE_Image_describer::Params(AKAZE::Params(), AKAZE_MLDB), !bUpRight);
            }
            if (!image_describer) {
                OPENMVG_LOG_ERROR << "Cannot create the designed Image_describer:"
        << sImage_Describer_Method << ".";
                return;
            } else {
                if (!sFeaturePreset.empty())
                    if (!image_describer->Set_configuration_preset(stringToEnum(sFeaturePreset))) {
                        OPENMVG_LOG_ERROR << "Preset configuration failed.";
                        return;
                    }
            }

            // Export the used Image_describer and region type for:
            // - dynamic future regions computation and/or loading
            {
                std::ofstream stream(sImage_describer.c_str());
                if (!stream)
                    return;

                cereal::JSONOutputArchive archive(stream);
                archive(cereal::make_nvp("image_describer", image_describer));
                auto regionsType = image_describer->Allocate();
                archive(cereal::make_nvp("regions_type", regionsType));
            }
        }

        // Feature extraction routines
        // For each View of the SfM_Data container:
        // - if regions file exists continue,
        // - if no file, compute features
        {
            system::Timer timer;
            Image<unsigned char> imageGray;

            // Use a boolean to track if we must stop feature extraction
            std::atomic<bool> preemptive_exit(false);
#ifdef OPENMVG_USE_OPENMP
    const unsigned int nb_max_thread = omp_get_max_threads();

    if (iNumThreads > 0) {
        omp_set_num_threads(iNumThreads);
    } else {
        omp_set_num_threads(nb_max_thread);
    }

    #pragma omp parallel for schedule(dynamic) if (iNumThreads > 0) private(imageGray)
#endif
            for (int i = 0; i < static_cast<int>(sfm_data.views.size()); ++i) {
                Views::const_iterator iterViews = sfm_data.views.begin();
                std::advance(iterViews, i);
                const View *view = iterViews->second.get();
                const std::string
                        sView_filename = fs::path(sfm_data.s_root_path) / view->s_Img_path,
                        sFeat = fs::path(sOutDir)/ (fs::path(sView_filename).stem().string() + ".feat"),
                        sDesc = fs::path(sOutDir)/ (fs::path(sView_filename).stem().string() + ".desc");

                // If features or descriptors file are missing, compute them
                if (!preemptive_exit && (bForce || !fs::is_regular_file(sFeat) || !fs::is_regular_file(sDesc))) {
                    if (!ReadImage(sView_filename.c_str(), &imageGray))
                        continue;

                    //
                    // Look if there is an occlusion feature mask
                    //
                    Image<unsigned char> *mask = nullptr; // The mask is null by default

                    const std::string
                            mask_filename_local = fs::path(sfm_data.s_root_path) / (fs::path(sView_filename).stem().string() + "_mask" + ".png"),
                            mask_filename_global = fs::path(sfm_data.s_root_path) / "mask.png";

                    Image<unsigned char> imageMask;
                    // Try to read the local mask
                    if (fs::is_regular_file(mask_filename_local)) {
                        if (!ReadImage(mask_filename_local.c_str(), &imageMask)) {
                            OPENMVG_LOG_ERROR
              << "Invalid mask: " << mask_filename_local << ';'
              << "Stopping feature extraction.";
                            preemptive_exit = true;
                            continue;
                        }
                        // Use the local mask only if it fits the current image size
                        if (imageMask.Width() == imageGray.Width() && imageMask.Height() == imageGray.Height())
                            mask = &imageMask;
                    } else {
                        // Try to read the global mask
                        if (fs::is_regular_file(mask_filename_global)) {
                            if (!ReadImage(mask_filename_global.c_str(), &imageMask)) {
                                OPENMVG_LOG_ERROR
                << "Invalid mask: " << mask_filename_global << ';'
                << "Stopping feature extraction.";
                                preemptive_exit = true;
                                continue;
                            }
                            // Use the global mask only if it fits the current image size
                            if (imageMask.Width() == imageGray.Width() && imageMask.Height() == imageGray.Height())
                                mask = &imageMask;
                        }
                    }

                    // Compute features and descriptors and export them to files
                    auto regions = image_describer->Describe(imageGray, mask);
                    if (regions && !image_describer->Save(regions.get(), sFeat, sDesc)) {
                        OPENMVG_LOG_ERROR
            << "Cannot save regions for image: " << sView_filename << ';'
            << "Stopping feature extraction.";
                        preemptive_exit = true;
                        continue;
                    }
                }
            }
            OPENMVG_LOG_INFO << "Task done in (s): " << timer.elapsed();
        }
        return;
    }

    void sparseBuilder::matchPair() {
        std::string sSfMDataFilename = matches_dir + "/sfm_data.json";
        std::string sOutputPairsFilename = matches_dir + "/pairs.bin";
        std::string sPairMode = "EXHAUSTIVE";
        int iContiguousCount = -1;

        EPairMode pairMode;
        if (sPairMode == "EXHAUSTIVE") {
            pairMode = PAIR_EXHAUSTIVE;
        } else if (sPairMode == "CONTIGUOUS") {
            pairMode = PAIR_CONTIGUOUS;
        }

        // 1. Load SfM data scene
        std::cout << "Loading scene.";
        SfM_Data sfm_data;
        if (!Load(sfm_data, sSfMDataFilename, ESfM_Data(VIEWS | INTRINSICS))) {
            std::cerr << std::endl
                    << "The input SfM_Data file \"" << sSfMDataFilename << "\" cannot be read." << std::endl;
            exit(EXIT_FAILURE);
        }
        const size_t NImage = sfm_data.GetViews().size();

        // 2. Compute pairs
        std::cout << "Computing pairs." << std::endl;
        Pair_Set pairs;
        switch (pairMode) {
            case PAIR_EXHAUSTIVE: {
                pairs = exhaustivePairs(NImage);
                break;
            }
            case PAIR_CONTIGUOUS: {
                pairs = contiguousWithOverlap(NImage, iContiguousCount);
                break;
            }
            default: {
                std::cerr << "Unknown pair mode" << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // 3. Save pairs
        std::cout << "Saving pairs." << std::endl;
        if (!savePairs(sOutputPairsFilename, pairs)) {
            std::cerr << "Failed to save pairs to file: \"" << sOutputPairsFilename << "\"" << std::endl;
            exit(EXIT_FAILURE);
        }

        return;
    }

    void sparseBuilder::match() {
        std::string sSfM_Data_Filename = matches_dir + "/sfm_data.json";
        std::string sOutputMatchesFilename = matches_dir + "/matches.putative.bin";
        float fDistRatio = 0.8f;
        std::string sPredefinedPairList = matches_dir + "/pairs.bin";
        std::string sNearestMatchingMethod = "AUTO";
        bool bForce = false;
        unsigned int ui_max_cache_size = 0;

        // Pre-emptive matching parameters
        unsigned int ui_preemptive_feature_count = 200;
        double preemptive_matching_percentage_threshold = 0.08;

        bool use_p = false;

        if (sOutputMatchesFilename.empty()) {
            OPENMVG_LOG_ERROR << "No output file set.";
            return;
        }

        // -----------------------------
        // . Load SfM_Data Views & intrinsics data
        // . Compute putative descriptor matches
        // + Export some statistics
        // -----------------------------

        //---------------------------------------
        // Read SfM Scene (image view & intrinsics data)
        //---------------------------------------
        SfM_Data sfm_data;
        if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS | INTRINSICS))) {
            OPENMVG_LOG_ERROR << "The input SfM_Data file \"" << sSfM_Data_Filename << "\" cannot be read.";
            return;
        }
        const std::string sMatchesDirectory = fs::path(sOutputMatchesFilename).parent_path();

        //---------------------------------------
        // Load SfM Scene regions
        //---------------------------------------
        // Init the regions_type from the image describer file (used for image regions extraction)
        using namespace openMVG::features;
        const std::string sImage_describer = fs::path(sMatchesDirectory) / "image_describer.json";
        std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
        if (!regions_type) {
            OPENMVG_LOG_ERROR << "Invalid: " << sImage_describer << " regions type file.";
            return;
        }

        //---------------------------------------
        // a. Compute putative descriptor matches
        //    - Descriptor matching (according user method choice)
        //    - Keep correspondences only if NearestNeighbor ratio is ok
        //---------------------------------------

        // Load the corresponding view regions
        std::shared_ptr<Regions_Provider> regions_provider;
        if (ui_max_cache_size == 0) {
            // Default regions provider (load & store all regions in memory)
            regions_provider = std::make_shared<Regions_Provider>();
        } else {
            // Cached regions provider (load & store regions on demand)
            regions_provider = std::make_shared<Regions_Provider_Cache>(ui_max_cache_size);
        }
        // If we use pre-emptive matching, we load less regions:
        if (ui_preemptive_feature_count > 0 && use_p) {
            regions_provider = std::make_shared<Preemptive_Regions_Provider>(ui_preemptive_feature_count);
        }

        if (!regions_provider->load(sfm_data, sMatchesDirectory, regions_type)) {
            OPENMVG_LOG_ERROR << "Cannot load view regions from: " << sMatchesDirectory << ".";
            return;
        }

        PairWiseMatches map_PutativeMatches;

        // Build some alias from SfM_Data Views data:
        // - List views as a vector of filenames & image sizes
        std::vector<std::string> vec_fileNames;
        std::vector<std::pair<size_t, size_t> > vec_imagesSize; {
            vec_fileNames.reserve(sfm_data.GetViews().size());
            vec_imagesSize.reserve(sfm_data.GetViews().size());
            for (const auto view_it: sfm_data.GetViews()) {
                const View *v = view_it.second.get();
                vec_fileNames.emplace_back(fs::path(sfm_data.s_root_path) / v->s_Img_path);
                vec_imagesSize.emplace_back(v->ui_width, v->ui_height);
            }
        }

        OPENMVG_LOG_INFO << " - PUTATIVE MATCHES - ";
        // If the matches already exists, reload them
        if (!bForce && (fs::is_regular_file(sOutputMatchesFilename))) {
            if (!(Load(map_PutativeMatches, sOutputMatchesFilename))) {
                OPENMVG_LOG_ERROR << "Cannot load input matches file";
                return;
            }
            OPENMVG_LOG_INFO
      << "\t PREVIOUS RESULTS LOADED;"
      << " #pair: " << map_PutativeMatches.size();
        } else // Compute the putative matches
        {
            // Allocate the right Matcher according the Matching requested method
            std::unique_ptr<Matcher> collectionMatcher;
            if (sNearestMatchingMethod == "AUTO") {
                if (regions_type->IsScalar()) {
                    OPENMVG_LOG_INFO << "Using FAST_CASCADE_HASHING_L2 matcher";
                    collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions(fDistRatio));
                } else if (regions_type->IsBinary()) {
                    OPENMVG_LOG_INFO << "Using HNSWHAMMING matcher";
                    collectionMatcher.reset(new Matcher_Regions(fDistRatio, HNSW_HAMMING));
                }
            } else if (sNearestMatchingMethod == "BRUTEFORCEL2") {
                OPENMVG_LOG_INFO << "Using BRUTE_FORCE_L2 matcher";
                collectionMatcher.reset(new Matcher_Regions(fDistRatio, BRUTE_FORCE_L2));
            } else if (sNearestMatchingMethod == "BRUTEFORCEHAMMING") {
                OPENMVG_LOG_INFO << "Using BRUTE_FORCE_HAMMING matcher";
                collectionMatcher.reset(new Matcher_Regions(fDistRatio, BRUTE_FORCE_HAMMING));
            } else if (sNearestMatchingMethod == "HNSWL2") {
                OPENMVG_LOG_INFO << "Using HNSWL2 matcher";
                collectionMatcher.reset(new Matcher_Regions(fDistRatio, HNSW_L2));
            }
            if (sNearestMatchingMethod == "HNSWL1") {
                OPENMVG_LOG_INFO << "Using HNSWL1 matcher";
                collectionMatcher.reset(new Matcher_Regions(fDistRatio, HNSW_L1));
            } else if (sNearestMatchingMethod == "HNSWHAMMING") {
                OPENMVG_LOG_INFO << "Using HNSWHAMMING matcher";
                collectionMatcher.reset(new Matcher_Regions(fDistRatio, HNSW_HAMMING));
            } else if (sNearestMatchingMethod == "ANNL2") {
                OPENMVG_LOG_INFO << "Using ANN_L2 matcher";
                collectionMatcher.reset(new Matcher_Regions(fDistRatio, ANN_L2));
            } else if (sNearestMatchingMethod == "CASCADEHASHINGL2") {
                OPENMVG_LOG_INFO << "Using CASCADE_HASHING_L2 matcher";
                collectionMatcher.reset(new Matcher_Regions(fDistRatio, CASCADE_HASHING_L2));
            } else if (sNearestMatchingMethod == "FASTCASCADEHASHINGL2") {
                OPENMVG_LOG_INFO << "Using FAST_CASCADE_HASHING_L2 matcher";
                collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions(fDistRatio));
            }
            if (!collectionMatcher) {
                OPENMVG_LOG_ERROR << "Invalid Nearest Neighbor method: " << sNearestMatchingMethod;
                return;
            }
            // Perform the matching
            system::Timer timer; {
                // From matching mode compute the pair list that have to be matched:
                Pair_Set pairs;
                if (sPredefinedPairList.empty()) {
                    OPENMVG_LOG_INFO << "No input pair file set. Use exhaustive match by default.";
                    const size_t NImage = sfm_data.GetViews().size();
                    pairs = exhaustivePairs(NImage);
                } else if (!loadPairs(sfm_data.GetViews().size(), sPredefinedPairList, pairs)) {
                    OPENMVG_LOG_ERROR << "Failed to load pairs from file: \"" << sPredefinedPairList << "\"";
                    return;
                }
                OPENMVG_LOG_INFO << "Running matching on #pairs: " << pairs.size();
                // Photometric matching of putative pairs
                collectionMatcher->Match(regions_provider, pairs, map_PutativeMatches);

                if (use_p) // Preemptive filter
                {
                    // Keep putative matches only if there is more than X matches
                    PairWiseMatches map_filtered_matches;
                    for (const auto &pairwisematches_it: map_PutativeMatches) {
                        const size_t putative_match_count = pairwisematches_it.second.size();
                        const int match_count_threshold =
                                preemptive_matching_percentage_threshold * ui_preemptive_feature_count;
                        // TODO: Add an option to keeping X Best pairs
                        if (putative_match_count >= match_count_threshold) {
                            // the pair will be kept
                            map_filtered_matches.insert(pairwisematches_it);
                        }
                    }
                    map_PutativeMatches.clear();
                    std::swap(map_filtered_matches, map_PutativeMatches);
                }

                //---------------------------------------
                //-- Export putative matches & pairs
                //---------------------------------------
                if (!Save(map_PutativeMatches, std::string(sOutputMatchesFilename))) {
                    OPENMVG_LOG_ERROR
          << "Cannot save computed matches in: "
          << sOutputMatchesFilename;
                    return;
                }
                // Save pairs
                const std::string sOutputPairFilename = fs::path(sMatchesDirectory) / "preemptive_pairs.txt";

                if (!savePairs(
                    sOutputPairFilename,
                    getPairs(map_PutativeMatches))) {
                    OPENMVG_LOG_ERROR
          << "Cannot save computed matches pairs in: "
          << sOutputPairFilename;
                    return;
                }
            }
            OPENMVG_LOG_INFO << "Task (Regions Matching) done in (s): " << timer.elapsed();
        }

        OPENMVG_LOG_INFO << "#Putative pairs: " << map_PutativeMatches.size();

        // -- export Putative View Graph statistics
        graph::getGraphStatistics(sfm_data.GetViews().size(), getPairs(map_PutativeMatches));

        //-- export putative matches Adjacency matrix
        PairWiseMatchingToAdjacencyMatrixSVG(vec_fileNames.size(), map_PutativeMatches, fs::path(sMatchesDirectory) / "PutativeAdjacencyMatrix.svg");
        //-- export view pair graph once putative graph matches has been computed
        {
            std::set<IndexT> set_ViewIds;
            std::transform(sfm_data.GetViews().begin(), sfm_data.GetViews().end(), std::inserter(set_ViewIds, set_ViewIds.begin()), stl::RetrieveKey());
            graph::indexedGraph putativeGraph(set_ViewIds, getPairs(map_PutativeMatches));
            graph::exportToGraphvizData( fs::path(sMatchesDirectory) / "putative_matches", putativeGraph);
        }

        return;
    }

    void sparseBuilder::filter() {
        // The scene
        std::string sSfM_Data_Filename = matches_dir + "/sfm_data.json";
        // The input matches
        std::string sPutativeMatchesFilename = matches_dir + "/matches.putative.bin";
        // The output matches
        std::string sFilteredMatchesFilename = matches_dir + "/matches.f.bin";
        // The input pairs
        std::string sInputPairsFilename;
        // The output pairs
        std::string sOutputPairsFilename;

        std::string sGeometricModel = "f";
        bool bForce = false;
        bool bGuided_matching = false;
        int imax_iteration = 2048;
        unsigned int ui_max_cache_size = 0;

        if (sFilteredMatchesFilename.empty()) {
            OPENMVG_LOG_ERROR << "It is an invalid output file";
            return;
        }
        if (sSfM_Data_Filename.empty()) {
            OPENMVG_LOG_ERROR << "It is an invalid SfM file";
            return;
        }
        if (sPutativeMatchesFilename.empty()) {
            OPENMVG_LOG_ERROR << "It is an invalid putative matche file";
            return;
        }

        const std::string sMatchesDirectory = fs::path(sPutativeMatchesFilename).parent_path();

        EGeometricModel eGeometricModelToCompute = FUNDAMENTAL_MATRIX;
        switch (std::tolower(sGeometricModel[0], std::locale())) {
            case 'f':
                eGeometricModelToCompute = FUNDAMENTAL_MATRIX;
                break;
            case 'e':
                eGeometricModelToCompute = ESSENTIAL_MATRIX;
                break;
            case 'h':
                eGeometricModelToCompute = HOMOGRAPHY_MATRIX;
                break;
            case 'a':
                eGeometricModelToCompute = ESSENTIAL_MATRIX_ANGULAR;
                break;
            case 'u':
                eGeometricModelToCompute = ESSENTIAL_MATRIX_UPRIGHT;
                break;
            case 'o':
                eGeometricModelToCompute = ESSENTIAL_MATRIX_ORTHO;
                break;
            default:
                OPENMVG_LOG_ERROR << "Unknown geometric model";
                return;
        }

        // -----------------------------
        // - Load SfM_Data Views & intrinsics data
        // a. Load putative descriptor matches
        // [a.1] Filter matches with input pairs
        // b. Geometric filtering of putative matches
        // + Export some statistics
        // -----------------------------

        //---------------------------------------
        // Read SfM Scene (image view & intrinsics data)
        //---------------------------------------
        SfM_Data sfm_data;
        if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS | INTRINSICS))) {
            OPENMVG_LOG_ERROR << "The input SfM_Data file \"" << sSfM_Data_Filename << "\" cannot be read.";
            return;
        }

        //---------------------------------------
        // Load SfM Scene regions
        //---------------------------------------
        // Init the regions_type from the image describer file (used for image regions extraction)
        using namespace openMVG::features;
        // Consider that the image_describer.json is inside the matches directory (which is bellow the sfm_data.bin)
        const std::string sImage_describer = fs::path(sMatchesDirectory) / "image_describer.json";
        std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
        if (!regions_type) {
            OPENMVG_LOG_ERROR << "Invalid: " << sImage_describer << " regions type file.";
            return;
        }

        //---------------------------------------
        // a. Compute putative descriptor matches
        //    - Descriptor matching (according user method choice)
        //    - Keep correspondences only if NearestNeighbor ratio is ok
        //---------------------------------------

        // Load the corresponding view regions
        std::shared_ptr<Regions_Provider> regions_provider;
        if (ui_max_cache_size == 0) {
            // Default regions provider (load & store all regions in memory)
            regions_provider = std::make_shared<Regions_Provider>();
        } else {
            // Cached regions provider (load & store regions on demand)
            regions_provider = std::make_shared<Regions_Provider_Cache>(ui_max_cache_size);
        }

        if (!regions_provider->load(sfm_data, sMatchesDirectory, regions_type)) {
            OPENMVG_LOG_ERROR << "Invalid regions.";
            return;
        }

        PairWiseMatches map_PutativeMatches;
        //---------------------------------------
        // A. Load initial matches
        //---------------------------------------
        if (!Load(map_PutativeMatches, sPutativeMatchesFilename)) {
            OPENMVG_LOG_ERROR << "Failed to load the initial matches file.";
            return;
        }

        if (!sInputPairsFilename.empty()) {
            // Load input pairs
            OPENMVG_LOG_INFO << "Loading input pairs ...";
            Pair_Set input_pairs;
            loadPairs(sfm_data.GetViews().size(), sInputPairsFilename, input_pairs);

            // Filter matches with the given pairs
            OPENMVG_LOG_INFO << "Filtering matches with the given pairs.";
            map_PutativeMatches = getPairs(map_PutativeMatches, input_pairs);
        }

        //---------------------------------------
        // b. Geometric filtering of putative matches
        //    - AContrario Estimation of the desired geometric model
        //    - Use an upper bound for the a contrario estimated threshold
        //---------------------------------------

        std::unique_ptr<ImageCollectionGeometricFilter> filter_ptr(
            new ImageCollectionGeometricFilter(&sfm_data, regions_provider));

        if (filter_ptr) {
            system::Timer timer;
            const double d_distance_ratio = 0.6;

            PairWiseMatches map_GeometricMatches;
            switch (eGeometricModelToCompute) {
                case HOMOGRAPHY_MATRIX: {
                    const bool bGeometric_only_guided_matching = true;
                    filter_ptr->Robust_model_estimation(
                        GeometricFilter_HMatrix_AC(4.0, imax_iteration),
                        map_PutativeMatches,
                        bGuided_matching,
                        bGeometric_only_guided_matching ? -1.0 : d_distance_ratio);
                    map_GeometricMatches = filter_ptr->Get_geometric_matches();
                }
                break;
                case FUNDAMENTAL_MATRIX: {
                    filter_ptr->Robust_model_estimation(
                        GeometricFilter_FMatrix_AC(4.0, imax_iteration),
                        map_PutativeMatches,
                        bGuided_matching,
                        d_distance_ratio);
                    map_GeometricMatches = filter_ptr->Get_geometric_matches();
                }
                break;
                case ESSENTIAL_MATRIX: {
                    filter_ptr->Robust_model_estimation(
                        GeometricFilter_EMatrix_AC(4.0, imax_iteration),
                        map_PutativeMatches,
                        bGuided_matching,
                        d_distance_ratio);
                    map_GeometricMatches = filter_ptr->Get_geometric_matches();

                    //-- Perform an additional check to remove pairs with poor overlap
                    std::vector<PairWiseMatches::key_type> vec_toRemove;
                    for (const auto &pairwisematches_it: map_GeometricMatches) {
                        const size_t putativePhotometricCount = map_PutativeMatches.find(pairwisematches_it.first)->
                                second.
                                size();
                        const size_t putativeGeometricCount = pairwisematches_it.second.size();
                        const float ratio = putativeGeometricCount / static_cast<float>(putativePhotometricCount);
                        if (putativeGeometricCount < 50 || ratio < .3f) {
                            // the pair will be removed
                            vec_toRemove.push_back(pairwisematches_it.first);
                        }
                    }
                    //-- remove discarded pairs
                    for (const auto &pair_to_remove_it: vec_toRemove) {
                        map_GeometricMatches.erase(pair_to_remove_it);
                    }
                }
                break;
                case ESSENTIAL_MATRIX_ANGULAR: {
                    filter_ptr->Robust_model_estimation(
                        GeometricFilter_ESphericalMatrix_AC_Angular<false>(4.0, imax_iteration),
                        map_PutativeMatches, bGuided_matching, d_distance_ratio);
                    map_GeometricMatches = filter_ptr->Get_geometric_matches();
                }
                break;
                case ESSENTIAL_MATRIX_UPRIGHT: {
                    filter_ptr->Robust_model_estimation(
                        GeometricFilter_ESphericalMatrix_AC_Angular<true>(4.0, imax_iteration),
                        map_PutativeMatches, bGuided_matching, d_distance_ratio);
                    map_GeometricMatches = filter_ptr->Get_geometric_matches();
                }
                break;
                case ESSENTIAL_MATRIX_ORTHO: {
                    filter_ptr->Robust_model_estimation(
                        GeometricFilter_EOMatrix_RA(2.0, imax_iteration),
                        map_PutativeMatches,
                        bGuided_matching,
                        d_distance_ratio);
                    map_GeometricMatches = filter_ptr->Get_geometric_matches();
                }
                break;
            }

            //---------------------------------------
            //-- Export geometric filtered matches
            //---------------------------------------
            if (!Save(map_GeometricMatches, sFilteredMatchesFilename)) {
                OPENMVG_LOG_ERROR << "Cannot save filtered matches in: " << sFilteredMatchesFilename;
                return;
            }

            // -- export Geometric View Graph statistics
            graph::getGraphStatistics(sfm_data.GetViews().size(), getPairs(map_GeometricMatches));

            OPENMVG_LOG_INFO << "Task done in (s): " << timer.elapsed();

            //-- export Adjacency matrix
            OPENMVG_LOG_INFO << "\n Export Adjacency Matrix of the pairwise's geometric matches";

            PairWiseMatchingToAdjacencyMatrixSVG(sfm_data.GetViews().size(),
                                                 map_GeometricMatches,
                                                 fs::path(sMatchesDirectory) / "GeometricAdjacencyMatrix.svg");

            const Pair_Set outputPairs = getPairs(map_GeometricMatches);

            //-- export view pair graph once geometric filter have been done
            {
                std::set<IndexT> set_ViewIds;
                std::transform(sfm_data.GetViews().begin(), sfm_data.GetViews().end(),
                               std::inserter(set_ViewIds, set_ViewIds.begin()), stl::RetrieveKey());
                graph::indexedGraph putativeGraph(set_ViewIds, outputPairs);
                graph::exportToGraphvizData(fs::path(sMatchesDirectory) / "geometric_matches", putativeGraph);
            }

            // Write pairs
            if (!sOutputPairsFilename.empty()) {
                OPENMVG_LOG_INFO << "Saving pairs to: " << sOutputPairsFilename;
                if (!savePairs(sOutputPairsFilename, outputPairs)) {
                    OPENMVG_LOG_ERROR << "Failed to write pairs file";
                    return;
                }
            }
        }
        return;
    }

    void sparseBuilder::reconstruction() {
        std::string
                filename_sfm_data = matches_dir + "/sfm_data.json",
                directory_match = matches_dir,
                filename_match,
                directory_output = reconstruction_dir,
                engine_name = "INCREMENTAL";

        // Bundle adjustment options:
        std::string sIntrinsic_refinement_options = "ADJUST_ALL";
        std::string sExtrinsic_refinement_options = "ADJUST_ALL";
        bool b_use_motion_priors = false;

        // Incremental SfM options
        int triangulation_method = static_cast<int>(ETriangulationMethod::DEFAULT);
        int resection_method = static_cast<int>(resection::SolverType::DEFAULT);
        int user_camera_model = PINHOLE_CAMERA_RADIAL3;

        // SfM v1
        std::pair<std::string, std::string> initial_pair_string("", "");

        // SfM v2
        std::string sfm_initializer_method = "STELLAR";

        // Global SfM
        int rotation_averaging_method = int(ROTATION_AVERAGING_L2);
        int translation_averaging_method = int(TRANSLATION_AVERAGING_SOFTL1);

        std::string graph_simplification = "MST_X";
        int graph_simplification_value = 5;

        b_use_motion_priors = false;

        // Check validity of command line parameters:
        if (!isValid(static_cast<ETriangulationMethod>(triangulation_method))) {
            OPENMVG_LOG_ERROR << "Invalid triangulation method";
            return;
        }

        if (!isValid(openMVG::cameras::EINTRINSIC(user_camera_model))) {
            OPENMVG_LOG_ERROR << "Invalid camera type";
            return;
        }

        const cameras::Intrinsic_Parameter_Type intrinsic_refinement_options =
                cameras::StringTo_Intrinsic_Parameter_Type(sIntrinsic_refinement_options);
        if (intrinsic_refinement_options == static_cast<cameras::Intrinsic_Parameter_Type>(0)) {
            OPENMVG_LOG_ERROR << "Invalid input for Bundle Adjustment Intrinsic parameter refinement option";
            return;
        }

        const sfm::Extrinsic_Parameter_Type extrinsic_refinement_options =
                sfm::StringTo_Extrinsic_Parameter_Type(sExtrinsic_refinement_options);
        if (extrinsic_refinement_options == static_cast<sfm::Extrinsic_Parameter_Type>(0)) {
            OPENMVG_LOG_ERROR << "Invalid input for the Bundle Adjustment Extrinsic parameter refinement option";
            return;
        }

        ESfMSceneInitializer scene_initializer_enum;
        if (!StringToEnum(sfm_initializer_method, scene_initializer_enum)) {
            OPENMVG_LOG_ERROR << "Invalid input for the SfM initializer option";
            return;
        }

        ESfMEngine sfm_engine_type;
        if (!StringToEnum(engine_name, sfm_engine_type)) {
            OPENMVG_LOG_ERROR << "Invalid input for the SfM Engine type";
            return;
        }

        if (rotation_averaging_method < ROTATION_AVERAGING_L1 ||
            rotation_averaging_method > ROTATION_AVERAGING_L2) {
            OPENMVG_LOG_ERROR << "Rotation averaging method is invalid";
            return;
        }

#ifndef USE_PATENTED_LIGT
        if (translation_averaging_method == TRANSLATION_LIGT) {
            OPENMVG_LOG_ERROR << "OpenMVG was not compiled with USE_PATENTED_LIGT cmake option";
            return;
        }
#endif
        if (translation_averaging_method < TRANSLATION_AVERAGING_L1 ||
            translation_averaging_method > TRANSLATION_LIGT) {
            OPENMVG_LOG_ERROR << "Translation averaging method is invalid";
            return;
        }

        EGraphSimplification graph_simplification_method;
        if (!StringToEnum_EGraphSimplification(graph_simplification, graph_simplification_method)) {
            OPENMVG_LOG_ERROR << "Cannot recognize graph simplification method";
            return;
        }
        if (graph_simplification_value <= 1) {
            OPENMVG_LOG_ERROR << "graph simplification value must be > 1";
            return;
        }

        if (directory_output.empty()) {
            OPENMVG_LOG_ERROR << "It is an invalid output directory";
            return;
        }

        // SfM related

        // Load input SfM_Data scene
        SfM_Data sfm_data;
        const ESfM_Data sfm_data_loading_etypes =
                scene_initializer_enum == ESfMSceneInitializer::INITIALIZE_EXISTING_POSES
                    ? ESfM_Data(VIEWS | INTRINSICS | EXTRINSICS)
                    : ESfM_Data(VIEWS | INTRINSICS);
        if (!Load(sfm_data, filename_sfm_data, sfm_data_loading_etypes)) {
            OPENMVG_LOG_ERROR << "The input SfM_Data file \"" << filename_sfm_data << "\" cannot be read.";
            return;
        }

        if (!std::filesystem::is_directory(directory_output)) {
            if (!std::filesystem::create_directories(directory_output)) {
                OPENMVG_LOG_ERROR << "Cannot create the output directory";
                return;
            }
        }

        //
        // Match and features
        //
        if (directory_match.empty() && !filename_match.empty() && fs::is_regular_file(filename_match)) {
            directory_match = fs::path(filename_match).parent_path();
            filename_match = fs::path(filename_match).filename();
        }

        // Init the regions_type from the image describer file (used for image regions extraction)
        using namespace openMVG::features;
        const std::string sImage_describer = fs::path(directory_match) / "image_describer.json";
        std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
        if (!regions_type) {
            OPENMVG_LOG_ERROR << "Invalid: " << sImage_describer << " regions type file.";
            return;
        }

        // Features reading
        std::shared_ptr<Features_Provider> feats_provider = std::make_shared<Features_Provider>();
        if (!feats_provider->load(sfm_data, directory_match, regions_type)) {
            OPENMVG_LOG_ERROR << "Cannot load view corresponding features in directory: " << directory_match << ".";
            return;
        }
        // Matches reading
        std::shared_ptr<Matches_Provider> matches_provider = std::make_shared<Matches_Provider>();
        if // Try to read the provided match filename or the default one (matches.f.txt/bin)
        (
            !(matches_provider->load(sfm_data, fs::path(directory_match) / filename_match) ||
              matches_provider->load(sfm_data, fs::path(directory_match) / "matches.f.txt") ||
              matches_provider->load(sfm_data, fs::path(directory_match) / "matches.f.bin") ||
              matches_provider->load(sfm_data, fs::path(directory_match) / "matches.e.txt") ||
              matches_provider->load(sfm_data, fs::path(directory_match) / "matches.e.bin"))
        ) {
            OPENMVG_LOG_ERROR << "Cannot load the match file.";
            return;
        }

        std::unique_ptr<SfMSceneInitializer> scene_initializer;
        switch (scene_initializer_enum) {
            case ESfMSceneInitializer::INITIALIZE_AUTO_PAIR:
                OPENMVG_LOG_ERROR << "Not yet implemented.";
                return;
                break;
            case ESfMSceneInitializer::INITIALIZE_MAX_PAIR:
                scene_initializer.reset(new SfMSceneInitializerMaxPair(sfm_data,
                                                                       feats_provider.get(),
                                                                       matches_provider.get()));
                break;
            case ESfMSceneInitializer::INITIALIZE_EXISTING_POSES:
                scene_initializer.reset(new SfMSceneInitializer(sfm_data,
                                                                feats_provider.get(),
                                                                matches_provider.get()));
                break;
            case ESfMSceneInitializer::INITIALIZE_STELLAR:
                scene_initializer.reset(new SfMSceneInitializerStellar(sfm_data,
                                                                       feats_provider.get(),
                                                                       matches_provider.get()));
                break;
            default:
                OPENMVG_LOG_ERROR << "Unknown SFM Scene initializer method";
                return;
        }
        if (!scene_initializer) {
            OPENMVG_LOG_ERROR << "Invalid scene initializer.";
            return;
        }


        std::unique_ptr<ReconstructionEngine> sfm_engine;
        switch (sfm_engine_type) {
            case ESfMEngine::INCREMENTAL: {
                SequentialSfMReconstructionEngine *engine =
                        new SequentialSfMReconstructionEngine(
                            sfm_data,
                            directory_output,
                            fs::path(directory_output) / "Reconstruction_Report.html");

                // Configuration:
                engine->SetFeaturesProvider(feats_provider.get());
                engine->SetMatchesProvider(matches_provider.get());

                // Configure reconstruction parameters
                engine->SetUnknownCameraType(EINTRINSIC(user_camera_model));
                engine->SetTriangulationMethod(static_cast<ETriangulationMethod>(triangulation_method));
                engine->SetResectionMethod(static_cast<resection::SolverType>(resection_method));

                // Handle Initial pair parameter
                if (!initial_pair_string.first.empty() && !initial_pair_string.second.empty()) {
                    Pair initial_pair_index;
                    if (!computeIndexFromImageNames(sfm_data, initial_pair_string, initial_pair_index)) {
                        OPENMVG_LOG_ERROR << "Could not find the initial pairs <" << initial_pair_string.first
                  << ", " << initial_pair_string.second << ">!";
                        return;
                    }
                    engine->setInitialPair(initial_pair_index);
                }

                sfm_engine.reset(engine);
            }
            break;
            case ESfMEngine::INCREMENTALV2: {
                SequentialSfMReconstructionEngine2 *engine =
                        new SequentialSfMReconstructionEngine2(
                            scene_initializer.get(),
                            sfm_data,
                            directory_output,
                            fs::path(directory_output) / "Reconstruction_Report.html");

                // Configuration:
                engine->SetFeaturesProvider(feats_provider.get());
                engine->SetMatchesProvider(matches_provider.get());

                // Configure reconstruction parameters
                engine->SetTriangulationMethod(static_cast<ETriangulationMethod>(triangulation_method));
                engine->SetUnknownCameraType(EINTRINSIC(user_camera_model));
                engine->SetResectionMethod(static_cast<resection::SolverType>(resection_method));

                sfm_engine.reset(engine);
            }
            break;
            case ESfMEngine::GLOBAL: {
                GlobalSfMReconstructionEngine_RelativeMotions *engine =
                        new GlobalSfMReconstructionEngine_RelativeMotions(
                            sfm_data,
                            directory_output,
                            fs::path(directory_output) / "Reconstruction_Report.html");

                // Configuration:
                engine->SetFeaturesProvider(feats_provider.get());
                engine->SetMatchesProvider(matches_provider.get());

                // Configure motion averaging method
                engine->SetRotationAveragingMethod(ERotationAveragingMethod(rotation_averaging_method));
                engine->SetTranslationAveragingMethod(ETranslationAveragingMethod(translation_averaging_method));

                sfm_engine.reset(engine);
            }
            break;
            case ESfMEngine::STELLAR: {
                StellarSfMReconstructionEngine *engine =
                        new StellarSfMReconstructionEngine(
                            sfm_data,
                            directory_output,
                            fs::path(directory_output) / "Reconstruction_Report.html");

                // Configure the features_provider & the matches_provider
                engine->SetFeaturesProvider(feats_provider.get());
                engine->SetMatchesProvider(matches_provider.get());

                // Configure reconstruction parameters
                engine->SetGraphSimplification(graph_simplification_method, graph_simplification_value);

                sfm_engine.reset(engine);
            }
            break;
            default:
                break;
        }
        if (!sfm_engine) {
            OPENMVG_LOG_ERROR << "Cannot create the requested SfM Engine.";
            return;
        }

        sfm_engine->Set_Intrinsics_Refinement_Type(intrinsic_refinement_options);
        sfm_engine->Set_Extrinsics_Refinement_Type(extrinsic_refinement_options);
        sfm_engine->Set_Use_Motion_Prior(b_use_motion_priors);

        //---------------------------------------
        // Sequential reconstruction process
        //---------------------------------------

        openMVG::system::Timer timer;

        if (sfm_engine->Process()) {
            OPENMVG_LOG_INFO << " Total Sfm took (s): " << timer.elapsed();

            OPENMVG_LOG_INFO << "...Generating SfM_Report.html";
            Generate_SfM_Report(sfm_engine->Get_SfM_Data(),
                                fs::path(directory_output) / "SfMReconstruction_Report.html");

            //-- Export to disk computed scene (data & viewable results)
            OPENMVG_LOG_INFO << "...Export SfM_Data to disk.";
            Save(sfm_engine->Get_SfM_Data(),
                 fs::path(directory_output) / "sfm_data.bin",
                 ESfM_Data(ALL));

            Save(sfm_engine->Get_SfM_Data(),
                 fs::path(directory_output) / "cloud_and_poses.ply",
                 ESfM_Data(ALL));

            return;
        }
        return;
    }

    void sparseBuilder::colorize() {
        std::string
                sSfM_Data_Filename_In = reconstruction_dir + "/sfm_data.bin",
                sOutputPLY_Out = reconstruction_dir + "colorized.ply";

        if (sOutputPLY_Out.empty()) {
            OPENMVG_LOG_ERROR << "No output PLY filename specified.";
            return;
        }

        // Load input SfM_Data scene
        SfM_Data sfm_data;
        if (!Load(sfm_data, sSfM_Data_Filename_In, ESfM_Data(ALL))) {
            OPENMVG_LOG_ERROR << "The input SfM_Data file \"" << sSfM_Data_Filename_In << "\" cannot be read.";
            return;
        }

        // Compute the scene structure color
        std::vector<Vec3> vec_3dPoints, vec_tracksColor, vec_camPosition;
        if (ColorizeTracks(sfm_data, vec_3dPoints, vec_tracksColor)) {
            GetCameraPositions(sfm_data, vec_camPosition);

            // Export the SfM_Data scene in the expected format
            if (plyHelper::exportToPly(vec_3dPoints, vec_camPosition, sOutputPLY_Out, &vec_tracksColor)) {
                return;
            }
        }

        return;
    }
} // sparse
