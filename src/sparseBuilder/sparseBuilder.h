//
// Created by rainbowx on 25-5-9.
//

#ifndef SPARSEBUILDER_H
#define SPARSEBUILDER_H
#include <string>

#include <filesystem>
#include <utility>

namespace sparse {

class sparseBuilder {
protected:
    std::string input_dir = "/home/rainbowx/Documents/Projects/3DRebuild/images3";
    std::string output_dir = "/home/rainbowx/Documents/Projects/3DRebuild/output";
    std::string matches_dir = "/home/rainbowx/Documents/Projects/3DRebuild/output/matches";
    std::string reconstruction_dir = "/home/rainbowx/Documents/Projects/3DRebuild/output/reconstruction_sequential";
    std::string camera_file_params = "/home/rainbowx/Documents/Projects/3DReconstruction/3rdparty/openMVG/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt";
public:
    explicit sparseBuilder(
        const std::string& base_path
        ):
            input_dir(base_path+"/images"),
            output_dir(base_path+"/output"),
            matches_dir(output_dir+"/matches"),
            reconstruction_dir(output_dir+"/reconstruction_sequential"),
            camera_file_params(base_path+"/data/sensor_width_camera_database.txt")
    {}
    void readImagesCluster(double f = -1) const;

    void detectFeature();
    void matchPair();
    void match();
    void filter();
    void reconstruction();
    void colorize();
};

} // sparse

#endif //SPARSEBUILDER_H
