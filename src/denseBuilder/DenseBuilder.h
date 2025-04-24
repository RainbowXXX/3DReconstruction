//
// Created by rainbowx on 25-4-22.
//

#ifndef DENSEBUILDER_H
#define DENSEBUILDER_H

#include <map>

#include "Interface.h"

#include "../component/Camera.h"
#include "../world/WorldStructure.h"

class DenseBuilder {
    const WorldStructure::Ptr structure_;

public:
    [[nodiscard]] explicit DenseBuilder(const WorldStructure::Ptr &structure)
        : structure_(structure) {
    }

    auto save(const std::string &save_path) {
        auto world = structure_;

        auto &images = world->getImages(); // map<Image::Idx, Image::Ptr>
        auto &worldPoints = world->getWorldPoints(); // map<WorldPoint::Idx, WorldPoint::Ptr>

        MVS::Interface scene; // the OpenMVS scene
        std::size_t nPoses(0);
        const uint32_t nViews(static_cast<uint32_t>(world->getImages().size()));

        std::map<Image::Ptr, uint32_t> map_view;
        std::map<Camera::Ptr, uint32_t> map_intrinsic;

        for (const auto &image: world->getImages() | std::ranges::views::values) {
            auto intrinsic = image->getCamera();
            if (not map_intrinsic.contains(intrinsic)) {
                map_intrinsic.insert(std::make_pair(intrinsic, scene.platforms.size()));
                _INTERFACE_NAMESPACE::Interface::Platform platform;
                // add the camera
                _INTERFACE_NAMESPACE::Interface::Platform::Camera camera;
                camera.width = image->width();
                camera.height = image->height();
                camera.K = intrinsic->getKMatxCV();

                // sub-pose
                camera.R = cv::Matx33d::eye();
                camera.C = {};
                platform.cameras.push_back(camera);
                scene.platforms.push_back(platform);
            }
        }

        // define images & poses
        scene.images.reserve(nViews);

        for (const auto &view: world->getImages() | std::ranges::views::values) {
            map_view[view] = scene.images.size();

            _INTERFACE_NAMESPACE::Interface::Image image;
            image.name = view->getName();
            image.platformID = map_intrinsic.at(view->getCamera());
            _INTERFACE_NAMESPACE::Interface::Platform &platform = scene.platforms[image.platformID];
            image.cameraID = 0;
            _INTERFACE_NAMESPACE::Interface::Platform::Pose pose;
            image.poseID = platform.poses.size();
            image.ID = map_view[view];

            pose.R = view->getRotationMatrix();
            pose.C = view->getTranslation();
            platform.poses.push_back(pose);
            ++nPoses;

            scene.images.emplace_back(image);
        }

        std::map<int,int> mp;

        // define structure
        scene.vertices.reserve(world->getWorldPoints().size());
        for (const auto& vertex: world->getWorldPoints() | std::ranges::views::values)
        {
            const auto & landmark = vertex->observed_frames_;
            _INTERFACE_NAMESPACE::Interface::Vertex vert;
            _INTERFACE_NAMESPACE::Interface::Vertex::ViewArr& views = vert.views;
            for (const auto &key: landmark | std::views::keys)
            {
                const auto it(map_view.find(key));
                if (it != map_view.end()) {
                    _INTERFACE_NAMESPACE::Interface::Vertex::View view;
                    view.imageID = it->second;
                    view.confidence = 0;
                    views.push_back(view);
                    mp[it->second]++;
                }
            }
            if (views.size() < 2)
                continue;
            std::ranges::sort(views,
                  [] (const _INTERFACE_NAMESPACE::Interface::Vertex::View& view0, const _INTERFACE_NAMESPACE::Interface::Vertex::View& view1){
                      return view0.imageID < view1.imageID;
                  }
            );

            auto pos = vertex->world_pos_;
            vert.X = {static_cast<float>(pos[0]), static_cast<float>(pos[1]), static_cast<float>(pos[2])};
            scene.vertices.push_back(vert);
        }

        if (!MVS::ARCHIVE::SerializeSave(scene, save_path)) {
            return false;
        }
        return true;
    }
};

#endif //DENSEBUILDER_H
