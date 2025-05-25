#include <vector>
#include <fstream>
#include <exception>
#include <filesystem>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

// spdlog
#include <spdlog/spdlog.h>

// cpp-httplib
// #define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

// #include <vtkAutoInit.h>

#include "common/common.h"
#include "utils/Event.h"
#include "utils/ConcurrentQueue.h"
// #include "frame/LocalFrame.h"
// #include "denseBuilder/DenseBuilder.h"
// #include "actuator/SequentialActuator.h"

// #include "example/OpenMVGExample.h"
#include "sparseBuilder/sparseBuilder.h"

// VTK_MODULE_INIT(vtkRenderingOpenGL2)
//
// VTK_MODULE_INIT(vtkInteractionStyle);
// VTK_MODULE_INIT(vtkRenderingFreeType);

using namespace std::string_literals;
namespace fs = std::filesystem;

fs::path url_base_path = "/www/wwwroot/proj.rainbowx.site/public";
fs::path base_path = url_base_path / "my";

auto images_path = base_path / "images";
auto chessboards_path = base_path / "chessboard";
auto out_path = base_path / "output";
auto reconstruction_out_path = out_path / "reconstruction_sequential";

auto tools_path = base_path/"tools";

ConcurrentQueue<::Event> event_queue;

int test() {
    return 0;
    // namespace fs = std::filesystem;
    //
    // DenseBuilder::writeToPLYFile("test.ply");
    // return 0;
    //
    // cv::Mat camera_matrix{}, dist_coeffs{};
    //
    // camera_matrix  = (cv::Mat_<double>(3,3)<<2905.88,0,1416,0,2905.88,1064,0,0,1);
    // // {
    // //     // 计算相机内参
    // //     spdlog::info("Start to calibrate camera.");
    // //     float square_size = 1.f; // 方格尺寸，单位：毫米
    // //     cv::Size board_size(8, 11); // 棋盘格内角点数量
    // //     std::vector<cv::Mat> chessboards_images;
    // //
    // //     for (const auto& entry : fs::directory_iterator(chessboards_path)) {
    // //         if (entry.is_regular_file()) {
    // //             chessboards_images.emplace_back(cv::imread(entry.path()));
    // //         }
    // //     }
    // //
    // //     [[maybe_unused]]auto rms = calibrateCameraFromImages(chessboards_images, board_size, square_size, camera_matrix, dist_coeffs);
    // //
    // //     spdlog::info("Calibrate camera done.");
    // //     std::cout<< camera_matrix<<std::endl;
    // //     std::cout<< dist_coeffs<<std::endl;
    // // }
    //
    // std::vector<Image> images;
    // {
    //     // 读取图像文件
    //     spdlog::info("Start to read images.");
    //     std::vector<std::string> image_paths;
    //     for (const auto& entry : fs::directory_iterator(images_path)) {
    //         if (entry.is_regular_file() and (entry.path().extension() != ".txt")) {
    //             image_paths.emplace_back(entry.path());
    //         }
    //     }
    //     std::ranges::sort(image_paths);
    //     auto camera = std::make_shared<Camera>(camera_matrix);
    //     for (const auto& image_path: image_paths) {
    //         cv::Mat img = cv::imread(image_path);
    //         images.emplace_back(img, image_path, camera);
    //     }
    //     spdlog::info("Read images done.");
    // }
    //
    // SequentialActuator actuator;
    //
    // actuator.init(images[0], images[1]);
    // actuator.bundleAdjustment();
    // // actuator.show(true);
    // for (int i = 2;i< images.size();i++) {
    //     actuator.addSingleImage(images[i]);
    //     actuator.bundleAdjustment();
    //     // actuator.show(true);
    // }
    //
    // auto world = actuator.getWorld();
    // world->writeToPLYFile("test.ply");
    // actuator.show();
    //
    // // DenseBuilder builder{actuator.getWorld()};
    // // builder.save("./output.mvs");
    //
    // return 0;
}

int preload(const std::shared_ptr<sparse::sparseBuilder>& builder) {
    std::string type = "preprocessing";

    event_queue.push(::Event::create().setType(type).setProgress(50*0));
    builder->readImagesCluster(2905.88);
    event_queue.push(::Event::create().setType(type).setProgress(50*1));
    builder->detectFeature();
    event_queue.push(::Event::create().setType(type).setProgress(50*2));
    return 0;
}

int sparseWork(const std::shared_ptr<sparse::sparseBuilder>& builder) {
    std::string type = "sparse";

    event_queue.push(::Event::create().setType(type).setProgress(20*0));
    builder->matchPair();
    event_queue.push(::Event::create().setType(type).setProgress(20*1));
    builder->match();
    event_queue.push(::Event::create().setType(type).setProgress(20*2));
    builder->filter();
    event_queue.push(::Event::create().setType(type).setProgress(20*3));
    builder->reconstruction();
    event_queue.push(::Event::create().setType(type).setProgress(20*4));
    builder->colorize();
    event_queue.push(::Event::create().setType(type).setProgress(20*5));
    return 0;
}

int denseWork(const std::string& output_data_path) {
    std::string type = "dense";
    fs::path data_dir = fs::path(output_data_path);

    std::string tool_densify_point_cloud = tools_path/ "DensifyPointCloud";
    std::string tool_export_path = tools_path/ "openMVG_main_openMVG2openMVS";

    event_queue.push(::Event::create().setType(type).setProgress(50*0));

    std::string command = std::format("{} -i {} -o {} -d {}", tool_export_path, (data_dir/"sfm_data.bin").string(), (data_dir/"output.mvs").string(), (data_dir/"undistorted_images").string());
    ensure(system(command.c_str()) == 0);
    event_queue.push(::Event::create().setType(type).setProgress(50*1));

    command = std::format("{} -i {} -w {}", tool_densify_point_cloud, "output.mvs", data_dir.string());
    ensure(system(command.c_str()) == 0);
    event_queue.push(::Event::create().setType(type).setProgress(50*2));

    return 0;
}

int meshWork(const std::string& output_data_path) {
    std::string type = "mesh";

    fs::path data_dir = fs::path(output_data_path);

    std::string tool_refine_mesh = tools_path/ "RefineMesh";
    std::string tool_texture_mesh = tools_path/ "TextureMesh";
    std::string tool_reconstruct_mesh = tools_path/ "ReconstructMesh";

    event_queue.push(::Event::create().setType(type).setProgress(0));

    std::string command;
    command = std::format("{} -i {} -w {}", tool_reconstruct_mesh, "output_dense.mvs", data_dir.string());
    ensure(system(command.c_str()) == 0);
    event_queue.push(::Event::create().setType(type).setProgress(33.33));

    command = std::format("{} -i {} -m {} -w {}", tool_refine_mesh, "output_dense.mvs", "output_dense_mesh.ply", data_dir.string());
    ensure(system(command.c_str()) == 0);
    event_queue.push(::Event::create().setType(type).setProgress(66.66));

    command = std::format("{} -i {} -m {} -w {}", tool_texture_mesh, "output_dense.mvs", "output_dense_refine.ply", data_dir.string());
    ensure(system(command.c_str()) == 0);
    event_queue.push(::Event::create().setType(type).setProgress(100));

    return 0;
}

bool create_event(uint64_t offset, httplib::DataSink& sink){
    bool isOk = true;
    while (isOk) {
        auto event = event_queue.pop();

        std::string eventdata = "data: " + event.toJson().dump() + "\n\n";
        isOk = sink.write(eventdata.c_str(), eventdata.length());
    }

    return true;
}

void add_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

int main() {
    std::shared_ptr<sparse::sparseBuilder> builder = std::make_shared<sparse::sparseBuilder>(base_path);

    // HTTP
    httplib::Server svr;

    svr.Options("/.*", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204; // No Content
    });

    // 事件总线
    svr.Get("/event",[](const httplib::Request &req, httplib::Response &res){
        add_cors_headers(res);
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider("text/event-stream", create_event);
    });

    svr.Post("/upload", [](const httplib::Request &req, httplib::Response &res) {
        add_cors_headers(res);
        if (!fs::is_directory(images_path)) {
            if (!fs::exists(images_path)) {
                fs::create_directories(images_path);
            }
        }

        if (!fs::is_directory(images_path)) {
            nlohmann::json j{{"status", "error"}, {"reason", "Fail to stat tmp dir:"+images_path.string()}};
            res.set_content(j.dump(), "application/json");
        }

        try {
            fs::remove_all(out_path); // 删除文件或子目录
            for (const auto &entry: fs::directory_iterator(images_path)) {
                fs::remove_all(entry.path()); // 删除文件或子目录
            }
        } catch (const std::exception &e) {
            nlohmann::json j{{"status", "error"}, {"reason", std::string("Failed to clean directory: ") + e.what()}};
            res.set_content(j.dump(), "application/json");
            return;
        }

        for (const auto& file_entry : req.files) {
            const auto& file = file_entry.second;
            std::ofstream ofs(images_path / file.filename, std::ios::binary);
            ofs << file.content;
            std::cout << "Saved: " << file.filename << std::endl;
        }

        nlohmann::json j{{"status", "ok"}};
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/preprocessing", [&builder](const httplib::Request &req, httplib::Response &res) {
        add_cors_headers(res);
        preload(builder);

        nlohmann::json j{{"status", "ok"}};
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/sparse", [&builder](const httplib::Request &req, httplib::Response &res) {
        add_cors_headers(res);
        sparseWork(builder);
        const auto abs_path = out_path / "reconstruction_sequentialcolorized.ply";
        const auto rel_path = fs::relative(abs_path, url_base_path);

        nlohmann::json j{{"status", "ok"}, {"data", "/"+rel_path.string()}};
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/dense", [](const httplib::Request &req, httplib::Response &res) {
        add_cors_headers(res);
        denseWork(reconstruction_out_path);

        const auto abs_path = reconstruction_out_path / "output_dense.ply";
        const auto rel_path = fs::relative(abs_path, url_base_path);

        nlohmann::json j{{"status", "ok"}, {"data", "/"+rel_path.string()}};
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/mesh", [](const httplib::Request &req, httplib::Response &res) {
        add_cors_headers(res);
        meshWork(reconstruction_out_path);

        const auto abs_path1 = reconstruction_out_path / "output_dense_texture.ply";
        const auto rel_path1 = fs::relative(abs_path1, url_base_path);
        const auto abs_path2 = reconstruction_out_path / "output_dense_texture0.png";
        const auto rel_path2 = fs::relative(abs_path2, url_base_path);

        nlohmann::json j{{"status", "ok"}, {"data", "/"+rel_path1.string()}, {"texture_url", "/"+rel_path2.string()}};
        res.set_content(j.dump(), "application/json");
    });

    svr.listen("localhost", 8080);

    return 0;
}
