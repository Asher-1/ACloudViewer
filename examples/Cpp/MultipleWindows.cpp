// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>

#include "CloudViewer.h"

using namespace cloudViewer;
using namespace cloudViewer::visualization;

const int WIDTH = 1024;
const int HEIGHT = 768;
const Eigen::Vector3f CENTER_OFFSET(0.0f, 0.0f, -3.0f);
const std::string CLOUD_NAME = "points";

class MultipleWindowsApp {
public:
    MultipleWindowsApp() {
        is_done_ = false;

        gui::Application::GetInstance().Initialize();
    }

    void Run() {
        main_vis_ = cloudViewer::make_shared<visualizer::O3DVisualizer>(
                "CloudViewer - Multi-Window Demo", WIDTH, HEIGHT);
        main_vis_->AddAction(
                "Take snapshot in new window",
                [this](visualizer::O3DVisualizer &) { this->OnSnapshot(); });
        main_vis_->SetOnClose([this]() { return this->OnMainWindowClosing(); });

        gui::Application::GetInstance().AddWindow(main_vis_);
        auto r = main_vis_->GetOSFrame();
        snapshot_pos_ = gui::Point(r.x, r.y);

        std::thread read_thread([this]() { this->ReadThreadMain(); });
        gui::Application::GetInstance().Run();
        read_thread.join();
    }

private:
    void OnSnapshot() {
        n_snapshots_ += 1;
        snapshot_pos_ = gui::Point(snapshot_pos_.x + 50, snapshot_pos_.y + 50);
        auto title = std::string("CloudViewer - Multi-Window Demo (Snapshot #") +
                     std::to_string(n_snapshots_) + ")";
        auto new_vis = cloudViewer::make_shared<visualizer::O3DVisualizer>(
                title, WIDTH, HEIGHT);

        ccBBox bounds;
        {
            std::lock_guard<std::mutex> lock(cloud_lock_);
            auto mat = rendering::MaterialRecord();
            mat.shader = "defaultUnlit";
            new_vis->AddGeometry(
                    CLOUD_NAME + " #" + std::to_string(n_snapshots_), cloud_,
                    &mat);
            bounds = cloud_->getAxisAlignedBoundingBox();
        }

        new_vis->ResetCameraToDefault();
        auto center = bounds.getGeometryCenter().cast<float>();
        new_vis->SetupCamera(60, center, center + CENTER_OFFSET,
                             {0.0f, -1.0f, 0.0f});
        gui::Application::GetInstance().AddWindow(new_vis);
        auto r = new_vis->GetOSFrame();
        new_vis->SetOSFrame(
                gui::Rect(snapshot_pos_.x, snapshot_pos_.y, r.width, r.height));
    }

    bool OnMainWindowClosing() {
        // Ensure object is free so Filament can clean up without crashing.
        // Also signals to the "reading" thread that it is finished.
        main_vis_.reset();
        return true;  // false would cancel the close
    }

private:
    void ReadThreadMain() {
        // This is NOT the UI thread, need to call PostToMainThread() to
        // update the scene or any part of the UI.

        ccBBox bounds;
        Eigen::Vector3d extent;
        data::DemoICPPointClouds demo_icp_pointclouds;
        {
            std::lock_guard<std::mutex> lock(cloud_lock_);
            cloud_ = cloudViewer::make_shared<ccPointCloud>();
            io::ReadPointCloud(demo_icp_pointclouds.GetPaths(0), *cloud_);
            bounds = cloud_->getAxisAlignedBoundingBox();
            extent = bounds.getExtent();
        }

        auto mat = rendering::MaterialRecord();
        mat.shader = "defaultUnlit";

        gui::Application::GetInstance().PostToMainThread(
                main_vis_.get(), [this, bounds, mat]() {
                    std::lock_guard<std::mutex> lock(cloud_lock_);
                    main_vis_->AddGeometry(CLOUD_NAME, cloud_, &mat);
                    main_vis_->ResetCameraToDefault();
                    Eigen::Vector3f center = bounds.getGeometryCenter().cast<float>();
                    main_vis_->SetupCamera(60, center, center + CENTER_OFFSET,
                                           {0.0f, -1.0f, 0.0f});
                });

        Eigen::Vector3d magnitude = 0.005 * extent;
        auto seed = std::random_device()();
        std::mt19937 gen_algo(seed);
        std::uniform_real_distribution<> random(-0.5, 0.5);

        while (main_vis_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Perturb the cloud with a random walk to simulate an actual read
            {
                std::lock_guard<std::mutex> lock(cloud_lock_);
                for (size_t i = 0; i < cloud_->size(); ++i) {
                    Eigen::Vector3d perturb(magnitude[0] * random(gen_algo),
                                            magnitude[1] * random(gen_algo),
                                            magnitude[2] * random(gen_algo));
                    *cloud_->getPointPtr(i) += perturb;
                }
            }

            if (!main_vis_) {  // might have changed while sleeping
                break;
            }
            gui::Application::GetInstance().PostToMainThread(
                    main_vis_.get(), [this, mat]() {
                        std::lock_guard<std::mutex> lock(cloud_lock_);
                        // Note: if the number of points is less than or equal
                        // to the number of points in the original object that
                        // was added, using Scene::UpdateGeometry() will be
                        // faster. Requires that the point cloud be a
                        // t::PointCloud.
                        main_vis_->RemoveGeometry(CLOUD_NAME);
                        main_vis_->AddGeometry(CLOUD_NAME, cloud_, &mat);
                    });
        }
    }

private:
    std::mutex cloud_lock_;
    std::shared_ptr<ccPointCloud> cloud_;

    std::atomic<bool> is_done_;
    std::shared_ptr<visualizer::O3DVisualizer> main_vis_;
    int n_snapshots_ = 0;
    gui::Point snapshot_pos_;
};

int main(int argc, char *argv[]) {
    MultipleWindowsApp().Run();
    return 0;
}
