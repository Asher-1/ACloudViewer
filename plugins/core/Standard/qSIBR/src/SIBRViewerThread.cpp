// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SIBRViewerThread.h"

#include <algorithm>
#include <cmath>
#include <core/graphics/Window.hpp>
#include <core/raycaster/Raycaster.hpp>
#include <core/renderer/PointBasedRenderer.hpp>
#include <core/scene/BasicIBRScene.hpp>
#include <core/system/CommandLineArgs.hpp>
#include <core/system/Utils.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/view/SceneDebugView.hpp>
#include <projects/ulr/renderer/Config.hpp>
#include <projects/ulr/renderer/TexturedMeshView.hpp>
#include <projects/ulr/renderer/ULRV3View.hpp>
#include <projects/ulr/renderer/ULRView.hpp>

#ifdef SIBR_HAS_REMOTE
#include <projects/remote/renderer/Config.hpp>
#include <projects/remote/renderer/RemotePointView.hpp>
#endif

#ifdef SIBR_HAS_CUDA
#include <projects/gaussianviewer/renderer/Config.hpp>
#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#endif

std::mutex SIBRViewerThread::s_sibrGlobalMutex;
std::atomic<int> SIBRViewerThread::s_activeViewerCount{0};

SIBRViewerThread::SIBRViewerThread(ViewerMode mode,
                                   const QString& datasetPath,
                                   int width,
                                   int height,
                                   QObject* parent)
    : QThread(parent),
      m_mode(mode),
      m_datasetPath(datasetPath),
      m_width(width),
      m_height(height) {}

SIBRViewerThread::SIBRViewerThread(const QString& toolName,
                                   const QStringList& toolArgs,
                                   QObject* parent)
    : QThread(parent),
      m_mode(ViewerMode::DatasetTool),
      m_toolName(toolName),
      m_toolArgs(toolArgs),
      m_width(0),
      m_height(0) {}

SIBRViewerThread::~SIBRViewerThread() {
    requestStop();
    wait(5000);
}

void SIBRViewerThread::requestStop() { m_stopRequested.store(true); }

bool SIBRViewerThread::isStopRequested() const {
    return m_stopRequested.load();
}

bool SIBRViewerThread::hasActiveViewer() {
    return s_activeViewerCount.load() > 0;
}

void SIBRViewerThread::FakeArgs::build(
        const std::string& appName,
        const std::vector<std::pair<std::string, std::string>>& params) {
    storage.clear();
    argv.clear();
    storage.push_back(appName);
    for (const auto& p : params) {
        storage.push_back("--" + p.first);
        if (!p.second.empty()) {
            storage.push_back(p.second);
        }
    }
    argc = static_cast<int>(storage.size());
    argv.resize(argc);
    for (int i = 0; i < argc; ++i) {
        argv[i] = storage[i].c_str();
    }
}

// Clear any stale GL errors left over from a previous context / viewer
// session.  The very first CHECK_GL_ERROR in SIBR would otherwise pick
// up the leftover error code and throw, crashing the new viewer.
static void drainPendingGLErrors() {
    for (int i = 0; i < 256 && glGetError() != GL_NO_ERROR; ++i) {
    }
}

namespace {
class PointBasedIBRView : public sibr::ViewBase {
public:
    explicit PointBasedIBRView(const sibr::BasicIBRScene::Ptr& scene,
                               int pointSize = 5)
        : _scene(scene), _pointSize(pointSize) {
        _shader.init("PointBasedIBR",
                     sibr::loadFile(sibr::getShadersDirectory("core") +
                                    "/alpha_points.vert"),
                     sibr::loadFile(sibr::getShadersDirectory("core") +
                                    "/alpha_points.frag"));
        _mvp.init(_shader, "mvp");
        _alpha.init(_shader, "alpha");
        _radius.init(_shader, "radius");

        auto& proxy = _scene->proxies()->proxy();
        if (!proxy.hasColors() && proxy.vertices().size() > 0) {
            sibr::Mesh::Colors cols(proxy.vertices().size());
            for (size_t i = 0; i < cols.size(); ++i)
                cols[i] = sibr::Vector3f(0.4f, 0.8f, 1.0f);
            const_cast<sibr::Mesh&>(proxy).colors(cols);
        }
    }

    void onRenderIBR(sibr::IRenderTarget& dst,
                     const sibr::Camera& eye) override {
        if (_frameCount == 0) {
            auto& proxy = _scene->proxies()->proxy();
            SIBR_LOG << "[PointBasedIBRView] First render | "
                     << proxy.vertices().size()
                     << " verts, hasColors=" << proxy.hasColors()
                     << ", pointSize=" << _pointSize << std::endl;
        }
        ++_frameCount;

        dst.bind();
        glClearColor(0.12f, 0.12f, 0.14f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        auto& proxy = _scene->proxies()->proxy();
        if (proxy.vertices().size() > 0) {
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_PROGRAM_POINT_SIZE);
            _shader.begin();
            _mvp.set(eye.viewproj());
            _alpha.set(1.0f);
            _radius.set(_pointSize);
            proxy.render_points();
            _shader.end();
            glDisable(GL_PROGRAM_POINT_SIZE);
            glDisable(GL_DEPTH_TEST);
        }

        dst.unbind();
    }

    void onGUI() override {
        if (ImGui::Begin("Point Settings")) {
            ImGui::SliderInt("Point Size", &_pointSize, 1, 30);
        }
        ImGui::End();
    }

private:
    sibr::BasicIBRScene::Ptr _scene;
    sibr::GLShader _shader;
    sibr::GLuniform<sibr::Matrix4f> _mvp;
    sibr::GLuniform<float> _alpha;
    sibr::GLuniform<int> _radius;
    int _pointSize;
    uint64_t _frameCount = 0;
};
}  // anonymous namespace

void SIBRViewerThread::run() {
    s_activeViewerCount.fetch_add(1);

    QString modeName;
    int exitCode = 1;

    try {
        switch (m_mode) {
            case ViewerMode::ULR:
                modeName = "ULR Viewer";
                emit viewerStarted(modeName);
                exitCode = runULRViewer();
                break;
            case ViewerMode::ULRv2:
                modeName = "ULR v2 Viewer";
                emit viewerStarted(modeName);
                exitCode = runULRv2Viewer();
                break;
            case ViewerMode::TexturedMesh:
                modeName = "Textured Mesh Viewer";
                emit viewerStarted(modeName);
                exitCode = runTexturedMeshViewer();
                break;
            case ViewerMode::PointBased:
                modeName = "Point-Based Viewer";
                emit viewerStarted(modeName);
                exitCode = runPointBasedViewer();
                break;
            case ViewerMode::GaussianSplatting:
                modeName = "3D Gaussian Splatting";
                emit viewerStarted(modeName);
                exitCode = runGaussianViewer();
                break;
            case ViewerMode::RemoteGaussian:
                modeName = "Remote Gaussian";
                emit viewerStarted(modeName);
                exitCode = runRemoteGaussianViewer();
                break;
            case ViewerMode::DatasetTool:
                modeName = m_toolName;
                emit viewerStarted(modeName);
                exitCode = runDatasetTool();
                break;
        }
    } catch (const std::exception& e) {
        emit viewerError(
                QString("[SIBR] %1 error: %2").arg(modeName).arg(e.what()));
        exitCode = -1;
    } catch (...) {
        emit viewerError(QString("[SIBR] %1 unknown error").arg(modeName));
        exitCode = -1;
    }

    s_activeViewerCount.fetch_sub(1);
    emit viewerFinished(modeName, exitCode);
}

int SIBRViewerThread::runULRViewer() {
    using namespace sibr;

    FakeArgs fargs;
    std::vector<std::pair<std::string, std::string>> params;
    params.push_back({"path", m_datasetPath.toStdString()});
    params.push_back({"width", std::to_string(m_width)});
    params.push_back({"height", std::to_string(m_height)});
    fargs.build("SIBR_ulr_app", params);

    std::unique_ptr<BasicIBRAppArgs> argsPtr;
    {
        std::lock_guard<std::mutex> lock(s_sibrGlobalMutex);
        CommandLineArgs::parseMainArgs(fargs.argc, fargs.argv.data());
        argsPtr = std::make_unique<BasicIBRAppArgs>();
    }
    auto& myArgs = *argsPtr;

    Window window("SIBR ULR Viewer", sibr::Vector2i(50, 50), myArgs);
    drainPendingGLErrors();

    BasicIBRScene::Ptr scene(new BasicIBRScene(myArgs));

    uint rw = scene->cameras()->inputCameras()[0]->w();
    uint rh = scene->cameras()->inputCameras()[0]->h();
    Vector2u usedRes(rw, rh);

    ULRView::Ptr ulrView(new ULRView(scene, usedRes.x(), usedRes.y()));
    ulrView->setNumBlend(50, 50);

    auto raycaster = std::make_shared<Raycaster>();
    raycaster->init();
    raycaster->addMesh(scene->proxies()->proxy());

    InteractiveCameraHandler::Ptr cam(new InteractiveCameraHandler());
    cam->setup(scene->cameras()->inputCameras(),
               Viewport(0, 0, (float)usedRes.x(), (float)usedRes.y()),
               raycaster);

    MultiViewManager mvm(window, false);
    mvm.addIBRSubView("ULR view", ulrView, usedRes,
                      ImGuiWindowFlags_ResizeFromAnySide);
    mvm.addCameraForView("ULR view", cam);

    auto topView = std::make_shared<SceneDebugView>(scene, cam, myArgs);
    mvm.addSubView("Top view", topView, usedRes);

    emit viewerLog(
            QString("[SIBR] ULR Viewer ready | %1 cameras | resolution %2x%3")
                    .arg(scene->cameras()->inputCameras().size())
                    .arg(usedRes.x())
                    .arg(usedRes.y()));
    emit viewerLog(QString("[SIBR] Mesh: %1 vertices")
                           .arg(scene->proxies()->proxy().vertices().size()));

    while (window.isOpened() && !m_stopRequested.load()) {
        Input::poll();
        window.makeContextCurrent();
        if (Input::global().key().isPressed(Key::Escape)) window.close();
        mvm.onUpdate(Input::global());
        mvm.onRender(window);
        window.swapBuffer();
    }

    auto meshFile = QString::fromStdString(scene->data()->meshPath());
    if (!meshFile.isEmpty())
        emit viewerResultReady(meshFile, tr("ULR Scene Mesh"));

    return EXIT_SUCCESS;
}

int SIBRViewerThread::runULRv2Viewer() {
    using namespace sibr;

    FakeArgs fargs;
    std::vector<std::pair<std::string, std::string>> params;
    params.push_back({"path", m_datasetPath.toStdString()});
    params.push_back({"width", std::to_string(m_width)});
    params.push_back({"height", std::to_string(m_height)});
    fargs.build("SIBR_ulrv2_app", params);

    std::unique_ptr<ULRAppArgs> argsPtr;
    {
        std::lock_guard<std::mutex> lock(s_sibrGlobalMutex);
        CommandLineArgs::parseMainArgs(fargs.argc, fargs.argv.data());
        argsPtr = std::make_unique<ULRAppArgs>();
    }
    auto& myArgs = *argsPtr;

    Window window("SIBR ULR v2 Viewer", sibr::Vector2i(50, 50), myArgs);
    drainPendingGLErrors();

    BasicIBRScene::Ptr scene(new BasicIBRScene(myArgs, true));

    const uint flags = SIBR_GPU_LINEAR_SAMPLING | SIBR_FLIP_TEXTURE;
    uint rw = scene->cameras()->inputCameras()[0]->w();
    uint rh = scene->cameras()->inputCameras()[0]->h();
    Vector2u usedRes(rw, rh);

    scene->renderTargets()->initRGBandDepthTextureArrays(
            scene->cameras(), scene->images(), scene->proxies(), flags);

    ULRV3View::Ptr ulrView(new ULRV3View(scene, usedRes.x(), usedRes.y()));

    if (myArgs.masks) {
        if (!myArgs.maskParams.get().empty()) {
            ulrView->getULRrenderer()->loadMasks(
                    scene, myArgs.maskParams.get(), "",
                    myArgs.maskParamsExtra.get().empty()
                            ? ".png"
                            : myArgs.maskParamsExtra.get());
        } else {
            ulrView->getULRrenderer()->loadMasks(scene);
        }
        ulrView->getULRrenderer()->useMasks() = true;
    }

    auto raycaster = std::make_shared<Raycaster>();
    raycaster->init();
    raycaster->addMesh(scene->proxies()->proxy());

    InteractiveCameraHandler::Ptr cam(new InteractiveCameraHandler());
    cam->setup(scene->cameras()->inputCameras(),
               Viewport(0, 0, (float)usedRes.x(), (float)usedRes.y()),
               raycaster);

    MultiViewManager mvm(window, false);
    mvm.addIBRSubView("ULR view", ulrView, usedRes,
                      ImGuiWindowFlags_ResizeFromAnySide);
    mvm.addCameraForView("ULR view", cam);

    auto topView = std::make_shared<SceneDebugView>(scene, cam, myArgs);
    mvm.addSubView("Top view", topView, usedRes);

    emit viewerLog(QString("[SIBR] ULR v2 Viewer ready | %1 cameras | "
                           "resolution %2x%3")
                           .arg(scene->cameras()->inputCameras().size())
                           .arg(usedRes.x())
                           .arg(usedRes.y()));
    emit viewerLog(QString("[SIBR] Mesh: %1 vertices")
                           .arg(scene->proxies()->proxy().vertices().size()));

    while (window.isOpened() && !m_stopRequested.load()) {
        Input::poll();
        window.makeContextCurrent();
        if (Input::global().key().isPressed(Key::Escape)) window.close();
        mvm.onUpdate(Input::global());
        mvm.onRender(window);
        window.swapBuffer();
    }

    auto meshFile = QString::fromStdString(scene->data()->meshPath());
    if (!meshFile.isEmpty())
        emit viewerResultReady(meshFile, tr("ULR v2 Scene Mesh"));

    return EXIT_SUCCESS;
}

int SIBRViewerThread::runTexturedMeshViewer() {
    using namespace sibr;

    FakeArgs fargs;
    std::vector<std::pair<std::string, std::string>> params;
    params.push_back({"path", m_datasetPath.toStdString()});
    params.push_back({"width", std::to_string(m_width)});
    params.push_back({"height", std::to_string(m_height)});
    fargs.build("SIBR_texturedMesh_app", params);

    std::unique_ptr<BasicIBRAppArgs> argsPtr;
    {
        std::lock_guard<std::mutex> lock(s_sibrGlobalMutex);
        CommandLineArgs::parseMainArgs(fargs.argc, fargs.argv.data());
        argsPtr = std::make_unique<BasicIBRAppArgs>();
    }
    auto& myArgs = *argsPtr;

    Window window("SIBR Textured Mesh", sibr::Vector2i(50, 50), myArgs);
    drainPendingGLErrors();

    BasicIBRScene::Ptr scene(new BasicIBRScene(myArgs));

    if (scene->proxies()->hasProxy() &&
        scene->proxies()->proxy().triangles().empty()) {
        std::string dp = myArgs.dataset_path.get();
        const std::vector<std::string> candidates = {
                dp + "/meshed-delaunay.ply",
                dp + "/meshed-poisson.ply",
                dp + "/mesh.ply",
                dp + "/mesh.obj",
                dp + "/recon.ply",
                dp + "/sfm_mvs_cm/recon.ply",
                dp + "/colmap/stereo/meshed-delaunay.ply",
        };
        for (const auto& c : candidates) {
            if (fileExists(c)) {
                Mesh::Ptr realMesh(new Mesh());
                realMesh->load(c);
                if (!realMesh->triangles().empty()) {
                    SIBR_LOG << "[TexturedMeshViewer] Using mesh: " << c
                             << std::endl;
                    scene->proxies()->replaceProxyPtr(realMesh);
                    break;
                }
            }
        }
    }

    if (!scene->inputMeshTextures()) {
        std::string dp = myArgs.dataset_path.get();
        std::string caprealDir = dp + "/capreal";
        if (!directoryExists(caprealDir))
            caprealDir = parentDirectory(dp) + "/capreal";

        const std::vector<std::string> texCandidates = {
                caprealDir + "/texture.png", caprealDir + "/texture_u1_v1.png",
                caprealDir + "/mesh.png",    caprealDir + "/mesh_u1_v1.png",
                dp + "/texture.png",         dp + "/mesh_u1_v1.png",
                dp + "/textured_u1_v1.png",
        };
        for (const auto& tp : texCandidates) {
            if (fileExists(tp)) {
                ImageRGB texImg;
                texImg.load(tp);
                scene->inputMeshTextures().reset(
                        new Texture2DRGB(texImg, SIBR_GPU_LINEAR_SAMPLING));
                SIBR_LOG << "[TexturedMeshViewer] Loaded texture: " << tp
                         << std::endl;
                break;
            }
        }
    }

    if (!scene->inputMeshTextures()) {
        SIBR_LOG << "[TexturedMeshViewer] No texture found, creating "
                    "fallback (mesh will render with solid color)"
                 << std::endl;
        ImageRGB::Pixel gray(200, 200, 200);
        ImageRGB fallback(1, 1, gray);
        scene->inputMeshTextures().reset(
                new Texture2DRGB(fallback, SIBR_GPU_LINEAR_SAMPLING));
    }

    uint rw = scene->cameras()->inputCameras()[0]->w();
    uint rh = scene->cameras()->inputCameras()[0]->h();
    Vector2u usedRes(rw, rh);
    // float aspect = (float)rw / rh;
    // uint uw = std::min(1920u, rw);
    // uint uh = (uint)(uw / aspect);
    // Vector2u usedRes(uw, uh);

    TexturedMeshView::Ptr texView(
            new TexturedMeshView(scene, usedRes.x(), usedRes.y()));

    auto raycaster = std::make_shared<Raycaster>();
    raycaster->init();
    raycaster->addMesh(scene->proxies()->proxy());

    InteractiveCameraHandler::Ptr cam(new InteractiveCameraHandler());
    cam->setup(scene->cameras()->inputCameras(),
               Viewport(0, 0, (float)usedRes.x(), (float)usedRes.y()),
               raycaster);

    MultiViewManager mvm(window, false);
    mvm.addIBRSubView("Textured Mesh", texView, usedRes,
                      ImGuiWindowFlags_ResizeFromAnySide);
    mvm.addCameraForView("Textured Mesh", cam);

    auto topView = std::make_shared<SceneDebugView>(scene, cam, myArgs);
    mvm.addSubView("Top view", topView, usedRes);

    emit viewerLog(QString("[SIBR] Textured Mesh Viewer ready | %1 cameras | "
                           "resolution %2x%3")
                           .arg(scene->cameras()->inputCameras().size())
                           .arg(usedRes.x())
                           .arg(usedRes.y()));
    emit viewerLog(QString("[SIBR] Mesh: %1 vertices | %2 triangles")
                           .arg(scene->proxies()->proxy().vertices().size())
                           .arg(scene->proxies()->proxy().triangles().size()));

    while (window.isOpened() && !m_stopRequested.load()) {
        Input::poll();
        window.makeContextCurrent();
        if (Input::global().key().isPressed(Key::Escape)) window.close();
        mvm.onUpdate(Input::global());
        mvm.onRender(window);
        window.swapBuffer();
    }

    auto meshFile = QString::fromStdString(scene->data()->meshPath());
    if (!meshFile.isEmpty())
        emit viewerResultReady(meshFile, tr("Textured Mesh"));

    return EXIT_SUCCESS;
}

int SIBRViewerThread::runPointBasedViewer() {
    using namespace sibr;

    FakeArgs fargs;
    std::vector<std::pair<std::string, std::string>> params;
    params.push_back({"path", m_datasetPath.toStdString()});
    params.push_back({"width", std::to_string(m_width)});
    params.push_back({"height", std::to_string(m_height)});
    fargs.build("SIBR_pointBased_app", params);

    std::unique_ptr<BasicIBRAppArgs> argsPtr;
    {
        std::lock_guard<std::mutex> lock(s_sibrGlobalMutex);
        CommandLineArgs::parseMainArgs(fargs.argc, fargs.argv.data());
        argsPtr = std::make_unique<BasicIBRAppArgs>();
    }
    auto& myArgs = *argsPtr;

    Window window("SIBR Point-Based Viewer", sibr::Vector2i(50, 50), myArgs);
    drainPendingGLErrors();

    BasicIBRScene::SceneOptions opts;
    opts.renderTargets = false;
    opts.mesh = true;
    opts.images = false;
    opts.cameras = true;
    opts.texture = false;

    BasicIBRScene::Ptr scene(new BasicIBRScene(myArgs, opts));
    uint rw = scene->cameras()->inputCameras()[0]->w();
    uint rh = scene->cameras()->inputCameras()[0]->h();
    float aspect = (float)rw / rh;
    uint uw = std::min(1200u, rw);
    uint uh = (uint)(uw / aspect);
    Vector2u usedRes(uw, uh);

    auto pbView = std::make_shared<PointBasedIBRView>(scene);

    auto raycaster = std::make_shared<Raycaster>();
    raycaster->init();
    raycaster->addMesh(scene->proxies()->proxy());

    InteractiveCameraHandler::Ptr cam(new InteractiveCameraHandler());
    cam->setup(scene->cameras()->inputCameras(),
               Viewport(0, 0, (float)usedRes.x(), (float)usedRes.y()),
               raycaster);

    MultiViewManager mvm(window, false);
    mvm.addIBRSubView("Point view", pbView, usedRes,
                      ImGuiWindowFlags_ResizeFromAnySide);
    mvm.addCameraForView("Point view", cam);

    auto topView = std::make_shared<SceneDebugView>(scene, cam, myArgs);
    mvm.addSubView("Top view", topView, usedRes);
    topView->active(false);

    emit viewerLog(QString("[SIBR] Point-Based Viewer ready | %1 cameras | "
                           "resolution %2x%3")
                           .arg(scene->cameras()->inputCameras().size())
                           .arg(usedRes.x())
                           .arg(usedRes.y()));
    emit viewerLog(QString("[SIBR] Proxy: %1 vertices")
                           .arg(scene->proxies()->proxy().vertices().size()));

    while (window.isOpened() && !m_stopRequested.load()) {
        Input::poll();
        window.makeContextCurrent();
        if (Input::global().key().isPressed(Key::Escape)) window.close();
        mvm.onUpdate(Input::global());
        mvm.onRender(window);
        window.swapBuffer();
    }

    auto meshFile = QString::fromStdString(scene->data()->meshPath());
    if (!meshFile.isEmpty())
        emit viewerResultReady(meshFile, tr("SfM Point Cloud"));

    return EXIT_SUCCESS;
}

int SIBRViewerThread::runGaussianViewer() {
#ifdef SIBR_HAS_CUDA
    using namespace sibr;

    FakeArgs fargs;
    std::vector<std::pair<std::string, std::string>> params;
    if (!m_datasetPath.isEmpty())
        params.push_back({"path", m_datasetPath.toStdString()});
    params.push_back({"model-path", m_modelPath.toStdString()});
    params.push_back({"width", std::to_string(m_width)});
    params.push_back({"height", std::to_string(m_height)});
    params.push_back({"device", std::to_string(m_cudaDevice)});
    if (!m_iteration.isEmpty())
        params.push_back({"iteration", m_iteration.toStdString()});
    if (m_noInterop) params.push_back({"no_interop", ""});
    fargs.build("SIBR_gaussianViewer_app", params);

    std::unique_ptr<GaussianAppArgs> argsPtr;
    {
        std::lock_guard<std::mutex> lock(s_sibrGlobalMutex);
        CommandLineArgs::parseMainArgs(fargs.argc, fargs.argv.data());
        argsPtr = std::make_unique<GaussianAppArgs>();
    }
    auto& myArgs = *argsPtr;

    if (!myArgs.modelPath.isInit() && myArgs.modelPathShort.isInit())
        myArgs.modelPath = myArgs.modelPathShort.get();
    if (!myArgs.dataset_path.isInit() && myArgs.pathShort.isInit())
        myArgs.dataset_path = myArgs.pathShort.get();

    int device = myArgs.device;

    uint rendering_width = myArgs.rendering_size.get()[0];
    uint rendering_height = myArgs.rendering_size.get()[1];

    Window window("SIBR 3D Gaussian Splatting", sibr::Vector2i(50, 50), myArgs,
                  getResourcesDirectory() + "/gaussians/sibr_3Dgaussian.ini");
    drainPendingGLErrors();

    std::string cfgLine;
    std::ifstream cfgFile(myArgs.modelPath.get() + "/cfg_args");
    if (cfgFile.good()) {
        std::getline(cfgFile, cfgLine);

        if (!myArgs.dataset_path.isInit()) {
            auto findArgLambda = [](const std::string& line,
                                    const std::string& name) {
                int start = line.find(name, 0);
                start = line.find("=", start);
                start += 1;
                int end = line.find_first_of(",)", start);
                return std::make_pair(start, end);
            };
            auto rng = findArgLambda(cfgLine, "source_path");
            myArgs.dataset_path =
                    cfgLine.substr(rng.first + 1, rng.second - rng.first - 2);
        }
    }

    int sh_degree = 3;
    bool white_background = false;
    if (!cfgLine.empty()) {
        auto findArgLambda = [](const std::string& line,
                                const std::string& name) {
            int start = line.find(name, 0);
            start = line.find("=", start);
            start += 1;
            int end = line.find_first_of(",)", start);
            return std::make_pair(start, end);
        };
        auto rng = findArgLambda(cfgLine, "sh_degree");
        sh_degree =
                std::stoi(cfgLine.substr(rng.first, rng.second - rng.first));
        rng = findArgLambda(cfgLine, "white_background");
        white_background = cfgLine.substr(rng.first, rng.second - rng.first)
                                   .find("True") != std::string::npos;
    }

    BasicIBRScene::SceneOptions sceneOpts;
    sceneOpts.renderTargets = myArgs.loadImages;
    sceneOpts.mesh = true;
    sceneOpts.images = myArgs.loadImages;
    sceneOpts.cameras = true;
    sceneOpts.texture = false;

    BasicIBRScene::Ptr scene;
    try {
        scene.reset(new BasicIBRScene(myArgs, sceneOpts));
    } catch (...) {
        myArgs.dataset_path = myArgs.modelPath.get();
        scene.reset(new BasicIBRScene(myArgs, sceneOpts));
    }

    std::string plyfile = myArgs.modelPath.get();
    if (plyfile.back() != '/') plyfile += "/";
    plyfile += "point_cloud";

    namespace fs = boost::filesystem;
    if (!myArgs.iteration.isInit()) {
        int largest = -1;
        std::string largestDir;
        for (auto& entry : fs::directory_iterator(plyfile)) {
            if (fs::is_directory(entry)) {
                std::string name = entry.path().filename().string();
                std::regex re(R"_(iteration_(\d+))_");
                std::smatch m;
                if (std::regex_match(name, m, re)) {
                    int num = std::stoi(m[1]);
                    if (num > largest) {
                        largest = num;
                        largestDir = name;
                    }
                }
            }
        }
        plyfile += "/" + largestDir + "/point_cloud.ply";
    } else {
        plyfile += "/iteration_" + myArgs.iteration.get() + "/point_cloud.ply";
    }

    uint sw = scene->cameras()->inputCameras()[0]->w();
    uint sh = scene->cameras()->inputCameras()[0]->h();
    float sa = (float)sw / sh;

    rendering_width =
            (rendering_width <= 0) ? std::min(1200u, sw) : rendering_width;
    rendering_height = (rendering_height <= 0) ? (uint)(rendering_width / sa)
                                               : rendering_height;
    Vector2u usedRes(rendering_width, rendering_height);

    bool messageRead = false;
    GaussianView::Ptr gaussianView(new GaussianView(
            scene, usedRes.x(), usedRes.y(), plyfile.c_str(), &messageRead,
            sh_degree, white_background, !myArgs.noInterop, device));

    InteractiveCameraHandler::Ptr cam(new InteractiveCameraHandler());
    cam->setup(scene->cameras()->inputCameras(),
               Viewport(0, 0, (float)usedRes.x(), (float)usedRes.y()), nullptr);

    MultiViewManager mvm(window, false);
    mvm.addIBRSubView("Point view", gaussianView, usedRes,
                      ImGuiWindowFlags_ResizeFromAnySide |
                              ImGuiWindowFlags_NoBringToFrontOnFocus);
    mvm.addCameraForView("Point view", cam);

    auto topView = std::make_shared<SceneDebugView>(scene, cam, myArgs,
                                                    myArgs.imagesPath.get());
    mvm.addSubView("Top view", topView, usedRes);
    topView->active(false);

    cam->getCameraRecorder().setViewPath(gaussianView,
                                         myArgs.dataset_path.get());

    emit viewerLog(QString("[SIBR] 3D Gaussian Splatting Viewer ready | "
                           "%1 cameras | resolution %2x%3")
                           .arg(scene->cameras()->inputCameras().size())
                           .arg(usedRes.x())
                           .arg(usedRes.y()));
    emit viewerLog(QString("[SIBR] SH degree: %1 | white_bg: %2 | device: %3")
                           .arg(sh_degree)
                           .arg(white_background ? "yes" : "no")
                           .arg(device));
    emit viewerLog(
            QString("[SIBR] PLY: %1").arg(QString::fromStdString(plyfile)));

    while (window.isOpened() && !m_stopRequested.load()) {
        Input::poll();
        window.makeContextCurrent();
        if (Input::global().key().isPressed(Key::Escape)) window.close();
        mvm.onUpdate(Input::global());
        mvm.onRender(window);
        window.swapBuffer();
    }

    emit viewerResultReady(QString::fromStdString(plyfile),
                           tr("3DGS Model (Gaussian Splats)"));

    auto meshFile = QString::fromStdString(scene->data()->meshPath());
    if (!meshFile.isEmpty())
        emit viewerResultReady(meshFile, tr("SfM Point Cloud"));

    return EXIT_SUCCESS;
#else
    emit viewerError("[SIBR] CUDA not available");
    return 1;
#endif
}

int SIBRViewerThread::runRemoteGaussianViewer() {
#ifdef SIBR_HAS_REMOTE
    using namespace sibr;

    FakeArgs fargs;
    std::vector<std::pair<std::string, std::string>> params;
    if (!m_datasetPath.isEmpty())
        params.push_back({"path", m_datasetPath.toStdString()});
    params.push_back({"ip", m_remoteIP.toStdString()});
    params.push_back({"port", std::to_string(m_remotePort)});
    params.push_back({"width", std::to_string(m_width)});
    params.push_back({"height", std::to_string(m_height)});
    fargs.build("SIBR_remoteGaussian_app", params);

    std::unique_ptr<RemoteAppArgs> argsPtr;
    {
        std::lock_guard<std::mutex> lock(s_sibrGlobalMutex);
        CommandLineArgs::parseMainArgs(fargs.argc, fargs.argv.data());
        argsPtr = std::make_unique<RemoteAppArgs>();
    }
    auto& myArgs = *argsPtr;

    if (!myArgs.dataset_path.isInit() && myArgs.pathShort.isInit())
        myArgs.dataset_path = myArgs.pathShort.get();

    uint rendering_width = myArgs.rendering_size.get()[0];
    uint rendering_height = myArgs.rendering_size.get()[1];
    if (rendering_width == 0) rendering_width = 1200;
    if (rendering_height == 0) rendering_height = 800;

    Window window("SIBR Remote Gaussian", sibr::Vector2i(50, 50), myArgs);
    drainPendingGLErrors();

    MultiViewManager mvm(window, false);
    BasicIBRScene::Ptr scene;
    RemotePointView::Ptr remoteView(
            new RemotePointView(myArgs.ip.get(), myArgs.port.get()));
    std::shared_ptr<SceneDebugView> topView;

    auto resetScene = [&]() {
        drainPendingGLErrors();

        if (mvm.numSubViews() > 0) {
            mvm.removeSubView("Point view");
            mvm.removeSubView("Top view");
        }

        BasicIBRScene::SceneOptions sceneOpts;
        sceneOpts.renderTargets = myArgs.loadImages;
        sceneOpts.mesh = true;
        sceneOpts.images = myArgs.loadImages;
        sceneOpts.cameras = true;
        sceneOpts.texture = false;

        scene.reset(new BasicIBRScene(myArgs, sceneOpts));

        float divider =
                scene->cameras()->inputCameras()[0]->w() /
                std::min(1200.f,
                         (float)scene->cameras()->inputCameras()[0]->w());
        uint sw = scene->cameras()->inputCameras()[0]->w();
        uint sh = scene->cameras()->inputCameras()[0]->h();
        float scene_ar = sw * 1.0f / sh;
        float render_ar =
                rendering_width * 1.0f / std::max(1u, rendering_height);

        uint rw = rendering_width;
        uint rh = rendering_height;
        if ((rw > 0) && !myArgs.force_aspect_ratio) {
            if (std::abs(scene_ar - render_ar) > 0.001f) {
                if (sw > sh)
                    rh = static_cast<uint>(rw / scene_ar);
                else
                    rw = static_cast<uint>(rh * scene_ar);
            }
        }
        rw = (rw <= 0) ? static_cast<uint>(sw / divider) : rw;
        rh = (rh <= 0) ? static_cast<uint>(sh / divider) : rh;
        Vector2u usedRes(rw, rh);

        const unsigned int sceneResWidth = usedRes.x();
        const unsigned int sceneResHeight = usedRes.y();

        remoteView->setScene(scene);
        remoteView->setResolution({(int)sceneResWidth, (int)sceneResHeight});

        auto raycaster = std::make_shared<Raycaster>();
        raycaster->init();
        raycaster->addMesh(scene->proxies()->proxy());

        InteractiveCameraHandler::Ptr cam(new InteractiveCameraHandler());
        cam->setup(scene->cameras()->inputCameras(),
                   Viewport(0, 0, (float)usedRes.x(), (float)usedRes.y()),
                   nullptr);

        topView.reset(new SceneDebugView(scene, cam, myArgs));
        mvm.addSubView("Top view", topView, usedRes);
        topView->active(false);

        mvm.addIBRSubView("Point view", remoteView,
                          {sceneResWidth, sceneResHeight},
                          ImGuiWindowFlags_NoBringToFrontOnFocus);
        mvm.addCameraForView("Point view", cam);

        CHECK_GL_ERROR;

        cam->getCameraRecorder().setViewPath(remoteView,
                                             myArgs.dataset_path.get());

        SIBR_LOG << "[qSIBR] resetScene: res=" << sceneResWidth << "x"
                 << sceneResHeight << " proxy_verts="
                 << scene->proxies()->proxy().vertices().size() << std::endl;
    };

    bool pathOverride = myArgs.dataset_path.isInit();
    if (pathOverride) {
        resetScene();
    }

    std::string currentName;

    emit viewerLog(QString("[SIBR] Remote Gaussian Viewer ready | %1:%2 | "
                           "resolution %3x%4")
                           .arg(QString::fromStdString(myArgs.ip.get()))
                           .arg(myArgs.port.get())
                           .arg(rendering_width)
                           .arg(rendering_height));
    if (pathOverride)
        emit viewerLog(QString("[SIBR] Dataset: %1")
                               .arg(QString::fromStdString(
                                       myArgs.dataset_path.get())));

    while (window.isOpened() && !m_stopRequested.load()) {
        if (!pathOverride && !remoteView->sceneName().empty() &&
            remoteView->sceneName() != currentName) {
            currentName = remoteView->sceneName();
            myArgs.dataset_path = currentName;
            resetScene();
            emit viewerLog(QString("[SIBR] Scene loaded: %1")
                                   .arg(QString::fromStdString(currentName)));
        }

        Input::poll();
        window.makeContextCurrent();
        if (Input::global().key().isPressed(Key::Escape)) window.close();
        mvm.onUpdate(Input::global());
        mvm.onRender(window);
        window.swapBuffer();
    }

    if (scene) {
        auto meshFile = QString::fromStdString(scene->data()->meshPath());
        if (!meshFile.isEmpty())
            emit viewerResultReady(meshFile, tr("Remote Scene Point Cloud"));
    }

    return EXIT_SUCCESS;
#else
    emit viewerError("[SIBR] Remote viewer not available");
    return 1;
#endif
}

// Embedded tool entry points (compiled from dataset_tools/preprocess/*/main.cpp
// with SIBR_TOOL_EMBEDDED defined).
extern "C" {
int sibr_tool_alignMeshes(int, char**);
int sibr_tool_cameraConverter(int, char**);
int sibr_tool_clippingPlanes(int, const char**);
int sibr_tool_cropFromCenter(int, const char**);
int sibr_tool_distordCrop(int, const char* const*);
int sibr_tool_nvmToSIBR(int, const char**);
int sibr_tool_prepareColmap4Sibr(int, const char**);
int sibr_tool_textureMesh(int, char**);
int sibr_tool_tonemapper(int, char**);
int sibr_tool_unwrapMesh(int, char**);
}

int SIBRViewerThread::runDatasetTool() {
    emit viewerLog(
            QString("[SIBR] Running tool %1 in-process ...").arg(m_toolName));

    std::vector<std::string> storage;
    storage.push_back(m_toolName.toStdString());
    for (const auto& a : m_toolArgs) storage.push_back(a.toStdString());
    int argc = static_cast<int>(storage.size());

    std::vector<const char*> cargv(argc);
    for (int i = 0; i < argc; ++i) cargv[i] = storage[i].c_str();

    using CharPP = char**;
    using ConstCharPP = const char**;
    using ConstCharCPP = const char* const*;

    CharPP argv_m = const_cast<CharPP>(cargv.data());
    ConstCharPP argv_c = cargv.data();

    struct ToolEntry {
        const char* name;
        std::function<int()> fn;
    };
    ToolEntry table[] = {
            {"alignMeshes",
             [&] { return sibr_tool_alignMeshes(argc, argv_m); }},
            {"cameraConverter",
             [&] { return sibr_tool_cameraConverter(argc, argv_m); }},
            {"clippingPlanes",
             [&] { return sibr_tool_clippingPlanes(argc, argv_c); }},
            {"cropFromCenter",
             [&] { return sibr_tool_cropFromCenter(argc, argv_c); }},
            {"distordCrop",
             [&] {
                 return sibr_tool_distordCrop(
                         argc, static_cast<ConstCharCPP>(argv_c));
             }},
            {"nvmToSIBR", [&] { return sibr_tool_nvmToSIBR(argc, argv_c); }},
            {"prepareColmap4Sibr",
             [&] { return sibr_tool_prepareColmap4Sibr(argc, argv_c); }},
            {"textureMesh",
             [&] { return sibr_tool_textureMesh(argc, argv_m); }},
            {"tonemapper", [&] { return sibr_tool_tonemapper(argc, argv_m); }},
            {"unwrapMesh", [&] { return sibr_tool_unwrapMesh(argc, argv_m); }},
    };

    auto name = m_toolName.toStdString();
    for (auto& entry : table) {
        if (name == entry.name) {
            int rc = entry.fn();
            emit viewerLog(QString("[SIBR] Tool %1 finished (exit=%2)")
                                   .arg(m_toolName)
                                   .arg(rc));
            return rc;
        }
    }

    emit viewerError(QString("[SIBR] Unknown tool: %1").arg(m_toolName));
    return 1;
}
