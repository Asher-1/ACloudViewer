// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/visualization/visualizer/MessageProcessor.h"

#include <zmq.hpp>

#include "cloudViewer/geometry/PointCloud.h"
#include "cloudViewer/geometry/TriangleMesh.h"
#include "cloudViewer/io/rpc/MessageUtils.h"
#include "cloudViewer/io/rpc/Messages.h"
#include "cloudViewer/visualization/gui/Application.h"
#include "cloudViewer/visualization/gui/Window.h"
#include "cloudViewer/visualization/rendering/Material.h"

using namespace cloudViewer::io::rpc;
using namespace cloudViewer::utility;

namespace cloudViewer {
namespace visualization {

std::shared_ptr<zmq::message_t> MessageProcessor::ProcessMessage(
        const messages::Request& req,
        const messages::SetMeshData& msg,
        const msgpack::object_handle& obj) {
    std::string errstr(":");
    if (!msg.data.CheckMessage(errstr)) {
        auto status_err = messages::Status::ErrorProcessingMessage();
        status_err.str += errstr;
        msgpack::sbuffer sbuf;
        messages::Reply reply{status_err.MsgId()};
        msgpack::pack(sbuf, reply);
        msgpack::pack(sbuf, status_err);
        return std::shared_ptr<zmq::message_t>(
                new zmq::message_t(sbuf.data(), sbuf.size()));
    }

    if (msg.data.faces.CheckNonEmpty()) {
        // create a TriangleMesh
        ccPointCloud* baseVertices = new ccPointCloud("vertices");
        assert(baseVertices);
        baseVertices->setEnabled(false);
        // DGM: no need to lock it as it is only used by one mesh!
        baseVertices->setLocked(false);
        auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);
        mesh->addChild(baseVertices);

        errstr = "";
        if (!msg.data.vertices.CheckType(
                    {messages::TypeStr<float>(), messages::TypeStr<double>()},
                    errstr)) {
            errstr = "Ignoring vertices. vertices have wrong data type:" +
                     errstr;
            LogInfo(errstr.c_str());
        } else {
            baseVertices->reserveThePointsTable(
                    static_cast<unsigned>(msg.data.vertices.shape[0]));
            if (msg.data.vertices.type == messages::TypeStr<float>()) {
                const float* ptr = msg.data.vertices.Ptr<float>();
                for (int64_t i = 0; i < msg.data.vertices.shape[0]; ++i) {
                    baseVertices->addPoint(CCVector3::fromArray(ptr));
                    ptr += 3;
                }
            }
            if (msg.data.vertices.type == messages::TypeStr<double>()) {
                const double* ptr = msg.data.vertices.Ptr<double>();
                for (int64_t i = 0; i < msg.data.vertices.shape[0]; ++i) {
                    baseVertices->addPoint(CCVector3::fromArray(ptr));
                    ptr += 3;
                }
            }
        }

        errstr = "";
        if (msg.data.vertex_attributes.count("normals")) {
            const auto& attr_arr = msg.data.vertex_attributes.at("normals");
            if (!attr_arr.CheckType({messages::TypeStr<float>(),
                                     messages::TypeStr<double>()},
                                    errstr)) {
                errstr = "Ignoring normals. normals have wrong data type:" +
                         errstr;
                LogInfo(errstr.c_str());
            } else if (!attr_arr.CheckShape({-1, 3}, errstr)) {
                errstr = "Ignoring normals. normals have wrong shape:" + errstr;
                LogInfo(errstr.c_str());
            } else {
                if (!baseVertices->hasNormals()) {
                    if (!baseVertices->reserveTheNormsTable()) {
                        errstr =
                                "Ignoring normals. not enough memory:" + errstr;
                        LogInfo(errstr.c_str());
                    }
                }

                if (attr_arr.type == messages::TypeStr<float>()) {
                    const float* ptr = attr_arr.Ptr<float>();
                    for (int64_t i = 0; i < attr_arr.shape[0]; ++i) {
                        baseVertices->addNorm(CCVector3::fromArray(ptr));
                        ptr += 3;
                    }
                }
                if (attr_arr.type == messages::TypeStr<double>()) {
                    const double* ptr = attr_arr.Ptr<double>();
                    for (int64_t i = 0; i < attr_arr.shape[0]; ++i) {
                        baseVertices->addNorm(CCVector3::fromArray(ptr));
                        ptr += 3;
                    }
                }
            }
        }

        errstr = "";
        if (msg.data.vertex_attributes.count("colors")) {
            const auto& attr_arr = msg.data.vertex_attributes.at("colors");
            if (!attr_arr.CheckType({messages::TypeStr<float>(),
                                     messages::TypeStr<double>()},
                                    errstr)) {
                errstr = "Ignoring colors. colors have wrong data type:" +
                         errstr;
                LogInfo(errstr.c_str());
            } else if (!attr_arr.CheckShape({-1, 3}, errstr)) {
                errstr = "Ignoring colors. colors have wrong shape:" + errstr;
                LogInfo(errstr.c_str());
            } else {
                if (!baseVertices->hasColors()) {
                    if (!baseVertices->reserveTheRGBTable()) {
                        errstr = "Ignoring colors. not enough memory:" + errstr;
                        LogInfo(errstr.c_str());
                    }
                }

                if (attr_arr.type == messages::TypeStr<float>()) {
                    const float* ptr = attr_arr.Ptr<float>();
                    for (int64_t i = 0; i < attr_arr.shape[0]; ++i) {
                        baseVertices->addRGBColor(ecvColor::Rgb(
                                static_cast<ColorCompType>(ptr[0] * 255),
                                static_cast<ColorCompType>(ptr[1] * 255),
                                static_cast<ColorCompType>(ptr[2] * 255)));
                        ptr += 3;
                    }
                }
                if (attr_arr.type == messages::TypeStr<double>()) {
                    const double* ptr = attr_arr.Ptr<double>();
                    for (int64_t i = 0; i < attr_arr.shape[0]; ++i) {
                        baseVertices->addRGBColor(ecvColor::Rgb(
                                static_cast<ColorCompType>(ptr[0] * 255),
                                static_cast<ColorCompType>(ptr[1] * 255),
                                static_cast<ColorCompType>(ptr[2] * 255)));
                        ptr += 3;
                    }
                }
            }
        }

        errstr = "";
        if (!msg.data.faces.CheckShape({-1, 3}, errstr)) {
            errstr = "Ignoring faces. Only triangular faces are supported:" +
                     errstr;
            LogInfo(errstr.c_str());
        } else if (!msg.data.faces.CheckType({messages::TypeStr<int32_t>(),
                                              messages::TypeStr<int64_t>()},
                                             errstr)) {
            errstr = "Ignoring faces. Triangles have wrong data type:" + errstr;
            LogInfo(errstr.c_str());
        } else {
            mesh->reserve(static_cast<std::size_t>(msg.data.faces.shape[0]));
            if (msg.data.faces.type == messages::TypeStr<int32_t>()) {
                const int32_t* ptr = msg.data.faces.Ptr<int32_t>();
                for (int64_t i = 0; i < msg.data.faces.shape[0]; ++i) {
                    mesh->addTriangle(static_cast<unsigned int>(ptr[0]),
                                      static_cast<unsigned int>(ptr[1]),
                                      static_cast<unsigned int>(ptr[2]));
                    ptr += 3;
                }
            }
            if (msg.data.faces.type == messages::TypeStr<int64_t>()) {
                const int64_t* ptr = msg.data.faces.Ptr<int64_t>();
                for (int64_t i = 0; i < msg.data.faces.shape[0]; ++i) {
                    mesh->addTriangle(static_cast<unsigned int>(ptr[0]),
                                      static_cast<unsigned int>(ptr[1]),
                                      static_cast<unsigned int>(ptr[2]));
                    ptr += 3;
                }
            }
        }

        SetGeometry(mesh, msg.path, msg.time, msg.layer);
    } else {
        // create a PointCloud
        auto pcd = cloudViewer::make_shared<ccPointCloud>();
        if (!msg.data.vertices.CheckType(
                    {messages::TypeStr<float>(), messages::TypeStr<double>()},
                    errstr)) {
            errstr = "Ignoring vertices. vertices have wrong data type:" +
                     errstr;
            LogInfo(errstr.c_str());
        } else {
            pcd->reserveThePointsTable(
                    static_cast<unsigned>(msg.data.vertices.shape[0]));
            if (msg.data.vertices.type == messages::TypeStr<float>()) {
                const float* ptr = msg.data.vertices.Ptr<float>();
                for (int64_t i = 0; i < msg.data.vertices.shape[0]; ++i) {
                    pcd->addPoint(CCVector3::fromArray(ptr));
                    ptr += 3;
                }
            }
            if (msg.data.vertices.type == messages::TypeStr<double>()) {
                const double* ptr = msg.data.vertices.Ptr<double>();
                for (int64_t i = 0; i < msg.data.vertices.shape[0]; ++i) {
                    pcd->addPoint(CCVector3::fromArray(ptr));
                    ptr += 3;
                }
            }

            errstr = "";
            if (msg.data.vertex_attributes.count("normals")) {
                const auto& attr_arr = msg.data.vertex_attributes.at("normals");
                if (!attr_arr.CheckType({messages::TypeStr<float>(),
                                         messages::TypeStr<double>()},
                                        errstr)) {
                    errstr = "Ignoring normals. normals have wrong data type:" +
                             errstr;
                    LogInfo(errstr.c_str());
                } else if (!attr_arr.CheckShape({-1, 3}, errstr)) {
                    errstr = "Ignoring normals. normals have wrong shape:" +
                             errstr;
                    LogInfo(errstr.c_str());
                } else {
                    if (!pcd->hasNormals()) {
                        if (!pcd->reserveTheNormsTable()) {
                            errstr = "Ignoring normals. not enough memory:" +
                                     errstr;
                            LogInfo(errstr.c_str());
                        }
                    }

                    if (attr_arr.type == messages::TypeStr<float>()) {
                        const float* ptr = attr_arr.Ptr<float>();
                        for (int64_t i = 0; i < attr_arr.shape[0]; ++i) {
                            pcd->addNorm(CCVector3::fromArray(ptr));
                            ptr += 3;
                        }
                    }
                    if (attr_arr.type == messages::TypeStr<double>()) {
                        const double* ptr = attr_arr.Ptr<double>();
                        for (int64_t i = 0; i < attr_arr.shape[0]; ++i) {
                            pcd->addNorm(CCVector3::fromArray(ptr));
                            ptr += 3;
                        }
                    }
                }
            }

            errstr = "";
            if (msg.data.vertex_attributes.count("colors")) {
                const auto& attr_arr = msg.data.vertex_attributes.at("colors");
                if (!attr_arr.CheckType({messages::TypeStr<float>(),
                                         messages::TypeStr<double>()},
                                        errstr)) {
                    errstr = "Ignoring colors. colors have wrong data type:" +
                             errstr;
                    LogInfo(errstr.c_str());
                } else if (!attr_arr.CheckShape({-1, 3}, errstr)) {
                    errstr = "Ignoring colors. colors have wrong shape:" +
                             errstr;
                    LogInfo(errstr.c_str());
                } else {
                    if (!pcd->hasColors()) {
                        if (!pcd->reserveTheRGBTable()) {
                            errstr = "Ignoring colors. not enough memory:" +
                                     errstr;
                            LogInfo(errstr.c_str());
                        }
                    }

                    if (attr_arr.type == messages::TypeStr<float>()) {
                        const float* ptr = attr_arr.Ptr<float>();
                        for (int64_t i = 0; i < attr_arr.shape[0]; ++i) {
                            pcd->addRGBColor(ecvColor::Rgb(
                                    static_cast<ColorCompType>(ptr[0] * 255),
                                    static_cast<ColorCompType>(ptr[1] * 255),
                                    static_cast<ColorCompType>(ptr[2] * 255)));
                            ptr += 3;
                        }
                    }
                    if (attr_arr.type == messages::TypeStr<double>()) {
                        const double* ptr = attr_arr.Ptr<double>();
                        for (int64_t i = 0; i < attr_arr.shape[0]; ++i) {
                            pcd->addRGBColor(ecvColor::Rgb(
                                    static_cast<ColorCompType>(ptr[0] * 255),
                                    static_cast<ColorCompType>(ptr[1] * 255),
                                    static_cast<ColorCompType>(ptr[2] * 255)));
                            ptr += 3;
                        }
                    }
                }
            }
        }
        SetGeometry(pcd, msg.path, msg.time, msg.layer);
    }

    return CreateStatusOKMsg();
}

void MessageProcessor::SetGeometry(std::shared_ptr<ccHObject> geom,
                                   const std::string& path,
                                   int time,
                                   const std::string& layer) {
    gui::Application::GetInstance().PostToMainThread(
            window_, [this, geom, path, time, layer]() {
                on_geometry_(geom, path, time, layer);
            });
}

}  // namespace visualization
}  // namespace cloudViewer
