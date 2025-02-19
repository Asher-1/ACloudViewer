// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "io/rpc/RemoteFunctions.h"

#include <Logging.h>
#include <ecvHObjectCaster.h>

#include <Eigen/Geometry>
#include <zmq.hpp>

#include "core/Dispatch.h"
#include "io/rpc/Connection.h"
#include "io/rpc/MessageUtils.h"
#include "io/rpc/Messages.h"

using namespace cloudViewer::utility;

namespace cloudViewer {
namespace io {
namespace rpc {

bool SetPointCloud(const ccPointCloud& pcd,
                   const std::string& path,
                   int time,
                   const std::string& layer,
                   std::shared_ptr<ConnectionBase> connection) {
    // TODO use SetMeshData here after switching to the new PointCloud class.
    if (!pcd.hasPoints()) {
        LogInfo("SetMeshData: point cloud is empty");
        return false;
    }

    messages::SetMeshData msg;
    msg.path = path;
    msg.time = time;
    msg.layer = layer;

    msg.data.vertices = messages::Array::FromPtr(
            (PointCoordinateType*)pcd.getPoints().data(),
            {int64_t(pcd.size()), 3});
    if (pcd.hasNormals()) {
        std::vector<CCVector3*> normals = pcd.getPointNormalsPtr();
        msg.data.vertex_attributes["normals"] = messages::Array::FromPtr(
                (PointCoordinateType*)(*normals.data()),
                {int64_t(normals.size()), 3});
    }
    if (pcd.hasColors()) {
        msg.data.vertex_attributes["colors"] = messages::Array::FromPtr(
                (ColorCompType*)pcd.getPointColors().data(),
                {int64_t(pcd.getPointColors().size()), 3});
    }

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

bool SetTriangleMesh(const ccMesh& mesh,
                     const std::string& path,
                     int time,
                     const std::string& layer,
                     std::shared_ptr<ConnectionBase> connection) {
    // TODO use SetMeshData here after switching to the new TriangleMesh class.
    if (!mesh.hasTriangles()) {
        LogInfo("SetMeshData: triangle mesh is empty");
        return false;
    }

    messages::SetMeshData msg;
    msg.path = path;
    msg.time = time;
    msg.layer = layer;

    if (!mesh.getAssociatedCloud()) {
        LogInfo("SetMeshData: triangle vertices is empty");
        return false;
    }

    ccPointCloud& associatedCloud =
            *ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());

    msg.data.vertices = messages::Array::FromPtr(
            (PointCoordinateType*)associatedCloud.getPoints().data(),
            {int64_t(associatedCloud.size()), 3});

    msg.data.faces = messages::Array::FromPtr(
            (unsigned*)(*mesh.getTrianglesPtr()).data(),
            {int64_t((*mesh.getTrianglesPtr()).size()), 3});
    if (mesh.hasNormals()) {
        std::vector<CCVector3*> normals = associatedCloud.getPointNormalsPtr();
        msg.data.vertex_attributes["normals"] = messages::Array::FromPtr(
                (PointCoordinateType*)(*normals.data()),
                {int64_t(normals.size()), 3});
    }
    if (mesh.hasColors()) {
        msg.data.vertex_attributes["colors"] = messages::Array::FromPtr(
                (ColorCompType*)associatedCloud.getPointColors().data(),
                {int64_t(associatedCloud.getPointColors().size()), 3});
    }
    if (mesh.hasTriNormals()) {
        std::vector<CCVector3*> triNormals = mesh.getTriangleNormalsPtr();
        msg.data.face_attributes["normals"] = messages::Array::FromPtr(
                (PointCoordinateType*)(*triNormals.data()),
                {int64_t(triNormals.size()), 3});
    }
    if (mesh.hasTriangleUvs()) {
        msg.data.face_attributes["uvs"] = messages::Array::FromPtr(
                (double*)mesh.triangle_uvs_.data(),
                {int64_t(mesh.triangle_uvs_.size()), 2});
    }
    if (mesh.hasTextures()) {
        int tex_id = 0;
        for (const auto& image : mesh.textures_) {
            if (!image.isEmpty()) {
                std::vector<int64_t> shape(
                        {image.height_, image.width_, image.num_of_channels_});
                if (image.bytes_per_channel_ == sizeof(uint8_t)) {
                    msg.data.textures[std::to_string(tex_id)] =
                            messages::Array::FromPtr(
                                    (uint8_t*)image.data_.data(), shape);
                } else if (image.bytes_per_channel_ == sizeof(float)) {
                    msg.data.textures[std::to_string(tex_id)] =
                            messages::Array::FromPtr((float*)image.data_.data(),
                                                     shape);
                } else if (image.bytes_per_channel_ == sizeof(double)) {
                    msg.data.textures[std::to_string(tex_id)] =
                            messages::Array::FromPtr(
                                    (double*)image.data_.data(), shape);
                }
            }
            ++tex_id;
        }
    }

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

bool SetMeshData(const core::Tensor& vertices,
                 const std::string& path,
                 int time,
                 const std::string& layer,
                 const std::map<std::string, core::Tensor>& vertex_attributes,
                 const core::Tensor& faces,
                 const std::map<std::string, core::Tensor>& face_attributes,
                 const core::Tensor& lines,
                 const std::map<std::string, core::Tensor>& line_attributes,
                 const std::map<std::string, core::Tensor>& textures,
                 std::shared_ptr<ConnectionBase> connection) {
    if (vertices.NumElements() == 0) {
        LogInfo("SetMeshData: vertices Tensor is empty");
        return false;
    }
    if (vertices.NumDims() != 2) {
        LogInfo("SetMeshData: vertices ndim must be 2 but is {}",
                vertices.NumDims());
        return false;
    }
    if (vertices.GetDtype() != core::Float32 &&
        vertices.GetDtype() != core::Float64) {
        LogError(
                "SetMeshData: vertices must have dtype Float32 or Float64 but "
                "is {}",
                vertices.GetDtype().ToString());
    }

    auto PrepareTensor = [](const core::Tensor& a) {
        return a.To(core::Device("CPU:0")).Contiguous();
    };

    auto CreateArray = [](const core::Tensor& a) {
        return DISPATCH_DTYPE_TO_TEMPLATE(a.GetDtype(), [&]() {
            return messages::Array::FromPtr(
                    (scalar_t*)a.GetDataPtr(),
                    static_cast<std::vector<int64_t>>(a.GetShape()));
        });
    };

    messages::SetMeshData msg;
    msg.path = path;
    msg.time = time;
    msg.layer = layer;

    const core::Tensor vertices_ok = PrepareTensor(vertices);
    msg.data.vertices = CreateArray(vertices_ok);

    // store tensors in this vector to make sure the memory blob is alive
    // for tensors where a deep copy was necessary.
    std::vector<core::Tensor> tensor_cache;
    for (const auto& item : vertex_attributes) {
        tensor_cache.push_back(PrepareTensor(item.second));
        const core::Tensor& tensor = tensor_cache.back();
        if (tensor.NumDims() >= 1 &&
            tensor.GetShape()[0] == vertices.GetShape()[0]) {
            msg.data.vertex_attributes[item.first] = CreateArray(tensor);
        } else {
            LogError("SetMeshData: Attribute {} has incompatible shape {}",
                     item.first, tensor.GetShape().ToString());
        }
    }

    if (faces.NumElements()) {
        if (faces.GetDtype() != core::Int32 &&
            faces.GetDtype() != core::Int64) {
            LogError(
                    "SetMeshData: faces must have dtype Int32 or Int64 but "
                    "is {}",
                    vertices.GetDtype().ToString());
        } else if (faces.NumDims() != 2) {
            LogError("SetMeshData: faces must have rank 2 but is {}",
                     faces.NumDims());
        } else if (faces.GetShape()[1] < 3) {
            LogError("SetMeshData: last dim of faces must be >=3 but is {}",
                     faces.GetShape()[1]);
        } else {
            tensor_cache.push_back(PrepareTensor(faces));
            const core::Tensor& faces_ok = tensor_cache.back();
            msg.data.faces = CreateArray(faces_ok);

            for (const auto& item : face_attributes) {
                tensor_cache.push_back(PrepareTensor(item.second));
                const core::Tensor& tensor = tensor_cache.back();
                if (tensor.NumDims() >= 1 &&
                    tensor.GetShape()[0] == faces.GetShape()[0]) {
                    msg.data.face_attributes[item.first] = CreateArray(tensor);
                } else {
                    LogError(
                            "SetMeshData: Attribute {} has incompatible shape "
                            "{}",
                            item.first, tensor.GetShape().ToString());
                }
            }
        }
    }

    if (lines.NumElements()) {
        if (lines.GetDtype() != core::Int32 &&
            lines.GetDtype() != core::Int64) {
            LogError(
                    "SetMeshData: lines must have dtype Int32 or Int64 but "
                    "is {}",
                    vertices.GetDtype().ToString());
        } else if (lines.NumDims() != 2) {
            LogError("SetMeshData: lines must have rank 2 but is {}",
                     lines.NumDims());
        } else if (lines.GetShape()[1] < 2) {
            LogError("SetMeshData: last dim of lines must be >=2 but is {}",
                     lines.GetShape()[1]);
        } else {
            tensor_cache.push_back(PrepareTensor(lines));
            const core::Tensor& lines_ok = tensor_cache.back();
            msg.data.lines = CreateArray(lines_ok);

            for (const auto& item : line_attributes) {
                tensor_cache.push_back(PrepareTensor(item.second));
                const core::Tensor& tensor = tensor_cache.back();
                if (tensor.NumDims() >= 1 &&
                    tensor.GetShape()[0] == lines.GetShape()[0]) {
                    msg.data.line_attributes[item.first] = CreateArray(tensor);
                } else {
                    LogError(
                            "SetMeshData: Attribute {} has incompatible shape "
                            "{}",
                            item.first, tensor.GetShape().ToString());
                }
            }
        }
    }

    for (const auto& item : textures) {
        tensor_cache.push_back(PrepareTensor(item.second));
        const core::Tensor& tensor = tensor_cache.back();
        if (tensor.NumElements()) {
            msg.data.textures[item.first] = CreateArray(tensor);
        } else {
            LogError("SetMeshData: Texture {} is empty", item.first);
        }
    }

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

bool SetLegacyCamera(const camera::PinholeCameraParameters& camera,
                     const std::string& path,
                     int time,
                     const std::string& layer,
                     std::shared_ptr<ConnectionBase> connection) {
    messages::SetCameraData msg;
    msg.path = path;
    msg.time = time;
    msg.layer = layer;

    // convert extrinsics
    Eigen::Matrix3d R = camera.extrinsic_.block<3, 3>(0, 0);
    Eigen::Vector3d t = camera.extrinsic_.block<3, 1>(0, 3);
    Eigen::Quaterniond q(R);
    msg.data.R[0] = q.x();
    msg.data.R[1] = q.y();
    msg.data.R[2] = q.z();
    msg.data.R[3] = q.w();

    msg.data.t[0] = t[0];
    msg.data.t[1] = t[1];
    msg.data.t[2] = t[2];

    // convert intrinsics
    if (camera.intrinsic_.IsValid()) {
        msg.data.width = camera.intrinsic_.width_;
        msg.data.height = camera.intrinsic_.height_;
        msg.data.intrinsic_model = "PINHOLE";
        msg.data.intrinsic_parameters.resize(4);
        msg.data.intrinsic_parameters[0] =
                camera.intrinsic_.intrinsic_matrix_(0, 0);
        msg.data.intrinsic_parameters[1] =
                camera.intrinsic_.intrinsic_matrix_(1, 1);
        msg.data.intrinsic_parameters[2] =
                camera.intrinsic_.intrinsic_matrix_(0, 2);
        msg.data.intrinsic_parameters[3] =
                camera.intrinsic_.intrinsic_matrix_(1, 2);
        if (camera.intrinsic_.GetSkew() != 0.0) {
            LogWarning(
                    "SetLegacyCamera: Nonzero skew parameteer in "
                    "PinholeCameraParameters will be ignored!");
        }
    }

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

bool SetTime(int time, std::shared_ptr<ConnectionBase> connection) {
    messages::SetTime msg;
    msg.time = time;

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

bool SetActiveCamera(const std::string& path,
                     std::shared_ptr<ConnectionBase> connection) {
    messages::SetActiveCamera msg;
    msg.path = path;

    msgpack::sbuffer sbuf;
    messages::Request request{msg.MsgId()};
    msgpack::pack(sbuf, request);
    msgpack::pack(sbuf, msg);

    zmq::message_t send_msg(sbuf.data(), sbuf.size());
    if (!connection) {
        connection = std::shared_ptr<Connection>(new Connection());
    }
    auto reply = connection->Send(send_msg);
    return ReplyIsOKStatus(*reply);
}

}  // namespace rpc
}  // namespace io
}  // namespace cloudViewer
