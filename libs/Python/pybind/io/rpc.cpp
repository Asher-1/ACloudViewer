// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 asher-1.github.io
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

#include <memory>

#include "io/rpc/BufferConnection.h"
#include "io/rpc/Connection.h"
#include "io/rpc/DummyReceiver.h"
#include "io/rpc/RemoteFunctions.h"
#include "io/rpc/ZMQContext.h"
#include "io/rpc/Messages.h"

#include "cloudViewer/core/Tensor.h"
#include "cloudViewer/t/geometry/PointCloud.h"
#include "cloudViewer/t/geometry/TriangleMesh.h"
#include "cloudViewer/t/geometry/LineSet.h"
#include "pybind/cloudViewer_pybind.h"
#include "pybind/core/tensor_type_caster.h"
#include "pybind/docstring.h"

namespace cloudViewer {
namespace io {

void pybind_rpc(py::module& m_io) {
    py::module m = m_io.def_submodule("rpc");

    // this is to cleanly shutdown the zeromq context on windows.
    auto atexit = py::module::import("atexit");
    atexit.attr("register")(
            py::cpp_function([]() { rpc::DestroyZMQContext(); }));

    py::class_<rpc::ConnectionBase, std::shared_ptr<rpc::ConnectionBase>>(
            m, "_ConnectionBase");

    py::class_<rpc::Connection, std::shared_ptr<rpc::Connection>,
               rpc::ConnectionBase>(m, "Connection")
            .def(py::init([](std::string address, int connect_timeout,
                             int timeout) {
                     return std::make_shared<rpc::Connection>(
                             address, connect_timeout, timeout);
                 }),
                 "Creates a connection object",
                 "address"_a = "tcp://127.0.0.1:51454",
                 "connect_timeout"_a = 5000, "timeout"_a = 10000);

    py::class_<rpc::BufferConnection, std::shared_ptr<rpc::BufferConnection>,
               rpc::ConnectionBase>(m, "BufferConnection")
            .def(py::init<>())
            .def(
                    "get_buffer",
                    [](const rpc::BufferConnection& self) {
                        return py::bytes(self.buffer().str());
                    },
                    "Returns a copy of the buffer.");

    py::class_<rpc::DummyReceiver, std::shared_ptr<rpc::DummyReceiver>>(
            m, "_DummyReceiver",
            "Dummy receiver for the server side receiving requests from a "
            "client.")
            .def(py::init([](const std::string& address, int timeout) {
                     return std::make_shared<rpc::DummyReceiver>(address,
                                                                 timeout);
                 }),
                 "Creates the receiver object which can be used for testing "
                 "connections.",
                 "address"_a = "tcp://127.0.0.1:51454", "timeout"_a = 10000)
            .def("start", &rpc::DummyReceiver::Start,
                 "Starts the receiver mainloop in a new thread.")
            .def("stop", &rpc::DummyReceiver::Stop,
                 "Stops the receiver mainloop and joins the thread. This "
                 "function blocks until the mainloop is done with processing "
                 "messages that have already been received.");

    m.def("destroy_zmq_context", &rpc::DestroyZMQContext,
          "Destroys the ZMQ context.");

    m.def("set_point_cloud", &rpc::SetPointCloud, "pcd"_a, "path"_a = "",
          "time"_a = 0, "layer"_a = "",
          "connection"_a = std::shared_ptr<rpc::Connection>(),
          "Sends a point cloud message to a viewer.");
    docstring::FunctionDocInject(
            m, "set_point_cloud",
            {
                    {"pcd", "Point cloud object."},
                    {"path", "A path descriptor, e.g., 'mygroup/points'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m.def("set_triangle_mesh", &rpc::SetTriangleMesh, "mesh"_a, "path"_a = "",
          "time"_a = 0, "layer"_a = "",
          "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
          "Sends a point cloud message to a viewer.");
    docstring::FunctionDocInject(
            m, "set_triangle_mesh",
            {
                    {"mesh", "The TriangleMesh object."},
                    {"path", "A path descriptor, e.g., 'mygroup/mesh'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m.def("set_mesh_data", &rpc::SetMeshData, "vertices"_a, "path"_a = "",
          "time"_a = 0, "layer"_a = "",
          "vertex_attributes"_a = std::map<std::string, core::Tensor>(),
          "faces"_a = core::Tensor({0}, core::Int32),
          "face_attributes"_a = std::map<std::string, core::Tensor>(),
          "lines"_a = core::Tensor({0}, core::Int32),
          "line_attributes"_a = std::map<std::string, core::Tensor>(),
          "textures"_a = std::map<std::string, core::Tensor>(),
          "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
          "Sends a set_mesh_data message.");
    docstring::FunctionDocInject(
            m, "set_mesh_data",
            {
                    {"vertices", "Tensor defining the vertices."},
                    {"path", "A path descriptor, e.g., 'mygroup/points'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"vertex_attributes",
                     "dict of Tensors with vertex attributes."},
                    {"faces", "Tensor defining the faces with vertex indices."},
                    {"face_attributes",
                     "dict of Tensors with face attributes."},
                    {"lines", "Tensor defining lines with vertex indices."},
                    {"line_attributes",
                     "dict of Tensors with line attributes."},
                    {"textures", "dict of Tensors with textures."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m.def("set_legacy_camera", &rpc::SetLegacyCamera, "camera"_a, "path"_a = "",
          "time"_a = 0, "layer"_a = "",
          "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
          "Sends a PinholeCameraParameters object.");
    docstring::FunctionDocInject(
            m, "set_legacy_camera",
            {
                    {"path", "A path descriptor, e.g., 'mygroup/camera'."},
                    {"time", "The time associated with this data."},
                    {"layer", "The layer associated with this data."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m.def("set_time", &rpc::SetTime, "time"_a,
          "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
          "Sets the time in the external visualizer.");
    docstring::FunctionDocInject(
            m, "set_time",
            {
                    {"time", "The time value to set."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    m.def("set_active_camera", &rpc::SetActiveCamera, "path"_a,
          "connection"_a = std::shared_ptr<rpc::ConnectionBase>(),
          "Sets the object with the specified path as the active camera.");
    docstring::FunctionDocInject(
            m, "set_active_camera",
            {
                    {"path", "A path descriptor, e.g., 'mygroup/camera'."},
                    {"connection",
                     "A Connection object. Use None to automatically create "
                     "the connection."},
            });

    // Convert a serialized SetMeshData msgpack buffer into meta (tag, step) and geometry
    // for TensorBoard plugin reading path. Exposed as data_buffer_to_meta_geometry(buf: bytes)
    m.def(
            "data_buffer_to_meta_geometry",
            [](py::bytes py_buf) {
                // Convert Python bytes to a contiguous buffer
                std::string buf = py_buf;
                msgpack::object_handle oh = msgpack::unpack(buf.data(), buf.size());
                // Expect a Reply followed by Status OK and then SetMeshData
                msgpack::object obj = oh.get();
                // The buffer is packed as [Reply, Status, SetMeshData]
                // We walk it as an array to access elements by index safely.
                if (obj.type != msgpack::type::ARRAY || obj.via.array.size < 3) {
                    return py::make_tuple("", 0, py::none());
                }
                // Element 2 should be SetMeshData
                messages::SetMeshData mesh_msg{};
                try {
                    obj.via.array.ptr[2].convert(mesh_msg);
                } catch (const std::exception&) {
                    return py::make_tuple("", 0, py::none());
                }

                // Construct Tensor-based geometry from MeshData
                using cloudViewer::core::Tensor;
                using cloudViewer::core::Device;
                using cloudViewer::core::Float32;
                using cloudViewer::core::Int32;

                auto to_tensor = [](const messages::Array& arr, cloudViewer::core::Dtype dtype) {
                    return Tensor(arr.Ptr<uint8_t>(),
                                  {arr.shape.begin(), arr.shape.end()},
                                  dtype, Device("CPU:0")).Contiguous();
                };

                // Primary vertices are required; infer geometry type by presence of faces/lines
                Tensor vertices;
                if (!mesh_msg.data.vertices.CheckNonEmpty()) {
                    return py::make_tuple("", 0, py::none());
                }
                // dtype: try float32 by default
                vertices = to_tensor(mesh_msg.data.vertices, Float32);

                // Collect attributes
                std::map<std::string, Tensor> vattrs;
                for (const auto& kv : mesh_msg.data.vertex_attributes) {
                    vattrs.emplace(kv.first, to_tensor(kv.second, Float32));
                }

                // Triangle mesh path
                if (mesh_msg.data.faces.CheckNonEmpty()) {
                    Tensor faces_t = to_tensor(mesh_msg.data.faces, Int32);
                    cloudViewer::t::geometry::TriangleMesh tmesh;
                    tmesh.SetVertexPositions(vertices);
                    if (faces_t.NumElements()) {
                        tmesh.SetTriangleIndices(faces_t);
                    }
                    for (const auto& kv : vattrs) {
                        if (kv.first == "colors") {
                            tmesh.SetVertexColors(kv.second);
                        } else if (kv.first == "normals") {
                            tmesh.SetVertexNormals(kv.second);
                        } else {
                            tmesh.SetVertexAttr(kv.first, kv.second);
                        }
                    }
                    return py::make_tuple(mesh_msg.path, mesh_msg.time, tmesh);
                }

                // LineSet path
                if (mesh_msg.data.lines.CheckNonEmpty()) {
                    Tensor lines_t = to_tensor(mesh_msg.data.lines, Int32);
                    cloudViewer::t::geometry::LineSet lset;
                    lset.SetPointPositions(vertices);
                    if (lines_t.NumElements()) {
                        lset.SetLineIndices(lines_t);
                    }
                    for (const auto& kv : vattrs) {
                        if (kv.first == "colors") {
                            lset.SetLineColors(kv.second);
                        } else {
                            lset.SetPointAttr(kv.first, kv.second);
                        }
                    }
                    return py::make_tuple(mesh_msg.path, mesh_msg.time, lset);
                }

                // Default to PointCloud
                cloudViewer::t::geometry::PointCloud pcd(vertices);
                for (const auto& kv : vattrs) {
                    if (kv.first == "colors") {
                        pcd.SetPointColors(kv.second);
                    } else if (kv.first == "normals") {
                        pcd.SetPointNormals(kv.second);
                    } else {
                        pcd.SetPointAttr(kv.first, kv.second);
                    }
                }
                return py::make_tuple(mesh_msg.path, mesh_msg.time, pcd);
            },
            "Parse a serialized SetMeshData msgpack buffer into (path, time, geometry)."
    );
}

}  // namespace io
}  // namespace cloudViewer
