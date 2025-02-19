// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
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

#pragma once

#include <Eigen/Core>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

class ccHObject;

namespace cloudViewer {
namespace visualization {

class Visualizer;

/// \brief Function to draw a list of geometry objects
///
/// The convenient function of drawing something
/// This function is a wrapper that calls the core functions of Visualizer.
/// This function MUST be called from the main thread. It blocks the main thread
/// until the window is closed.
///
/// \param geometry_ptrs List of geometries to be visualized.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
/// \param point_show_normal visualize point normals if set to true.
/// \param mesh_show_wireframe visualize mesh wireframe if set to true.
/// \param mesh_show_back_face visualize also the back face of the mesh
/// triangles.
/// \param lookat The lookat vector of the camera.
/// \param up The up vector of the camera.
/// \param front The front vector of the camera.
/// \param zoom The zoom of the camera.
bool DrawGeometries(
	const std::vector<std::shared_ptr<const ccHObject>> &geometry_ptrs,
	const std::string &window_name = "cloudViewer",
	int width = 640,
	int height = 480,
	int left = 50,
	int top = 50,
	bool point_show_normal = false,
	bool mesh_show_wireframe = false,
	bool mesh_show_back_face = false,
	Eigen::Vector3d *lookat = nullptr,
	Eigen::Vector3d *up = nullptr,
	Eigen::Vector3d *front = nullptr,
	double *zoom = nullptr);

/// \brief Function to draw a list of geometry objects with a GUI that
/// supports animation.
///
/// \param geometry_ptrs List of geometries to be visualized.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
/// \param json_filename Camera trajectory json file path
/// for custom animation.
bool DrawGeometriesWithCustomAnimation(
        const std::vector<std::shared_ptr<const ccHObject>>
                &geometry_ptrs,
        const std::string &window_name = "CloudViewer",
        int width = 640,
        int height = 480,
        int left = 50,
        int top = 50,
        const std::string &json_filename = "");

/// \brief Function to draw a list of geometry objects with a
/// customized animation callback function.
///
/// \param geometry_ptrs List of geometries to be visualized.
/// \param callback_func Call back function to be triggered at a key press
/// event.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
bool DrawGeometriesWithAnimationCallback(
        const std::vector<std::shared_ptr<const ccHObject>>
                &geometry_ptrs,
        std::function<bool(Visualizer *)> callback_func,
        const std::string &window_name = "CloudViewer",
        int width = 640,
        int height = 480,
        int left = 50,
        int top = 50);

/// \brief Function to draw a list of geometry.
///
/// Geometry objects with a customized key-callback mapping
///
/// \param geometry_ptrs List of geometries to be visualized.
/// \param key_to_callback Map of key to call back functions.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
bool DrawGeometriesWithKeyCallbacks(
        const std::vector<std::shared_ptr<const ccHObject>>
                &geometry_ptrs,
        const std::map<int, std::function<bool(Visualizer *)>> &key_to_callback,
        const std::string &window_name = "CloudViewer",
        int width = 640,
        int height = 480,
        int left = 50,
        int top = 50);

/// \brief Function to draw a list of geometry.
///
/// Geometry providing user interaction.
///
/// \param geometry_ptrs List of geometries to be visualized.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
bool DrawGeometriesWithEditing(
        const std::vector<std::shared_ptr<const ccHObject>>
                &geometry_ptrs,
        const std::string &window_name = "CloudViewer",
        int width = 640,
        int height = 480,
        int left = 50,
        int top = 50);

bool DrawGeometriesWithVertexSelection(
        const std::vector<std::shared_ptr<const ccHObject>>
                &geometry_ptrs,
        const std::string &window_name = "CloudViewer",
        int width = 640,
        int height = 480,
        int left = 50,
        int top = 50);

}  // namespace visualization
}  // namespace cloudViewer
