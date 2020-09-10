// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

// Note: do not modify CloudViewer.h, modify CloudViewer.h.in instead
#include "Console.h"
#include "CVTools.h"
#include "FileSystem.h"
#include "Eigen.h"
#include "Helper.h"
#include "Timer.h"
#include "CVGeom.h"
#include "CVLog.h"
#include "IJsonConvertibleIO.h"
#include "Visualization/Visualizer/Visualizer.h" // must include first

#include "ecvMesh.h"
#include "LineSet.h"
#include "Octree.h"
#include "Image.h"
#include "RGBDImage.h"
#include "ecvFeature.h"
#include "VoxelGrid.h"
#include "ecvKDTreeFlann.h"
#include "ecvPointCloud.h"

#include "ImageIO.h"
#include "FeatureIO.h"
#include "VoxelGridIO.h"
#include "PointCloudIO.h"
#include "TriangleMeshIO.h"
#include "IJsonConvertibleIO.h"
#include "Integration/ScalableTSDFVolume.h"
#include "Integration/TSDFVolume.h"
#include "Integration/UniformTSDFVolume.h"
#include "Camera/PinholeCameraIntrinsic.h"
#include "Camera/PinholeCameraParameters.h"
#include "Camera/PinholeCameraTrajectory.h"
#include "Camera/PinholeCameraTrajectoryIO.h"
#include "ColorMap/ColorMapOptimization.h"
#include "ColorMap/ImageWarpingField.h"
#include "Registration/PoseGraph.h"
#include "Registration/PoseGraphIO.h"
#include "Registration/Registration.h"
#include "Registration/TransformationEstimation.h"
#include "Odometry/Odometry.h"
#include "Visualization/Utility/DrawGeometry.h"
#include "Visualization/Utility/SelectionPolygon.h"
#include "Visualization/Utility/SelectionPolygonVolume.h"
#include "Visualization/Visualizer/ViewControl.h"
#include "Visualization/Visualizer/ViewControlWithCustomAnimation.h"
#include "Visualization/Visualizer/ViewControlWithEditing.h"
#include "Visualization/Visualizer/VisualizerWithCustomAnimation.h"
#include "Visualization/Visualizer/VisualizerWithEditing.h"
#include "Visualization/Visualizer/VisualizerWithKeyCallback.h"
#include "Visualization/Visualizer/VisualizerWithVertexSelection.h"
