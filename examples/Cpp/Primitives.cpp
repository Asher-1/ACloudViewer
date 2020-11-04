// ----------------------------------------------------------------------------
// -                        ErowCloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#include <Eigen/Dense>
#include <cstdio>
#include <vector>

#include "CloudViewer.h"

#include <ecvPointCloud.h>
#include <ecvMesh.h>
#include <ecvBox.h>
#include <ecvCone.h>
#include <ecvDish.h>
#include <ecvExtru.h>
#include <ecvFacet.h>
#include <ecvPlane.h>
#include <ecvSphere.h>
#include <ecvTorus.h>
#include <ecvQuadric.h>
#include <ecvCylinder.h>
#include <ecvHObjectCaster.h>
#include <ecvGenericPrimitive.h>
#include <ecvPlanarEntityInterface.h>

using namespace cloudViewer;

auto getUnions()
{
	double d = 4;
	std::shared_ptr<ccGenericPrimitive> mesh = nullptr;
	mesh = std::make_shared<ccPlane>(2.0f, 4.0f);
	mesh->setColor(ecvColor::Rgb(125, 255, 0));
	mesh->translate(Eigen::Vector3d(-d, 0.0, 0.0));

	auto box = std::make_shared<ccBox>(CCVector3(2.0f, 2.0f, 2.0f));
	box->setColor(ecvColor::Rgb(0, 0, 255));
	mesh->merge(&box->translate(Eigen::Vector3d(0.0, 0.0, 0.0)), false);

	auto sphere = std::make_shared<ccSphere>(2.0f);
	sphere->setDrawingPrecision(96);
	sphere->setColor(ecvColor::Rgb(255, 0, 0));
	mesh->merge(&sphere->translate(Eigen::Vector3d(0.0, -d, 0.0)), false);

	auto torus = std::make_shared<ccTorus>(1.0f, 1.5f);
	torus->setDrawingPrecision(96);
	torus->setColor(ecvColor::Rgb(125, 0, 255));
	mesh->merge(&torus->translate(Eigen::Vector3d(-d, -d, 0.0)), false);

	auto truncatedCone = std::make_shared<ccCone>(2.0f, 1.0f, 4.0f);
	truncatedCone->setDrawingPrecision(128);
	truncatedCone->setColor(ecvColor::Rgb(255, 0, 255));
	mesh->merge(&truncatedCone->translate(Eigen::Vector3d(d, -d, 0.0)), false);

	auto cone = std::make_shared<ccCone>(2.0f, 0.0f, 4.0f);
	cone->setDrawingPrecision(128);
	cone->setColor(ecvColor::Rgb(255, 0, 255));
	mesh->merge(&cone->translate(Eigen::Vector3d(-d, d, 0.0)), false);

	auto cylinder = std::make_shared<ccCylinder>(2.0f, 4.0f);
	cylinder->setDrawingPrecision(128);
	cylinder->setColor(ecvColor::Rgb(0, 255, 0));
	mesh->merge(&cylinder->translate(Eigen::Vector3d(d, d, 0.0)), false);

	PointCoordinateType equation[6] = { 1.0f, 1.0f, 1.0f , 1.0f , 1.0f , 1.0f };
	auto quadric = std::make_shared<ccQuadric>(	CCVector2(-1.0f, -1.0f), 
												CCVector2(1.0f, 1.0f), equation);
	quadric->setDrawingPrecision(96);
	quadric->setColor(ecvColor::Rgb(0, 255, 125));
	mesh->merge(&quadric->translate(Eigen::Vector3d(d, 0.0, 0.0)), false);

	return mesh;
}

int main(int argc, char **argv) {

    CVLib::utility::SetVerbosityLevel(CVLib::utility::VerbosityLevel::Debug);

    if (argc < 1) {
        // clang-format off
        CVLib::utility::LogInfo("Usage:");
        CVLib::utility::LogInfo("    > Primitives sphere");
		CVLib::utility::LogInfo("    > Primitives cone <file1>");
        // clang-format on
        return 1;
    }

	std::string option = "unions";
	if (argc > 1)
	{
		option = argv[1];
	}
	
	std::shared_ptr<ccGenericPrimitive> mesh = nullptr;
	if (option == "plane") {
		mesh = std::make_shared<ccPlane>(2.0f, 4.0f);
	} else if (option == "box") {
		mesh = std::make_shared<ccBox>(CCVector3(2.0f, 2.0f, 2.0f));
	} else if (option == "sphere") {
		mesh = std::make_shared<ccSphere>(2.0f);
		mesh->setDrawingPrecision(96);
	} else if (option == "torus") {
		mesh = std::make_shared<ccTorus>(1.0f, 1.5f);
		mesh->setDrawingPrecision(128);
	} else if (option == "quadric") {
		PointCoordinateType equation[6] = {1.0f, 1.0f, 1.0f , 1.0f , 1.0f , 1.0f };
		mesh = std::make_shared<ccQuadric>(	CCVector2(-1.0f, -1.0f),
											CCVector2(1.0f, 1.0f), equation);
		mesh->setDrawingPrecision(96);
		mesh->setColor(ecvColor::Rgb(0, 255, 125));
	}
	else if (option == "cone") {
		mesh = std::make_shared<ccCone>(2.0f, 1.0f, 4.0f);
		mesh->setDrawingPrecision(128);
	} else if (option == "truncated_cone") {
		mesh = std::make_shared<ccCone>(2.0f, 0.0f, 4.0f);
		mesh->setDrawingPrecision(128);
	} else if (option == "cylinder") {
		mesh = std::make_shared<ccCylinder>(2.0f, 4.0f);
		mesh->setDrawingPrecision(128);
	} else { // union modes
		mesh = getUnions();
	}

	if (mesh)
	{
		mesh->clearTriNormals();
		mesh->computeVertexNormals();
		visualization::DrawGeometries({ mesh });
		if (argc > 2)
		{
			io::WriteTriangleMesh(argv[2], *mesh, true, true);
		}
	}

    return 0;
}
