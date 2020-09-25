// ----------------------------------------------------------------------------
// -                        ErowCloudViewer: www.erow.cn                            -
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
#include <ecvGenericPrimitive.h>
#include <ecvPlanarEntityInterface.h>

using namespace cloudViewer;

int main(int argc, char **argv) {

    CVLib::utility::SetVerbosityLevel(CVLib::utility::VerbosityLevel::Debug);

    if (argc < 2) {
        // clang-format off
        CVLib::utility::LogInfo("Usage:");
        CVLib::utility::LogInfo("    > Primitives sphere");
		CVLib::utility::LogInfo("    > Primitives cone <file1>");
        // clang-format on
        return 1;
    }

	std::string option(argv[1]);
	std::shared_ptr<ccGenericPrimitive> mesh = nullptr;
	if (option == "plane") {
		mesh = std::make_shared<ccPlane>(2.0f, 4.0f);
	} else if (option == "sphere") {
		mesh = std::make_shared<ccSphere>(2.0f);
		mesh->setDrawingPrecision(96);
	} else if (option == "cylinder") {
		mesh = std::make_shared<ccCylinder>(2.0f, 4.0f);
	} else if (option == "box") {
		mesh = std::make_shared<ccBox>(CCVector3(2.0f, 2.0f, 2.0f));
	} else if (option == "torus") {
		mesh = std::make_shared<ccTorus>(1.0f, 1.5f);
	} else if (option == "cone") {
		mesh = std::make_shared<ccCone>(2.0f, 1.0f, 4.0f);
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
