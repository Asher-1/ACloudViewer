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
#include <ecvHObjectCaster.h>

using namespace cloudViewer;

void testFromPointClouds(const std::string& filename)
{
	auto cloud_ptr = io::CreatePointCloudFromFile(filename);
	PointCoordinateType maxEdgeLength = 0;
	ccFacet* facet = ccFacet::Create(cloud_ptr.get(), maxEdgeLength);
	if (facet)
	{
		facet->getPolygon()->setOpacity(0.5f);
		facet->getPolygon()->clearTriNormals();
		facet->getPolygon()->computeVertexNormals();
		facet->getPolygon()->setTempColor(ecvColor::darkGrey);
		facet->getContour()->setColor(ecvColor::green);
		facet->showNormalVector(true);
		facet->getContour()->showColors(true);
		double rms = 0.0;
		CCVector3 C, N;
		N = facet->getNormal();
		C = facet->getCenter();
		rms = facet->getRMS();

        cloudViewer::utility::LogInfo("RMS: {:d}", rms);

		//hack: output the transformation matrix that would make this normal points towards +Z
		ccGLMatrix makeZPosMatrix = ccGLMatrix::FromToRotation(N, CCVector3(0, 0, PC_ONE));
		CCVector3 Gt = C;
		makeZPosMatrix.applyRotation(Gt);
		makeZPosMatrix.setTranslation(C - Gt);
		cloudViewer::utility::LogInfo("[Orientation] A matrix that would make this plane horizontal (normal towards Z+) is:");
		cloudViewer::utility::LogInfo(makeZPosMatrix.toString(12, ' ').toStdString().c_str()); //full precision
		visualization::DrawGeometries({ std::shared_ptr<ccFacet>(facet) });
	}
}

void testFromFile(const std::string& filename)
{
	auto cloud_ptr = io::CreateEntityFromFile(filename);
	ccHObject::Container facets;
	cloud_ptr->filterChildren(facets, false, CV_TYPES::FACET);
	for (size_t i = 0; i < facets.size(); ++i)
	{
		auto facet = cloudViewer::make_shared<ccFacet>();
		*facet = *ccHObjectCaster::ToFacet(facets[i]);
		if (facet)
		{
			facet->getPolygon()->setOpacity(0.5f);
			facet->getPolygon()->clearTriNormals();
			facet->getPolygon()->computeVertexNormals();
			facet->getPolygon()->setTempColor(ecvColor::blue);
			facet->getContour()->setColor(ecvColor::green);
			facet->getContour()->setWidth(10);
			facet->showNormalVector(true);
			facet->getContour()->showColors(true);
			double rms = 0.0;
			CCVector3 C, N;
			N = facet->getNormal();
			C = facet->getCenter();
			rms = facet->getRMS();

            cloudViewer::utility::LogInfo("RMS: {:d}", rms);

			//hack: output the transformation matrix that would make this normal points towards +Z
			ccGLMatrix makeZPosMatrix = ccGLMatrix::FromToRotation(N, CCVector3(0, 0, PC_ONE));
			CCVector3 Gt = C;
			makeZPosMatrix.applyRotation(Gt);
			makeZPosMatrix.setTranslation(C - Gt);
			cloudViewer::utility::LogInfo("[Orientation] A matrix that would make this plane horizontal (normal towards Z+) is:");
			cloudViewer::utility::LogInfo(makeZPosMatrix.toString(12, ' ').toStdString().c_str()); //full precision
			visualization::DrawGeometries({ facet });
		}
	}
	
	if (facets.size() >= 2)
	{
		auto facet1 = std::shared_ptr<ccFacet>(ccHObjectCaster::ToFacet(facets[0]));
		auto facet2 = std::shared_ptr<ccFacet>(ccHObjectCaster::ToFacet(facets[1]));

		facet1->getPolygon()->clearTriNormals();
		facet2->getPolygon()->clearTriNormals();
		facet1->getPolygon()->computeVertexNormals();
		facet2->getPolygon()->computeVertexNormals();
		facet1->showNormalVector(true);
		facet2->showNormalVector(true);
		facet1->getContour()->showColors(true);
		facet2->getContour()->showColors(true);
		visualization::DrawGeometries({ facet1, facet2 });
	}

}

int main(int argc, char **argv) {

    cloudViewer::utility::SetVerbosityLevel(cloudViewer::utility::VerbosityLevel::Debug);

    if (argc < 2) {
        // clang-format off
        cloudViewer::utility::LogInfo("Usage:");
        cloudViewer::utility::LogInfo("    > Facets [filename]");
        cloudViewer::utility::LogInfo("    The program will :");
        cloudViewer::utility::LogInfo("    1. load the facets in [filename].");
        // clang-format on
        return 1;
    }

	testFromFile(argv[1]);
	//testFromPointClouds(argv[1]);
    return 0;
}
