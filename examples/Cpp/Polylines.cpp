// ----------------------------------------------------------------------------
// -                        ACloudViewer: asher-1.github.io                    -
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

// clang-format off
#include "CloudViewer.h"
// clang-format on

#include <FileIOFilter.h>
#include <ecvPolyline.h>

#include <Eigen/Dense>
#include <cstdio>
#include <vector>

using namespace cloudViewer;
void testFromFile(const std::string& filename) {
    auto cloud_ptr = io::CreateEntityFromFile(filename);
    ccHObject::Container polylines;
    cloud_ptr->filterChildren(polylines, false, CV_TYPES::POLY_LINE);
    for (size_t i = 0; i < polylines.size(); ++i) {
        auto poly = cloudViewer::make_shared<ccPolyline>(nullptr);
        *poly = *ccHObjectCaster::ToPolyline(polylines[i]);
        poly->setColor(ecvColor::blue);
        if (poly) {
            visualization::DrawGeometries({poly});
        }
    }

    if (polylines.size() >= 2) {
        ccPolyline* poly1 = ccHObjectCaster::ToPolyline(polylines[0]);
        ccPolyline* poly2 = ccHObjectCaster::ToPolyline(polylines[1]);

        auto poly = cloudViewer::make_shared<ccPolyline>(nullptr);
        *poly = *poly1 + *poly2;
        poly->setWidth(10);
        unsigned vertCount = poly->getAssociatedCloud()->size();
        CCVector3* lastP = const_cast<CCVector3*>(
                poly->getAssociatedCloud()->getPointPersistentPtr(vertCount -
                                                                  1));
        CCVector3* lastQ = const_cast<CCVector3*>(
                poly->getAssociatedCloud()->getPointPersistentPtr(vertCount -
                                                                  2));
        PointCoordinateType tipLength = (*lastQ - *lastP).norm();
        PointCoordinateType defaultArrowSize = std::min(20.0f, tipLength / 2);
        poly->showArrow(true, vertCount - 1, defaultArrowSize);
        visualization::DrawGeometries({poly});
        if (poly1) {
            delete poly1;
            poly1 = nullptr;
        }
        if (poly2) {
            delete poly2;
            poly2 = nullptr;
        }
    }
}

int main(int argc, char** argv) {
    cloudViewer::utility::SetVerbosityLevel(
            cloudViewer::utility::VerbosityLevel::Debug);

    if (argc < 2) {
        // clang-format off
        cloudViewer::utility::LogInfo("Usage:");
        cloudViewer::utility::LogInfo("    > Polylines [filename]");
        cloudViewer::utility::LogInfo("    The program will :");
        cloudViewer::utility::LogInfo("    1. load the polyline in [filename].");
        // clang-format on
        return 1;
    }

    // Test
    testFromFile(argv[1]);

    return 0;
}
