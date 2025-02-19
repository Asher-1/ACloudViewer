// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 asher-1.github.io
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

#ifndef ECV_QHULL_HEADER
#define ECV_QHULL_HEADER

#include <Eigen/Core>
#include <memory>
#include <vector>

class ccMesh;

namespace cloudViewer {

	namespace geometry {
		class TetraMesh;
	}

namespace utility {

class Qhull {
public:
	static std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
	ComputeConvexHull(const std::vector<Eigen::Vector3d>& points);

	static std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
	ComputeConvexHull(const std::vector<CCVector3>& points);

	static std::tuple<std::shared_ptr<geometry::TetraMesh>, std::vector<size_t>>
	ComputeDelaunayTetrahedralization(const std::vector<Eigen::Vector3d>& points);

	static std::tuple<std::shared_ptr<geometry::TetraMesh>, std::vector<size_t>>
		ComputeDelaunayTetrahedralization(const std::vector<CCVector3>& points);
};

}  // namespace utility
}  // namespace cloudViewer

#endif // ECV_QHULL_HEADER