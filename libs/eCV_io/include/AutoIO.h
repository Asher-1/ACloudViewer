// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
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

#pragma once

#include <memory>
#include <string>

class ccHObject;

//! Generic file read and write utility for python interface
/** Gives static access to file loader and writer
**/
namespace cloudViewer
{
	namespace io
	{
		bool AutoReadEntity(const std::string &filename,
			ccHObject& entity,
			bool print_progress = false);

		bool AutoWriteMesh(const std::string &filename,
			const ccHObject& entity,
			bool write_ascii = false,
			bool compressed = false,
			bool write_vertex_normals = true,
			bool write_vertex_colors = true,
			bool write_triangle_uvs = true,
			bool print_progress = false);

		bool AutoWriteEntity(const std::string &filename,
			const ccHObject& entity,
			bool write_ascii = false,
			bool compressed = false,
			bool print_progress = false);
	}
}


namespace cloudViewer {
namespace io {

/// Factory function to create a ccHObject from a file.
/// \return return an empty ccHObject if fail to read the file.
std::shared_ptr<ccHObject> CreateEntityFromFile(
        const std::string &filename,
        const std::string &format = "auto",
        bool print_progress = false);

/// The general entrance for reading a ccHObject from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ReadEntity(const std::string &filename,
				 ccHObject &obj,
                 const std::string &format = "auto",
                 bool print_progress = false);

/// The general entrance for writing a ccHObject to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports binary encoding and compression, the later
/// two parameters will be used. Otherwise they will be ignored.
/// \return return true if the write function is successful, false otherwise.
bool WriteEntity(const std::string &filename,
                  const ccHObject &obj,
                  bool write_ascii = false,
                  bool compressed = false,
                  bool print_progress = false);


}  // namespace io
}  // namespace cloudViewer
