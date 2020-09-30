// ----------------------------------------------------------------------------
// -                                    ECV_DB                           -
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

// LOCAL
#include "IJsonConvertibleIO.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <FileSystem.h>


// JSON_LIB
#include <json/json.h>

// SYSTEM
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace cloudViewer {

	namespace {
		using namespace io;

		static const std::unordered_map<
			std::string,
			std::function<bool(const std::string &, CVLib::utility::IJsonConvertible &)>>
			file_extension_to_ijsonconvertible_read_function{
					{"json", ReadIJsonConvertibleFromJSON},
		};

		static const std::unordered_map<
			std::string,
			std::function<bool(const std::string &,
				const CVLib::utility::IJsonConvertible &)>>
			file_extension_to_ijsonconvertible_write_function{
					{"json", WriteIJsonConvertibleToJSON},
		};


		bool ReadIJsonConvertibleFromJSONStream(std::istream &json_stream,
			CVLib::utility::IJsonConvertible &object) {
			Json::Value root_object;
			Json::CharReaderBuilder builder;
			builder["collectComments"] = false;
			JSONCPP_STRING errs;
			bool is_parse_successful =
				parseFromStream(builder, json_stream, &root_object, &errs);
			if (is_parse_successful == false) {
				CVLog::Warning("Read JSON failed: {}.", errs);
				return false;
			}
			return object.ConvertFromJsonValue(root_object);
		}

		bool WriteIJsonConvertibleToJSONStream(
			std::ostream &json_stream, const CVLib::utility::IJsonConvertible &object) {
			Json::Value root_object;
			if (object.ConvertToJsonValue(root_object) == false) {
				return false;
			}
			Json::StreamWriterBuilder builder;
			builder["commentStyle"] = "None";
			builder["indentation"] = "\t";
			auto writer = builder.newStreamWriter();
			writer->write(root_object, &json_stream);
			return true;
		}

	}  // unnamed namespace

	namespace io
	{

		bool ReadIJsonConvertibleFromJSON(const std::string &filename,
			CVLib::utility::IJsonConvertible &object) {
			std::ifstream file_in(filename);
			if (file_in.is_open() == false) {
				CVLog::Warning("Read JSON failed: unable to open file: {}",
					filename);
				return false;
			}
			bool success = ReadIJsonConvertibleFromJSONStream(file_in, object);
			file_in.close();
			return success;
		}

		bool WriteIJsonConvertibleToJSON(const std::string &filename,
			const CVLib::utility::IJsonConvertible &object) {
			std::ofstream file_out(filename);
			if (file_out.is_open() == false) {
				CVLog::Warning("Write JSON failed: unable to open file: {}",
					filename);
				return false;
			}
			bool success = WriteIJsonConvertibleToJSONStream(file_out, object);
			file_out.close();
			return success;
		}

		bool ReadIJsonConvertibleFromJSONString(const std::string &json_string,
			CVLib::utility::IJsonConvertible &object) {
			std::istringstream iss(json_string);
			return ReadIJsonConvertibleFromJSONStream(iss, object);
		}

		bool WriteIJsonConvertibleToJSONString(
			std::string &json_string, const CVLib::utility::IJsonConvertible &object) {
			std::ostringstream oss;
			bool success = WriteIJsonConvertibleToJSONStream(oss, object);
			json_string = oss.str();
			return success;
		}

		bool ReadIJsonConvertible(const std::string &filename,
			CVLib::utility::IJsonConvertible &object) {
			std::string filename_ext =
				CVLib::utility::filesystem::GetFileExtensionInLowerCase(filename);
			if (filename_ext.empty()) {
				CVLog::Warning(
					"Read utility::IJsonConvertible failed: unknown file "
					"extension.");
				return false;
			}
			auto map_itr =
				file_extension_to_ijsonconvertible_read_function.find(filename_ext);
			if (map_itr == file_extension_to_ijsonconvertible_read_function.end()) {
				CVLog::Warning(
					"Read utility::IJsonConvertible failed: unknown file "
					"extension.");
				return false;
			}
			return map_itr->second(filename, object);
		}

		bool WriteIJsonConvertible(const std::string &filename,
			const CVLib::utility::IJsonConvertible &object) {
			std::string filename_ext =
				CVLib::utility::filesystem::GetFileExtensionInLowerCase(filename);
			if (filename_ext.empty()) {
				CVLog::Warning(
					"Write utility::IJsonConvertible failed: unknown file "
					"extension.");
				return false;
			}
			auto map_itr = file_extension_to_ijsonconvertible_write_function.find(
				filename_ext);
			if (map_itr == file_extension_to_ijsonconvertible_write_function.end()) {
				CVLog::Warning(
					"Write utility::IJsonConvertible failed: unknown file "
					"extension.");
				return false;
			}
			return map_itr->second(filename, object);
		}
	}

}