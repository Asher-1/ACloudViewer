// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "dtype_contract.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "common.hpp"
#include "ggml.h"

namespace sam3d {
namespace {

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += character;
                break;
        }
    }
    return escaped;
}

std::string type_name(enum ggml_type type) {
    const char* name = ggml_type_name(type);
    return name ? name : "unknown";
}

std::string op_name(enum ggml_op op) {
    const char* name = ggml_op_name(op);
    return name ? name : "unknown";
}

std::string contract_key(const std::string& stage,
                         const std::string& op,
                         const std::string& output_type,
                         const std::vector<std::string>& input_types) {
    std::ostringstream stream;
    stream << stage << '\x1f' << op << '\x1f' << output_type;
    for (const std::string& input_type : input_types)
        stream << '\x1f' << input_type;
    return stream.str();
}

struct ContractEntry {
    std::string stage;
    std::string op;
    std::string output_type;
    std::vector<std::string> input_types;
    int64_t node_count = 0;
    int64_t max_output_elements = 0;
};

class DtypeContractRecorder {
public:
    DtypeContractRecorder(std::string output_path,
                          DtypeContractMetadata metadata)
        : output_path_(std::move(output_path)),
          metadata_(std::move(metadata)) {}

    void record(const char* stage, ggml_cgraph* graph) {
        if (graph == nullptr || stage == nullptr) return;
        const int node_count = ggml_graph_n_nodes(graph);
        for (int index = 0; index < node_count; ++index) {
            ggml_tensor* node = ggml_graph_node(graph, index);
            if (node == nullptr) continue;
            std::vector<std::string> input_types;
            for (int source_index = 0; source_index < GGML_MAX_SRC;
                 ++source_index) {
                const ggml_tensor* source = node->src[source_index];
                if (source != nullptr)
                    input_types.push_back(type_name(source->type));
            }
            const std::string stage_name(stage);
            const std::string operation = op_name(node->op);
            const std::string output_type = type_name(node->type);
            const std::string key = contract_key(stage_name, operation,
                                                 output_type, input_types);
            ContractEntry& entry = entries_[key];
            if (entry.node_count == 0) {
                entry.stage = stage_name;
                entry.op = operation;
                entry.output_type = output_type;
                entry.input_types = std::move(input_types);
            }
            ++entry.node_count;
            entry.max_output_elements =
                    std::max(entry.max_output_elements, ggml_nelements(node));
        }
    }

    bool flush(bool complete, std::string& error) const {
        if (output_path_.empty()) return true;
        const std::filesystem::path output(output_path_);
        const std::filesystem::path parent = output.parent_path();
        std::error_code filesystem_error;
        if (!parent.empty())
            std::filesystem::create_directories(parent, filesystem_error);
        if (filesystem_error) {
            error = "cannot create dtype contract directory: " +
                    filesystem_error.message();
            return false;
        }
        std::ofstream stream(output, std::ios::out | std::ios::trunc);
        if (!stream) {
            error = "cannot write dtype contract: " + output_path_;
            return false;
        }
        stream << "{\n"
               << "  \"schema\": \"sam3d.e2e.dtype-contract.v1\",\n"
               << "  \"complete\": " << (complete ? "true" : "false") << ",\n"
               << "  \"backend\": \"" << json_escape(metadata_.backend)
               << "\",\n"
               << "  \"model_dtypes\": {\n"
               << "    \"base\": \"" << json_escape(metadata_.base_dtype)
               << "\",\n"
               << "    \"ss\": \"" << json_escape(metadata_.ss_dtype) << "\",\n"
               << "    \"ss_decoder\": \""
               << json_escape(metadata_.ss_decoder_dtype) << "\",\n"
               << "    \"slat\": \"" << json_escape(metadata_.slat_dtype)
               << "\",\n"
               << "    \"gs\": \"" << json_escape(metadata_.gs_dtype) << "\",\n"
               << "    \"mesh\": \"" << json_escape(metadata_.mesh_dtype)
               << "\"\n"
               << "  },\n"
               << "  \"entries\": [\n";
        bool first_entry = true;
        for (const auto& [unused_key, entry] : entries_) {
            (void)unused_key;
            if (!first_entry) stream << ",\n";
            first_entry = false;
            stream << "    {\"stage\": \"" << json_escape(entry.stage)
                   << "\", \"op\": \"" << json_escape(entry.op)
                   << "\", \"input_types\": [";
            for (size_t index = 0; index < entry.input_types.size(); ++index) {
                if (index != 0) stream << ", ";
                stream << "\"" << json_escape(entry.input_types[index]) << "\"";
            }
            stream << "], \"output_type\": \"" << json_escape(entry.output_type)
                   << "\", \"node_count\": " << entry.node_count
                   << ", \"max_output_elements\": " << entry.max_output_elements
                   << "}";
        }
        stream << "\n  ]\n}\n";
        if (!stream) {
            error = "failed while writing dtype contract: " + output_path_;
            return false;
        }
        return true;
    }

private:
    std::string output_path_;
    DtypeContractMetadata metadata_;
    std::map<std::string, ContractEntry> entries_;
};

std::unique_ptr<DtypeContractRecorder> active_recorder;

}  // namespace

ScopedDtypeContractRecording::ScopedDtypeContractRecording(
        const std::string& output_path,
        const DtypeContractMetadata& metadata,
        std::string& error) {
    if (output_path.empty()) return;
    if (active_recorder) {
        error = "a dtype contract recording is already active";
        return;
    }
    active_recorder =
            std::make_unique<DtypeContractRecorder>(output_path, metadata);
    enabled_ = true;
}

ScopedDtypeContractRecording::~ScopedDtypeContractRecording() {
    if (!enabled_ || closed_) return;
    std::string ignored_error;
    if (!active_recorder->flush(false, ignored_error)) {
        LOGE("e2e: failed to write incomplete dtype contract: %s",
             ignored_error.c_str());
    }
    active_recorder.reset();
}

bool ScopedDtypeContractRecording::complete(std::string& error) {
    if (!enabled_) return true;
    if (closed_) {
        error = "dtype contract recording was already closed";
        return false;
    }
    closed_ = true;
    const bool written = active_recorder->flush(true, error);
    active_recorder.reset();
    return written;
}

void record_dtype_contract_graph(const char* stage, ggml_cgraph* graph) {
    if (active_recorder) active_recorder->record(stage, graph);
}

}  // namespace sam3d
