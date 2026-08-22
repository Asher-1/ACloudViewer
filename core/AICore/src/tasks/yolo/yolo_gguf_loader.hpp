#pragma once


// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: AGPL-3.0
// ----------------------------------------------------------------------------
//
// YOLO GGUF parser: reads the yolo.* metadata KV set plus the op-graph
// (op.N.type / op.N.inputs / params) and every tensor's host data.
// In-tree port of ultralytics-ggml cpp_ggml/src/gguf_loader.{hpp,cpp}.

#include <memory>
#include <string>

#include "tasks/yolo/yolo_common.hpp"


namespace yolo {

// Read only the metadata header (cheap: no tensor data). Used to inspect a
// custom GGUF's task ("detect" | "depth") before building a session.
ModelMeta read_gguf_meta(const std::string& path);

// Load and parse a GGUF file. Returns nullptr and logs on failure.
std::unique_ptr<ModelDef> load_gguf(const std::string& path);

}  // namespace yolo
