// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"
#ifdef AICore_ENABLED
#include "ecvAICoreBatchCommand.h"
#endif

#define COMMAND_LLM_GENERATE "LLM_GENERATE"

// 文本生成与标注 TASK 分离；执行、模型和后端仍由公共 AICore 契约持有。
struct CommandLLMGenerate : public ccCommandLineInterface::Command {
    CommandLLMGenerate()
        : ccCommandLineInterface::Command(QObject::tr("Local LLM/VLM generation"),
                                          COMMAND_LLM_GENERATE) {}
#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"PROMPT", "string", "", "User prompt (required)"},
                {"SYSTEM", "string", "", "Optional system message"},
                {"MMPROJ", "path", "",
                 "Projector path/catalog filename; with images, unset uses the model catalog recommendation"},
                {"MAX_TOKENS", "int", "",
                 "Positive output token limit; unset uses the public LLM request default"},
                {"TEMPERATURE", "float", "",
                 "Nonnegative sampling temperature; unset uses the public LLM request default"},
        };
        return table;
    }
#endif
    bool process(ccCommandLineInterface& cmd) override;
};
