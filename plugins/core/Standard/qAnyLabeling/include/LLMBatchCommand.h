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

// Keep text generation separate from annotation TASKs; execution, models and
// backends remain owned by the public AICore contract.
struct CommandLLMGenerate : public ccCommandLineInterface::Command {
    CommandLLMGenerate()
        : ccCommandLineInterface::Command(
                  QObject::tr("Local LLM/VLM generation"),
                  COMMAND_LLM_GENERATE) {}
#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"PROMPT", "string", "", "User prompt (required)"},
                {"SYSTEM", "string", "", "Optional system message"},
                {"MMPROJ", "path", "",
                 "Projector path/catalog filename; with images, unset uses the "
                 "model catalog recommendation"},
                {"MAX_TOKENS", "int", "",
                 "Positive output token limit; unset uses the public LLM "
                 "request default"},
                {"TEMPERATURE", "float", "",
                 "Nonnegative sampling temperature; unset uses the public LLM "
                 "request default"},
        };
        return table;
    }
#endif
    bool process(ccCommandLineInterface& cmd) override;
};
