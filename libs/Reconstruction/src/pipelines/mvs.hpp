// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "exe/mvs.h"
#include "option_utils.hpp"

namespace cloudViewer {

int RunDelaunayMesher(const std::string& input_path,
                      const std::string& output_path,
                      const std::string& input_type = "dense") {
//  colmap::OptionManager options;
//  options.AddDelaunayMeshingOptions();
  OptionsParser parser;
  // Path to either the dense workspace folder or the sparse reconstruction
  parser.registerOption("input_path", &input_path);
  parser.registerOption("output_path", &output_path);
  // supported {dense, sparse}
  parser.registerOption("input_type", &input_type);
  if (!parser.parseOptions())
    return EXIT_FAILURE;

  return colmap::RunDelaunayMesher(parser.getArgc(), parser.getArgv());
}

int RunPatchMatchStereo(std::string workspace_path,
                        std::string config_path,
                        std::string workspace_format = "COLMAP",
                        std::string pmvs_option_name = "option-all") {
//  colmap::OptionManager options;
//  options.AddPatchMatchStereoOptions();
  OptionsParser parser;
  // Path to the folder containing the undistorted images
  parser.registerOption("workspace_path", &workspace_path);
  parser.registerOption("config_path", &config_path);
  // supported {COLMAP, PMVS}
  parser.registerOption("workspace_format", &workspace_format);
  parser.registerOption("pmvs_option_name", &pmvs_option_name);
  if (!parser.parseOptions())
    return EXIT_FAILURE;

  return colmap::RunPatchMatchStereo(parser.getArgc(), parser.getArgv());
}

int RunPoissonMesher(const std::string& input_path,
                     const std::string& output_path) {
  //  colmap::OptionManager options;
  //  options.AddPoissonMeshingOptions();
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    if (!parser.parseOptions())
      return EXIT_FAILURE;

    return colmap::RunPoissonMesher(parser.getArgc(), parser.getArgv());
}

int RunStereoFuser(const std::string& workspace_path,
                   const std::string& bbox_path,
                   const std::string& output_path,
                   const std::string& input_type = "geometric",
                   const std::string& output_type = "PLY",
                   const std::string& workspace_format = "COLMAP",
                   const std::string& pmvs_option_name = "option-all") {
//  colmap::OptionManager options;
//  options.AddStereoFusionOptions();
  OptionsParser parser;
  // Path to the folder containing the undistorted images
  parser.registerOption("workspace_path", &workspace_path);
  parser.registerOption("bbox_path", &bbox_path);
  parser.registerOption("output_path", &output_path);
  // supported {photometric, geometric}
  parser.registerOption("input_type", &input_type);
  // supported {BIN, TXT, PLY}
  parser.registerOption("output_type", &output_type);
  // supported {COLMAP, PMVS}
  parser.registerOption("workspace_format", &workspace_format);
  parser.registerOption("pmvs_option_name", &pmvs_option_name);
  if (!parser.parseOptions())
    return EXIT_FAILURE;

  return colmap::RunStereoFuser(parser.getArgc(), parser.getArgv());
}

}  // namespace cloudViewer
