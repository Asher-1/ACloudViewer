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

// Port of upstream COLMAP (d3ccaf35) src/colmap/mvs/poisson_meshing.cc. The
// solver entry points come from the unified vendored Adaptive Multigrid
// Solvers v18.75 tree (3rdparty/PoissonRecon ExternalProject), which is the
// byte-parity target of the reconstruction output alignment.

#include "mvs/poisson_meshing.h"

#include <filesystem>
#include <string>
#include <vector>

#include "PoissonRecon.h"
#include "SurfaceTrimmer.h"
#include "MultiThreading.h"
#include "util/file.h"
#include "util/logging.h"
#include "util/threading.h"

namespace colmap {
namespace mvs {

namespace {

// The fork's util layer does not carry the upstream THROW_CHECK_HAS_FILE_
// EXTENSION/FILE_EXISTS/PATH_OPEN helpers, so the equivalent guards live here.
bool HasPlyExtension(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  for (char& c : extension) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return extension == ".ply";
}

bool ParentDirExists(const std::filesystem::path& path) {
  return std::filesystem::exists(path.parent_path());
}

}  // namespace

bool PoissonMeshingOptions::Check() const {
  CHECK_OPTION_GE(point_weight, 0);
  CHECK_OPTION_GT(depth, 0);
  CHECK_OPTION_GE(trim, 0);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_NE(num_threads, 0);
  return true;
}

bool PoissonMeshing(const PoissonMeshingOptions& options,
                    const std::filesystem::path& input_path,
                    const std::filesystem::path& output_path) {
  THROW_CHECK(options.Check());
  CHECK(HasPlyExtension(input_path))
      << "Input path must have a .ply extension: " << input_path;
  CHECK(ExistsFile(input_path)) << "Input path does not exist: " << input_path;
  CHECK(HasPlyExtension(output_path))
      << "Output path must have a .ply extension: " << output_path;
  CHECK(ParentDirExists(output_path))
      << "Output directory does not exist: " << output_path.parent_path();

  bool success = true;

  const int num_effective_threads = GetEffectiveNumThreads(options.num_threads);

  // Configure PoissonRecon's internal thread pool directly, since it uses its
  // own threading mechanism that is not controlled by OMP settings.
  PoissonRecon::ThreadPool::SetNumThreads(num_effective_threads);
  if (num_effective_threads > 1) {
#ifdef _OPENMP
    PoissonRecon::ThreadPool::ParallelizationType =
        PoissonRecon::ThreadPool::ParallelType::OPEN_MP;
#else
    PoissonRecon::ThreadPool::ParallelizationType =
        PoissonRecon::ThreadPool::ParallelType::ASYNC;
#endif
  } else {
    PoissonRecon::ThreadPool::ParallelizationType =
        PoissonRecon::ThreadPool::ParallelType::NONE;
  }

  try {
    std::vector<std::string> args;

    args.push_back("./poisson_recon");

    args.push_back("--in");
    args.push_back(input_path.string());

    args.push_back("--out");
    args.push_back(output_path.string());

    args.push_back("--pointWeight");
    args.push_back(std::to_string(options.point_weight));

    args.push_back("--depth");
    args.push_back(std::to_string(options.depth));

    // Full depth cannot exceed system depth.
    if (options.depth < 5) {
      args.push_back("--fullDepth");
      args.push_back(std::to_string(options.depth));
    }

    if (options.color) {
      args.push_back("--colors");
    }

    if (options.trim > 0) {
      args.push_back("--density");
    }

    std::vector<const char*> args_cstr;
    args_cstr.reserve(args.size());
    for (const auto& arg : args) {
      args_cstr.push_back(arg.c_str());
    }

    if (RunPoissonRecon(args_cstr.size(),
                        const_cast<char**>(args_cstr.data())) != EXIT_SUCCESS) {
      success = false;
    }

    if (success && options.trim != 0) {
      args.clear();
      args_cstr.clear();

      args.push_back("./surface_trimmer");

      args.push_back("--in");
      args.push_back(output_path.string());

      args.push_back("--out");
      args.push_back(output_path.string());

      args.push_back("--trim");
      args.push_back(std::to_string(options.trim));

      args_cstr.reserve(args.size());
      for (const auto& arg : args) {
        args_cstr.push_back(arg.c_str());
      }

      if (RunSurfaceTrimmer(args_cstr.size(),
                            const_cast<char**>(args_cstr.data())) !=
          EXIT_SUCCESS) {
        success = false;
      }
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "PoissonRecon failed with exception: " << e.what();
    success = false;
  }

  return success;
}

}  // namespace mvs
}  // namespace colmap
