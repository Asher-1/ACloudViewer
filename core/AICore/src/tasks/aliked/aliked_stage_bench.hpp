#pragma once

#include <chrono>
#include <cstdio>
#include <string>

namespace lightglue::aliked_internal {

// Stage timing probe. The historical AICORE_ALIKED_STAGE_BENCH env gate was
// development scaffolding and is removed; the probe is dormant (enabled_ is
// always false) until an explicit API re-enables it.
class StageBench {
public:
  explicit StageBench(const char *name)
      : name_(name), enabled_(false),
        t0_(std::chrono::steady_clock::now()) {}

  ~StageBench() {
    if (!enabled_) {
      return;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0_).count();
    std::fprintf(stderr, "  [stage] %-18s %8.2f ms\n", name_, ms);
  }

  StageBench(const StageBench &) = delete;
  StageBench &operator=(const StageBench &) = delete;

private:
  const char *name_ = nullptr;
  bool enabled_ = false;
  std::chrono::steady_clock::time_point t0_;
};

} // namespace lightglue::aliked_internal
