#pragma once

namespace aicore {
namespace depth {

// Kept for source compatibility with existing white-box clients. Production
// graph selection no longer depends on this process state; it inspects the
// backend buffer attached to the weight tensor instead.
void set_gpu_mode(bool on);
bool gpu_mode();

}  // namespace depth
}  // namespace aicore
