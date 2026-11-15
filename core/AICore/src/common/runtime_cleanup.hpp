#pragma once

namespace aicore {
namespace runtime {

using CleanupFn = void (*)();

// Register an idempotent process-cache cleanup. Registration is deduplicated;
// callbacks run outside the registry mutex so they may take their own locks.
void register_cleanup(CleanupFn cleanup);
void run_cleanups();

}  // namespace runtime
}  // namespace aicore
