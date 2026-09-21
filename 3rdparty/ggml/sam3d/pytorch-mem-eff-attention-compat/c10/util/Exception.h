#pragma once

#include <cassert>

// The imported forward-only kernel uses TORCH_CHECK only for host-side input
// contracts. ggml validates the same contracts before launch, so an assert is
// sufficient and avoids linking any PyTorch runtime.
#define TORCH_CHECK(condition, ...) assert(condition)
