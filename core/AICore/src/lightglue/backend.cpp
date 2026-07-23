// Backend initialization for LightGlue — uses shared AICore ggml utilities.

#include "backend.hpp"

#include "ggml_backend_utils.hpp"

#include <ggml-backend.h>
#if !defined(GGML_BACKEND_DL)
#include <ggml-cpu.h>
#endif

#include <cstdlib>

namespace aicore {
namespace lightglue {

bool engine_backend::init(const std::string& device_req, int n_threads) {
    release();
    ggml_common::load_backends_once();

    std::string name;
    int want_idx = 0;
    ggml_common::parse_device(device_req, name, want_idx);
    if (n_threads <= 0) {
        n_threads = static_cast<int>(ggml_common::default_cpu_threads());
    }

    if (name.empty() || name == "auto") {
        std::string resolved;
        if (ggml_backend_t accelerator =
                    ggml_common::find_auto_backend(resolved)) {
            be = accelerator;
            device = resolved;
        } else {
            return init("cpu", n_threads);
        }
    } else if (name == "cpu") {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!be) {
            error = "CPU backend init failed";
            return false;
        }
        device = "cpu";
        if (const char* env = std::getenv("LIGHTGLUE_NTHREADS")) {
            if (int v = std::atoi(env)) n_threads = v;
        }
        ggml_common::set_cpu_threads(be, n_threads);
    } else if (name == "gpu" || name == "cuda" || name == "opencl" ||
               name == "metal" || name == "sycl" || name == "vulkan") {
        be = ggml_common::find_gpu_backend(name, want_idx, device);
        if (!be) {
            error = "no usable '" + name +
                    "' device (backend built and runtime driver present?)";
            return false;
        }
    } else {
        error = "unknown device '" + device_req +
                "' (want auto|cpu|gpu|sycl|vulkan|cuda|metal)";
        return false;
    }

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!galloc) {
        error = "gallocr init failed";
        release();
        return false;
    }
    return true;
}

void engine_backend::release() {
    if (galloc) {
        ggml_gallocr_free(galloc);
        galloc = nullptr;
    }
    if (be) {
        ggml_backend_free(be);
        be = nullptr;
    }
    device.clear();
    error.clear();
}

bool engine_backend::is_cpu() const { return ggml_common::is_cpu_backend(be); }

}  // namespace lightglue
}  // namespace aicore
