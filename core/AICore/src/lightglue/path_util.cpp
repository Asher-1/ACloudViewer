#include "path_util.hpp"

#include "model_cache.hpp"

namespace aicore {
namespace lightglue {

std::string default_model_cache_dir() {
    return aicore::lightglue_model_cache_dir();
}

}  // namespace lightglue
}  // namespace aicore
