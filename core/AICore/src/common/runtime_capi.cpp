// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aicore/runtime_capi.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "aicore/backend_capi.h"

namespace {

std::mutex g_inference_mutex;
thread_local bool g_inference_lock_held = false;
thread_local std::vector<aicore_cancel_token*> g_cancel_scope_stack;

std::mutex g_device_queues_mutex;
std::unordered_map<std::string, std::shared_ptr<std::mutex>> g_device_queues;
thread_local std::shared_ptr<std::mutex> g_device_queue_held;

std::string lowercase(const char* raw) {
    std::string value = raw && raw[0] ? raw : "auto";
    std::transform(
            value.begin(), value.end(), value.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string device_queue_key(const char* device) {
    std::string request = lowercase(device);
    if (request == "auto" || request == "gpu") {
        const int count = aicore_device_count();
        for (int i = 0; i < count; ++i) {
            const aicore_device_info* info = aicore_device_at(i);
            if (info && info->id && std::string(info->id) != "auto" &&
                std::string(info->id) != "cpu" &&
                aicore_device_available(info->id)) {
                request = info->id;
                break;
            }
        }
        if (request == "auto" || request == "gpu") request = "cpu";
    }
    if (request == "mtl") request = "metal";
    for (const char* family : {"cuda", "vulkan", "metal"}) {
        const std::string name(family);
        if (request == name + ":0" || request == name + "0") {
            return name;
        }
        if (request.size() > name.size() &&
            request.compare(0, name.size(), name) == 0) {
            const std::string suffix = request.substr(name.size());
            if (!suffix.empty() &&
                suffix.find_first_not_of("0123456789") == std::string::npos) {
                return name + ":" + suffix;
            }
        }
    }
    return request;
}

std::shared_ptr<std::mutex> device_queue(const char* device) {
    const std::string key = device_queue_key(device);
    std::lock_guard<std::mutex> lock(g_device_queues_mutex);
    std::shared_ptr<std::mutex>& queue = g_device_queues[key];
    if (!queue) queue = std::make_shared<std::mutex>();
    return queue;
}

}  // namespace

struct aicore_cancel_token {
    std::atomic<bool> requested{false};
};

namespace {
aicore_cancel_token g_legacy_cancel_token;
}

AICORE_CAPI aicore_cancel_token* aicore_cancel_token_new(void) {
    return new (std::nothrow) aicore_cancel_token();
}

AICORE_CAPI void aicore_cancel_token_free(aicore_cancel_token* token) {
    g_cancel_scope_stack.erase(std::remove(g_cancel_scope_stack.begin(),
                                           g_cancel_scope_stack.end(), token),
                               g_cancel_scope_stack.end());
    delete token;
}

AICORE_CAPI void aicore_cancel_token_reset(aicore_cancel_token* token) {
    if (token) token->requested.store(false, std::memory_order_release);
}

AICORE_CAPI void aicore_cancel_token_request(aicore_cancel_token* token) {
    if (token) token->requested.store(true, std::memory_order_release);
}

AICORE_CAPI int aicore_cancel_token_requested(
        const aicore_cancel_token* token) {
    return token && token->requested.load(std::memory_order_acquire) ? 1 : 0;
}

AICORE_CAPI void aicore_cancel_scope_begin(aicore_cancel_token* token) {
    if (token) g_cancel_scope_stack.push_back(token);
}

AICORE_CAPI void aicore_cancel_scope_end(aicore_cancel_token* token) {
    if (!g_cancel_scope_stack.empty() && g_cancel_scope_stack.back() == token) {
        g_cancel_scope_stack.pop_back();
    }
}

AICORE_CAPI void aicore_cancel_begin(void) {
    aicore_cancel_token_reset(&g_legacy_cancel_token);
    aicore_cancel_scope_begin(&g_legacy_cancel_token);
}

AICORE_CAPI void aicore_cancel_end(void) {
    aicore_cancel_token_reset(&g_legacy_cancel_token);
    aicore_cancel_scope_end(&g_legacy_cancel_token);
}

AICORE_CAPI void aicore_cancel_request(void) {
    aicore_cancel_token_request(&g_legacy_cancel_token);
}

AICORE_CAPI int aicore_cancel_requested(void) {
    return aicore_cancel_token_requested(g_cancel_scope_stack.empty()
                                                 ? &g_legacy_cancel_token
                                                 : g_cancel_scope_stack.back());
}

AICORE_CAPI int aicore_inference_lock(void) {
    g_inference_mutex.lock();
    g_inference_lock_held = true;
    return 0;
}

AICORE_CAPI void aicore_inference_unlock(void) {
    if (!g_inference_lock_held) return;
    g_inference_lock_held = false;
    g_inference_mutex.unlock();
}

AICORE_CAPI int aicore_inference_try_lock(void) {
    if (!g_inference_mutex.try_lock()) return -1;
    g_inference_lock_held = true;
    return 0;
}

AICORE_CAPI int aicore_device_task_lock(const char* device) {
    if (g_device_queue_held) return -1;
    std::shared_ptr<std::mutex> queue = device_queue(device);
    queue->lock();
    g_device_queue_held = std::move(queue);
    return 0;
}

AICORE_CAPI int aicore_device_task_lock_cancelable(
        const char* device, const aicore_cancel_token* token) {
    if (g_device_queue_held) return -1;
    std::shared_ptr<std::mutex> queue = device_queue(device);
    while (!queue->try_lock()) {
        if (aicore_cancel_token_requested(token)) return 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (aicore_cancel_token_requested(token)) {
        queue->unlock();
        return 1;
    }
    g_device_queue_held = std::move(queue);
    return 0;
}

AICORE_CAPI int aicore_device_task_try_lock(const char* device) {
    if (g_device_queue_held) return -1;
    std::shared_ptr<std::mutex> queue = device_queue(device);
    if (!queue->try_lock()) return -1;
    g_device_queue_held = std::move(queue);
    return 0;
}

AICORE_CAPI void aicore_device_task_unlock(void) {
    if (!g_device_queue_held) return;
    g_device_queue_held->unlock();
    g_device_queue_held.reset();
}
