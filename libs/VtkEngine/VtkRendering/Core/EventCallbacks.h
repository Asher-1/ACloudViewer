// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file EventCallbacks.h
 *  @brief Connection and Signal classes for callback registration (replaces
 * boost::signals2)
 */

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace VtkRendering {

/** @class Connection
 *  @brief Lightweight replacement for boost::signals2::connection.
 *  Holds a numeric ID that can be used to disconnect a callback.
 */
class Connection {
public:
    Connection() = default;
    /// @param id Connection identifier
    explicit Connection(uint64_t id) : id_(id) {}
    /// @param id Connection identifier
    /// @param disconnect_fn Function to call on disconnect
    Connection(uint64_t id, std::function<void()> disconnect_fn)
        : id_(id), disconnect_fn_(std::move(disconnect_fn)) {}

    /// @return Connection ID
    uint64_t id() const { return id_; }
    /// @return true if connected
    bool connected() const { return id_ != 0; }
    void reset() {
        id_ = 0;
        disconnect_fn_ = nullptr;
    }
    /// Disconnects the callback from the signal. No-op if not connected.
    void disconnect() {
        if (disconnect_fn_) {
            disconnect_fn_();
            disconnect_fn_ = nullptr;
        }
        id_ = 0;
    }

private:
    uint64_t id_ = 0;
    std::function<void()> disconnect_fn_;
};

/** @class Signal
 *  @brief Lightweight replacement for boost::signals2::signal.
 *  Thread-safe signal that stores callbacks identified by numeric IDs.
 */
template <typename... Args>
class Signal {
public:
    using Callback = std::function<void(Args...)>;

    /// @param cb Callback to invoke when signal is emitted
    /// @return Connection for disconnect
    Connection connect(Callback cb) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t id = ++next_id_;
        slots_.push_back({id, std::move(cb)});
        auto disconnect_fn = [this, id]() { disconnect(Connection(id)); };
        return Connection(id, std::move(disconnect_fn));
    }

    /// @param conn Connection to disconnect
    void disconnect(const Connection& conn) {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.erase(std::remove_if(
                             slots_.begin(), slots_.end(),
                             [&](const Slot& s) { return s.id == conn.id(); }),
                     slots_.end());
    }

    void disconnect_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.clear();
    }

    /// @param args Arguments to pass to all connected callbacks
    void operator()(Args... args) const {
        std::vector<Slot> copy;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            copy = slots_;
        }
        for (const auto& slot : copy) {
            slot.callback(args...);
        }
    }

    /// @return true if no callbacks connected
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slots_.empty();
    }

private:
    struct Slot {
        uint64_t id;
        Callback callback;
    };

    mutable std::mutex mutex_;
    std::vector<Slot> slots_;
    uint64_t next_id_ = 0;
};

}  // namespace VtkRendering
