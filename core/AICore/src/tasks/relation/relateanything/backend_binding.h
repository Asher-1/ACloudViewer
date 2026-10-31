// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <stdint.h>
/* Optional host-owned backend handles from the same patched GGML ABI.
 * The host acquires a shared BackendLease and locks its device group around
 * create/infer/free. These callbacks retain/release that lease, never the bare
 * handles. They must not throw. No discovery, thread-policy changes or backend
 * free occur for a binding. The session still owns its scheduler and buffers.
 */
typedef struct ra_backend_binding {
    uint32_t struct_size, abi_version;
    void* accelerator_handle;
    void* cpu_handle;
    void* lease_user;
    void (*retain_lease)(void* user);
    void (*release_lease)(void* user);
} ra_backend_binding;
