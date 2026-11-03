// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <unordered_map>

#include <onnxruntime_c_api.h>

namespace gpu_ep {

// Session allocators may be returned directly by a backend instead of going
// through the umbrella's factory-owned Allocator wrappers. Remember their
// originating factory: the currently selected backend can change before release.
class BackendAllocatorRegistry {
public:
    ~BackendAllocatorRegistry() { ReleaseAll(); }

    bool Track(OrtAllocator* allocator, OrtEpFactory* factory) {
        if (allocator == nullptr || factory == nullptr || factory->ReleaseAllocator == nullptr) {
            return false;
        }
        std::lock_guard lock{mutex_};
        auto [it, inserted] = owners_.try_emplace(allocator, Owner{factory, 1});
        if (!inserted) {
            if (it->second.factory != factory) {
                return false;
            }
            // A backend may return the same shared allocator for several creates.
            ++it->second.references;
        }
        return true;
    }

    bool Release(OrtAllocator* allocator) {
        OrtEpFactory* factory = nullptr;
        {
            std::lock_guard lock{mutex_};
            auto it = owners_.find(allocator);
            if (it == owners_.end()) {
                return false;
            }
            factory = it->second.factory;
            if (--it->second.references == 0) {
                owners_.erase(it);
            }
        }
        // Backend callbacks must not run under our mutex (they may re-enter).
        factory->ReleaseAllocator(factory, allocator);
        return true;
    }

    // Called before backend factories or DLLs are released. Normally sessions
    // have already returned every allocator; also drain any outstanding entries.
    void ReleaseAll() {
        for (;;) {
            OrtAllocator* allocator = nullptr;
            Owner owner{};
            {
                std::lock_guard lock{mutex_};
                if (owners_.empty()) {
                    return;
                }
                auto it = owners_.begin();
                allocator = it->first;
                owner = it->second;
                owners_.erase(it);
            }
            for (size_t i = 0; i < owner.references; ++i) {
                owner.factory->ReleaseAllocator(owner.factory, allocator);
            }
        }
    }

private:
    struct Owner {
        OrtEpFactory* factory;
        size_t references;
    };
    std::mutex mutex_;
    std::unordered_map<OrtAllocator*, Owner> owners_;
};

}  // namespace gpu_ep
