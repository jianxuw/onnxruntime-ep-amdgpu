// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backend_allocator_registry.h"

#include <array>
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using gpu_ep::BackendAllocatorRegistry;

void Check(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error{message};
    }
}

struct Factory : OrtEpFactory {
    Factory() : OrtEpFactory{} {
        ReleaseAllocator = [](OrtEpFactory* base, OrtAllocator* allocator) noexcept {
            auto& self = *static_cast<Factory*>(base);
            ++self.releases;
            self.last = allocator;
            if (self.reenter != nullptr) {
                auto* registry = self.reenter;
                self.reenter = nullptr;
                registry->Release(self.other);
            }
        };
    }
    std::atomic<size_t> releases{0};
    OrtAllocator* last = nullptr;
    BackendAllocatorRegistry* reenter = nullptr;
    OrtAllocator* other = nullptr;
};

void TestSessionOwnersAndSharedPointers() {
    Factory hip, other;
    OrtAllocator gpu{}, pinned{}, shared_wrapper{};
    BackendAllocatorRegistry registry;
    Check(registry.Track(&gpu, &hip), "track HIP session allocator");
    Check(registry.Track(&pinned, &other), "track another backend");
    Check(!registry.Release(&shared_wrapper), "factory-owned wrapper must be left alone");
    Check(registry.Release(&gpu), "release HIP allocator after backend switch");
    Check(hip.releases == 1 && hip.last == &gpu && other.releases == 0,
        "release must use originating factory");
    Check(!registry.Release(&gpu), "do not release an allocator twice");
    Check(registry.Track(&pinned, &other), "backend may share allocator across creates");
    Check(registry.Release(&pinned) && registry.Release(&pinned), "pair both shared creates");
    Check(other.releases == 2, "forward one release for each successful create");
    Check(!registry.Release(&pinned), "shared reference count must reach zero");
}

void TestInvalidOwnersAndTeardown() {
    Factory hip, other;
    OrtEpFactory missing_release{};
    OrtAllocator allocator{};
    {
        BackendAllocatorRegistry registry;
        Check(!registry.Track(nullptr, &hip), "reject null allocator");
        Check(!registry.Track(&allocator, nullptr), "reject null factory");
        Check(!registry.Track(&allocator, &missing_release), "reject missing release callback");
        Check(registry.Track(&allocator, &hip), "track outstanding session allocation");
        Check(!registry.Track(&allocator, &other), "must not overwrite original owner");
        Check(registry.Track(&allocator, &hip), "track shared outstanding reference");
        registry.ReleaseAll();
        Check(hip.releases == 2 && other.releases == 0, "drain through original factory");
        Check(registry.Track(&allocator, &hip), "registry can track after explicit drain");
    }
    Check(hip.releases == 3, "destruction drains remaining references exactly once");
}

void TestReentrantRelease() {
    Factory hip;
    OrtAllocator first{}, second{};
    BackendAllocatorRegistry registry;
    Check(registry.Track(&first, &hip) && registry.Track(&second, &hip), "track reentrant pair");
    hip.reenter = &registry;
    hip.other = &second;
    Check(registry.Release(&first), "release callback can re-enter registry without deadlock");
    Check(hip.releases == 2, "both allocations released by reentrant callback");
}

void TestConcurrentSessions() {
    // Each backend has its own allocator and callback state, like separate sessions.
    std::array<Factory, 8> factories;
    std::array<OrtAllocator, 8> allocators{};
    BackendAllocatorRegistry registry;
    std::atomic<bool> passed{true};
    std::vector<std::thread> workers;
    for (size_t i = 0; i < factories.size(); ++i) {
        workers.emplace_back([&, i] {
            for (size_t j = 0; j < 100; ++j) {
                if (!registry.Track(&allocators[i], &factories[i]) ||
                    !registry.Release(&allocators[i])) {
                    passed = false;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    Check(passed, "concurrent session tracking and release");
    for (const auto& factory : factories) {
        Check(factory.releases == 100, "each session has paired releases");
    }
}

int main() {
    try {
        TestSessionOwnersAndSharedPointers();
        TestInvalidOwnersAndTeardown();
        TestReentrantRelease();
        TestConcurrentSessions();
        std::cout << "allocator ownership tests passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
