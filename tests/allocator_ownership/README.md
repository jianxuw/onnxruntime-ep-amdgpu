# Allocator ownership regression tests

These tests use fake backend factories and the real ORT C API types. They check
release routing across backends, shared allocator references, factory-owned
wrappers, teardown, reentrant release callbacks, and concurrent sessions. They do
not require a GPU or link an ORT/backend library.

```sh
cmake -S tests/allocator_ownership -B build/allocator-ownership \
  -DORT_INCLUDE_DIR=/path/to/onnxruntime/include
cmake --build build/allocator-ownership
ctest --test-dir build/allocator-ownership --output-on-failure
```

After building the Windows provider, also run EpCert with OGA 0.16.0 and only the
Application Verifier Leak layer enabled. Keep the HIP backend and fixed
`custom_kernels` DLL unchanged when comparing the old and new `amdgpu-ep.dll`.
Verify that session allocators enter their backend's release callback before the
backend DLL is unloaded, and check the allocation stacks for remaining leaks.
Exercise shared allocators and simultaneous sessions selecting different
backends as well; an overall debugger exit code of zero is not a verifier pass.
