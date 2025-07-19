// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <immintrin.h>
#include <cstring>
#include <algorithm>

namespace cloudViewer {
namespace utility {

// Cache line size optimization
constexpr size_t CACHE_LINE_SIZE = 64;

// Force inline for performance-critical functions
#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif

// Memory prefetch hints
#ifdef _MSC_VER
#define PREFETCH_READ(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#define PREFETCH_WRITE(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
#define PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
#define PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)
#endif

// Branch prediction hints
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

// Memory alignment macros
#define ALIGN_TO_CACHE_LINE alignas(CACHE_LINE_SIZE)
#define ALIGN_TO_SIMD alignas(32)

// SIMD-optimized vector operations
namespace simd {

#if defined(__AVX2__)
// AVX2-optimized vector addition
FORCE_INLINE void VectorAdd(const float* a, const float* b, float* result, size_t count) {
    const size_t simd_count = count & ~7; // Process 8 elements at a time
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

// AVX2-optimized dot product
FORCE_INLINE float DotProduct(const float* a, const float* b, size_t count) {
    __m256 sum = _mm256_setzero_ps();
    const size_t simd_count = count & ~7;
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal addition
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), 
                               _mm256_extractf128_ps(sum, 1));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    float result = _mm_cvtss_f32(sum128);
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

#elif defined(__SSE4_2__)
// SSE-optimized vector operations for older CPUs
FORCE_INLINE void VectorAdd(const float* a, const float* b, float* result, size_t count) {
    const size_t simd_count = count & ~3; // Process 4 elements at a time
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_storeu_ps(&result[i], vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

FORCE_INLINE float DotProduct(const float* a, const float* b, size_t count) {
    __m128 sum = _mm_setzero_ps();
    const size_t simd_count = count & ~3;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }
    
    // Horizontal addition
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    
    float result = _mm_cvtss_f32(sum);
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

#else
// Fallback scalar implementations
FORCE_INLINE void VectorAdd(const float* a, const float* b, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

FORCE_INLINE float DotProduct(const float* a, const float* b, size_t count) {
    float result = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

} // namespace simd

// Cache-friendly memory operations
namespace memory {

// Cache-aligned memory allocation
FORCE_INLINE void* AlignedAlloc(size_t size, size_t alignment = CACHE_LINE_SIZE) {
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

FORCE_INLINE void AlignedFree(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Optimized memory copy with prefetching
FORCE_INLINE void OptimizedMemcpy(void* dest, const void* src, size_t size) {
    const char* s = static_cast<const char*>(src);
    char* d = static_cast<char*>(dest);
    
    // Prefetch source data
    for (size_t i = 0; i < size; i += CACHE_LINE_SIZE) {
        PREFETCH_READ(s + i);
    }
    
    std::memcpy(dest, src, size);
}

} // namespace memory

// Parallel processing utilities
namespace parallel {

// Get optimal number of threads for parallel processing
FORCE_INLINE size_t GetOptimalThreadCount() {
#ifdef _OPENMP
    return static_cast<size_t>(omp_get_max_threads());
#else
    return std::thread::hardware_concurrency();
#endif
}

// Parallel for loop with optimal chunk size
template <typename Func>
FORCE_INLINE void ParallelFor(size_t start, size_t end, Func&& func) {
    const size_t range = end - start;
    const size_t num_threads = GetOptimalThreadCount();
    const size_t chunk_size = std::max(size_t(1), range / (num_threads * 4));
    
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (size_t i = start; i < end; ++i) {
        func(i);
    }
#else
    // Fallback to serial execution
    for (size_t i = start; i < end; ++i) {
        func(i);
    }
#endif
}

} // namespace parallel

// Branch-free utilities for better performance
namespace branchfree {

FORCE_INLINE int Min(int a, int b) {
    return a + ((b - a) & ((b - a) >> 31));
}

FORCE_INLINE int Max(int a, int b) {
    return a - ((a - b) & ((a - b) >> 31));
}

FORCE_INLINE int Clamp(int value, int min_val, int max_val) {
    return Min(Max(value, min_val), max_val);
}

} // namespace branchfree

} // namespace utility
} // namespace cloudViewer