// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "util/cache"
#include "util/cache.h"

#include "util/testing.h"

using namespace colmap;

TEST(util_cache, TestLRUCacheEmpty) {
    LRUCache<int, int> cache(
            5, [](const int key) { return std::make_shared<int>(key); });
    EXPECT_EQ(cache.NumElems(), 0);
    EXPECT_EQ(cache.MaxNumElems(), 5);
}

TEST(util_cache, TestLRUCacheGet) {
    LRUCache<int, int> cache(
            5, [](const int key) { return std::make_shared<int>(key); });
    EXPECT_EQ(cache.NumElems(), 0);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(*cache.Get(i), i);
        EXPECT_EQ(cache.NumElems(), i + 1);
        EXPECT_TRUE(cache.Exists(i));
    }

    EXPECT_EQ(*cache.Get(5), 5);
    EXPECT_EQ(cache.NumElems(), 5);
    EXPECT_FALSE(cache.Exists(0));
    EXPECT_TRUE(cache.Exists(5));

    EXPECT_EQ(*cache.Get(5), 5);
    EXPECT_EQ(cache.NumElems(), 5);
    EXPECT_FALSE(cache.Exists(0));
    EXPECT_TRUE(cache.Exists(5));

    EXPECT_EQ(*cache.Get(6), 6);
    EXPECT_EQ(cache.NumElems(), 5);
    EXPECT_FALSE(cache.Exists(0));
    EXPECT_FALSE(cache.Exists(1));
    EXPECT_TRUE(cache.Exists(6));

    *cache.Get(6) = 66;
    EXPECT_EQ(*cache.Get(6), 66);
    EXPECT_EQ(cache.NumElems(), 5);
    EXPECT_FALSE(cache.Exists(0));
    EXPECT_FALSE(cache.Exists(1));
    EXPECT_TRUE(cache.Exists(6));
}

TEST(util_cache, TestLRUCachePop) {
    LRUCache<int, int> cache(
            5, [](const int key) { return std::make_shared<int>(key); });
    EXPECT_EQ(cache.NumElems(), 0);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(*cache.Get(i), i);
        EXPECT_EQ(cache.NumElems(), i + 1);
        EXPECT_TRUE(cache.Exists(i));
    }

    EXPECT_EQ(*cache.Get(5), 5);
    EXPECT_EQ(cache.NumElems(), 5);
    EXPECT_FALSE(cache.Exists(0));
    EXPECT_TRUE(cache.Exists(5));

    cache.Pop();
    EXPECT_EQ(cache.NumElems(), 4);
    cache.Pop();
    EXPECT_EQ(cache.NumElems(), 3);
    cache.Pop();
    EXPECT_EQ(cache.NumElems(), 2);
    cache.Pop();
    EXPECT_EQ(cache.NumElems(), 1);
    cache.Pop();
    EXPECT_EQ(cache.NumElems(), 0);
    cache.Pop();
    EXPECT_EQ(cache.NumElems(), 0);
}

TEST(util_cache, TestLRUCacheClear) {
    LRUCache<int, int> cache(
            5, [](const int key) { return std::make_shared<int>(key); });
    EXPECT_EQ(cache.NumElems(), 0);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(*cache.Get(i), i);
        EXPECT_EQ(cache.NumElems(), i + 1);
        EXPECT_TRUE(cache.Exists(i));
    }

    cache.Clear();
    EXPECT_EQ(cache.NumElems(), 0);

    EXPECT_EQ(*cache.Get(0), 0);
    EXPECT_EQ(cache.NumElems(), 1);
    EXPECT_TRUE(cache.Exists(0));
}

struct SizedElem {
    SizedElem(const size_t num_bytes_) : num_bytes(num_bytes_) {}
    size_t NumBytes() const { return num_bytes; }
    size_t num_bytes;
};

TEST(util_cache, TestMemoryConstrainedLRUCacheEmpty) {
    MemoryConstrainedLRUCache<int, SizedElem> cache(
            5, [](const int key) { return std::make_shared<SizedElem>(key); });
    EXPECT_EQ(cache.NumElems(), 0);
    EXPECT_EQ(cache.MaxNumElems(), std::numeric_limits<size_t>::max());
    EXPECT_EQ(cache.NumBytes(), 0);
    EXPECT_EQ(cache.MaxNumBytes(), 5);
}

TEST(util_cache, TestMemoryConstrainedLRUCacheGet) {
    MemoryConstrainedLRUCache<int, SizedElem> cache(
            10, [](const int key) { return std::make_shared<SizedElem>(key); });
    EXPECT_EQ(cache.NumElems(), 0);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(cache.Get(i)->NumBytes(), i);
        EXPECT_EQ(cache.NumElems(), i + 1);
        EXPECT_TRUE(cache.Exists(i));
    }

    EXPECT_EQ(cache.Get(5)->NumBytes(), 5);
    EXPECT_EQ(cache.NumElems(), 2);
    EXPECT_EQ(cache.NumBytes(), 9);
    EXPECT_FALSE(cache.Exists(0));
    EXPECT_TRUE(cache.Exists(5));

    EXPECT_EQ(cache.Get(5)->NumBytes(), 5);
    EXPECT_EQ(cache.NumElems(), 2);
    EXPECT_EQ(cache.NumBytes(), 9);
    EXPECT_FALSE(cache.Exists(0));
    EXPECT_TRUE(cache.Exists(5));

    EXPECT_EQ(cache.Get(6)->NumBytes(), 6);
    EXPECT_EQ(cache.NumElems(), 1);
    EXPECT_EQ(cache.NumBytes(), 6);
    EXPECT_FALSE(cache.Exists(0));
    EXPECT_FALSE(cache.Exists(1));
    EXPECT_TRUE(cache.Exists(6));

    EXPECT_EQ(cache.Get(1)->NumBytes(), 1);
    EXPECT_EQ(cache.NumElems(), 2);
    EXPECT_EQ(cache.NumBytes(), 7);
    EXPECT_FALSE(cache.Exists(0));
    EXPECT_TRUE(cache.Exists(1));
    EXPECT_TRUE(cache.Exists(6));
}

TEST(util_cache, TestMemoryConstrainedLRUCacheClear) {
    MemoryConstrainedLRUCache<int, SizedElem> cache(
            10, [](const int key) { return std::make_shared<SizedElem>(key); });
    EXPECT_EQ(cache.NumElems(), 0);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(cache.Get(i)->NumBytes(), i);
        EXPECT_EQ(cache.NumElems(), i + 1);
        EXPECT_TRUE(cache.Exists(i));
    }

    cache.Clear();
    EXPECT_EQ(cache.NumElems(), 0);
    EXPECT_EQ(cache.NumBytes(), 0);

    EXPECT_EQ(cache.Get(1)->NumBytes(), 1);
    EXPECT_EQ(cache.NumBytes(), 1);
    EXPECT_EQ(cache.NumElems(), 1);
    EXPECT_TRUE(cache.Exists(1));
}

TEST(util_cache, TestMemoryConstrainedLRUCacheUpdateNumBytes) {
    MemoryConstrainedLRUCache<int, SizedElem> cache(
            50, [](const int key) { return std::make_shared<SizedElem>(key); });
    EXPECT_EQ(cache.NumElems(), 0);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(cache.Get(i)->NumBytes(), i);
        EXPECT_EQ(cache.NumElems(), i + 1);
        EXPECT_TRUE(cache.Exists(i));
    }

    EXPECT_EQ(cache.NumBytes(), 10);

    cache.Get(4)->num_bytes = 3;
    EXPECT_EQ(cache.NumBytes(), 10);
    cache.UpdateNumBytes(4);
    EXPECT_EQ(cache.NumBytes(), 9);

    cache.Get(2)->num_bytes = 3;
    EXPECT_EQ(cache.NumBytes(), 9);
    cache.UpdateNumBytes(2);
    EXPECT_EQ(cache.NumBytes(), 10);

    cache.Get(0)->num_bytes = 40;
    EXPECT_EQ(cache.NumBytes(), 10);
    cache.UpdateNumBytes(0);
    EXPECT_EQ(cache.NumBytes(), 50);

    cache.Clear();
    EXPECT_EQ(cache.NumBytes(), 0);
    EXPECT_EQ(cache.Get(2)->NumBytes(), 2);
    EXPECT_EQ(cache.NumBytes(), 2);
}
