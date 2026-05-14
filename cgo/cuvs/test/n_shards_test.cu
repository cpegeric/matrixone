/*
 * Copyright 2021 Matrix Origin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Tests for the n_shards parameter — letting SHARDED indexes use fewer
 * shards than physical GPUs. Tests are grouped by what they require:
 *   - validation tests (no GPU work) run anywhere
 *   - multi-GPU build/search/save tests skip on single-GPU hosts
 */

#include "test_framework.hpp"
#include "helper.h"
#include "ivf_flat.hpp"
#include "ivf_pq.hpp"
#include "cagra.hpp"
#include <stdexcept>
#include <filesystem>

using namespace matrixone;

// ---------------------------------------------------------------------------
// Validation tests — these never start a worker, so they exercise pure
// constructor argument checks and run on any host.
// ---------------------------------------------------------------------------

// SHARDED + n_shards > devices.size() must throw before allocating anything.
TEST(NShardsTest, ConstructorRejectsTooMany) {
    const uint32_t dimension = 8;
    const uint64_t count = 32;
    std::vector<float> dataset(count * dimension, 0.0f);
    std::vector<int> devices = {0};  // 1 device

    ivf_flat_build_params_t bp = ivf_flat_build_params_default();
    bp.n_lists = 4;

    ASSERT_THROW(
        (gpu_ivf_flat_t<float>(dataset.data(), count, dimension,
            DistanceType_L2Expanded, bp, devices, 1,
            DistributionMode_SHARDED, /*ids=*/nullptr, /*n_shards=*/2)),
        std::invalid_argument);
}

// SHARDED + n_shards == 1 with a single-device list is valid and assigns
// effective_n_shards() == 1.
TEST(NShardsTest, ConstructorAcceptsExplicitOne) {
    const uint32_t dimension = 8;
    const uint64_t count = 32;
    std::vector<float> dataset(count * dimension, 0.0f);
    std::vector<int> devices = {0};

    ivf_flat_build_params_t bp = ivf_flat_build_params_default();
    bp.n_lists = 4;

    gpu_ivf_flat_t<float> index(dataset.data(), count, dimension,
        DistanceType_L2Expanded, bp, devices, 1,
        DistributionMode_SHARDED, /*ids=*/nullptr, /*n_shards=*/1);
    ASSERT_EQ((size_t)index.effective_n_shards(), (size_t)1);
}

// SHARDED + n_shards == 0 defaults to devices.size() (legacy behaviour).
TEST(NShardsTest, ConstructorDefaultsToDeviceCount) {
    const uint32_t dimension = 8;
    const uint64_t count = 32;
    std::vector<float> dataset(count * dimension, 0.0f);
    std::vector<int> devices = {0};

    ivf_flat_build_params_t bp = ivf_flat_build_params_default();
    bp.n_lists = 4;

    gpu_ivf_flat_t<float> index(dataset.data(), count, dimension,
        DistanceType_L2Expanded, bp, devices, 1,
        DistributionMode_SHARDED, /*ids=*/nullptr, /*n_shards=*/0);
    ASSERT_EQ((size_t)index.effective_n_shards(), (size_t)1);
}

// Non-SHARDED modes ignore any n_shards value the caller passes.
TEST(NShardsTest, NonShardedIgnoresNShards) {
    const uint32_t dimension = 8;
    const uint64_t count = 32;
    std::vector<float> dataset(count * dimension, 0.0f);
    std::vector<int> devices = {0};

    ivf_flat_build_params_t bp = ivf_flat_build_params_default();
    bp.n_lists = 4;

    gpu_ivf_flat_t<float> index(dataset.data(), count, dimension,
        DistanceType_L2Expanded, bp, devices, 1,
        DistributionMode_SINGLE_GPU, /*ids=*/nullptr, /*n_shards=*/7);
    // effective_n_shards() returns 1 for non-SHARDED regardless of input.
    ASSERT_EQ((size_t)index.effective_n_shards(), (size_t)1);
}

// Same validation applies to gpu_ivf_pq_t and gpu_cagra_t — sanity-check
// each so a future refactor doesn't accidentally drop the call in one
// constructor.
TEST(NShardsTest, ConstructorValidationIvfPq) {
    const uint32_t dimension = 8;
    const uint64_t count = 32;
    std::vector<float> dataset(count * dimension, 0.0f);
    std::vector<int> devices = {0};
    ivf_pq_build_params_t bp = ivf_pq_build_params_default();
    bp.n_lists = 4; bp.m = 4;
    ASSERT_THROW(
        (gpu_ivf_pq_t<float>(dataset.data(), count, dimension,
            DistanceType_L2Expanded, bp, devices, 1,
            DistributionMode_SHARDED, /*ids=*/nullptr, /*n_shards=*/2)),
        std::invalid_argument);
}

TEST(NShardsTest, ConstructorValidationCagra) {
    const uint32_t dimension = 8;
    const uint64_t count = 32;
    std::vector<float> dataset(count * dimension, 0.0f);
    std::vector<int> devices = {0};
    cagra_build_params_t bp = cagra_build_params_default();
    ASSERT_THROW(
        (gpu_cagra_t<float>(dataset.data(), count, dimension,
            DistanceType_L2Expanded, bp, devices, 1,
            DistributionMode_SHARDED, /*ids=*/nullptr, /*n_shards=*/3)),
        std::invalid_argument);
}

// dynb_concurrency_hint divides nthread by n_shards (SHARDED) rather than
// devices.size(). Pure-arithmetic check on a constructed (but unstarted)
// index; no GPU activity, so safe to run anywhere.
TEST(NShardsTest, DynbConcurrencyHintUsesNShards) {
    const uint32_t dimension = 8;
    const uint64_t count = 32;
    std::vector<float> dataset(count * dimension, 0.0f);
    std::vector<int> devices = {0};  // pretend we have many; we don't, but
                                     // the hint is host-side math.
    ivf_flat_build_params_t bp = ivf_flat_build_params_default();
    bp.n_lists = 4;

    // Non-SHARDED control: hint == nthread / devices.size().
    {
        gpu_ivf_flat_t<float> idx(dataset.data(), count, dimension,
            DistanceType_L2Expanded, bp, devices, /*nthread=*/8,
            DistributionMode_SINGLE_GPU);
        idx.start();
        // SINGLE_GPU spawns one worker thread on devices[0]; nthread arg is
        // an upper bound, so the denom is devices.size() == 1.
        ASSERT_EQ((int)idx.dynb_concurrency_hint(),
                  (int)(idx.worker->nthread() / 1));
        idx.destroy();
    }
    // SHARDED + explicit n_shards=1: hint divides by 1, not by anything
    // larger. (We can only test n_shards == 1 on a single-GPU host; the
    // multi-GPU divisor cases require an actual multi-GPU box.)
    {
        gpu_ivf_flat_t<float> idx(dataset.data(), count, dimension,
            DistanceType_L2Expanded, bp, devices, /*nthread=*/8,
            DistributionMode_SHARDED, /*ids=*/nullptr, /*n_shards=*/1);
        idx.start();
        ASSERT_EQ((int)idx.dynb_concurrency_hint(),
                  (int)(idx.worker->nthread() / 1));
        idx.destroy();
    }
}

// ---------------------------------------------------------------------------
// Multi-GPU tests — these need ≥ 2 GPUs to actually exercise the case
// where n_shards < devices.size(). Skip on single-GPU hosts.
// ---------------------------------------------------------------------------

// Build an IVF-Flat index with devices=[0..N-1] but n_shards=2. Only the
// first 2 device queues should host shards; the rest stay idle.
TEST(NShardsTest, BuildWithFewerShardsThanGpus) {
    int dev_count = gpu_get_device_count();
    if (dev_count < 2) {
        TEST_LOG("Skipping BuildWithFewerShardsThanGpus: Need at least 2 GPUs");
        return;
    }
    if (dev_count > 4) dev_count = 4;

    std::vector<int> devices(dev_count);
    gpu_get_device_list(devices.data(), dev_count);

    const uint32_t dimension = 8;
    const uint64_t count = 1024;
    std::vector<float> dataset(count * dimension);
    for (size_t i = 0; i < dataset.size(); ++i) dataset[i] = (float)i / dataset.size();

    ivf_flat_build_params_t bp = ivf_flat_build_params_default();
    bp.n_lists = 16;

    gpu_ivf_flat_t<float> index(dataset.data(), count, dimension,
        DistanceType_L2Expanded, bp, devices, /*nthread=*/(uint32_t)dev_count,
        DistributionMode_SHARDED, /*ids=*/nullptr, /*n_shards=*/2);
    index.start();
    index.build();

    // effective_n_shards() should report 2 (not devices.size()).
    ASSERT_EQ((size_t)index.effective_n_shards(), (size_t)2);
    // shard_sizes_.size() == n_shards (NOT devices.size()).
    ASSERT_EQ(index.shard_sizes_.size(), (size_t)2);
    // The two shard sizes must sum to count.
    ASSERT_EQ((uint64_t)(index.shard_sizes_[0] + index.shard_sizes_[1]), count);
    // First n_shards-1 shards are rounded down to a multiple of 32.
    ASSERT_EQ((uint64_t)(index.shard_sizes_[0] & 31), (uint64_t)0);
    // Only devices_[0] and devices_[1] participate — devices_[2..] never
    // populated their replicated_indices_ slot.
    {
        std::shared_lock<std::shared_mutex> lock(index.mutex_);
        ASSERT_EQ(index.replicated_indices_.count(devices[0]), (size_t)1);
        ASSERT_EQ(index.replicated_indices_.count(devices[1]), (size_t)1);
        for (int d = 2; d < dev_count; ++d) {
            ASSERT_EQ(index.replicated_indices_.count(devices[d]), (size_t)0);
        }
    }

    index.destroy();
}

// Search a 2-shard / 4-GPU index and confirm a near-trivial top-1 still
// resolves (recall sanity, not exact match — both shards probed via
// submit_first_n_no_wait).
TEST(NShardsTest, SearchWithFewerShardsThanGpus) {
    int dev_count = gpu_get_device_count();
    if (dev_count < 2) {
        TEST_LOG("Skipping SearchWithFewerShardsThanGpus: Need at least 2 GPUs");
        return;
    }
    if (dev_count > 4) dev_count = 4;

    std::vector<int> devices(dev_count);
    gpu_get_device_list(devices.data(), dev_count);

    const uint32_t dimension = 8;
    const uint64_t count = 1024;
    std::vector<float> dataset(count * dimension);
    for (size_t i = 0; i < dataset.size(); ++i) dataset[i] = (float)i / dataset.size();

    ivf_flat_build_params_t bp = ivf_flat_build_params_default();
    bp.n_lists = 16;

    gpu_ivf_flat_t<float> index(dataset.data(), count, dimension,
        DistanceType_L2Expanded, bp, devices, /*nthread=*/(uint32_t)dev_count,
        DistributionMode_SHARDED, /*ids=*/nullptr, /*n_shards=*/2);
    index.start();
    index.build();

    // Query the very first vector; expect it back in top-k.
    std::vector<float> q(dataset.begin(), dataset.begin() + dimension);
    ivf_flat_search_params_t sp = ivf_flat_search_params_default();
    sp.n_probes = 16;
    auto r = index.search(q.data(), 1, dimension, 5, sp);
    ASSERT_EQ(r.neighbors.size(), (size_t)5);
    bool found_zero = false;
    for (auto n : r.neighbors) if (n == 0) { found_zero = true; break; }
    ASSERT_TRUE(found_zero);

    index.destroy();
}

// Save with n_shards=2, devices=4. Reload with the same setup must succeed.
// Reload with n_shards=3 (mismatch) must throw on load_dir.
TEST(NShardsTest, SaveLoadRoundtripWithFewerShards) {
    int dev_count = gpu_get_device_count();
    if (dev_count < 2) {
        TEST_LOG("Skipping SaveLoadRoundtripWithFewerShards: Need at least 2 GPUs");
        return;
    }
    if (dev_count > 4) dev_count = 4;

    std::vector<int> devices(dev_count);
    gpu_get_device_list(devices.data(), dev_count);

    const uint32_t dimension = 8;
    const uint64_t count = 1024;
    std::vector<float> dataset(count * dimension);
    for (size_t i = 0; i < dataset.size(); ++i) dataset[i] = (float)i / dataset.size();

    ivf_flat_build_params_t bp = ivf_flat_build_params_default();
    bp.n_lists = 16;

    auto save_dir = std::filesystem::temp_directory_path() / "n_shards_save_test";
    std::filesystem::remove_all(save_dir);
    std::filesystem::create_directories(save_dir);

    // Save phase.
    {
        gpu_ivf_flat_t<float> src(dataset.data(), count, dimension,
            DistanceType_L2Expanded, bp, devices, (uint32_t)dev_count,
            DistributionMode_SHARDED, nullptr, /*n_shards=*/2);
        src.start();
        src.build();
        src.save_dir(save_dir.string());
        src.destroy();
    }

    // Reload phase — matching n_shards (=2). Should succeed.
    {
        gpu_ivf_flat_t<float> dst((uint64_t)0, dimension, DistanceType_L2Expanded,
            bp, devices, (uint32_t)dev_count, DistributionMode_SHARDED, nullptr,
            /*n_shards=*/2);
        dst.start();
        dst.load_dir(save_dir.string(), DistributionMode_SHARDED);
        ASSERT_EQ((size_t)dst.effective_n_shards(), (size_t)2);
        dst.destroy();
    }

    // Reload phase — n_shards == 0 (auto-derive from manifest).
    {
        gpu_ivf_flat_t<float> dst((uint64_t)0, dimension, DistanceType_L2Expanded,
            bp, devices, (uint32_t)dev_count, DistributionMode_SHARDED, nullptr,
            /*n_shards=*/0);
        dst.start();
        dst.load_dir(save_dir.string(), DistributionMode_SHARDED);
        ASSERT_EQ((size_t)dst.effective_n_shards(), (size_t)2);
        dst.destroy();
    }

    // Reload phase — n_shards mismatch (=3) must throw on load_dir.
    if (dev_count >= 3) {
        gpu_ivf_flat_t<float> dst((uint64_t)0, dimension, DistanceType_L2Expanded,
            bp, devices, (uint32_t)dev_count, DistributionMode_SHARDED, nullptr,
            /*n_shards=*/3);
        dst.start();
        ASSERT_THROW(dst.load_dir(save_dir.string(), DistributionMode_SHARDED),
                     std::invalid_argument);
        dst.destroy();
    }

    std::filesystem::remove_all(save_dir);
}
