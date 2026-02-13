#pragma once

#include "cuvs_worker.hpp" // For CuvsWorker and RaftHandleWrapper
#include <raft/util/cudart_utils.hpp> // For RAFT_CUDA_TRY

// Standard library includes
#include <algorithm>   // For std::copy
#include <iostream>    // For simulation debug logs
#include <memory>
#include <numeric>     // For std::iota
#include <stdexcept>   // For std::runtime_error
#include <string>      // Corrected: was #string
#include <type_traits> // For std::is_floating_point
#include <vector>
#include <future>      // For std::promise and std::future
#include <limits>      // For std::numeric_limits

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
// RAFT includes
#include <raft/core/device_mdarray.hpp> // For raft::device_matrix
#include <raft/core/device_mdspan.hpp>   // Required for device_matrix_view
#include <raft/core/host_mdarray.hpp> // For raft::host_matrix
#include <raft/core/resources.hpp>       // Core resource handle
#include <raft/linalg/map.cuh>           // RESTORED: map.cuh


// cuVS includes
#include <cuvs/distance/distance.hpp>    // cuVS distance API
#include <cuvs/neighbors/brute_force.hpp> // Correct include
#pragma GCC diagnostic pop


namespace matrixone {

// --- GpuBruteForceIndex Class ---
template <typename T>
class GpuBruteForceIndex {
    static_assert(std::is_floating_point<T>::value, "T must be a floating-point type.");

public:
    std::vector<std::vector<T>> HostDataset; // Store raw data as std::vector
    std::unique_ptr<cuvs::neighbors::brute_force::index<T, float>> Index; // Corrected Index type to float
    cuvs::distance::DistanceType Metric;
    uint32_t Dimension;
    uint32_t Count;
    uint32_t ElementSize;
    std::unique_ptr<CuvsWorker> Worker;

    ~GpuBruteForceIndex() {
        Destroy();
    }

    GpuBruteForceIndex(const std::vector<std::vector<T>>& dataset_data, uint32_t dimension, cuvs::distance::DistanceType m,
                       uint32_t elemsz, uint32_t nthread)
        : Dimension(dimension), ElementSize(elemsz), HostDataset(dataset_data) { // Initialize HostDataset directly
        Worker = std::make_unique<CuvsWorker>(nthread);
        Count = static_cast<uint32_t>(dataset_data.size());
        Metric = m;
    }

    void Load() {
        std::promise<bool> init_complete_promise;
        std::future<bool> init_complete_future = init_complete_promise.get_future();

        auto init_fn = [&](RaftHandleWrapper& handle) -> std::any {
            if (HostDataset.empty()) {
                Index = nullptr; // Ensure Index is null if no data
                init_complete_promise.set_value(true); // Signal completion even if empty
                return std::any();
            }

            // Create host_matrix from HostDataset
            auto dataset_host_matrix = raft::make_host_matrix<T, int64_t, raft::layout_c_contiguous>(*handle.get_raft_resources(), static_cast<int64_t>(HostDataset.size()), static_cast<int64_t>(HostDataset[0].size()));
            for (size_t i = 0; i < HostDataset.size(); ++i) {
                if (HostDataset[i].size() != HostDataset[0].size()) {
                    throw std::runtime_error("Ragged array not supported for raft::host_matrix conversion.");
                }
                std::copy(HostDataset[i].begin(), HostDataset[i].end(), dataset_host_matrix.data_handle() + i * HostDataset[0].size());
            }

            auto dataset_device = raft::make_device_matrix<T, int64_t, raft::layout_c_contiguous>(*handle.get_raft_resources(), static_cast<int64_t>(dataset_host_matrix.extent(0)), static_cast<int64_t>(dataset_host_matrix.extent(1)));
            RAFT_CUDA_TRY(cudaMemcpy(dataset_device.data_handle(), dataset_host_matrix.data_handle(), dataset_host_matrix.size() * sizeof(T), cudaMemcpyHostToDevice));

            cuvs::neighbors::brute_force::index_params index_params; // Correct brute_force namespace
            index_params.metric = Metric;

            Index = std::make_unique<cuvs::neighbors::brute_force::index<T, float>>( // Corrected Index type to float
                cuvs::neighbors::brute_force::build(*handle.get_raft_resources(), index_params, raft::make_const_mdspan(dataset_device.view()))); // Use raft::make_const_mdspan

            raft::resource::sync_stream(*handle.get_raft_resources()); // Synchronize after build

            init_complete_promise.set_value(true); // Signal that initialization is complete
            return std::any();
        };
        auto stop_fn = [&](RaftHandleWrapper& handle) -> std::any {
            if (Index) { // Check if unique_ptr holds an object
                Index.reset();
            }
            return std::any();
        };
        Worker->Start(init_fn, stop_fn);

        init_complete_future.get(); // Wait for the init_fn to complete
    }

    struct SearchResult {
        std::vector<std::vector<int64_t>> Neighbors;
        std::vector<std::vector<float>> Distances;
    };

    SearchResult Search(const std::vector<std::vector<T>>& queries_data, uint32_t limit) {
        if (queries_data.empty() || queries_data[0].empty()) {
            return SearchResult{};
        }
        if (limit == 0) { // Handle limit = 0 explicitly as cuVS requires k > 0
            // Return empty vectors of correct dimensions for the number of queries
            std::vector<std::vector<int64_t>> neighbors_vec(queries_data.size());
            std::vector<std::vector<float>> distances_vec(queries_data.size());
            return SearchResult{neighbors_vec, distances_vec};
        }
        if (!Index) {
            return SearchResult{};
        }

        size_t queries_rows = queries_data.size();
        size_t queries_cols = queries_data[0].size();

        uint64_t jobID = Worker->Submit(
            [&](RaftHandleWrapper& handle) -> std::any {
                // Create host_matrix from queries_data
                auto queries_host_matrix = raft::make_host_matrix<T, int64_t, raft::layout_c_contiguous>(*handle.get_raft_resources(), static_cast<int64_t>(queries_rows), static_cast<int64_t>(queries_cols));
                for (size_t i = 0; i < queries_rows; ++i) {
                    if (queries_data[i].size() != queries_cols) {
                        throw std::runtime_error("Ragged array not supported for raft::host_matrix conversion for queries.");
                    }
                    std::copy(queries_data[i].begin(), queries_data[i].end(), queries_host_matrix.data_handle() + i * queries_cols);
                }

                auto queries_device = raft::make_device_matrix<T, int64_t, raft::layout_c_contiguous>(*handle.get_raft_resources(), static_cast<int64_t>(queries_host_matrix.extent(0)), static_cast<int64_t>(queries_host_matrix.extent(1)));
                RAFT_CUDA_TRY(cudaMemcpy(queries_device.data_handle(), queries_host_matrix.data_handle(), queries_host_matrix.size() * sizeof(T), cudaMemcpyHostToDevice));

                auto neighbors_device = raft::make_device_matrix<int64_t, int64_t, raft::layout_c_contiguous>(*handle.get_raft_resources(), static_cast<int64_t>(queries_rows), static_cast<int64_t>(limit));
                auto distances_device = raft::make_device_matrix<float, int64_t, raft::layout_c_contiguous>(*handle.get_raft_resources(), static_cast<int64_t>(queries_rows), static_cast<int64_t>(limit));

                cuvs::neighbors::brute_force::search_params search_params; // Correct brute_force namespace
                
                // Get the index object from the unique_ptr
                cuvs::neighbors::brute_force::index<T, float>& index_obj = *Index; // Use the actual Index member

                cuvs::neighbors::brute_force::search(*handle.get_raft_resources(), search_params, index_obj, raft::make_const_mdspan(queries_device.view()), neighbors_device.view(), distances_device.view()); // Use raft::make_const_mdspan

                // Synchronize the CUDA stream before copying results back to host
                raft::resource::sync_stream(*handle.get_raft_resources()); // Corrected to use raft::resource::sync_stream with resources object

                auto neighbors_host = raft::make_host_matrix<int64_t, int64_t, raft::layout_c_contiguous>(*handle.get_raft_resources(), static_cast<int64_t>(neighbors_device.extent(0)), static_cast<int64_t>(neighbors_device.extent(1)));
                auto distances_host = raft::make_host_matrix<float, int64_t, raft::layout_c_contiguous>(*handle.get_raft_resources(), static_cast<int64_t>(distances_device.extent(0)), static_cast<int64_t>(distances_device.extent(1)));
                
                RAFT_CUDA_TRY(cudaMemcpy(neighbors_host.data_handle(), neighbors_device.data_handle(), neighbors_host.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
                RAFT_CUDA_TRY(cudaMemcpy(distances_host.data_handle(), distances_device.data_handle(), distances_host.size() * sizeof(float), cudaMemcpyDeviceToHost));

                std::vector<std::vector<int64_t>> neighbors_vec;
                std::vector<std::vector<float>> distances_vec;
                neighbors_vec.reserve(queries_rows);
                distances_vec.reserve(queries_rows);

                for (size_t i = 0; i < queries_rows; ++i) {
                    std::vector<int64_t> current_neighbors;
                    std::vector<float> current_distances;
                    current_neighbors.reserve(limit);
                    current_distances.reserve(limit);

                    for (size_t j = 0; j < limit; ++j) {
                        int64_t neighbor_idx = neighbors_host(i, j);
                        float distance_val = distances_host(i, j);

                        // Filter out invalid neighbors (UINT_MAX and FLT_MAX)
                        // cuVS uses numeric_limits<int64_t>::max() for invalid indices and numeric_limits<float>::max() for invalid distances
                        if (neighbor_idx != std::numeric_limits<int64_t>::max() &&
                            !std::isinf(distance_val) && // Check for infinity
                            distance_val != std::numeric_limits<float>::max()) {
                            current_neighbors.push_back(neighbor_idx);
                            current_distances.push_back(distance_val);
                        }
                    }
                    neighbors_vec.push_back(current_neighbors);
                    distances_vec.push_back(current_distances);
                }
                
                return SearchResult{neighbors_vec, distances_vec};
            }
        );

        CuvsTaskResult result = Worker->Wait(jobID).get();
        if (result.Error) {
            std::rethrow_exception(result.Error);
        }

        return std::any_cast<SearchResult>(result.Result);
    }

    void Destroy() {
        if (Worker) {
            Worker->Stop();
        }
    }
};

} // namespace matrixone
