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
#include <shared_mutex> // For std::shared_mutex

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
    std::vector<T> flattened_host_dataset; // Store flattened data as std::vector
    std::unique_ptr<cuvs::neighbors::brute_force::index<T, float>> Index; // Corrected Index type to float
    cuvs::distance::DistanceType Metric;
    uint32_t Dimension;
    uint32_t Count;
    std::unique_ptr<CuvsWorker> Worker;
    std::shared_mutex mutex_; // Mutex to protect Load() and Search()

    ~GpuBruteForceIndex() {
        Destroy();
    }

    GpuBruteForceIndex(const T* dataset_data, uint64_t count_vectors, uint32_t dimension, cuvs::distance::DistanceType m,
                       uint32_t nthread)
        : Dimension(dimension), Count(static_cast<uint32_t>(count_vectors)), Metric(m) {
        Worker = std::make_unique<CuvsWorker>(nthread);

        // Resize flattened_host_dataset and copy data from the flattened array
        flattened_host_dataset.resize(Count * Dimension); // Total elements
        std::copy(dataset_data, dataset_data + (Count * Dimension), flattened_host_dataset.begin());
    }

    void Load() {
        std::unique_lock<std::shared_mutex> lock(mutex_); // Acquire exclusive lock
        std::promise<bool> init_complete_promise;
        std::future<bool> init_complete_future = init_complete_promise.get_future();

        auto init_fn = [&](RaftHandleWrapper& handle) -> std::any {
            if (flattened_host_dataset.empty()) { // Use new member
                Index = nullptr; // Ensure Index is null if no data
                init_complete_promise.set_value(true); // Signal completion even if empty
                return std::any();
            }

            // Create host_matrix from flattened_host_dataset
            // HostDataset.size() is Count, HostDataset[0].size() is Dimension
            auto dataset_host_matrix = raft::make_host_matrix<T, int64_t, raft::layout_c_contiguous>(
                *handle.get_raft_resources(), static_cast<int64_t>(Count), static_cast<int64_t>(Dimension));
            
            // Single std::copy from flattened_host_dataset to dataset_host_matrix
            std::copy(flattened_host_dataset.begin(), flattened_host_dataset.end(), dataset_host_matrix.data_handle());

            auto dataset_device = raft::make_device_matrix<T, int64_t, raft::layout_c_contiguous>(
                *handle.get_raft_resources(), static_cast<int64_t>(dataset_host_matrix.extent(0)), static_cast<int64_t>(dataset_host_matrix.extent(1)));
            RAFT_CUDA_TRY(cudaMemcpy(dataset_device.data_handle(), dataset_host_matrix.data_handle(),
                                     dataset_host_matrix.size() * sizeof(T), cudaMemcpyHostToDevice));

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

    SearchResult Search(const T* queries_data, uint64_t num_queries, uint32_t query_dimension, uint32_t limit) {
        if (!queries_data || num_queries == 0 || Dimension == 0) { // Check for invalid input
            return SearchResult{};
        }
        if (query_dimension != this->Dimension) {
            throw std::runtime_error("Query dimension does not match index dimension.");
        }
        if (limit == 0) {
            // Return empty vectors of correct dimensions for the number of queries
            std::vector<std::vector<int64_t>> neighbors_vec(num_queries);
            std::vector<std::vector<float>> distances_vec(num_queries);
            return SearchResult{neighbors_vec, distances_vec};
        }
        if (!Index) {
            return SearchResult{};
        }

        size_t queries_rows = num_queries;
        size_t queries_cols = Dimension; // Use the class's Dimension

        uint64_t jobID = Worker->Submit(
            [&](RaftHandleWrapper& handle) -> std::any {
                std::shared_lock<std::shared_mutex> lock(mutex_); // Acquire shared read-only lock inside worker thread
                // Create host_matrix directly from flattened queries_data
                // No need for intermediate std::vector<std::vector<T>>
                auto queries_host_matrix = raft::make_host_matrix<T, int64_t, raft::layout_c_contiguous>(
                    *handle.get_raft_resources(), static_cast<int64_t>(queries_rows), static_cast<int64_t>(queries_cols));
                
                // Copy the flattened data to queries_host_matrix
                std::copy(queries_data, queries_data + (queries_rows * queries_cols), queries_host_matrix.data_handle());

                auto queries_device = raft::make_device_matrix<T, int64_t, raft::layout_c_contiguous>(
                    *handle.get_raft_resources(), static_cast<int64_t>(queries_host_matrix.extent(0)), static_cast<int64_t>(queries_host_matrix.extent(1)));
                RAFT_CUDA_TRY(cudaMemcpy(queries_device.data_handle(), queries_host_matrix.data_handle(),
                                         queries_host_matrix.size() * sizeof(T), cudaMemcpyHostToDevice));

                auto neighbors_device = raft::make_device_matrix<int64_t, int64_t, raft::layout_c_contiguous>(
                    *handle.get_raft_resources(), static_cast<int64_t>(queries_rows), static_cast<int64_t>(limit));
                auto distances_device = raft::make_device_matrix<float, int64_t, raft::layout_c_contiguous>(
                    *handle.get_raft_resources(), static_cast<int64_t>(queries_rows), static_cast<int64_t>(limit));

                cuvs::neighbors::brute_force::search_params search_params;
                cuvs::neighbors::brute_force::index<T, float>& index_obj = *Index;
                cuvs::neighbors::brute_force::search(*handle.get_raft_resources(), search_params, index_obj,
                                                     raft::make_const_mdspan(queries_device.view()), neighbors_device.view(), distances_device.view());

                raft::resource::sync_stream(*handle.get_raft_resources());

                auto neighbors_host = raft::make_host_matrix<int64_t, int64_t, raft::layout_c_contiguous>(
                    *handle.get_raft_resources(), static_cast<int64_t>(neighbors_device.extent(0)), static_cast<int64_t>(neighbors_device.extent(1)));
                auto distances_host = raft::make_host_matrix<float, int64_t, raft::layout_c_contiguous>(
                    *handle.get_raft_resources(), static_cast<int64_t>(distances_device.extent(0)), static_cast<int64_t>(distances_device.extent(1)));
                
                RAFT_CUDA_TRY(cudaMemcpy(neighbors_host.data_handle(), neighbors_device.data_handle(),
                                         neighbors_host.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
                RAFT_CUDA_TRY(cudaMemcpy(distances_host.data_handle(), distances_device.data_handle(),
                                         distances_host.size() * sizeof(float), cudaMemcpyDeviceToHost));

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

                        if (neighbor_idx != std::numeric_limits<int64_t>::max() &&
                            !std::isinf(distance_val) &&
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
