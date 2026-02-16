#include "brute_force_c.h"
#include "../cpp/brute_force.hpp" // For C++ GpuBruteForceIndex
#include <iostream>    // For error logging
#include <stdexcept>   // For std::runtime_error
#include <vector>      // For std::vector
#include <algorithm>   // For std::copy
#include <cstdlib>     // For malloc, free

// Helper to convert C enum to C++ enum
cuvs::distance::DistanceType convert_distance_type(CuvsDistanceTypeC metric_c) {
    switch (metric_c) {
        case DistanceType_L2Expanded: return cuvs::distance::DistanceType::L2Expanded;
        case DistanceType_L1: return cuvs::distance::DistanceType::L1;
        case DistanceType_InnerProduct: return cuvs::distance::DistanceType::InnerProduct;
        // Add other cases as needed
        default:
            std::cerr << "Error: Unknown distance type: " << metric_c << std::endl;
            throw std::runtime_error("Unknown distance type");
    }
}

// Constructor for GpuBruteForceIndex
GpuBruteForceIndexC GpuBruteForceIndex_New(const float* dataset_data, uint64_t count_vectors, uint32_t dimension, CuvsDistanceTypeC metric_c, uint32_t nthread) {
    try {
        cuvs::distance::DistanceType metric = convert_distance_type(metric_c);
        matrixone::GpuBruteForceIndex<float>* index = new matrixone::GpuBruteForceIndex<float>(dataset_data, count_vectors, dimension, metric, nthread);
        return static_cast<GpuBruteForceIndexC>(index);
    } catch (const std::exception& e) {
        std::cerr << "Error in GpuBruteForceIndex_New: " << e.what() << std::endl;
        return nullptr;
    }
}

// Loads the index to the GPU
void GpuBruteForceIndex_Load(GpuBruteForceIndexC index_c) {
    try {
        matrixone::GpuBruteForceIndex<float>* index = static_cast<matrixone::GpuBruteForceIndex<float>*>(index_c);
        if (index) {
            index->Load();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in GpuBruteForceIndex_Load: " << e.what() << std::endl;
    }
}

// Performs a search operation
CuvsSearchResultC GpuBruteForceIndex_Search(GpuBruteForceIndexC index_c, const float* queries_data, uint64_t num_queries, uint32_t query_dimension, uint32_t limit) {
    CuvsSearchResultC result_c = {nullptr, nullptr, 0, 0, 0}; // Initialize all members
    try {
        matrixone::GpuBruteForceIndex<float>* index = static_cast<matrixone::GpuBruteForceIndex<float>*>(index_c);
        if (index) {
            matrixone::GpuBruteForceIndex<float>::SearchResult search_result = index->Search(queries_data, num_queries, query_dimension, limit);

            uint64_t total_neighbors_elements = 0;
            uint64_t total_distances_elements = 0;
            
            // Calculate total elements needed for flattened arrays
            // Note: search_result.Neighbors[i].size() could be less than 'limit'
            for (size_t i = 0; i < search_result.Neighbors.size(); ++i) {
                total_neighbors_elements += search_result.Neighbors[i].size();
                total_distances_elements += search_result.Distances[i].size();
            }

            result_c.neighbors = (int64_t*)malloc(total_neighbors_elements * sizeof(int64_t));
            result_c.distances = (float*)malloc(total_distances_elements * sizeof(float));

            if (!result_c.neighbors || !result_c.distances) {
                // Handle malloc failure
                std::cerr << "Error: Memory allocation failed in GpuBruteForceIndex_Search." << std::endl;
                free(result_c.neighbors); // Free if one succeeded and other failed
                free(result_c.distances);
                return {nullptr, nullptr, 0, 0, 0};
            }

            uint64_t current_neighbor_offset = 0;
            uint64_t current_distance_offset = 0;
            for (size_t i = 0; i < search_result.Neighbors.size(); ++i) {
                std::copy(search_result.Neighbors[i].begin(), search_result.Neighbors[i].end(), result_c.neighbors + current_neighbor_offset);
                std::copy(search_result.Distances[i].begin(), search_result.Distances[i].end(), result_c.distances + current_distance_offset);
                current_neighbor_offset += search_result.Neighbors[i].size();
                current_distance_offset += search_result.Distances[i].size();
            }
            
            result_c.num_queries = num_queries;
            result_c.limit = limit;
            // The actual_k is per query, but we are returning flattened arrays.
            // For the C interface, it might be more useful to indicate the total number of found neighbors per query.
            // For now, let's keep it simple and assume the caller knows the structure.
            // If num_queries > 0, and search_result.Neighbors[0] is not empty, we can infer actual_k.
            // If the search result always returns 'limit' items per query, then actual_k = limit.
            // If it returns less, then actual_k for each query could be different.
            // For simplicity, let's assume actual_k is the number of results per query if it's consistent.
            // If the internal logic of SearchResult is always to return results up to 'limit' per query,
            // then actual_k would be 'limit', or the actual size of the first neighbor vector if results can vary.
            // Assuming that for a well-formed query, search_result.Neighbors[0].size() will give the actual count for that query.
            // However, this value could differ for different queries.
            // For simplicity in the C struct, actual_k will be the 'limit' requested unless modified.
            // A more robust solution might return an array of actual_k for each query.
            result_c.actual_k = search_result.Neighbors.empty() ? 0 : search_result.Neighbors[0].size();
            // This is a simplification; ideally, CuvsSearchResultC would store actual_k per query.
            // For a flattened array, to reconstruct, the caller needs to know how many elements belong to each query.
            // This assumes a consistent 'limit' for each query or requires an additional array of lengths.
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in GpuBruteForceIndex_Search: " << e.what() << std::endl;
        // Clean up any allocated memory if an error occurred after allocation
        if (result_c.neighbors) free(result_c.neighbors);
        if (result_c.distances) free(result_c.distances);
        result_c.neighbors = nullptr;
        result_c.distances = nullptr;
        result_c.num_queries = 0;
        result_c.limit = 0;
        result_c.actual_k = 0;
    }
    return result_c;
}

// Destroys the GpuBruteForceIndex object and frees associated resources
void GpuBruteForceIndex_Destroy(GpuBruteForceIndexC index_c) {
    try {
        matrixone::GpuBruteForceIndex<float>* index = static_cast<matrixone::GpuBruteForceIndex<float>*>(index_c);
        if (index) {
            delete index;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in GpuBruteForceIndex_Destroy: " << e.what() << std::endl;
    }
}

// Frees the memory allocated for a CuvsSearchResultC object
void CuvsSearchResult_Free(CuvsSearchResultC result_c) {
    free(result_c.neighbors);
    free(result_c.distances);
}