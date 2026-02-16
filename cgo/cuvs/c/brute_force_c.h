#ifndef BRUTE_FORCE_C_H
#define BRUTE_FORCE_C_H

#include <stdint.h> // For uint32_t, uint64_t
#include <stddef.h> // For size_t

#ifdef __cplusplus
extern "C" {
#endif

// Define a C-compatible enum for distance types
typedef enum {
    DistanceType_L2Expanded = 0,
    DistanceType_L1,
    DistanceType_InnerProduct,
    DistanceType_CosineSimilarity,
    DistanceType_Jaccard,
    DistanceType_Hamming,
    DistanceType_Unknown // Should not happen
} CuvsDistanceTypeC;

// Opaque pointer to the C++ GpuBruteForceIndex object
typedef void* GpuBruteForceIndexC;

// Structure to hold the search results
typedef struct {
    int64_t* neighbors;     // Flattened array of neighbor indices
    float* distances;       // Flattened array of distances
    uint64_t num_queries;   // Number of query vectors for this result
    uint32_t limit;         // Number of neighbors per query
    uint32_t actual_k;      // Actual number of neighbors found per query (min of limit and available)
} CuvsSearchResultC;

// Constructor for GpuBruteForceIndex
// dataset_data: Flattened array of dataset vectors
// count_vectors: Number of vectors in the dataset
// dimension: Dimension of each vector
// metric: Distance metric to use
// nthread: Number of worker threads
GpuBruteForceIndexC GpuBruteForceIndex_New(const float* dataset_data, uint64_t count_vectors, uint32_t dimension, CuvsDistanceTypeC metric, uint32_t nthread);

// Loads the index to the GPU
// index_c: Opaque pointer to the GpuBruteForceIndex object
void GpuBruteForceIndex_Load(GpuBruteForceIndexC index_c);

// Performs a search operation
// index_c: Opaque pointer to the GpuBruteForceIndex object
// queries_data: Flattened array of query vectors
// num_queries: Number of query vectors
// query_dimension: Dimension of each query vector (must match index dimension)
// limit: Maximum number of neighbors to return per query
CuvsSearchResultC GpuBruteForceIndex_Search(GpuBruteForceIndexC index_c, const float* queries_data, uint64_t num_queries, uint32_t query_dimension, uint32_t limit);

// Destroys the GpuBruteForceIndex object and frees associated resources
// index_c: Opaque pointer to the GpuBruteForceIndex object
void GpuBruteForceIndex_Destroy(GpuBruteForceIndexC index_c);

// Frees the memory allocated for a CuvsSearchResultC object
// result_c: The CuvsSearchResultC object whose internal arrays need to be freed
void CuvsSearchResult_Free(CuvsSearchResultC result_c);

#ifdef __cplusplus
}
#endif

#endif // BRUTE_FORCE_C_H
