package cuvs

/*
#cgo LDFLAGS: /home/eric/github/matrixone/cgo/cuvs/c/libbrute_force_c.so -Wl,-rpath=/home/eric/github/matrixone/cgo/cuvs/c
#cgo CFLAGS: -I../c

#include "brute_force_c.h"
*/
import "C"
import (
    "fmt"
    "unsafe"
)

// DistanceType maps to C.CuvsDistanceTypeC
type DistanceType C.CuvsDistanceTypeC

const (
    L2Expanded      DistanceType = C.DistanceType_L2Expanded
    L1              DistanceType = C.DistanceType_L1
    InnerProduct    DistanceType = C.DistanceType_InnerProduct
    CosineSimilarity DistanceType = C.DistanceType_CosineSimilarity
    Jaccard         DistanceType = C.DistanceType_Jaccard
    Hamming         DistanceType = C.DistanceType_Hamming
    Unknown         DistanceType = C.DistanceType_Unknown
)

// GpuBruteForceIndex represents the C++ GpuBruteForceIndex object
type GpuBruteForceIndex struct {
    cIndex C.GpuBruteForceIndexC
}

// SearchResult maps to C.CuvsSearchResultC
type SearchResult struct {
    Neighbors []int64
    Distances []float32
    NumQueries uint64
    Limit      uint32
    ActualK    uint32
    // Internally, keep the C struct pointer to free memory later
    cResult C.CuvsSearchResultC
}

// NewGpuBruteForceIndex creates a new GpuBruteForceIndex instance
func NewGpuBruteForceIndex(dataset []float32, countVectors uint64, dimension uint32, metric DistanceType, nthread uint32) (*GpuBruteForceIndex, error) {
    if len(dataset) == 0 || countVectors == 0 || dimension == 0 {
        return nil, fmt.Errorf("dataset, countVectors, and dimension cannot be zero")
    }
    if uint64(len(dataset)) != countVectors * uint64(dimension) {
        return nil, fmt.Errorf("dataset size (%d) does not match countVectors (%d) * dimension (%d)", len(dataset), countVectors, dimension)
    }

    cIndex := C.GpuBruteForceIndex_New(
        (*C.float)(&dataset[0]),
        C.uint64_t(countVectors),
        C.uint32_t(dimension),
        C.CuvsDistanceTypeC(metric),
        C.uint32_t(nthread),
    )
    if cIndex == nil {
        return nil, fmt.Errorf("failed to create GpuBruteForceIndex")
    }
    return &GpuBruteForceIndex{cIndex: cIndex}, nil
}

// Load loads the index to the GPU
func (gbi *GpuBruteForceIndex) Load() error {
    if gbi.cIndex == nil {
        return fmt.Errorf("GpuBruteForceIndex is not initialized")
    }
    C.GpuBruteForceIndex_Load(gbi.cIndex)
    // C functions print errors to stderr, more robust error handling could be added to C interface
    return nil
}

// Search performs a search operation
func (gbi *GpuBruteForceIndex) Search(queries []float32, numQueries uint64, queryDimension uint32, limit uint32) (SearchResult, error) {
    if gbi.cIndex == nil {
        return SearchResult{}, fmt.Errorf("GpuBruteForceIndex is not initialized")
    }
    if len(queries) == 0 || numQueries == 0 || queryDimension == 0 {
        return SearchResult{}, fmt.Errorf("queries, numQueries, and queryDimension cannot be zero")
    }
    if uint64(len(queries)) != numQueries * uint64(queryDimension) {
        return SearchResult{}, fmt.Errorf("queries size (%d) does not match numQueries (%d) * queryDimension (%d)", len(queries), numQueries, queryDimension)
    }

    var cQueries *C.float
    if len(queries) > 0 {
        cQueries = (*C.float)(&queries[0])
    }

    cResult := C.GpuBruteForceIndex_Search(
        gbi.cIndex,
        cQueries,
        C.uint64_t(numQueries),
        C.uint32_t(queryDimension),
        C.uint32_t(limit),
    )

    // Check for errors returned by C.GpuBruteForceIndex_Search
    if cResult.neighbors == nil && cResult.distances == nil && cResult.num_queries == 0 {
        // This is a simplistic error check. A more robust C interface would return error codes.
        return SearchResult{}, fmt.Errorf("C.GpuBruteForceIndex_Search returned empty result, possibly due to an internal C++ error")
    }

    // Determine the total size of the flattened arrays
    // This assumes the C side returned contiguous blocks of data
    totalNeighborsElements := uint64(cResult.num_queries) * uint64(cResult.actual_k)
    totalDistancesElements := uint64(cResult.num_queries) * uint64(cResult.actual_k)
    
    // Safely create Go slices from C arrays
    // C.int64_t and C.float are Go types wrapping the C types
    goNeighbors := unsafe.Slice((*int64)(unsafe.Pointer(cResult.neighbors)), totalNeighborsElements)
    goDistances := unsafe.Slice((*float32)(unsafe.Pointer(cResult.distances)), totalDistancesElements)

    return SearchResult{
        Neighbors:  goNeighbors,
        Distances:  goDistances,
        NumQueries: uint64(cResult.num_queries),
        Limit:      uint32(cResult.limit),
        ActualK:    uint32(cResult.actual_k),
        cResult:    cResult, // Store C result struct to free later
    }, nil
}

// Destroy frees the C++ GpuBruteForceIndex instance
func (gbi *GpuBruteForceIndex) Destroy() error {
    if gbi.cIndex == nil {
        return fmt.Errorf("GpuBruteForceIndex is not initialized")
    }
    C.GpuBruteForceIndex_Destroy(gbi.cIndex)
    gbi.cIndex = nil // Mark as destroyed
    return nil
}

// Free frees the memory allocated for the C SearchResult.
// This MUST be called by the Go code after it's done with the SearchResult.
func (sr *SearchResult) Free() {
    if sr.cResult.neighbors != nil || sr.cResult.distances != nil {
        C.CuvsSearchResult_Free(sr.cResult)
        sr.cResult.neighbors = nil // Mark as freed
        sr.cResult.distances = nil
    }
}
