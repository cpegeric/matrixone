// Copyright 2024 Matrix Origin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package aggexec

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/common/mpool"
	"github.com/matrixorigin/matrixone/pkg/common/util"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/sql/colexec/aggexec/algos/kmeans"
	"github.com/matrixorigin/matrixone/pkg/sql/colexec/aggexec/algos/kmeans/elkans"
	"github.com/matrixorigin/matrixone/pkg/vectorize/moarray"
)

const (
	defaultKmeansMaxIteration   = 500
	defaultKmeansDeltaThreshold = 0.01
	defaultKmeansDistanceType   = kmeans.L2Distance
	defaultKmeansInitType       = kmeans.Random
	defaultKmeansClusterCnt     = 1
	defaultKmeansNormalize      = false

	configSeparator = ","
)

var (
	ClusterCentersSupportTypes = []types.T{
		types.T_array_float32, types.T_array_float64,
	}

	distTypeStrToEnum = map[string]kmeans.DistanceType{
		"vector_l2_ops":     kmeans.L2Distance,
		"vector_ip_ops":     kmeans.InnerProduct,
		"vector_cosine_ops": kmeans.CosineDistance,
		"vector_l1_ops":     kmeans.L1Distance,
	}

	initTypeStrToEnum = map[string]kmeans.InitType{
		"random":         kmeans.Random,
		"kmeansplusplus": kmeans.KmeansPlusPlus,
	}
)

func ClusterCentersReturnType(argType []types.Type) types.Type {
	return types.T_varchar.ToType()
}

type clusterCentersExec struct {
	singleAggInfo
	arg sBytesArg
	ret aggResultWithBytesType

	// groupData hold the inputting []byte.
	// todo: there is a problem here, if the input is large, it will cause the memory overflow of one vector (memory usage > 1gb).
	groupData []*vector.Vector

	// Kmeans parameters.
	clusterCnt uint64
	distType   kmeans.DistanceType
	initType   kmeans.InitType
	normalize  bool
}

func (exec *clusterCentersExec) GetOptResult() SplitResult {
	return &exec.ret.optSplitResult
}

func (exec *clusterCentersExec) marshal() ([]byte, error) {
	d := exec.singleAggInfo.getEncoded()
	r, em, err := exec.ret.marshalToBytes()
	if err != nil {
		return nil, err
	}

	encoded := EncodedAgg{
		Info:    d,
		Result:  r,
		Empties: em,
		Groups:  nil,
	}

	encoded.Groups = make([][]byte, len(exec.groupData)+1)
	if len(exec.groupData) > 0 {
		for i := range exec.groupData {
			if encoded.Groups[i], err = exec.groupData[i].MarshalBinary(); err != nil {
				return nil, err
			}
		}
	}

	{
		t1 := uint16(exec.distType)
		t2 := uint16(exec.initType)

		bs := types.EncodeUint64(&exec.clusterCnt)
		bs = append(bs, types.EncodeUint16(&t1)...)
		bs = append(bs, types.EncodeUint16(&t2)...)
		bs = append(bs, types.EncodeBool(&exec.normalize)...)
		encoded.Groups[len(encoded.Groups)-1] = bs
	}
	return encoded.Marshal()
}

func (exec *clusterCentersExec) unmarshal(mp *mpool.MPool, result, empties, groups [][]byte) error {
	if err := exec.ret.unmarshalFromBytes(result, empties); err != nil {
		return err
	}
	if len(groups) > 0 {
		exec.groupData = make([]*vector.Vector, len(groups)-1)
		for i := range exec.groupData {
			exec.groupData[i] = vector.NewVec(exec.singleAggInfo.argType)
			if err := vectorUnmarshal(exec.groupData[i], groups[i], mp); err != nil {
				return err
			}
		}
		bs := groups[len(groups)-1]
		if len(bs) != 13 { // 8+2+2+1
			return moerr.NewInternalErrorNoCtx("invalid cluster center exec data")
		}
		exec.clusterCnt = types.DecodeUint64(bs[:8])
		exec.distType = kmeans.DistanceType(types.DecodeUint16(bs[8:10]))
		exec.initType = kmeans.InitType(types.DecodeUint16(bs[10:12]))
		exec.normalize = types.DecodeBool(bs[12:])
	}
	return nil
}

func newClusterCentersExecutor(mg AggMemoryManager, info singleAggInfo) (AggFuncExec, error) {
	if info.distinct {
		return nil, moerr.NewInternalErrorNoCtx("do not support distinct for cluster_centers()")
	}
	return &clusterCentersExec{
		singleAggInfo: info,
		ret:           initAggResultWithBytesTypeResult(mg, info.retType, true, ""),
		clusterCnt:    defaultKmeansClusterCnt,
		distType:      defaultKmeansDistanceType,
		initType:      defaultKmeansInitType,
		normalize:     defaultKmeansNormalize,
	}, nil
}

func (exec *clusterCentersExec) GroupGrow(more int) error {
	if err := exec.ret.grows(more); err != nil {
		return err
	}
	for i := 0; i < more; i++ {
		exec.groupData = append(exec.groupData, vector.NewVec(types.T_varchar.ToType()))
	}
	return nil
}

func (exec *clusterCentersExec) PreAllocateGroups(more int) error {
	return exec.ret.preExtend(more)
}

func (exec *clusterCentersExec) Fill(groupIndex int, row int, vectors []*vector.Vector) error {
	if vectors[0].IsNull(uint64(row)) {
		return nil
	}
	if vectors[0].IsConst() {
		row = 0
	}

	x, y := exec.ret.updateNextAccessIdx(groupIndex)
	exec.ret.setGroupNotEmpty(x, y)
	return vectorAppendBytesWildly(exec.groupData[groupIndex], exec.ret.mp, vectors[0].GetBytesAt(row))
}

func (exec *clusterCentersExec) BulkFill(groupIndex int, vectors []*vector.Vector) error {
	if vectors[0].IsConstNull() {
		return nil
	}

	x, y := exec.ret.updateNextAccessIdx(groupIndex)

	if vectors[0].IsConst() {
		value := vectors[0].GetBytesAt(0)
		exec.ret.setGroupNotEmpty(x, y)
		for i, j := uint64(0), uint64(vectors[0].Length()); i < j; i++ {
			if err := vectorAppendBytesWildly(exec.groupData[groupIndex], exec.ret.mp, value); err != nil {
				return err
			}
		}
		return nil
	}

	exec.arg.prepare(vectors[0])
	for i, j := uint64(0), uint64(vectors[0].Length()); i < j; i++ {
		v, null := exec.arg.w.GetStrValue(i)
		if null {
			continue
		}
		exec.ret.setGroupNotEmpty(x, y)
		if err := vectorAppendBytesWildly(exec.groupData[groupIndex], exec.ret.mp, v); err != nil {
			return err
		}
	}
	return nil
}

func (exec *clusterCentersExec) BatchFill(offset int, groups []uint64, vectors []*vector.Vector) error {
	if vectors[0].IsConstNull() {
		return nil
	}

	if vectors[0].IsConst() {
		value := vectors[0].GetBytesAt(0)
		for i := 0; i < len(groups); i++ {
			if groups[i] != GroupNotMatched {
				groupIndex := int(groups[i] - 1)
				x, y := exec.ret.updateNextAccessIdx(groupIndex)

				exec.ret.setGroupNotEmpty(x, y)
				if err := vectorAppendBytesWildly(
					exec.groupData[groupIndex],
					exec.ret.mp, value); err != nil {
					return err
				}
			}
		}
		return nil
	}

	exec.arg.prepare(vectors[0])
	for i, j, idx := uint64(offset), uint64(offset+len(groups)), 0; i < j; i++ {
		if groups[idx] != GroupNotMatched {
			v, null := exec.arg.w.GetStrValue(i)
			if !null {
				groupIndex := int(groups[idx] - 1)
				x, y := exec.ret.updateNextAccessIdx(groupIndex)

				exec.ret.setGroupNotEmpty(x, y)
				if err := vectorAppendBytesWildly(
					exec.groupData[groupIndex],
					exec.ret.mp, v); err != nil {
					return err
				}
			}
		}
		idx++
	}
	return nil
}

func (exec *clusterCentersExec) Merge(next AggFuncExec, groupIdx1 int, groupIdx2 int) error {
	other := next.(*clusterCentersExec)
	if other.groupData[groupIdx2] == nil || other.groupData[groupIdx2].Length() == 0 {
		return nil
	}

	if exec.groupData[groupIdx1] == nil || exec.groupData[groupIdx1].Length() == 0 {
		exec.groupData[groupIdx1] = other.groupData[groupIdx2]
		other.groupData[groupIdx2] = nil
		return nil
	}

	otherBat := other.groupData[groupIdx2]
	if err := exec.groupData[groupIdx1].UnionBatch(otherBat, 0, otherBat.Length(), nil, exec.ret.mp); err != nil {
		return err
	}
	other.groupData[groupIdx2] = nil

	x1, y1 := exec.ret.updateNextAccessIdx(groupIdx1)
	x2, y2 := other.ret.updateNextAccessIdx(groupIdx2)

	exec.ret.MergeAnotherEmpty(x1, y1, other.ret.isGroupEmpty(x2, y2))
	return nil
}

func (exec *clusterCentersExec) BatchMerge(next AggFuncExec, offset int, groups []uint64) error {
	for i, group := range groups {
		if group != GroupNotMatched {
			if err := exec.Merge(next, int(group)-1, i+offset); err != nil {
				return err
			}
		}
	}
	return nil
}

func (exec *clusterCentersExec) Flush() ([]*vector.Vector, error) {
	switch exec.singleAggInfo.argType.Oid {
	case types.T_array_float32:
		if err := exec.flushArray32(); err != nil {
			return nil, err
		}
	case types.T_array_float64:
		if err := exec.flushArray64(); err != nil {
			return nil, err
		}
	default:
		return nil, moerr.NewInternalErrorNoCtxf(
			"unsupported type '%s' for cluster_centers", exec.singleAggInfo.argType.String())
	}

	return exec.ret.flushAll(), nil
}

func (exec *clusterCentersExec) flushArray32() error {
	for i, group := range exec.groupData {
		exec.ret.updateNextAccessIdx(i)

		if group == nil || group.Length() == 0 {
			continue
		}

		bts, area := vector.MustVarlenaRawData(group)
		// todo: it's bad here this f64s is out of the memory control.
		f64s := make([][]float64, group.Length())
		for m := range f64s {
			f32s := types.BytesToArray[float32](bts[m].GetByteSlice(area))
			f64s[m] = make([]float64, len(f32s))
			for n := range f32s {
				f64s[m][n] = float64(f32s[n])
			}
		}

		centers, err := exec.getCentersByKmeansAlgorithm(f64s)
		if err != nil {
			return err
		}
		res, err := exec.arraysToString(centers)
		if err != nil {
			return err
		}
		if err = exec.ret.set(util.UnsafeStringToBytes(res)); err != nil {
			return err
		}
	}
	return nil
}

func (exec *clusterCentersExec) flushArray64() error {
	for i, group := range exec.groupData {
		exec.ret.updateNextAccessIdx(i)

		if group == nil || group.Length() == 0 {
			continue
		}

		bts, area := vector.MustVarlenaRawData(group)
		f64s := make([][]float64, group.Length())
		for m := range f64s {
			f64s[m] = types.BytesToArray[float64](bts[m].GetByteSlice(area))
		}

		centers, err := exec.getCentersByKmeansAlgorithm(f64s)
		if err != nil {
			return err
		}
		res, err := exec.arraysToString(centers)
		if err != nil {
			return err
		}
		if err = exec.ret.set(util.UnsafeStringToBytes(res)); err != nil {
			return err
		}
	}
	return nil
}

func (exec *clusterCentersExec) getCentersByKmeansAlgorithm(f64s [][]float64) ([][]float64, error) {
	var clusterer kmeans.Clusterer
	var centers [][]float64
	var err error

	if clusterer, err = elkans.NewKMeans(
		f64s, int(exec.clusterCnt),
		defaultKmeansMaxIteration,
		defaultKmeansDeltaThreshold,
		exec.distType,
		exec.initType,
		exec.normalize); err != nil {
		return nil, err
	}
	if centers, err = clusterer.Cluster(); err != nil {
		return nil, err
	}

	return centers, nil
}

// converts [][]float64 to json string.
func (exec *clusterCentersExec) arraysToString(centers [][]float64) (res string, err error) {
	switch exec.singleAggInfo.argType.Oid {
	case types.T_array_float32:
		// cast [][]float64 to [][]float32
		_centers := make([][]float32, len(centers))
		for i, center := range centers {
			_centers[i], err = moarray.Cast[float64, float32](center)
			if err != nil {
				return "", err
			}
		}

		// comments that copied from old code.
		// create json string for [][]float32
		// NOTE: here we can't use jsonMarshall as it does not accept precision as done in ArraysToString
		// We need precision here, as it is the final output that will be printed on SQL console.
		res = fmt.Sprintf("[ %s ]", types.ArraysToString[float32](_centers, ","))

	case types.T_array_float64:
		res = fmt.Sprintf("[ %s ]", types.ArraysToString[float64](centers, ","))
	}
	return res, nil
}

func (exec *clusterCentersExec) Free() {
	exec.ret.free()
	if exec.ret.mp == nil {
		return
	}
	for _, v := range exec.groupData {
		if v == nil {
			continue
		}

		v.Free(exec.ret.mp)
	}
}

func (exec *clusterCentersExec) SetExtraInformation(partialResult any, groupIndex int) error {
	if bts, ok := partialResult.([]byte); ok {
		k, distType, initType, normalize, err := decodeConfig(bts)
		if err == nil {
			exec.clusterCnt = k
			exec.distType = distType
			exec.initType = initType
			exec.normalize = normalize
		}
		return err
	}
	return nil
}

// that's very bad codes here, because we cannot know how to encode this config.
// support an encode method here is a better way.
func decodeConfig(extra []byte) (
	clusterCount uint64, distType kmeans.DistanceType, initType kmeans.InitType, normalize bool, err error) {
	// decode methods.
	parseClusterCount := func(s string) (uint64, error) {
		return strconv.ParseUint(strings.TrimSpace(s), 10, 64)
	}
	parseDistType := func(s string) (kmeans.DistanceType, error) {
		v := strings.ToLower(s)
		if res, ok := distTypeStrToEnum[v]; ok {
			return res, nil
		}
		return 0, moerr.NewInternalErrorNoCtxf("unsupported distance_type '%s' for cluster_centers", v)
	}
	parseInitType := func(s string) (kmeans.InitType, error) {
		if res, ok := initTypeStrToEnum[s]; ok {
			return res, nil
		}
		return 0, moerr.NewInternalErrorNoCtxf("unsupported init_type '%s' for cluster_centers", s)
	}

	if len(extra) == 0 {
		return defaultKmeansClusterCnt, defaultKmeansDistanceType, defaultKmeansInitType, defaultKmeansNormalize, nil
	}

	configs := strings.Split(string(extra), configSeparator)
	for i := range configs {
		configs[i] = strings.TrimSpace(configs[i])

		switch i {
		case 0:
			clusterCount, err = parseClusterCount(configs[i])
		case 1:
			distType, err = parseDistType(configs[i])
		case 2:
			initType, err = parseInitType(configs[i])
		case 3:
			normalize, err = strconv.ParseBool(configs[i])
		}
		if err != nil {
			return defaultKmeansClusterCnt, defaultKmeansDistanceType, defaultKmeansInitType, defaultKmeansNormalize, err
		}
	}
	return
}
