// Copyright 2026 Matrix Origin
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

package objectio

import (
	"sync/atomic"
	"time"

	"github.com/matrixorigin/matrixone/pkg/fileservice"
)

const (
	metaTouchSlots    = 1 << 16
	metaTouchInterval = 10 * time.Second
)

// metaTouched holds, per slot of cache keys, when a hit of one of them last told the file
// service of a use.
var metaTouched [metaTouchSlots]atomic.Int64

// touchDue reports whether a hit of key in metaCache tells the file service of a use: at
// most once an interval per slot of keys.
func touchDue(key mataCacheKey) bool {
	slot := &metaTouched[shardMetaCacheKey(key)%metaTouchSlots]
	now := time.Now().UnixNano()
	last := slot.Load()
	return now-last >= int64(metaTouchInterval) && slot.CompareAndSwap(last, now)
}

// touchMeta tells fs, when due, of a use of the meta of name at extent that a hit of key
// in metaCache served.
func touchMeta(key mataCacheKey, fs fileservice.FileService, name *ObjectName, extent *Extent) {
	if touchDue(key) {
		fileservice.TouchMeta(fs, name.String(), int64(extent.Offset()), int64(extent.Length()))
	}
}

// touchBF tells fs, when due, of a use of the bloom filter of location that a hit of key
// in metaCache served, when the object's meta is cached to find its extent.
func touchBF(key mataCacheKey, fs fileservice.FileService, location Location, meta ObjectDataMeta) {
	if !touchDue(key) {
		return
	}
	if meta == nil {
		v, ok := getMeta(encodeCacheKey(*location.ShortName(), cacheKeyTypeMeta))
		if !ok {
			return
		}
		m, err := checkedObjectMeta(v)
		if err != nil {
			return
		}
		data, ok := m.DataMeta()
		if !ok {
			return
		}
		meta = data
	}
	extent := meta.BlockHeader().BFExtent()
	fileservice.TouchMeta(fs, location.Name().String(), int64(extent.Offset()), int64(extent.Length()))
}
