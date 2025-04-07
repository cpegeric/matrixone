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

package fifocache

import (
	"context"
	"sync"
	"sync/atomic"

	"golang.org/x/sys/cpu"

	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
)

const numShards = 256

// Cache implements an in-memory cache with FIFO-based eviction
// it's mostly like the S3-fifo, only without the ghost queue part
type Cache[K comparable, V any] struct {
	capacity     fscache.CapacityFunc
	capacity1    fscache.CapacityFunc
	keyShardFunc func(K) uint64

	postSet   func(ctx context.Context, key K, value V, size int64)
	postGet   func(ctx context.Context, key K, value V, size int64)
	postEvict func(ctx context.Context, key K, value V, size int64)

	shards [numShards]struct {
		sync.Mutex
		values map[K]*_CacheItem[K, V]
		_      cpu.CacheLinePad
	}

	queueLock sync.Mutex
	usedSmall atomic.Int64
	small     Queue[*_CacheItem[K, V]]
	usedMain  atomic.Int64
	main      Queue[*_CacheItem[K, V]]
	ghost     *ghost[K]

	capacityCut atomic.Int64
}

type _CacheItem[K comparable, V any] struct {
	key   K
	value V
	size  int64
	count atomic.Int32
}

func (c *_CacheItem[K, V]) inc() {
	for {
		cur := c.count.Load()
		if cur >= 3 {
			return
		}
		if c.count.CompareAndSwap(cur, cur+1) {
			return
		}
	}
}

func (c *_CacheItem[K, V]) dec() {
	for {
		cur := c.count.Load()
		if cur <= 0 {
			return
		}
		if c.count.CompareAndSwap(cur, cur-1) {
			return
		}
	}
}

// assume cache size is 256K
// if cache capacity is smaller than 4G, ghost size is 100%.  Otherwise, 50%
func estimateGhostSize(capacity int64) int {
	estimate := int(capacity / int64(256000))
	if capacity > 4000000000 { // 4G
		// only 50%
		estimate /= 2
	}
	if estimate < 1000 {
		estimate = 1000
	}
	return estimate

}

func New[K comparable, V any](
	capacity fscache.CapacityFunc,
	keyShardFunc func(K) uint64,
	postSet func(ctx context.Context, key K, value V, size int64),
	postGet func(ctx context.Context, key K, value V, size int64),
	postEvict func(ctx context.Context, key K, value V, size int64),
) *Cache[K, V] {

	ghostsize := estimateGhostSize(capacity())
	ret := &Cache[K, V]{
		capacity: capacity,
		capacity1: func() int64 {
			return capacity() / 10
		},
		small:        *NewQueue[*_CacheItem[K, V]](),
		main:         *NewQueue[*_CacheItem[K, V]](),
		ghost:        newGhost[K](ghostsize),
		keyShardFunc: keyShardFunc,
		postSet:      postSet,
		postGet:      postGet,
		postEvict:    postEvict,
	}
	for i := range ret.shards {
		ret.shards[i].values = make(map[K]*_CacheItem[K, V], 1024)
	}
	return ret
}

func (c *Cache[K, V]) set(ctx context.Context, key K, value V, size int64) *_CacheItem[K, V] {
	shard := &c.shards[c.keyShardFunc(key)%numShards]
	shard.Lock()
	defer shard.Unlock()
	_, ok := shard.values[key]
	if ok {
		// existed
		return nil
	}

	item := &_CacheItem[K, V]{
		key:   key,
		value: value,
		size:  size,
	}
	shard.values[key] = item
	if c.postSet != nil {
		c.postSet(ctx, key, value, size)
	}

	return item
}

func (c *Cache[K, V]) Set(ctx context.Context, key K, value V, size int64) {
	if item := c.set(ctx, key, value, size); item != nil {
		c.queueLock.Lock()
		defer c.queueLock.Unlock()
		// enqueue
		if c.ghost.contains(item.key) {
			c.ghost.remove(item.key)
			c.main.enqueue(item)
			c.usedMain.Add(item.size)
		} else {
			c.small.enqueue(item)
			c.usedSmall.Add(item.size)
		}

		// evict
		c.evictAll(ctx, nil, 0)
	}
}

func (c *Cache[K, V]) Get(ctx context.Context, key K) (value V, ok bool) {
	shard := &c.shards[c.keyShardFunc(key)%numShards]
	shard.Lock()
	defer shard.Unlock()

	var item *_CacheItem[K, V]
	item, ok = shard.values[key]
	if !ok {
		return
	}
	if c.postGet != nil {
		c.postGet(ctx, item.key, item.value, item.size)
	}
	item.inc()
	return item.value, true
}

func (c *Cache[K, V]) Delete(ctx context.Context, key K) {
	shard := &c.shards[c.keyShardFunc(key)%numShards]
	shard.Lock()
	defer shard.Unlock()
	item, ok := shard.values[key]
	if !ok {
		return
	}
	delete(shard.values, key)
	if c.postEvict != nil {
		c.postEvict(ctx, item.key, item.value, item.size)
	}
	// queues will be update in evict
}

func (c *Cache[K, V]) Evict(ctx context.Context, done chan int64, capacityCut int64) {
	if done == nil {
		// can be async
		if c.queueLock.TryLock() {
			defer c.queueLock.Unlock()
		} else {
			if capacityCut > 0 {
				// let the holder do more evict
				c.capacityCut.Add(capacityCut)
			}
			return
		}

	} else {
		if cap(done) < 1 {
			panic("should be buffered chan")
		}
		c.queueLock.Lock()
		defer c.queueLock.Unlock()
	}

	target := c.evictAll(ctx, done, capacityCut)
	if done != nil {
		done <- target
	}

}

// ForceEvict evicts n bytes despite capacity
func (c *Cache[K, V]) ForceEvict(ctx context.Context, n int64) {
	capacityCut := c.capacity() - c.used() + n
	c.Evict(ctx, nil, capacityCut)
}

func (c *Cache[K, V]) used() int64 {
	return c.usedSmall.Load() + c.usedMain.Load()
}

func (c *Cache[K, V]) evictAll(ctx context.Context, done chan int64, capacityCut int64) int64 {
	var target int64
	for {
		globalCapacityCut := c.capacityCut.Swap(0)
		target = c.capacity() - capacityCut - globalCapacityCut
		if target < 0 {
			target = 0
		}
		if c.used() <= target {
			break
		}
		target1 := c.capacity1() - capacityCut - globalCapacityCut
		if target1 < 0 {
			target1 = 0
		}
		if c.usedSmall.Load() > target1 {
			c.evictSmall(ctx)
		} else {
			c.evictMain(ctx)
		}
	}

	return target
}

func (c *Cache[K, V]) evictSmall(ctx context.Context) {
	// queue 1
	for {
		item, ok := c.small.dequeue()
		if !ok {
			// queue empty
			return
		}
		if item.count.Load() > 1 {
			// put main
			c.main.enqueue(item)
			c.usedSmall.Add(-item.size)
			c.usedMain.Add(item.size)
		} else {
			// evict
			c.deleteItem(ctx, item)
			c.usedSmall.Add(-item.size)
			c.ghost.add(item.key)
			return
		}
	}
}

func (c *Cache[K, V]) deleteItem(ctx context.Context, item *_CacheItem[K, V]) {
	shard := &c.shards[c.keyShardFunc(item.key)%numShards]
	shard.Lock()
	defer shard.Unlock()
	delete(shard.values, item.key)
	if c.postEvict != nil {
		c.postEvict(ctx, item.key, item.value, item.size)
	}
}

func (c *Cache[K, V]) evictMain(ctx context.Context) {
	// queue 2
	for {
		item, ok := c.main.dequeue()
		if !ok {
			// empty queue
			break
		}
		if item.count.Load() > 0 {
			// re-enqueue
			item.dec()
			c.main.enqueue(item)
		} else {
			// evict
			c.deleteItem(ctx, item)
			c.usedMain.Add(-item.size)
			return
		}
	}
}
