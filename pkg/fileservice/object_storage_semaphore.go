// Copyright 2023 Matrix Origin
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

package fileservice

import (
	"context"
	"io"
	"iter"
	"sync"
	"time"
)

type objectStorageSemaphore struct {
	upstream  ObjectStorage
	semaphore chan struct{}
}

func newObjectStorageSemaphore(
	upstream ObjectStorage,
	capacity int64,
) *objectStorageSemaphore {
	return &objectStorageSemaphore{
		upstream:  upstream,
		semaphore: make(chan struct{}, capacity),
	}
}

func (o *objectStorageSemaphore) acquire() {
	o.semaphore <- struct{}{}
}

func (o *objectStorageSemaphore) release() {
	<-o.semaphore
}

var _ ObjectStorage = new(objectStorageSemaphore)

func (o *objectStorageSemaphore) Delete(ctx context.Context, keys ...string) (err error) {
	o.acquire()
	defer o.release()
	return o.upstream.Delete(ctx, keys...)
}

func (o *objectStorageSemaphore) Exists(ctx context.Context, key string) (bool, error) {
	o.acquire()
	defer o.release()
	return o.upstream.Exists(ctx, key)
}

func (o *objectStorageSemaphore) List(ctx context.Context, prefix string) iter.Seq2[*DirEntry, error] {
	return func(yield func(*DirEntry, error) bool) {
		o.acquire()
		defer o.release()
		o.upstream.List(ctx, prefix)(yield)
	}
}

func (o *objectStorageSemaphore) Read(ctx context.Context, key string, min *int64, max *int64) (io.ReadCloser, error) {
	o.acquire()
	r, err := o.upstream.Read(ctx, key, min, max)
	if err != nil {
		o.release()
		return nil, err
	}

	release := sync.OnceFunc(func() {
		o.release()
	})

	return &readCloser{
		r: readerFunc(func(buf []byte) (n int, err error) {
			n, err = r.Read(buf)
			if err != nil {
				// release if error
				release()
			}
			return
		}),
		closeFunc: func() error {
			// release when close
			release()
			return r.Close()
		},
	}, nil
}

func (o *objectStorageSemaphore) Stat(ctx context.Context, key string) (size int64, err error) {
	o.acquire()
	defer o.release()
	return o.upstream.Stat(ctx, key)
}

func (o *objectStorageSemaphore) Write(ctx context.Context, key string, r io.Reader, sizeHint *int64, expire *time.Time) (err error) {
	o.acquire()
	defer o.release()
	return o.upstream.Write(ctx, key, r, sizeHint, expire)
}
