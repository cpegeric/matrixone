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

package fileservice

import "context"

// MetaToucher is a FileService told of the reads of an object's meta or bloom filter that
// a cache above it served, so its disk cache counts them as uses of the entry.
type MetaToucher interface {
	// TouchMeta counts a use of the ObjectMetaRead entry of filePath at offset, of size
	// bytes. It does no I/O.
	TouchMeta(filePath string, offset, size int64)
}

// TouchMeta tells fs of a use of an ObjectMetaRead entry, when fs is a MetaToucher.
func TouchMeta(fs FileService, filePath string, offset, size int64) {
	if t, ok := fs.(MetaToucher); ok {
		t.TouchMeta(filePath, offset, size)
	}
}

var _ MetaToucher = (*S3FS)(nil)
var _ MetaToucher = (*FileServices)(nil)
var _ MetaToucher = (*subPathFS)(nil)

func (s *S3FS) TouchMeta(filePath string, offset, size int64) {
	if s.diskCache == nil {
		return
	}
	path, err := parseFilePathAtService(filePath, s.name)
	if err != nil {
		return
	}
	s.diskCache.touch(s.diskCache.metaPathForIOEntry(path.File, IOEntry{Offset: offset, Size: size}))
}

func (f *FileServices) TouchMeta(filePath string, offset, size int64) {
	path, err := parseFilePathAtService(filePath, "")
	if err != nil {
		return
	}
	if path.Service == "" {
		path.Service = f.defaultName
	}
	fs, err := Get[FileService](f, path.Service)
	if err != nil {
		return
	}
	TouchMeta(fs, filePath, offset, size)
}

func (s *subPathFS) TouchMeta(filePath string, offset, size int64) {
	p, err := s.toUpstreamFilePath(filePath)
	if err != nil {
		return
	}
	TouchMeta(s.upstream, p, offset, size)
}

// touch makes the file at diskPath, if cached, the most recently used.
func (d *DiskCache) touch(diskPath string) {
	d.cache.Get(context.Background(), diskPath)
}
