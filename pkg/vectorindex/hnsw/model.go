// Copyright 2022 Matrix Origin
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

package hnsw

import (
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/detailyang/go-fallocate"
	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/util/executor"
	"github.com/matrixorigin/matrixone/pkg/vectorindex"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
	usearch "github.com/unum-cloud/usearch/golang"
)

// Hnsw Build index implementation
type HnswModel struct {
	Id       string
	Index    *usearch.Index
	Path     string
	FileSize int64

	// info required for build
	MaxCapacity uint

	// from metadata.  info required for search
	Timestamp int64
	Checksum  string

	// for cdc update
	Dirty      bool
	View       bool
	Len        uint
	InsertMeta bool
}

// New HnswModel struct
func NewHnswModelForBuild(id string, cfg vectorindex.IndexConfig, nthread int, max_capacity uint) (*HnswModel, error) {
	var err error
	idx := &HnswModel{}

	idx.Id = id

	idx.Index, err = usearch.NewIndex(cfg.Usearch)
	if err != nil {
		return nil, err
	}

	idx.MaxCapacity = max_capacity

	err = idx.Index.Reserve(idx.MaxCapacity)
	if err != nil {
		return nil, err
	}

	err = idx.Index.ChangeThreadsAdd(uint(nthread))
	if err != nil {
		return nil, err
	}
	return idx, nil
}

// Destroy the struct
func (idx *HnswModel) Destroy() error {
	if idx.Index != nil {
		err := idx.Index.Destroy()
		if err != nil {
			return err
		}
		idx.Index = nil
	}

	if len(idx.Path) > 0 {
		// remove the file
		if _, err := os.Stat(idx.Path); err == nil || os.IsExist(err) {
			err := os.Remove(idx.Path)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// Save the index to file
func (idx *HnswModel) SaveToFile() error {

	if !idx.Dirty {
		// nothing change. ignore
		return nil
	}

	// delete old file
	oldpath := idx.Path
	if len(oldpath) > 0 {
		// remove the file
		if _, err := os.Stat(oldpath); err == nil || os.IsExist(err) {
			err := os.Remove(oldpath)
			if err != nil {
				return err
			}
		}
	}
	idx.Path = ""

	// save to file
	f, err := os.CreateTemp("", "hnsw")
	if err != nil {
		return err
	}

	err = idx.Index.Save(f.Name())
	if err != nil {
		os.Remove(f.Name())
		return err
	}

	// free memory
	err = idx.Index.Destroy()
	if err != nil {
		return err
	}
	idx.Index = nil
	idx.Path = f.Name()
	return nil
}

// Generate the SQL to update the secondary index tables.
// 1. store the index file into the index table
func (idx *HnswModel) ToSql(cfg vectorindex.IndexTableConfig) ([]string, error) {

	err := idx.SaveToFile()
	if err != nil {
		return nil, err
	}

	if len(idx.Path) == 0 {
		// file path is empty string. No file is written
		return []string{}, nil
	}

	fi, err := os.Stat(idx.Path)
	if err != nil {
		return nil, err
	}

	filesz := fi.Size()
	offset := int64(0)
	chunksz := int64(0)
	chunkid := int64(0)

	idx.FileSize = filesz

	sqls := make([]string, 0, 5)

	sql := fmt.Sprintf("INSERT INTO `%s`.`%s` VALUES ", cfg.DbName, cfg.IndexTable)
	values := make([]string, 0, int64(math.Ceil(float64(filesz)/float64(vectorindex.MaxChunkSize))))
	n := 0
	for offset = 0; offset < filesz; {
		if offset+vectorindex.MaxChunkSize < filesz {
			chunksz = vectorindex.MaxChunkSize

		} else {
			chunksz = filesz - offset
		}

		url := fmt.Sprintf("file://%s?offset=%d&size=%d", idx.Path, offset, chunksz)
		tuple := fmt.Sprintf("('%s', %d, load_file(cast('%s' as datalink)), 0)", idx.Id, chunkid, url)
		values = append(values, tuple)

		// offset and chunksz
		offset += chunksz
		chunkid++

		n++
		if n == 10000 {
			newsql := sql + strings.Join(values, ", ")
			sqls = append(sqls, newsql)
			values = values[:0]
			n = 0
		}
	}

	if len(values) > 0 {
		newsql := sql + strings.Join(values, ", ")
		sqls = append(sqls, newsql)
	}

	//sql += strings.Join(values, ", ")
	//return []string{sql}, nil
	return sqls, nil
}

// is the index empty
func (idx *HnswModel) Empty() (bool, error) {
	if idx.Index == nil {
		return false, moerr.NewInternalErrorNoCtx("usearch index is nil")
	}

	sz, err := idx.Index.Len()
	if err != nil {
		return false, err
	}
	return (sz == 0), nil
}

// check the index is full, i.e. 10K vectors
func (idx *HnswModel) Full() (bool, error) {
	if idx.Index == nil {
		return false, moerr.NewInternalErrorNoCtx("usearch index is nil")
	}
	sz, err := idx.Index.Len()
	if err != nil {
		return false, err
	}
	return (sz == idx.MaxCapacity), nil
}

// add vector to the index
func (idx *HnswModel) Add(key int64, vec []float32) error {
	if idx.Index == nil {
		return moerr.NewInternalErrorNoCtx("usearch index is nil")
	}
	idx.Dirty = true
	return idx.Index.Add(uint64(key), vec)
}

// remove key
func (idx *HnswModel) Remove(key int64) error {
	if idx.Index == nil {
		return moerr.NewInternalErrorNoCtx("usearch index is nil")
	}
	idx.Dirty = true
	return idx.Index.Remove(uint64(key))
}

// contains key
func (idx *HnswModel) Contains(key int64) (found bool, err error) {
	if idx.Index == nil {
		return false, moerr.NewInternalErrorNoCtx("usearch index is nil")
	}
	return idx.Index.Contains(uint64(key))
}

// load chunk from database
func (idx *HnswModel) loadChunk(proc *process.Process, stream_chan chan executor.Result, error_chan chan error, fp *os.File) (stream_closed bool, err error) {
	var res executor.Result
	var ok bool

	select {
	case res, ok = <-stream_chan:
		if !ok {
			return true, nil
		}
	case err = <-error_chan:
		return false, err
	case <-proc.Ctx.Done():
		return false, moerr.NewInternalError(proc.Ctx, "context cancelled")
	}

	bat := res.Batches[0]
	defer res.Close()

	for i := 0; i < bat.RowCount(); i++ {
		chunk_id := vector.GetFixedAtWithTypeCheck[int64](bat.Vecs[0], i)
		data := bat.Vecs[1].GetRawBytesAt(i)

		offset := chunk_id * vectorindex.MaxChunkSize
		_, err = fp.Seek(offset, io.SeekStart)
		if err != nil {
			return false, err
		}

		_, err = fp.Write(data)
		if err != nil {
			return false, err
		}
	}
	return false, nil
}

// load index from database
// TODO: loading file is tricky.
// 1. we need to know the size of the file.
// 2. Write Zero to file to have a pre-allocated size
// 3. SELECT chunk_id, data from index_table WHERE index_id = id.  Result will be out of order
// 4. according to the chunk_id, seek to the offset and write the chunk
// 5. check the checksum to verify the correctness of the file
func (idx *HnswModel) LoadIndex(proc *process.Process, idxcfg vectorindex.IndexConfig, tblcfg vectorindex.IndexTableConfig, nthread int64, view bool) error {

	if idx.Index != nil {
		// index already loaded. ignore
		return nil

	}

	stream_chan := make(chan executor.Result, 2)
	error_chan := make(chan error)

	if len(idx.Path) == 0 {
		// create tempfile for writing
		fp, err := os.CreateTemp("", "hnswindx")
		if err != nil {
			return err
		}

		// load index to memory
		defer func() {
			if !view {
				// if view == false, remove the file
				os.Remove(fp.Name())
			}
		}()

		err = fallocate.Fallocate(fp, 0, idx.FileSize)
		if err != nil {
			fp.Close()
			return err
		}

		// run streaming sql
		sql := fmt.Sprintf("SELECT chunk_id, data from `%s`.`%s` WHERE index_id = '%s'", tblcfg.DbName, tblcfg.IndexTable, idx.Id)
		go func() {
			_, err := runSql_streaming(proc, sql, stream_chan, error_chan)
			if err != nil {
				error_chan <- err
				return
			}
		}()

		// incremental load from database
		sql_closed := false
		for !sql_closed {
			sql_closed, err = idx.loadChunk(proc, stream_chan, error_chan, fp)
			if err != nil {
				fp.Close()
				return err
			}
		}

		idx.Path = fp.Name()
		fp.Close()
	}

	// check checksum
	chksum, err := vectorindex.CheckSum(idx.Path)
	if err != nil {
		return err
	}
	if chksum != idx.Checksum {
		return moerr.NewInternalError(proc.Ctx, "Checksum mismatch with the index file")
	}

	usearchidx, err := usearch.NewIndex(idxcfg.Usearch)
	if err != nil {
		return err
	}

	err = usearchidx.ChangeThreadsSearch(uint(nthread))
	if err != nil {
		return err
	}

	if view {
		err = usearchidx.View(idx.Path)
		idx.View = true
	} else {
		err = usearchidx.Load(idx.Path)
	}
	if err != nil {
		return err
	}

	idx.Index = usearchidx

	return nil
}

// unload
func (idx *HnswModel) Unload() error {
	if idx.Index == nil {
		return moerr.NewInternalErrorNoCtx("usearch index is nil")
	}

	err := idx.Index.Destroy()
	if err != nil {
		return err
	}
	// reset variable
	idx.Index = nil
	idx.Dirty = false
	return nil
}

// Call usearch.Search
func (idx *HnswModel) Search(query []float32, limit uint) (keys []usearch.Key, distances []float32, err error) {
	if idx.Index == nil {
		return nil, nil, moerr.NewInternalErrorNoCtx("usearch index is nil")
	}
	return idx.Index.Search(query, limit)
}
