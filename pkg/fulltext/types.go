// Copyright 2022 Matrix Origin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package fulltext

import (
	"container/heap"

	"github.com/matrixorigin/matrixone/pkg/container/types"
)

/*
The following examples demonstrate some search strings that use boolean full-text operators:

'apple banana'

Find rows that contain at least one of the two words.

'+apple +juice'

Find rows that contain both words.

'+apple macintosh'

Find rows that contain the word “apple”, but rank rows higher if they also contain “macintosh”.

'+apple -macintosh'

Find rows that contain the word “apple” but not “macintosh”.

'+apple ~macintosh'

Find rows that contain the word “apple”, but if the row also contains the word “macintosh”, rate it lower than if row does not. This is “softer” than a search for '+apple -macintosh', for which the presence of “macintosh” causes the row not to be returned at all.

'+apple +(>turnover <strudel)'

Find rows that contain the words “apple” and “turnover”, or “apple” and “strudel” (in any order), but rank “apple turnover” higher than “apple strudel”.

'apple*'

Find rows that contain words such as “apple”, “apples”, “applesauce”, or “applet”.

'"some words"'

Find rows that contain the exact phrase “some words” (for example, rows that contain “some words of wisdom” but not “some noise words”). Note that the " characters that enclose the phrase are operator characters that delimit the phrase. They are not the quotation marks that enclose the search string itself.
*/

// Parser parameters
type FullTextParserParam struct {
	Parser string `json:"parser"`
}

// Word is associated with particular DocId (index.doc_id) and could have multiple positions
type Word struct {
	DocId    any
	Position []int32
	DocCount int32
}

// Word accumulator accumulate the same word appeared in multiple (index.doc_id).
type WordAccum struct {
	Words map[any]*Word
}

// Search accumulator is to parse the search string into list of pattern and each pattern will associate with WordAccum by pattern.Text
type SearchAccum struct {
	SrcTblName string
	TblName    string
	Mode       int64
	Pattern    []*Pattern
	Params     string
	WordAccums map[string]*WordAccum
	Nrow       int64
}

// Boolean mode search string parsing
type FullTextBooleanOperator int

var (
	TEXT        = 0
	STAR        = 1
	PLUS        = 2
	MINUS       = 3
	LESSTHAN    = 4
	GREATERTHAN = 5
	RANKLESS    = 6
	GROUP       = 7
	PHRASE      = 8
)

func OperatorToString(op int) string {
	switch op {
	case TEXT:
		return "text"
	case STAR:
		return "*"
	case PLUS:
		return "+"
	case MINUS:
		return "-"
	case LESSTHAN:
		return "<"
	case GREATERTHAN:
		return ">"
	case RANKLESS:
		return "~"
	case GROUP:
		return "group"
	case PHRASE:
		return "phrase"
	default:
		return ""
	}
}

// Pattern works on both Natural Language and Boolean mode
type Pattern struct {
	Text     string
	Operator int
	Children []*Pattern
	Position int32
}

// Bucket
type DocRow struct {
	Widx  int32
	Pos   int32
	Docid any // depends on the pk type.
}

type DocHeap []*DocRow

func (h DocHeap) Len() int {
	return len(h)
}

// TODO: Fill In all types
func (h DocHeap) Less(i, j int) bool {
	switch h[i].Docid.(type) {
	case int8:
		return h[i].Docid.(int8) < h[j].Docid.(int8)
	case int16:
		return h[i].Docid.(int16) < h[j].Docid.(int16)
	case int32:
		return h[i].Docid.(int32) < h[j].Docid.(int32)
	case int64:
		return h[i].Docid.(int64) < h[j].Docid.(int64)
	case uint8:
		return h[i].Docid.(uint8) < h[j].Docid.(uint8)
	case uint16:
		return h[i].Docid.(uint16) < h[j].Docid.(uint16)
	case uint32:
		return h[i].Docid.(uint32) < h[j].Docid.(uint32)
	case uint64:
		return h[i].Docid.(uint64) < h[j].Docid.(uint64)
	case string:
		return h[i].Docid.(string) < h[j].Docid.(string)
	case float32:
		return h[i].Docid.(float32) < h[j].Docid.(float32)
	case float64:
		return h[i].Docid.(float64) < h[j].Docid.(float64)
	case types.Date:
		return h[i].Docid.(types.Date) < h[j].Docid.(types.Date)
	default:
		return false
	}

	return false
}

func (h DocHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *DocHeap) Push(x any) {
	row := x.(*DocRow)
	*h = append(*h, row)

}

func (h *DocHeap) Pop() any {
	old := *h
	n := len(old)
	if n == 0 {
		return nil
	}
	row := old[n-1]
	old[n-1] = nil
	*h = old[0 : n-1]
	return row
}

type Bucket struct {
	Id      int
	Maxsize int
	Docheap DocHeap
}

func NewBucket(id int, maxsize int) *Bucket {
	return &Bucket{Id: id, Maxsize: maxsize, Docheap: make(DocHeap, 0, 1024)}
}

func (b *Bucket) Push(r *DocRow) {
	if b.Docheap.Len() >= b.Maxsize {
		b.spill()
	}
	heap.Push(&b.Docheap, r)
}

func (b *Bucket) Pop() *DocRow {
	if b.Docheap.Len() == 0 {
		return nil
	}
	return heap.Pop(&b.Docheap).(*DocRow)
}

func (b *Bucket) Len() int {

	return b.Docheap.Len()
}

func (b *Bucket) spill() error {

	return nil
}

// free the memory and spill files
func (b *Bucket) Free() {

}

type Cache struct {
	curr    int
	nbucket int
	buckets []*Bucket
}

func NewCache(nbucket int, maxsize int) *Cache {
	cache := &Cache{nbucket: nbucket, buckets: make([]*Bucket, nbucket)}
	for i := range cache.buckets {
		cache.buckets[i] = NewBucket(i, maxsize)
	}
	return cache
}

func (c *Cache) GetNBucket() int {
	return c.nbucket
}

func (c *Cache) Add(hashv uint64, row *DocRow) {
	bid := hashv % uint64(c.nbucket)
	c.buckets[bid].Push(row)
}

func (c *Cache) GetNext() *DocRow {
	for c.curr < c.nbucket {
		if c.buckets[c.curr].Len() > 0 {
			return c.buckets[c.curr].Pop()
		} else {
			c.curr++
		}
	}
	return nil
}

func (c *Cache) GetBuckets() []*Bucket {
	return c.buckets
}

// Insert End.  All data from SQL have been received.  Spill the bucket if necessary
func (c *Cache) InsertEnd() {

}

// Change the SQL to output word index 0,1,2,..etc. instead of WORD String
// Spill
type SpillDoc struct {
	widx  int16
	pos   int32
	docid any // depends on the pk type.
}

type SpillHeader struct {
	magic  [4]byte
	size   int64
	zsize  int64
	ndoc   int32
	nblock int32
}

// read buffer size keep IO Block 4096 bytes
type SpillBlock struct {
	size int32
	ndoc int32
	docs []*SpillDoc
}

type Spill struct {
	header SpillHeader
	blocks []*SpillBlock
}

type Zonemap struct {
	filename string
	mindoc   any
	maxdoc   any
}

type SpillInfo struct {
	pktype  types.Type // t.ProtoSize() == 20 bytes
	nword   int16
	words   []string
	zonemap []Zonemap
}
