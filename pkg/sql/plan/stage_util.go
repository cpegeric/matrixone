// Copyright 2021 Matrix Origin
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

package plan

import (
	//"context"
	"encoding/csv"
	"fmt"
	"path"
	"strings"

	//"github.com/matrixorigin/matrixone/pkg/common/moerr"
	//moruntime "github.com/matrixorigin/matrixone/pkg/common/runtime"
	//"github.com/matrixorigin/matrixone/pkg/vm/process"
	"github.com/matrixorigin/matrixone/pkg/logutil"
	//"github.com/matrixorigin/matrixone/pkg/util/executor"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/fileservice"
)

const STAGE_PROTOCOL = "stage://"
const S3_PROTOCOL = "s3://"
const FILE_PROTOCOL = "file://"

type StageKey struct {
	Db   string
	Name string
}

type StageDef struct {
	Id          uint32
	Name        string
	Db          string
	Url         string
	Credentials string
	Disabled    bool
}

func (s *StageDef) SplitURL() (protocol string, segments []string, err error) {
	protocol = ""
	err = nil
	segments = nil
	if strings.HasPrefix(s.Url, STAGE_PROTOCOL) {
		// e.g. stage://dbname/stagename/path
		segments = strings.SplitN(s.Url[len(STAGE_PROTOCOL):], "/", 3)
		if len(segments) < 3 {
			err = fmt.Errorf("stage: invalid stage:// Url %s", s)
		}
		protocol = STAGE_PROTOCOL

	} else if strings.HasPrefix(s.Url, S3_PROTOCOL) {
		// e.g. s3://endpoint/bucket/path
		segments = strings.SplitN(s.Url[len(S3_PROTOCOL):], "/", 3)
		if len(segments) < 3 {
			err = fmt.Errorf("stage: invalid s3:// Url %s", s)
		}
		protocol = S3_PROTOCOL

	} else if strings.HasPrefix(s.Url, FILE_PROTOCOL) {
		// e.g. file://path/to/somewhere
		p := s.Url[len(FILE_PROTOCOL):]
		segments = []string{p}
		protocol = FILE_PROTOCOL

	} else {
		err = fmt.Errorf("invalid protcol error")
	}

	return
}

func (s *StageDef) expandSubStage(stagemap map[StageKey]StageDef, defaultdb string) (StageDef, error) {
	if strings.HasPrefix(s.Url, STAGE_PROTOCOL) {
		// expand URL
		_, segments, err := s.SplitURL()
		if err != nil {
			return StageDef{}, err
		}

		dbname := ""
		stagename := ""
		prefix := ""
		if len(segments) == 2 {
			dbname = segments[0]
			stagename = segments[1]
		} else if len(segments) == 3 {
			dbname = segments[0]
			stagename = segments[1]
			prefix = segments[2]
		} else {
			return StageDef{}, fmt.Errorf("Invalid stage URL format")
		}

		if len(dbname) == 0 {
			dbname = defaultdb
		}

		key := StageKey{dbname, stagename}
		res, ok := stagemap[key]
		if !ok {
			return StageDef{}, fmt.Errorf("stage not found. stage://%s/%s", dbname, stagename)
		}

		if strings.HasSuffix(res.Url, "/") {
			res.Url = res.Url + prefix
		} else {
			res.Url = res.Url + "/" + prefix
		}
		return res.expandSubStage(stagemap, defaultdb)
	}

	return *s, nil
}

func getS3ServiceFromProvider(provider string) (string, error) {
	provider = strings.ToLower(provider)
	switch provider {
	case "amazon":
		return "s3", nil
	case "minio":
		return "minio", nil
	default:
		return "", fmt.Errorf("provider %s not supported", provider)
	}
}

// get stages and expand the path. stage may be a file or s3
// use the format of path  s3,<endpoint>,<region>,<bucket>,<key>,<secret>,<prefix>
// or minio,<endpoint>,<region>,<bucket>,<key>,<secret>,<prefix>
// expand the subpath to MO path.
// subpath is in the format like path or path with query like path?q1=v1&q2=v2...
func (s *StageDef) ToPath(subpath string) (mopath string, query string, err error) {

	if strings.HasPrefix(s.Url, S3_PROTOCOL) {
		protocol, segments, err := s.SplitURL()
		if err != nil {
			return "", "", err
		}
		logutil.Infof("proto = %s, segments = %s", protocol, segments)

		endpoint := segments[0]
		bucket := segments[1]
		prefix := ""
		if len(segments) == 3 {
			prefix = segments[2]
		}

		subp := ""
		query := ""
		idx := strings.LastIndex(subpath, "?")
		if idx == -1 {
			subp = subpath
		} else {
			subp = subpath[:idx]
			query = subpath[idx+1:]
		}
		p := path.Join(prefix, subp)

		// TODO: Decode credentials
		aws_key_id := "aws_key_id"
		aws_secret_key := "aws_secret_key"
		aws_region := "aws_region"
		provider := "amazon"

		service, err := getS3ServiceFromProvider(provider)
		if err != nil {
			return "", "", err
		}

		buf := new(strings.Builder)
		w := csv.NewWriter(buf)
		opts := []string{service, endpoint, aws_region, bucket, aws_key_id, aws_secret_key, ""}

		if err = w.Write(opts); err != nil {
			return "", "", err
		}
		w.Flush()
		return fileservice.JoinPath(buf.String(), p), query, nil
	} else if strings.HasPrefix(s.Url, FILE_PROTOCOL) {
		prefix := s.Url[len(FILE_PROTOCOL):]
		subp := ""
		query := ""
		idx := strings.LastIndex(subpath, "?")
		if idx == -1 {
			subp = subpath
		} else {
			subp = subpath[:idx]
			query = subpath[idx+1:]
		}
		p := path.Join(prefix, subp)

		logutil.Infof("ToPath: prefix = %s, query = %s, result = %s", prefix, query, p)
		return p, query, nil
	}

	return "", "", nil
}

func StageLoadCatalog(ctx CompilerContext) (stagemap map[StageKey]StageDef, err error) {
	getAllStagesSql := fmt.Sprintf("select stage_id, stage_name, url, stage_credentials, stage_status, 'dbname' from `%s`.`%s`;", "mo_catalog", "mo_stages")
	res, err := runSql(ctx, getAllStagesSql)
	if err != nil {
		return nil, err
	}
	defer res.Close()

	stagemap = make(map[StageKey]StageDef)
	const id_idx = 0
	const name_idx = 1
	const url_idx = 2
	const cred_idx = 3
	const status_idx = 4
	const db_idx = 5
	if res.Batches != nil {
		for _, batch := range res.Batches {
			if batch != nil && batch.Vecs[0] != nil && batch.Vecs[0].Length() > 0 {
				for i := 0; i < batch.Vecs[0].Length(); i++ {
					stage_id := vector.GetFixedAt[uint32](batch.Vecs[id_idx], i)
					stage_name := string(batch.Vecs[name_idx].GetBytesAt(i))
					stage_url := string(batch.Vecs[url_idx].GetBytesAt(i))
					stage_cred := string(batch.Vecs[cred_idx].GetBytesAt(i))
					stage_status := string(batch.Vecs[status_idx].GetBytesAt(i))
					dbname := string(batch.Vecs[db_idx].GetBytesAt(i))
					disabled := false
					if stage_status == "disabled" {
						disabled = true
					}

					key := StageKey{dbname, stage_name}
					stagemap[key] = StageDef{stage_id, stage_name, dbname, stage_url, stage_cred, disabled}
					logutil.Infof("CATALOG: ID %d,  stage %s url %s cred %s", stage_id, stage_name, stage_url, stage_cred)
				}
			}
		}
	}

	return stagemap, nil
}

func UrlToPath(url string, stagemap map[StageKey]StageDef, ctx CompilerContext) (path string, query string, err error) {

	curdb := ctx.GetProcess().GetSessionInfo().Database
	logutil.Infof("Current database = %s, URL = %s", curdb, url)

	if strings.HasPrefix(url, STAGE_PROTOCOL) {
		dbname := ""
		stagename := ""
		subpath := ""

		segments := strings.SplitN(url[len(STAGE_PROTOCOL):], "/", 3)
		if len(segments) == 2 {
			dbname = segments[0]
			stagename = segments[1]
		} else if len(segments) == 3 {
			dbname = segments[0]
			stagename = segments[1]
			subpath = segments[2]
		} else {
			return "", "", fmt.Errorf("Invalid stage URL format.  e.g. stage://dbname/stagename/path")
		}

		if len(dbname) == 0 {
			dbname = curdb
		}
		logutil.Infof("UrlToPath dbname %s, stagename %s, subpath %s", dbname, stagename, subpath)
		key := StageKey{dbname, stagename}
		s, ok := stagemap[key]
		if !ok {
			return "", "", fmt.Errorf("stage %s not found", stagename)
		}

		exs, err := s.expandSubStage(stagemap, curdb)
		if err != nil {
			return "", "", err
		}

		logutil.Infof("ExanpdSubStage Url=%s", exs.Url)

		return exs.ToPath(subpath)
	}

	return url, "", nil
}
