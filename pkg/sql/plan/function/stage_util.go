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

package function

import (
	//"context"
	"encoding/csv"
	"fmt"
	//"path"
	"net/url"
	"strings"

	//"github.com/matrixorigin/matrixone/pkg/common/moerr"
	moruntime "github.com/matrixorigin/matrixone/pkg/common/runtime"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
	"github.com/matrixorigin/matrixone/pkg/logutil"
	//"github.com/matrixorigin/matrixone/pkg/sql/parsers/tree"
	//"github.com/matrixorigin/matrixone/pkg/util/executor"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/fileservice"
        "github.com/matrixorigin/matrixone/pkg/util/executor"
)

const STAGE_PROTOCOL = "stage"
const S3_PROTOCOL = "s3"
const FILE_PROTOCOL = "file"

type StageDef struct {
	Id          uint32
	Name        string
	Url         *url.URL
	Credentials string
	Disabled    bool
}

func (s *StageDef) expandSubStage(stagemap map[string]StageDef) (StageDef, error) {
	if s.Url.Scheme == STAGE_PROTOCOL {
		stagename, prefix, query, err := ParseStageUrl(s.Url)
		if err != nil {
			return StageDef{}, err
		}

		res, ok := stagemap[stagename]
		if !ok {
			return StageDef{}, fmt.Errorf("stage not found. stage://%s", stagename)
		}

		res.Url = res.Url.JoinPath(prefix)
		res.Url.RawQuery = query
		return res.expandSubStage(stagemap)
	}

	return *s, nil
}

// get stages and expand the path. stage may be a file or s3
// use the format of path  s3,<endpoint>,<region>,<bucket>,<key>,<secret>,<prefix>
// or minio,<endpoint>,<region>,<bucket>,<key>,<secret>,<prefix>
// expand the subpath to MO path.
// subpath is in the format like path or path with query like path?q1=v1&q2=v2...
func (s *StageDef) ToPath() (mopath string, query string, err error) {

	if s.Url.Scheme == S3_PROTOCOL {
		bucket, prefix, query, err := ParseS3Url(s.Url)
		if err != nil {
			return "", "", err
		}

		// TODO: Decode credentials
		aws_key_id := "aws_key_id"
		aws_secret_key := "aws_secret_key"
		aws_region := "aws_region"
		provider := "amazon"
		endpoint := "endpoint"

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
		return fileservice.JoinPath(buf.String(), prefix), query, nil
	} else if s.Url.Scheme == FILE_PROTOCOL {
		logutil.Infof("ToPath: prefix = %s, query = %s", s.Url.Path, s.Url.RawQuery)
		return s.Url.Path, s.Url.RawQuery, nil
	}
	return "", "", nil
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

func runSql(proc *process.Process, sql string) (executor.Result, error) {
        v, ok := moruntime.ProcessLevelRuntime().GetGlobalVariables(moruntime.InternalSQLExecutor)
        if !ok {
                panic("missing lock service")
        }
        exec := v.(executor.SQLExecutor)
        opts := executor.Options{}.
                // All runSql and runSqlWithResult is a part of input sql, can not incr statement.
                // All these sub-sql's need to be rolled back and retried en masse when they conflict in pessimistic mode
                WithDisableIncrStatement().
                WithTxn(proc.GetTxnOperator()).
                WithDatabase(proc.GetSessionInfo().Database).
                WithTimeZone(proc.GetSessionInfo().TimeZone).
                WithAccountID(proc.GetSessionInfo().AccountId)
        return exec.Exec(proc.Ctx, sql, opts)
}

func StageLoadCatalog(proc *process.Process) (stagemap map[string]StageDef, err error) {
	getAllStagesSql := fmt.Sprintf("select stage_id, stage_name, url, stage_credentials, stage_status from `%s`.`%s`;", "mo_catalog", "mo_stages")
	res, err := runSql(proc, getAllStagesSql)
	if err != nil {
		return nil, err
	}
	defer res.Close()

	stagemap = make(map[string]StageDef)
	const id_idx = 0
	const name_idx = 1
	const url_idx = 2
	const cred_idx = 3
	const status_idx = 4
	if res.Batches != nil {
		for _, batch := range res.Batches {
			if batch != nil && batch.Vecs[0] != nil && batch.Vecs[0].Length() > 0 {
				for i := 0; i < batch.Vecs[0].Length(); i++ {
					stage_id := vector.GetFixedAt[uint32](batch.Vecs[id_idx], i)
					stage_name := string(batch.Vecs[name_idx].GetBytesAt(i))
					stage_url, err := url.Parse(string(batch.Vecs[url_idx].GetBytesAt(i)))
					if err != nil {
						return nil, err
					}
					stage_cred := string(batch.Vecs[cred_idx].GetBytesAt(i))
					stage_status := string(batch.Vecs[status_idx].GetBytesAt(i))
					disabled := false
					if stage_status == "disabled" {
						disabled = true
					}

					key := stage_name
					stagemap[key] = StageDef{stage_id, stage_name, stage_url, stage_cred, disabled}
					logutil.Infof("CATALOG: ID %d,  stage %s url %s cred %s", stage_id, stage_name, stage_url, stage_cred)
				}
			}
		}
	}

	return stagemap, nil
}

func UrlToPath(furl string, stagemap map[string]StageDef) (path string, query string, err error) {

	s, err := UrlToStageDef(furl, stagemap)
	if err != nil {
		return "", "", err
	}

	return s.ToPath()
}

func ParseStageUrl(u *url.URL) (stagename, prefix, query string, err error) {
	if u.Scheme != STAGE_PROTOCOL {
		return "", "", "", fmt.Errorf("URL protocol is not stage://")
	}

	stagename = u.Host
	if len(stagename) == 0 {
		return "", "", "", fmt.Errorf("Invalid stage URL: stage name is empty string")
	}

	prefix = u.Path
	query = u.RawQuery

	return
}

func ParseS3Url(u *url.URL) (bucket, fpath, query string, err error) {
	bucket = u.Host
	fpath = u.Path
	query = u.RawQuery
	err = nil

	if len(bucket) == 0 {
		err = fmt.Errorf("Invalid s3 URL: bucket is empty string")
		return "", "", "", err
	}

	return
}

func UrlToStageDef(furl string, stagemap map[string]StageDef) (s StageDef, err error) {

	aurl, err := url.Parse(furl)
	if err != nil {
		return StageDef{}, err
	}

	if aurl.Scheme != STAGE_PROTOCOL {
		return StageDef{}, fmt.Errorf("URL is not stage URL")
	}

	stagename, subpath, query, err := ParseStageUrl(aurl)
	if err != nil {
		return StageDef{}, err
	}

	logutil.Infof("URL = %s", aurl)
	logutil.Infof("UrlToPath stagename %s, subpath %s", stagename, subpath)

	s, ok := stagemap[stagename]
	if !ok {
		return StageDef{}, fmt.Errorf("stage %s not found", stagename)
	}

	exs, err := s.expandSubStage(stagemap)
	if err != nil {
		return StageDef{}, err
	}

	logutil.Infof("ExanpdSubStage Url=%s", exs.Url)

	exs.Url = exs.Url.JoinPath(subpath)
	exs.Url.RawQuery = query

	logutil.Infof("JoinPath Url=%s", exs.Url)

	return exs, nil
}

