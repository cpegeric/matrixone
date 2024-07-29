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
	"context"
	"encoding/csv"
	"fmt"
	"net/url"
	"strings"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	moruntime "github.com/matrixorigin/matrixone/pkg/common/runtime"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/fileservice"
	//"github.com/matrixorigin/matrixone/pkg/logutil"
	"github.com/matrixorigin/matrixone/pkg/util/executor"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
)

const STAGE_PROTOCOL = "stage"
const S3_PROTOCOL = "s3"
const FILE_PROTOCOL = "file"

const PARAMKEY_AWS_KEY_ID = "aws_key_id"
const PARAMKEY_AWS_SECRET_KEY = "aws_secret_key"
const PARAMKEY_AWS_REGION = "aws_region"
const PARAMKEY_ENDPOINT = "endpoint"
const PARAMKEY_COMPRESSION = "compression"
const PARAMKEY_PROVIDER = "provider"

const S3_PROVIDER_AMAZON = "amazon"
const S3_PROVIDER_MINIO = "minio"

const S3_SERVICE = "s3"
const MINIO_SERVICE = "minio"

type StageDef struct {
	Id          uint32
	Name        string
	Url         *url.URL
	Credentials string
	Disabled    bool
}

func (s *StageDef) GetCredentials(key string, defval string) (string, bool) {
	k := strings.ToLower(key)
	switch k {
	case PARAMKEY_AWS_KEY_ID:
		return "KEY123", true
	case PARAMKEY_AWS_SECRET_KEY:
		return "SECRET123", true
	case PARAMKEY_AWS_REGION:
		return "local", true
	case PARAMKEY_COMPRESSION:
		return "", true
	case PARAMKEY_PROVIDER:
		return "minio", true
	case PARAMKEY_ENDPOINT:
		return "127.0.0.1", true
	default:
		return defval, false
	}
}

func (s *StageDef) expandSubStage(proc *process.Process) (StageDef, error) {
	if s.Url.Scheme == STAGE_PROTOCOL {
		stagename, prefix, query, err := ParseStageUrl(s.Url)
		if err != nil {
			return StageDef{}, err
		}

		res, err := StageLoadCatalog(proc, stagename)
		if err != nil {
			return StageDef{}, err
		}

		res.Url = res.Url.JoinPath(prefix)
		res.Url.RawQuery = query
		return res.expandSubStage(proc)
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
		aws_key_id := "KEY123aws_key_id"
		aws_secret_key := "SECRET123"
		aws_region := "local"
		provider := "minio"
		endpoint := "127.0.0.1:9000"

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
		//logutil.Infof("ToPath: prefix = %s, query = %s", s.Url.Path, s.Url.RawQuery)
		return s.Url.Path, s.Url.RawQuery, nil
	}
	return "", "", nil
}

func getS3ServiceFromProvider(provider string) (string, error) {
	provider = strings.ToLower(provider)
	switch provider {
	case S3_PROVIDER_AMAZON:
		return S3_SERVICE, nil
	case S3_PROVIDER_MINIO:
		return MINIO_SERVICE, nil
	default:
		return "", moerr.NewBadConfig(context.TODO(), "provider %s not supported", provider)
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

func StageLoadCatalog(proc *process.Process, stagename string) (s StageDef, err error) {
	getAllStagesSql := fmt.Sprintf("select stage_id, stage_name, url, stage_credentials, stage_status from `%s`.`%s` WHERE stage_name = '%s';", "mo_catalog", "mo_stages", stagename)
	res, err := runSql(proc, getAllStagesSql)
	if err != nil {
		return StageDef{}, err
	}
	defer res.Close()

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
						return StageDef{}, err
					}
					stage_cred := string(batch.Vecs[cred_idx].GetBytesAt(i))
					stage_status := string(batch.Vecs[status_idx].GetBytesAt(i))
					disabled := false
					if stage_status == "disabled" {
						disabled = true
					}

					//logutil.Infof("CATALOG: ID %d,  stage %s url %s cred %s", stage_id, stage_name, stage_url, stage_cred)
					return StageDef{stage_id, stage_name, stage_url, stage_cred, disabled}, nil
				}
			}
		}
	}

	return StageDef{}, moerr.NewBadConfig(context.TODO(), "Stage %s not found", stagename)
}

func UrlToPath(furl string, proc *process.Process) (path string, query string, err error) {

	s, err := UrlToStageDef(furl, proc)
	if err != nil {
		return "", "", err
	}

	return s.ToPath()
}

func ParseStageUrl(u *url.URL) (stagename, prefix, query string, err error) {
	if u.Scheme != STAGE_PROTOCOL {
		return "", "", "", moerr.NewBadConfig(context.TODO(), "ParseStageUrl: URL protocol is not stage://")
	}

	stagename = u.Host
	if len(stagename) == 0 {
		return "", "", "", moerr.NewBadConfig(context.TODO(), "Invalid stage URL: stage name is empty string")
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
		err = moerr.NewBadConfig(context.TODO(), "Invalid s3 URL: bucket is empty string")
		return "", "", "", err
	}

	return
}

func UrlToStageDef(furl string, proc *process.Process) (s StageDef, err error) {

	aurl, err := url.Parse(furl)
	if err != nil {
		return StageDef{}, err
	}

	if aurl.Scheme != STAGE_PROTOCOL {
		return StageDef{}, moerr.NewBadConfig(context.TODO(), "URL is not stage URL")
	}

	stagename, subpath, query, err := ParseStageUrl(aurl)
	if err != nil {
		return StageDef{}, err
	}

	sdef, err := StageLoadCatalog(proc, stagename)
	if err != nil {
		return StageDef{}, err
	}

	s, err = sdef.expandSubStage(proc)
	if err != nil {
		return StageDef{}, err
	}

	s.Url = s.Url.JoinPath(subpath)
	s.Url.RawQuery = query

	return s, nil
}
