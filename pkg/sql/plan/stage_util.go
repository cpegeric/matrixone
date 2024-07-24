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
	//"encoding/csv"
	"fmt"
	//"path"
	//"net/url"
	"strings"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	//moruntime "github.com/matrixorigin/matrixone/pkg/common/runtime"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
	//"github.com/matrixorigin/matrixone/pkg/logutil"
	"github.com/matrixorigin/matrixone/pkg/sql/parsers/tree"
	"github.com/matrixorigin/matrixone/pkg/sql/plan/function"
	//"github.com/matrixorigin/matrixone/pkg/util/executor"
	//"github.com/matrixorigin/matrixone/pkg/container/vector"
	//"github.com/matrixorigin/matrixone/pkg/fileservice"
)

func GetFilePathFromParam(param *tree.ExternParam) string {
	fpath := param.Filepath
	for i := 0; i < len(param.Option); i += 2 {
		name := strings.ToLower(param.Option[i])
		if name == "filepath" {
			fpath = param.Option[i+1]
			break
		}
	}

	return fpath
}

func InitStageS3Param(param *tree.ExternParam, s function.StageDef) error {

	param.ScanType = tree.S3
	param.S3Param = &tree.S3Parameter{}

	if len(s.Url.RawQuery) > 0 {
		return fmt.Errorf("s3:// Query don't support in ExternParam.S3Param")
	}

	if s.Url.Scheme != function.S3_PROTOCOL {
		return fmt.Errorf("protocol is not S3")
	}

	bucket, prefix, _ := function.ParseS3Url(s.Url)

	param.S3Param.Endpoint = "endpoint"
	param.S3Param.Region = "region"
	param.S3Param.APIKey = "aws_key_id"
	param.S3Param.APISecret = "aws_secret_key"
	param.S3Param.Bucket = bucket
	param.S3Param.Provider = "minio"

	param.Filepath = prefix
	param.CompressType = "compression"

	for i := 0; i < len(param.Option); i += 2 {
		switch strings.ToLower(param.Option[i]) {
		case "format":
			format := strings.ToLower(param.Option[i+1])
			if format != tree.CSV && format != tree.JSONLINE {
				return moerr.NewBadConfig(param.Ctx, "the format '%s' is not supported", format)
			}
			param.Format = format
		case "jsondata":
			jsondata := strings.ToLower(param.Option[i+1])
			if jsondata != tree.OBJECT && jsondata != tree.ARRAY {
				return moerr.NewBadConfig(param.Ctx, "the jsondata '%s' is not supported", jsondata)
			}
			param.JsonData = jsondata
			param.Format = tree.JSONLINE

		default:
			return moerr.NewBadConfig(param.Ctx, "the keyword '%s' is not support", strings.ToLower(param.Option[i]))
		}
	}

	if param.Format == tree.JSONLINE && len(param.JsonData) == 0 {
		return moerr.NewBadConfig(param.Ctx, "the jsondata must be specified")
	}
	if len(param.Format) == 0 {
		param.Format = tree.CSV
	}

	return nil

}

func InitInfileOrStageParam(param *tree.ExternParam, proc *process.Process) error {

	fpath := GetFilePathFromParam(param)

	if !strings.HasPrefix(fpath, function.STAGE_PROTOCOL+"://") {
		return InitInfileParam(param)
	}

	stagemap, err := function.StageLoadCatalog(proc)
	if err != nil {
		return err
	}

	s, err := function.UrlToStageDef(fpath, stagemap, proc)
	if err != nil {
		return err
	}

	if len(s.Url.RawQuery) > 0 {
		return fmt.Errorf("Invalid URL: query not supported in ExternParam")
	}

	if s.Url.Scheme == function.S3_PROTOCOL {
		return InitStageS3Param(param, s)
	} else if s.Url.Scheme == function.FILE_PROTOCOL {

		err := InitInfileParam(param)
		if err != nil {
			return err
		}

		param.Filepath = s.Url.Path

	} else {
		return fmt.Errorf("invalid URL: protocol %s not supported", s.Url.Scheme)
	}

	return nil
}
