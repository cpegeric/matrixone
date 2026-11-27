// Copyright 2025 Matrix Origin
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
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/retry"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/matrixorigin/matrixone/pkg/perfcounter"
	metric "github.com/matrixorigin/matrixone/pkg/util/metric/v2"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	costypes "github.com/tencentyun/cos-go-sdk-v5"
)

// failAfterBytesReader wraps an io.Reader and returns errAfter once
// totalBytes have been read. Reads up to failAfter bytes succeed normally.
type failAfterBytesReader struct {
	r         io.Reader
	readSoFar int64
	failAfter int64
	errAfter  error
}

func (r *failAfterBytesReader) Read(p []byte) (int, error) {
	if r.readSoFar >= r.failAfter {
		return 0, r.errAfter
	}
	remaining := r.failAfter - r.readSoFar
	if int64(len(p)) > remaining {
		p = p[:remaining]
	}
	n, err := r.r.Read(p)
	r.readSoFar += int64(n)
	return n, err
}

// eofSignalReader notifies a waiting worker when the producer reaches EOF.
type eofSignalReader struct {
	r       io.Reader
	eof     chan struct{}
	eofOnce sync.Once
}

func (r *eofSignalReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	if errors.Is(err, io.EOF) {
		r.eofOnce.Do(func() { close(r.eof) })
	}
	return n, err
}

type waitAfterBytesReader struct {
	r         io.Reader
	readSoFar int64
	waitAfter int64
	waitCh    <-chan struct{}
	timeout   time.Duration
}

func (r *waitAfterBytesReader) Read(p []byte) (int, error) {
	if r.readSoFar >= r.waitAfter && r.waitCh != nil {
		select {
		case <-r.waitCh:
			r.waitCh = nil
		case <-time.After(r.timeout):
			return 0, fmt.Errorf("timed out waiting after %d bytes", r.waitAfter)
		}
	}
	n, err := r.r.Read(p)
	r.readSoFar += int64(n)
	return n, err
}

func newMockAWSServer(t *testing.T, failPart int32) (*httptest.Server, *awsServerState) {
	t.Helper()
	state := &awsServerState{
		failPart: failPart,
		parts:    make(map[int32][]byte),
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.Contains(r.URL.RawQuery, "uploads"):
			state.creates.Add(1)
			if state.failCreate {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			_, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/xml")
			_, _ = fmt.Fprintf(w, `<CreateMultipartUploadResult><UploadId>%s</UploadId></CreateMultipartUploadResult>`, state.uploadID)
		case r.Method == http.MethodPut && !strings.Contains(r.URL.RawQuery, "partNumber") && !strings.Contains(r.URL.RawQuery, "uploadId") && !strings.Contains(r.URL.RawQuery, "uploads"):
			body, _ := io.ReadAll(r.Body)
			state.mu.Lock()
			state.putCount++
			state.putBody = append([]byte{}, body...)
			state.mu.Unlock()
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodPut && strings.Contains(r.URL.RawQuery, "partNumber"):
			partStr := r.URL.Query().Get("partNumber")
			pn, _ := strconv.Atoi(partStr)
			if state.blockUploadPart != nil {
				if state.uploadPartStarted != nil {
					state.uploadPartStartOnce.Do(func() { close(state.uploadPartStarted) })
				}
				<-state.blockUploadPart
			}
			body, _ := io.ReadAll(r.Body)
			if state.failPart > 0 && int32(pn) == state.failPart {
				if state.failPartSeen != nil {
					state.failPartOnce.Do(func() { close(state.failPartSeen) })
				}
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			state.mu.Lock()
			state.parts[int32(pn)] = body
			state.mu.Unlock()
			w.Header().Set("ETag", fmt.Sprintf("\"etag-%d\"", pn))
		case r.Method == http.MethodPost && strings.Contains(r.URL.RawQuery, "uploadId"):
			body, _ := io.ReadAll(r.Body)
			if state.failComplete {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			state.mu.Lock()
			state.completeBody = append([]byte{}, body...)
			state.mu.Unlock()
			w.Header().Set("Content-Type", "application/xml")
			_, _ = w.Write([]byte(`<CompleteMultipartUploadResult><Location>loc</Location><Bucket>bucket</Bucket><Key>object</Key><ETag>"etag"</ETag></CompleteMultipartUploadResult>`))
		case r.Method == http.MethodPost && strings.Contains(r.URL.RawQuery, "delete"):
			_, _ = io.ReadAll(r.Body)
			state.mu.Lock()
			state.deleteMultiCount++
			state.deleteMultiHasContentMD5 = r.Header.Get("Content-MD5") != ""
			state.mu.Unlock()
			if state.failDeleteMultiMalformed {
				w.WriteHeader(http.StatusBadRequest)
				w.Header().Set("Content-Type", "application/xml")
				_, _ = w.Write([]byte(awsS3ErrorXML("MalformedXML", "The XML you provided was not well-formed")))
				return
			}
			if state.failDeleteMultiInternal {
				w.WriteHeader(http.StatusInternalServerError)
				w.Header().Set("Content-Type", "application/xml")
				_, _ = w.Write([]byte(awsS3ErrorXML("InternalError", "internal error")))
				return
			}
			if state.failDeleteMultiChecksum && !state.deleteMultiHasContentMD5 {
				w.WriteHeader(http.StatusBadRequest)
				w.Header().Set("Content-Type", "application/xml")
				_, _ = w.Write([]byte(awsS3ErrorXML("MissingContentMD5", "Content-MD5 is required")))
				return
			}
			if state.failDeleteMultiMissing && !state.deleteMultiHasContentMD5 {
				w.WriteHeader(http.StatusBadRequest)
				w.Header().Set("Content-Type", "application/xml")
				_, _ = w.Write([]byte(awsS3ErrorXML("MissingArgument", "Missing Some Required Arguments")))
				return
			}
			if state.failDeleteMultiPlain && !state.deleteMultiHasContentMD5 {
				w.Header().Set("Content-Type", "text/plain")
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte("Missing required header for this request: Content-MD5"))
				return
			}
			w.Header().Set("Content-Type", "application/xml")
			_, _ = w.Write([]byte(`<DeleteResult></DeleteResult>`))
		case r.Method == http.MethodDelete && strings.Contains(r.URL.RawQuery, "uploadId"):
			state.aborted.Store(true)
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodDelete:
			state.mu.Lock()
			state.deleteSingleKeys = append(state.deleteSingleKeys, awsObjectKeyFromPath(r.URL.Path))
			state.mu.Unlock()
			w.WriteHeader(http.StatusNoContent)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})
	return httptest.NewServer(handler), state
}

type awsServerState struct {
	mu                       sync.Mutex
	parts                    map[int32][]byte
	completeBody             []byte
	failPart                 int32
	failPartSeen             chan struct{}
	failPartOnce             sync.Once
	blockUploadPart          chan struct{}
	uploadPartStarted        chan struct{}
	uploadPartStartOnce      sync.Once
	failComplete             bool
	failCreate               bool
	uploadID                 string
	creates                  atomic.Int32
	aborted                  atomic.Bool
	putCount                 int
	putBody                  []byte
	deleteMultiCount         int
	deleteSingleKeys         []string
	failDeleteMultiMalformed bool
	failDeleteMultiInternal  bool
	failDeleteMultiChecksum  bool
	failDeleteMultiMissing   bool
	failDeleteMultiPlain     bool
	deleteMultiHasContentMD5 bool
}

func awsObjectKeyFromPath(path string) string {
	trimmed := strings.TrimPrefix(path, "/")
	parts := strings.SplitN(trimmed, "/", 2)
	if len(parts) != 2 {
		return ""
	}
	return parts[1]
}

func awsS3ErrorXML(code, message string) string {
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?><Error><Code>%s</Code><Message>%s</Message></Error>`, code, message)
}

// multipartFixture is a mock AWS or COS multipart server and the fields of its state.
type multipartFixture struct {
	provider     string
	server       *httptest.Server
	mu           *sync.Mutex
	failComplete *bool
	failCreate   *bool
	failPartSeen *chan struct{}
	creates      *atomic.Int32
	aborted      *atomic.Bool
	completeBody *[]byte
	parts        func() int
}

var multipartProviders = []string{"aws", "cos"}

func newMultipartFixture(t *testing.T, provider string, failPart int) *multipartFixture {
	f := &multipartFixture{provider: provider}
	if provider == "aws" {
		server, s := newMockAWSServer(t, int32(failPart))
		s.uploadID = "aws-upload"
		f.server, f.mu, f.failComplete, f.failCreate, f.failPartSeen = server, &s.mu, &s.failComplete, &s.failCreate, &s.failPartSeen
		f.creates, f.aborted, f.completeBody, f.parts = &s.creates, &s.aborted, &s.completeBody, func() int { return len(s.parts) }
	} else {
		server, s := newMockCOSServer(t, failPart)
		f.server, f.mu, f.failComplete, f.failCreate, f.failPartSeen = server, &s.mu, &s.failComplete, &s.failCreate, &s.failPartSeen
		f.creates, f.aborted, f.completeBody, f.parts = &s.initCalls, &s.aborted, &s.completeBody, func() int { return len(s.parts) }
	}
	t.Cleanup(f.server.Close)
	return f
}

// client returns a client of the server that sends requests through transport, or the
// transport of the server when it is nil.
func (f *multipartFixture) client(t *testing.T, transport http.RoundTripper) ObjectStorage {
	if transport == nil {
		transport = f.server.Client().Transport
	}
	if f.provider == "aws" {
		return newTestAWSClientWithTransport(t, f.server, transport)
	}
	return newTestCOSClientWithTransport(t, f.server, transport)
}

// requireAbortedNotCompleted requires that the upload was aborted with no part kept
// when noParts is set, and was never completed.
func (f *multipartFixture) requireAbortedNotCompleted(t *testing.T, noParts bool) {
	require.True(t, f.aborted.Load(), "abort")
	f.mu.Lock()
	defer f.mu.Unlock()
	require.Empty(t, *f.completeBody, "complete")
	if noParts {
		require.Zero(t, f.parts())
	}
}

func newTestAWSClient(t *testing.T, srv *httptest.Server) *AwsSDKv2 {
	return newTestAWSClientWithTransport(t, srv, srv.Client().Transport)
}

func newTestAWSClientWithTransport(t *testing.T, srv *httptest.Server, transport http.RoundTripper) *AwsSDKv2 {
	t.Helper()
	httpClient := srv.Client()
	httpClient.Transport = transport
	cfg := aws.Config{
		Region:      "us-east-1",
		Credentials: aws.NewCredentialsCache(credentials.NewStaticCredentialsProvider("id", "key", "")),
		HTTPClient:  httpClient,
		Retryer: func() aws.Retryer {
			return aws.NopRetryer{}
		},
	}
	client := s3.NewFromConfig(cfg, func(o *s3.Options) {
		o.UsePathStyle = true
		o.BaseEndpoint = aws.String(srv.URL)
	})

	return &AwsSDKv2{
		name:   "aws-test",
		bucket: "bucket",
		client: client,
	}
}

func TestAwsParallelMultipartSuccess(t *testing.T) {
	for _, known := range []bool{true, false} {
		t.Run(fmt.Sprintf("known size %v", known), func(t *testing.T) {
			server, state := newMockAWSServer(t, 0)
			defer server.Close()
			state.uploadID = "uid-success"

			upstream := server.Client().Transport
			var traceMu sync.Mutex
			traces := make(map[*httptrace.ClientTrace]struct{})
			transport := wrapRoundTripper(roundTripperFunc(func(req *http.Request) (*http.Response, error) {
				tr := httptrace.ContextClientTrace(req.Context())
				if tr == nil {
					return nil, errors.New("multipart request has no HTTP trace")
				}
				traceMu.Lock()
				traces[tr] = struct{}{}
				traceMu.Unlock()
				return upstream.RoundTrip(req)
			}))
			g := newObjectStorageServer(newTestAWSClientWithTransport(t, server, transport), minRequestRoom)
			counter := new(perfcounter.CounterSet)
			ctx := perfcounter.WithCounterSet(context.Background(), counter)
			var callerHooks atomic.Int64
			ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{GotConn: func(httptrace.GotConnInfo) { callerHooks.Add(1) }})

			data := bytes.Repeat([]byte("a"), int(minMultipartPartSize+1))
			size := int64(len(data))
			sizeHint := &size
			if !known {
				sizeHint = nil
			}
			err := writeParts(ctx, g, "object", bytes.NewReader(data), sizeHint, &ParallelMultipartOption{
				PartSize:    minMultipartPartSize,
				Concurrency: 2,
			})
			if err != nil {
				t.Fatalf("write failed: %v, parts=%d, complete=%s", err, len(state.parts), string(state.completeBody))
			}
			if len(state.parts) != 2 {
				t.Fatalf("expected 2 multipart parts, got %d", len(state.parts))
			}
			if len(state.parts[1])+len(state.parts[2]) != len(data) {
				t.Fatalf("expected total part size %d, got %d", len(data), len(state.parts[1])+len(state.parts[2]))
			}
			if len(state.completeBody) == 0 {
				t.Fatalf("complete body not recorded")
			}
			traceMu.Lock()
			traceCount := len(traces)
			traceMu.Unlock()
			require.GreaterOrEqual(t, traceCount, 4, "create, two parts and completion need independent traces")
			require.Equal(t, int64(traceCount), callerHooks.Load(), "caller hooks must be composed exactly once per request")
			if got := counter.FileService.S3.Put.Load(); got != 2 {
				t.Fatalf("expected 2 physical PUT requests, got %d", got)
			}
			if got := counter.FileService.S3WriteSize.Load(); got != size {
				t.Fatalf("expected %d accepted bytes, got %d", size, got)
			}
		})
	}
}

func TestAwsMultipartTooManyParts(t *testing.T) {
	server, _ := newMockAWSServer(t, 0)
	defer server.Close()

	sdk := newTestAWSClient(t, server)
	huge := int64(maxMultipartParts+1) * defaultParallelMultipartPartSize
	err := writeInParts(context.Background(), sdk, "object", bytes.NewReader(nil), &huge, &ParallelMultipartOption{
		PartSize: defaultParallelMultipartPartSize,
	})
	if err == nil {
		t.Fatalf("expected error for too many parts")
	}
}

func TestAwsMultipartUploadPartError(t *testing.T) {
	for _, provider := range multipartProviders {
		t.Run(provider, func(t *testing.T) {
			f := newMultipartFixture(t, provider, 2)
			counter := new(perfcounter.CounterSet)
			ctx := perfcounter.WithCounterSet(context.Background(), counter)
			data := bytes.Repeat([]byte("p"), int(minMultipartPartSize*2))
			size := int64(len(data))
			require.Error(t, writeInParts(ctx, f.client(t, nil), "object", bytes.NewReader(data), &size, &ParallelMultipartOption{
				PartSize:    minMultipartPartSize,
				Concurrency: 1,
			}))
			require.GreaterOrEqual(t, counter.FileService.S3.Put.Load(), int64(2), "successful and failed PUT attempts")
			require.Equal(t, int64(minMultipartPartSize), counter.FileService.S3WriteSize.Load(), "first accepted part bytes")
		})
	}
}

// A part or completion the server rejects aborts the upload it created and never
// completes it, also when the reader is still waiting on the rejected part.
func TestAwsMultipartSendJobFailureAfterPendingChunk(t *testing.T) {
	for _, provider := range multipartProviders {
		for _, c := range []struct {
			name         string
			failPart     int
			knownSize    bool
			failComplete bool
		}{
			{"pending part", 1, true, false},
			{"pending part unknown size", 1, false, false},
			{"complete rejected", 0, true, true},
		} {
			t.Run(provider+" "+c.name, func(t *testing.T) {
				f := newMultipartFixture(t, provider, c.failPart)
				*f.failComplete = c.failComplete
				data := bytes.Repeat([]byte("q"), int(minMultipartPartSize*3))
				var reader io.Reader = bytes.NewReader(data)
				if c.failPart > 0 {
					seen := make(chan struct{})
					*f.failPartSeen = seen
					reader = &waitAfterBytesReader{r: reader, waitAfter: minMultipartPartSize * 2, waitCh: seen, timeout: 3 * time.Second}
				}
				size := int64(len(data))
				sizeHint := &size
				if !c.knownSize {
					sizeHint = nil
				}
				require.Error(t, writeInParts(context.Background(), f.client(t, nil), "object", reader, sizeHint, &ParallelMultipartOption{
					PartSize:    minMultipartPartSize,
					Concurrency: 1,
				}))
				require.Equal(t, int32(1), f.creates.Load())
				f.requireAbortedNotCompleted(t, false)
			})
		}
	}
}

func TestAwsMultipartCreateFail(t *testing.T) {
	for _, provider := range multipartProviders {
		t.Run(provider, func(t *testing.T) {
			f := newMultipartFixture(t, provider, 0)
			*f.failCreate = true
			data := bytes.Repeat([]byte("i"), int(minMultipartPartSize+1))
			size := int64(len(data))
			require.Error(t, writeInParts(context.Background(), f.client(t, nil), "object", bytes.NewReader(data), &size, &ParallelMultipartOption{PartSize: minMultipartPartSize}))
		})
	}
}

func TestAwsDeleteMultiUsesBatchWhenSupported(t *testing.T) {
	server, state := newMockAWSServer(t, 0)
	defer server.Close()

	sdk := newTestAWSClient(t, server)
	if err := sdk.Delete(context.Background(), "a", "b", "c"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	if state.deleteMultiCount != 1 {
		t.Fatalf("expected 1 batch delete, got %d", state.deleteMultiCount)
	}
	if len(state.deleteSingleKeys) != 0 {
		t.Fatalf("expected no single-delete fallback, got %v", state.deleteSingleKeys)
	}
}

func TestAwsDeleteMultiFallsBackToSinglesOnMalformedXML(t *testing.T) {
	server, state := newMockAWSServer(t, 0)
	defer server.Close()
	state.failDeleteMultiMalformed = true

	sdk := newTestAWSClient(t, server)
	if err := sdk.Delete(context.Background(), "a", "b", "c"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	if state.deleteMultiCount != 1 {
		t.Fatalf("expected 1 batch delete attempt, got %d", state.deleteMultiCount)
	}
	if strings.Join(state.deleteSingleKeys, ",") != "a,b,c" {
		t.Fatalf("expected single-delete fallback for all keys, got %v", state.deleteSingleKeys)
	}

	if err := sdk.Delete(context.Background(), "d", "e"); err != nil {
		t.Fatalf("second delete failed: %v", err)
	}
	if state.deleteMultiCount != 1 {
		t.Fatalf("expected later deletes to skip batch after malformed xml, got %d batch attempts", state.deleteMultiCount)
	}
	if strings.Join(state.deleteSingleKeys, ",") != "a,b,c,d,e" {
		t.Fatalf("expected later deletes to go directly to single-delete path, got %v", state.deleteSingleKeys)
	}
}

func TestAwsDeleteMultiFallsBackToSinglesOnChecksumRejection(t *testing.T) {
	server, state := newMockAWSServer(t, 0)
	defer server.Close()
	state.failDeleteMultiChecksum = true

	sdk := newTestAWSClient(t, server)
	if err := sdk.Delete(context.Background(), "a", "b", "c"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	if state.deleteMultiCount != 1 {
		t.Fatalf("expected 1 batch delete attempt, got %d", state.deleteMultiCount)
	}
	if state.deleteMultiHasContentMD5 {
		t.Fatal("expected the compatibility endpoint to reject the SDK's non-MD5 batch checksum")
	}
	if strings.Join(state.deleteSingleKeys, ",") != "a,b,c" {
		t.Fatalf("expected single-delete fallback for all keys, got %v", state.deleteSingleKeys)
	}

	if err := sdk.Delete(context.Background(), "d", "e"); err != nil {
		t.Fatalf("second delete failed: %v", err)
	}
	if state.deleteMultiCount != 1 {
		t.Fatalf("expected later deletes to skip batch after checksum rejection, got %d batch attempts", state.deleteMultiCount)
	}
	if strings.Join(state.deleteSingleKeys, ",") != "a,b,c,d,e" {
		t.Fatalf("expected later deletes to go directly to single-delete path, got %v", state.deleteSingleKeys)
	}
}

func TestAwsDeleteMultiFallsBackToSinglesOnMissingArgument(t *testing.T) {
	server, state := newMockAWSServer(t, 0)
	defer server.Close()
	state.failDeleteMultiMissing = true

	sdk := newTestAWSClient(t, server)
	if err := sdk.Delete(context.Background(), "a", "b"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	if state.deleteMultiCount != 1 {
		t.Fatalf("expected 1 batch delete attempt, got %d", state.deleteMultiCount)
	}
	if strings.Join(state.deleteSingleKeys, ",") != "a,b" {
		t.Fatalf("expected single-delete fallback for all keys, got %v", state.deleteSingleKeys)
	}

	if err := sdk.Delete(context.Background(), "c", "d"); err != nil {
		t.Fatalf("second delete failed: %v", err)
	}
	if state.deleteMultiCount != 1 {
		t.Fatalf("expected later deletes to skip batch after missing-argument rejection, got %d batch attempts", state.deleteMultiCount)
	}
	if strings.Join(state.deleteSingleKeys, ",") != "a,b,c,d" {
		t.Fatalf("expected later deletes to go directly to single-delete path, got %v", state.deleteSingleKeys)
	}
}

func TestAwsDeleteMultiFallsBackToSinglesOnUnstructuredChecksumRejection(t *testing.T) {
	server, state := newMockAWSServer(t, 0)
	defer server.Close()
	state.failDeleteMultiPlain = true

	sdk := newTestAWSClient(t, server)
	if err := sdk.Delete(context.Background(), "a", "b"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	if state.deleteMultiCount != 1 {
		t.Fatalf("expected 1 batch delete attempt, got %d", state.deleteMultiCount)
	}
	if strings.Join(state.deleteSingleKeys, ",") != "a,b" {
		t.Fatalf("expected single-delete fallback for all keys, got %v", state.deleteSingleKeys)
	}
}

func TestAwsDeleteMultiDoesNotFallbackOnOtherErrors(t *testing.T) {
	server, state := newMockAWSServer(t, 0)
	defer server.Close()
	state.failDeleteMultiInternal = true

	sdk := newTestAWSClient(t, server)
	if err := sdk.Delete(context.Background(), "a", "b"); err == nil {
		t.Fatalf("expected delete error")
	}
	if state.deleteMultiCount != retry.DefaultMaxAttempts {
		t.Fatalf("expected %d batch delete attempts, got %d", retry.DefaultMaxAttempts, state.deleteMultiCount)
	}
	if len(state.deleteSingleKeys) != 0 {
		t.Fatalf("expected no single-delete fallback, got %v", state.deleteSingleKeys)
	}
}

func newMockCOSServer(t *testing.T, failPart int) (*httptest.Server, *cosServerState) {
	t.Helper()
	state := &cosServerState{
		failPart:           failPart,
		failPartStatus:     http.StatusBadRequest,
		failCompleteStatus: http.StatusBadRequest,
		failCreateStatus:   http.StatusBadRequest,
		uploadID:           "cos-upload",
		parts:              make(map[int][]byte),
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.Contains(r.URL.RawQuery, "uploads"):
			initCall := state.initCalls.Add(1)
			if state.failCreate {
				w.WriteHeader(state.failCreateStatus)
				return
			}
			_, _ = io.ReadAll(r.Body)
			uploadID := state.uploadID
			if state.generateUploadIDs {
				uploadID = fmt.Sprintf("%s-%d", uploadID, initCall)
			}
			w.Header().Set("Content-Type", "application/xml")
			_, _ = fmt.Fprintf(w, `<InitiateMultipartUploadResult><UploadId>%s</UploadId></InitiateMultipartUploadResult>`, uploadID)
		case r.Method == http.MethodPut && !strings.Contains(r.URL.RawQuery, "partNumber") && !strings.Contains(r.URL.RawQuery, "uploadId") && !strings.Contains(r.URL.RawQuery, "uploads"):
			body, _ := io.ReadAll(r.Body)
			state.mu.Lock()
			state.putCount++
			state.putBody = append([]byte{}, body...)
			state.mu.Unlock()
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodPut && strings.Contains(r.URL.RawQuery, "partNumber"):
			partStr := r.URL.Query().Get("partNumber")
			pn, _ := strconv.Atoi(partStr)
			body, _ := io.ReadAll(r.Body)
			if state.failPart > 0 && pn == state.failPart {
				if state.failPartSeen != nil {
					state.failPartOnce.Do(func() { close(state.failPartSeen) })
				}
				w.WriteHeader(state.failPartStatus)
				return
			}
			state.mu.Lock()
			state.parts[pn] = body
			state.mu.Unlock()
		case r.Method == http.MethodPost && strings.Contains(r.URL.RawQuery, "uploadId"):
			body, _ := io.ReadAll(r.Body)
			if state.failComplete {
				w.WriteHeader(state.failCompleteStatus)
				return
			}
			state.mu.Lock()
			state.completeBody = append([]byte{}, body...)
			state.completedUploadID = r.URL.Query().Get("uploadId")
			for partNum := 1; partNum < len(state.parts); partNum++ {
				if len(state.parts[partNum]) < int(minMultipartPartSize) {
					state.mu.Unlock()
					w.WriteHeader(http.StatusBadRequest)
					w.Header().Set("Content-Type", "application/xml")
					_, _ = w.Write([]byte(`<Error><Code>EntityTooSmall</Code><Message>Your proposed upload is smaller than the minimum allowed object size.</Message></Error>`))
					return
				}
			}
			state.mu.Unlock()
			w.Header().Set("Content-Type", "application/xml")
			state.respBody = `<CompleteMultipartUploadResult><Location>loc</Location><Bucket>bucket</Bucket><Key>object</Key><ETag>etag</ETag></CompleteMultipartUploadResult>`
			state.completed.Store(true)
			_, _ = w.Write([]byte(state.respBody))
		case r.Method == http.MethodDelete && strings.Contains(r.URL.RawQuery, "uploadId"):
			state.aborted.Store(true)
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})
	return httptest.NewServer(handler), state
}

type cosServerState struct {
	mu                 sync.Mutex
	parts              map[int][]byte
	completeBody       []byte
	failPart           int
	failPartSeen       chan struct{}
	failPartOnce       sync.Once
	failPartStatus     int
	failComplete       bool
	failCompleteStatus int
	failCreate         bool
	failCreateStatus   int
	uploadID           string
	generateUploadIDs  bool
	completedUploadID  string
	initCalls          atomic.Int32
	aborted            atomic.Bool
	completed          atomic.Bool
	respBody           string
	putCount           int
	putBody            []byte
}

func newTestCOSClient(t *testing.T, srv *httptest.Server) *QCloudSDK {
	t.Helper()
	baseURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}

	client := costypes.NewClient(
		&costypes.BaseURL{BucketURL: baseURL},
		srv.Client(),
	)
	client.Conf.EnableCRC = false
	client.Conf.RetryOpt.Count = 0

	return &QCloudSDK{
		name:   "cos-test",
		client: client,
	}
}

func TestCOSParallelMultipartSuccess(t *testing.T) {
	server, state := newMockCOSServer(t, 0)
	defer server.Close()
	state.uploadID = "cos-uid"

	sdk := newTestCOSClient(t, server)
	counter := new(perfcounter.CounterSet)
	ctx := perfcounter.WithCounterSet(context.Background(), counter)
	data := bytes.Repeat([]byte("c"), int(minMultipartPartSize+2))
	size := int64(len(data))
	err := writeInParts(ctx, sdk, "object", bytes.NewReader(data), &size, &ParallelMultipartOption{
		PartSize:    minMultipartPartSize,
		Concurrency: 2,
	})
	if err != nil {
		t.Fatalf("write failed: %v, parts=%d, complete=%s", err, len(state.parts), string(state.completeBody))
	}
	if len(state.parts) != 2 {
		t.Fatalf("expected 2 multipart parts, got %d parts", len(state.parts))
	}
	if len(state.parts[1])+len(state.parts[2]) != len(data) {
		t.Fatalf("expected total part size %d, got %d", len(data), len(state.parts[1])+len(state.parts[2]))
	}
	if len(state.completeBody) == 0 {
		t.Fatalf("complete body not recorded")
	}
	if got := counter.FileService.S3.Put.Load(); got != 2 {
		t.Fatalf("expected 2 physical PUT requests, got %d", got)
	}
	if got := counter.FileService.S3WriteSize.Load(); got != size {
		t.Fatalf("expected %d accepted bytes, got %d", size, got)
	}
}

func TestCOSMultipartUploadPartRetriesServerClosedIdleConnection(t *testing.T) {
	server, state := newMockCOSServer(t, 0)
	defer server.Close()
	state.uploadID = "cos-uid-retry-idle-conn"

	baseClient := server.Client()
	baseTransport := baseClient.Transport
	if baseTransport == nil {
		baseTransport = http.DefaultTransport
	}
	transport := &cosUploadPartIdleConnTransport{
		base: baseTransport,
		part: "1",
	}
	baseClient.Transport = transport

	baseURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}
	client := costypes.NewClient(
		&costypes.BaseURL{BucketURL: baseURL},
		baseClient,
	)
	client.Conf.EnableCRC = false
	client.Conf.RetryOpt.Count = 0
	sdk := &QCloudSDK{
		name:   "cos-retry-idle-conn-test",
		client: client,
	}

	data := bytes.Repeat([]byte("r"), int(minMultipartPartSize+1))
	size := int64(len(data))
	if err := writeInParts(context.Background(), sdk, "object", bytes.NewReader(data), &size, &ParallelMultipartOption{
		PartSize:    minMultipartPartSize,
		Concurrency: 1,
	}); err != nil {
		t.Fatalf("write failed after transient upload-part error: %v", err)
	}
	if transport.uploadPartCalls.Load() < 2 {
		t.Fatalf("expected upload part retry, got %d calls", transport.uploadPartCalls.Load())
	}
	if !state.completed.Load() {
		t.Fatalf("expected multipart upload complete")
	}
}

func TestCOSMultipartInitRecoversBeforeRequest(t *testing.T) {
	server, state := newMockCOSServer(t, 0)
	defer server.Close()
	state.uploadID = "cos-uid-init-before-request"
	sentinel := errors.New("http: server closed idle connection")
	transport := &cosMultipartInitBeforeRequestTransport{
		base:     server.Client().Transport,
		err:      sentinel,
		failures: 1,
	}
	sdk := newTestCOSClientWithTransport(t, server, transport)

	attemptsBefore := testutil.ToFloat64(metric.FSMultipartInitAttemptCounter)
	ambiguousBefore := testutil.ToFloat64(metric.FSMultipartInitAmbiguousCounter)
	recoveredBefore := testutil.ToFloat64(metric.FSMultipartInitRecoveredCounter)
	data := bytes.Repeat([]byte("r"), int(minMultipartPartSize+1))
	size := int64(len(data))
	require.NoError(t, writeInParts(context.Background(), sdk, "object", bytes.NewReader(data), &size, &ParallelMultipartOption{PartSize: minMultipartPartSize}))
	require.Equal(t, int32(2), transport.initCalls.Load())
	require.Equal(t, int32(1), state.initCalls.Load())
	require.True(t, state.completed.Load())
	require.Equal(t, attemptsBefore+2, testutil.ToFloat64(metric.FSMultipartInitAttemptCounter))
	require.Equal(t, ambiguousBefore, testutil.ToFloat64(metric.FSMultipartInitAmbiguousCounter))
	require.Equal(t, recoveredBefore+1, testutil.ToFloat64(metric.FSMultipartInitRecoveredCounter))
}

func TestCOSMultipartInitDoesNotAdoptAnotherInstanceUpload(t *testing.T) {
	server, state := newMockCOSServer(t, 0)
	defer server.Close()
	state.uploadID = "upload"
	state.generateUploadIDs = true
	sentinel := errors.New("http: server closed idle connection")
	transport := &cosMultipartInitBeforeRequestTransport{
		base:     server.Client().Transport,
		err:      sentinel,
		failures: 1,
		onFailure: func() {
			resp, err := http.Post(server.URL+"/object?uploads", "application/octet-stream", nil)
			require.NoError(t, err)
			_ = resp.Body.Close()
		},
	}
	sdk := newTestCOSClientWithTransport(t, server, transport)

	data := bytes.Repeat([]byte("r"), int(minMultipartPartSize+1))
	size := int64(len(data))
	require.NoError(t, writeInParts(context.Background(), sdk, "object", bytes.NewReader(data), &size, &ParallelMultipartOption{PartSize: minMultipartPartSize}))
	require.Equal(t, int32(2), state.initCalls.Load(), "one external init and one init owned by this writer")
	state.mu.Lock()
	require.Equal(t, "upload-2", state.completedUploadID)
	state.mu.Unlock()
	require.False(t, state.aborted.Load())
}

func TestCOSMultipartInitDoesNotRequireListPermission(t *testing.T) {
	server, state := newMockCOSServer(t, 0)
	defer server.Close()
	state.uploadID = "cos-uid-no-list-permission"
	transport := &denyMultipartListTransport{base: server.Client().Transport}
	sdk := newTestCOSClientWithTransport(t, server, transport)

	data := bytes.Repeat([]byte("r"), int(minMultipartPartSize+1))
	size := int64(len(data))
	require.NoError(t, writeInParts(context.Background(), sdk, "object", bytes.NewReader(data), &size, &ParallelMultipartOption{PartSize: minMultipartPartSize}))
	require.Zero(t, transport.listCalls.Load())
	require.Equal(t, int32(1), state.initCalls.Load())
}

func TestAwsMultipartWorkerCancellationAborts(t *testing.T) {
	for _, provider := range multipartProviders {
		t.Run(provider, func(t *testing.T) {
			f := newMultipartFixture(t, provider, 0)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			data := bytes.Repeat([]byte("r"), int(minMultipartPartSize)+1)
			reader := &eofSignalReader{r: bytes.NewReader(data), eof: make(chan struct{})}
			size := int64(len(data))
			err := writeInParts(ctx, f.client(t, nil), "object", reader, &size, &ParallelMultipartOption{
				PartSize: minMultipartPartSize,
				beforePartUpload: func() {
					<-reader.eof
					cancel()
				},
			})
			require.ErrorIs(t, err, context.Canceled)
			f.requireAbortedNotCompleted(t, true)
		})
	}
}

func TestAwsMultipartInitCancellationCleansOwnedUpload(t *testing.T) {
	for _, provider := range multipartProviders {
		t.Run(provider, func(t *testing.T) {
			f := newMultipartFixture(t, provider, 0)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			sdk := f.client(t, &multipartInitCancelAfterResponseTransport{base: f.server.Client().Transport, cancel: cancel})
			cleanupBefore := testutil.ToFloat64(metric.FSMultipartInitCleanupCounter)
			data := bytes.Repeat([]byte("r"), int(minMultipartPartSize+1))
			size := int64(len(data))
			err := writeInParts(ctx, sdk, "object", bytes.NewReader(data), &size, &ParallelMultipartOption{PartSize: minMultipartPartSize})
			require.ErrorIs(t, err, context.Canceled)
			f.requireAbortedNotCompleted(t, false)
			if provider == "cos" {
				require.Equal(t, cleanupBefore+1, testutil.ToFloat64(metric.FSMultipartInitCleanupCounter))
			}
		})
	}
}

func TestCOSMultipartInitCancellationOverridesRetryableError(t *testing.T) {
	server, state := newMockCOSServer(t, 0)
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	transport := &cosMultipartInitBeforeRequestTransport{
		base:      server.Client().Transport,
		err:       errors.New("http: server closed idle connection"),
		failures:  1,
		onFailure: cancel,
	}
	sdk := newTestCOSClientWithTransport(t, server, transport)

	data := bytes.Repeat([]byte("r"), int(minMultipartPartSize+1))
	size := int64(len(data))
	err := writeInParts(ctx, sdk, "object", bytes.NewReader(data), &size, &ParallelMultipartOption{PartSize: minMultipartPartSize})
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, int32(1), transport.initCalls.Load())
	require.Zero(t, state.initCalls.Load())
}

func TestCOSMultipartInitDefinitiveErrorOverridesEarlierRetryableError(t *testing.T) {
	server, state := newMockCOSServer(t, 0)
	defer server.Close()
	state.failCreate = true
	state.failCreateStatus = http.StatusForbidden
	sentinel := errors.New("http: server closed idle connection")
	transport := &cosMultipartInitBeforeRequestTransport{
		base:     server.Client().Transport,
		err:      sentinel,
		failures: 1,
	}
	sdk := newTestCOSClientWithTransport(t, server, transport)

	data := bytes.Repeat([]byte("r"), int(minMultipartPartSize+1))
	size := int64(len(data))
	err := writeInParts(context.Background(), sdk, "object", bytes.NewReader(data), &size, &ParallelMultipartOption{PartSize: minMultipartPartSize})
	require.Error(t, err)
	require.NotContains(t, err.Error(), sentinel.Error())
	require.Contains(t, err.Error(), "403")
	require.Equal(t, int32(2), transport.initCalls.Load())
	require.Equal(t, int32(1), state.initCalls.Load())
}

func TestCOSMultipartInitDoesNotRetryAfterRequestCommitted(t *testing.T) {
	invalidBody := func(r io.Reader) func() io.ReadCloser {
		return func() io.ReadCloser { return io.NopCloser(r) }
	}
	tests := []struct {
		name string
		err  error
		body func() io.ReadCloser
	}{
		{name: "EOF", err: io.EOF},
		{name: "unexpected EOF", err: io.ErrUnexpectedEOF},
		{name: "timeout", err: timeoutError{}},
		{name: "connection reset", err: errors.New("read: connection reset by peer")},
		{name: "broken pipe", err: errors.New("write: broken pipe")},
		{name: "server closed connection", err: errors.New("http: server closed idle connection")},
		{name: "empty success body", err: io.EOF, body: invalidBody(strings.NewReader(""))},
		{name: "truncated success body", err: io.ErrUnexpectedEOF, body: invalidBody(io.MultiReader(
			strings.NewReader(`<InitiateMultipartUploadResult><UploadId>`),
			multipartInitErrorReader{err: io.ErrUnexpectedEOF},
		))},
	}

	ambiguousBefore := testutil.ToFloat64(metric.FSMultipartInitAmbiguousCounter)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server, state := newMockCOSServer(t, 0)
			defer server.Close()
			state.generateUploadIDs = true
			var transport http.RoundTripper = &cosMultipartInitResponseLostTransport{base: server.Client().Transport, err: test.err}
			if test.body != nil {
				transport = &cosMultipartInitInvalidSuccessBodyTransport{base: server.Client().Transport, body: test.body}
			}
			sdk := newTestCOSClientWithTransport(t, server, transport)
			data := bytes.Repeat([]byte("r"), int(minMultipartPartSize+1))
			size := int64(len(data))
			err := writeInParts(context.Background(), sdk, "object", bytes.NewReader(data), &size, &ParallelMultipartOption{PartSize: minMultipartPartSize})
			require.ErrorContains(t, err, test.err.Error())
			require.Equal(t, int32(1), state.initCalls.Load())
			require.False(t, state.completed.Load())
			require.False(t, state.aborted.Load())
		})
	}
	require.Equal(t, ambiguousBefore+float64(len(tests)), testutil.ToFloat64(metric.FSMultipartInitAmbiguousCounter))
}

func TestCOSMultipartInitRetriesAreBoundedBeforeRequest(t *testing.T) {
	server, state := newMockCOSServer(t, 0)
	defer server.Close()
	sentinel := errors.New("http: server closed idle connection")
	transport := &cosMultipartInitBeforeRequestTransport{
		base:     server.Client().Transport,
		err:      sentinel,
		failures: qcloudMultipartInitMaxAttempts,
	}
	sdk := newTestCOSClientWithTransport(t, server, transport)

	data := bytes.Repeat([]byte("r"), int(minMultipartPartSize+1))
	size := int64(len(data))
	err := writeInParts(context.Background(), sdk, "object", bytes.NewReader(data), &size, &ParallelMultipartOption{PartSize: minMultipartPartSize})
	require.ErrorContains(t, err, sentinel.Error())
	require.Equal(t, int32(qcloudMultipartInitMaxAttempts), transport.initCalls.Load())
	require.Zero(t, state.initCalls.Load())
}

type cosMultipartInitBeforeRequestTransport struct {
	base      http.RoundTripper
	err       error
	failures  int32
	initCalls atomic.Int32
	onFailure func()
}

type cosMultipartInitResponseLostTransport struct {
	base http.RoundTripper
	err  error
}

type cosMultipartInitInvalidSuccessBodyTransport struct {
	base http.RoundTripper
	body func() io.ReadCloser
}

type multipartInitErrorReader struct {
	err error
}

func (r multipartInitErrorReader) Read([]byte) (int, error) {
	return 0, r.err
}

type denyMultipartListTransport struct {
	base      http.RoundTripper
	listCalls atomic.Int32
}

type multipartInitCancelAfterResponseTransport struct {
	base   http.RoundTripper
	cancel context.CancelFunc
}

func newTestCOSClientWithTransport(t *testing.T, server *httptest.Server, transport http.RoundTripper) *QCloudSDK {
	t.Helper()
	baseURL, err := url.Parse(server.URL)
	require.NoError(t, err)
	httpClient := server.Client()
	httpClient.Transport = transport
	client := costypes.NewClient(&costypes.BaseURL{BucketURL: baseURL}, httpClient)
	client.Conf.EnableCRC = false
	client.Conf.RetryOpt.Count = 0
	return &QCloudSDK{name: "cos-multipart-init-test", client: client}
}

func (t *cosMultipartInitResponseLostTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	if req.Method == http.MethodPost && req.URL.Query().Has("uploads") {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		return nil, t.err
	}
	return resp, nil
}

func (t *cosMultipartInitInvalidSuccessBodyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	if req.Method == http.MethodPost && req.URL.Query().Has("uploads") {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		resp.Body = t.body()
		resp.ContentLength = -1
	}
	return resp, nil
}

func (t *multipartInitCancelAfterResponseTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	if req.Method == http.MethodPost && req.URL.Query().Has("uploads") {
		body, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, readErr
		}
		resp.Body = io.NopCloser(bytes.NewReader(body))
		resp.ContentLength = int64(len(body))
		t.cancel()
	}
	return resp, nil
}

func (t *cosMultipartInitBeforeRequestTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Method == http.MethodPost && req.URL.Query().Has("uploads") {
		if t.initCalls.Add(1) <= t.failures {
			if req.Body != nil {
				_ = req.Body.Close()
			}
			if t.onFailure != nil {
				t.onFailure()
			}
			return nil, t.err
		}
	}
	return t.base.RoundTrip(req)
}

func (t *denyMultipartListTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Method == http.MethodGet && req.URL.Query().Has("uploads") {
		t.listCalls.Add(1)
		return &http.Response{
			StatusCode: http.StatusForbidden,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader("Access Denied")),
			Request:    req,
		}, nil
	}
	return t.base.RoundTrip(req)
}

type cosUploadPartIdleConnTransport struct {
	base            http.RoundTripper
	part            string
	uploadPartCalls atomic.Int32
}

func (t *cosUploadPartIdleConnTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Method == http.MethodPut && req.URL.Query().Get("partNumber") == t.part {
		if t.uploadPartCalls.Add(1) == 1 {
			return nil, fmt.Errorf("http: server closed idle connection")
		}
	}
	return t.base.RoundTrip(req)
}

// A read error after the upload is created aborts it with a context that is not canceled.
func TestAwsParallelMultipartAbortOnReadError(t *testing.T) {
	for _, provider := range multipartProviders {
		t.Run(provider, func(t *testing.T) {
			f := newMultipartFixture(t, provider, 0)
			data := bytes.Repeat([]byte("x"), int(minMultipartPartSize*2))
			r := &failAfterBytesReader{
				r:         bytes.NewReader(data),
				failAfter: minMultipartPartSize,
				errAfter:  errors.New("size does not match"),
			}
			size := int64(len(data))
			require.Error(t, writeInParts(context.Background(), f.client(t, nil), "object", r, &size, &ParallelMultipartOption{
				PartSize:    minMultipartPartSize,
				Concurrency: 2,
			}))
			f.requireAbortedNotCompleted(t, false)
		})
	}
}

// writeInParts writes r as key in parts through a server of s.
func writeInParts(ctx context.Context, s ObjectStorage, key string, r io.Reader, size *int64, opt *ParallelMultipartOption) error {
	return writeParts(ctx, newObjectStorageServer(s, 0), key, r, size, opt)
}
