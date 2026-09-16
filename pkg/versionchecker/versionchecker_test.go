// Copyright 2026 Matrix Origin
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

package versionchecker

import (
	"context"
	"errors"
	"testing"

	"github.com/matrixorigin/matrixone/pkg/clusterservice"
	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/common/mpool"
	moruntime "github.com/matrixorigin/matrixone/pkg/common/runtime"
	"github.com/matrixorigin/matrixone/pkg/pb/metadata"
	"github.com/matrixorigin/matrixone/pkg/pb/pipeline"
	querypb "github.com/matrixorigin/matrixone/pkg/pb/query"
	qclient "github.com/matrixorigin/matrixone/pkg/queryservice/client"
	"github.com/matrixorigin/matrixone/pkg/testutil"
	"github.com/matrixorigin/matrixone/pkg/vm/engine"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
	"github.com/stretchr/testify/require"
)

// mockQueryClient answers the GetProtocolVersion probe with a fixed version (or an error).
type mockQueryClient struct {
	version int64
	sendErr error
	nilResp bool
}

func (m *mockQueryClient) ServiceID() string { return "mock" }
func (m *mockQueryClient) NewRequest(cmd querypb.CmdMethod) *querypb.Request {
	return &querypb.Request{CmdMethod: cmd}
}
func (m *mockQueryClient) SendMessage(_ context.Context, _ string, _ *querypb.Request) (*querypb.Response, error) {
	if m.sendErr != nil {
		return nil, m.sendErr
	}
	if m.nilResp {
		return nil, nil
	}
	return &querypb.Response{GetProtocolVersion: &querypb.GetProtocolVersionResponse{Version: m.version}}, nil
}
func (m *mockQueryClient) Release(*querypb.Response) {}
func (m *mockQueryClient) Close() error              { return nil }

var _ qclient.QueryClient = (*mockQueryClient)(nil)

// vcTestEnv builds a process bound to a fresh service runtime with a known local protocol version,
// a static CN cluster, and a mock query client, so SupportProtocol's full probe loop is reachable.
func vcTestEnv(t *testing.T, localVersion int64, client qclient.QueryClient, cns []metadata.CNService) (*process.Process, string) {
	proc := testutil.NewProcessWithMPool(t, "vc_"+t.Name(), mpool.MustNewZero())
	proc.Base.QueryClient = client
	// SupportProtocol/ProtocolVersion read the runtime of proc.GetService(); set the globals and
	// build the cluster under that exact service so the test env is what the code observes.
	service := proc.GetService()
	rt := moruntime.ServiceRuntime(service)
	oldV, hadV := rt.GetGlobalVariables(moruntime.MOProtocolVersion)
	oldC, hadC := rt.GetGlobalVariables(moruntime.ClusterService)
	t.Cleanup(func() {
		if hadV {
			rt.SetGlobalVariables(moruntime.MOProtocolVersion, oldV)
		}
		if hadC {
			rt.SetGlobalVariables(moruntime.ClusterService, oldC)
		}
	})
	rt.SetGlobalVariables(moruntime.MOProtocolVersion, localVersion)
	if cns != nil {
		cluster := clusterservice.NewMOCluster(service, nil, 0,
			clusterservice.WithDisableRefresh(), clusterservice.WithServices(cns, nil))
		t.Cleanup(cluster.Close)
		rt.SetGlobalVariables(moruntime.ClusterService, cluster)
	}
	return proc, service
}

func TestProtocolVersion(t *testing.T) {
	_, service := vcTestEnv(t, 50, nil, nil)
	v, ok := ProtocolVersion(service)
	require.True(t, ok)
	require.Equal(t, int64(50), v)

	// Unknown service has no runtime -> not known.
	_, ok = ProtocolVersion("vc_no_such_service_xyz")
	require.False(t, ok)
}

func TestLocalAtLeast(t *testing.T) {
	_, service := vcTestEnv(t, 50, nil, nil)
	require.True(t, LocalAtLeast(service, 50))
	require.True(t, LocalAtLeast(service, 49))
	require.False(t, LocalAtLeast(service, 51))
	require.False(t, LocalAtLeast("vc_no_such_service_xyz", 1))
}

func TestSupportProtocolEarlyReturns(t *testing.T) {
	// nil proc -> not supported, no error.
	ok, err := SupportProtocol(nil, engine.Nodes{{Id: "w", Addr: "a"}}, 1)
	require.NoError(t, err)
	require.False(t, ok)

	// Local rollout gate below the minimum -> not supported, no probe.
	proc, _ := vcTestEnv(t, 50, &mockQueryClient{version: 99}, nil)
	ok, err = SupportProtocol(proc, engine.Nodes{{Id: "w", Addr: "a"}}, 60)
	require.NoError(t, err)
	require.False(t, ok)
}

func TestSupportProtocolProbesDestination(t *testing.T) {
	cns := []metadata.CNService{
		{ServiceID: "worker-1", PipelineServiceAddress: "pipe-1", QueryAddress: "query-1"},
	}

	// Destination supports the version.
	proc, _ := vcTestEnv(t, 60, &mockQueryClient{version: 60}, cns)
	ok, err := SupportProtocol(proc, engine.Nodes{{Id: "worker-1", Addr: "pipe-1"}}, 55)
	require.NoError(t, err)
	require.True(t, ok)

	// Destination is too old (probe returns a lower version) -> not supported.
	proc, _ = vcTestEnv(t, 60, &mockQueryClient{version: 50}, cns)
	ok, err = SupportProtocol(proc, engine.Nodes{{Id: "worker-1", Addr: "pipe-1"}}, 55)
	require.NoError(t, err)
	require.False(t, ok)

	// No CN matches the worker address -> unresolvable -> not supported.
	proc, _ = vcTestEnv(t, 60, &mockQueryClient{version: 60}, cns)
	ok, err = SupportProtocol(proc, engine.Nodes{{Id: "missing", Addr: "no-such-pipe"}}, 55)
	require.NoError(t, err)
	require.False(t, ok)

	// Empty worker address short-circuits to not-supported.
	proc, _ = vcTestEnv(t, 60, &mockQueryClient{version: 60}, cns)
	ok, err = SupportProtocol(proc, engine.Nodes{{Id: "worker-1", Addr: ""}}, 55)
	require.NoError(t, err)
	require.False(t, ok)
}

// Require is a no-op when the pipeline does not use the feature.
func TestRequireNoOpWhenFeatureAbsent(t *testing.T) {
	require.NoError(t, Require(nil, nil, false, 999, "must not fire"))
}

// When the feature IS present but the destination cannot be verified, Require fails closed with a
// message derived from name + version.
func TestRequireFailsClosedWithoutVerifiableDestination(t *testing.T) {
	err := Require(nil, nil, true, 999, "test feature")
	require.Error(t, err)
	require.ErrorContains(t, err, "remote destination does not support test feature (MORPC version 999)")
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrNotSupported))

	// A pipeline without a destination node is equally unverifiable.
	require.Error(t, Require(nil, &pipeline.Pipeline{}, true, 999, "test feature"))
}

func TestRequireSupportedDestination(t *testing.T) {
	cns := []metadata.CNService{{ServiceID: "worker-1", PipelineServiceAddress: "pipe-1", QueryAddress: "query-1"}}
	proc, _ := vcTestEnv(t, 60, &mockQueryClient{version: 60}, cns)
	p := &pipeline.Pipeline{Node: &pipeline.NodeInfo{Id: "worker-1", Addr: "pipe-1"}}
	require.NoError(t, Require(proc, p, true, 55, "test feature"))
}

func TestRegisterAndRunAll(t *testing.T) {
	before := len(Registered())

	sentinel := errors.New("custom fired")
	skipped := true
	Register(Check{ // data gate that is not needed -> RunAll skips it
		MinVer: 999,
		Needs:  func(*process.Process, *pipeline.Pipeline) bool { skipped = false; return false },
		Name:   "unused feature",
	})
	Register(Check{ // custom check that fails
		Custom: func(*process.Process, *pipeline.Pipeline) error { return sentinel },
	})

	got := Registered()
	require.Equal(t, before+2, len(got))

	err := RunAll(nil, nil)
	require.ErrorIs(t, err, sentinel)
	require.False(t, skipped, "the data check's detector must have been evaluated")
}
