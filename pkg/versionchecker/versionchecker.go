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

// Package versionchecker centralizes remote MORPC-protocol capability checks so a version-gated
// wire feature is fenced at the destination boundary from one place instead of a hand-rolled
// validator per feature. A feature registers a Check (its version + a detector); the sender loops
// the registry and fails closed for any destination that predates the feature.
package versionchecker

import (
	"context"
	"fmt"
	"time"

	"github.com/matrixorigin/matrixone/pkg/clusterservice"
	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	moruntime "github.com/matrixorigin/matrixone/pkg/common/runtime"
	"github.com/matrixorigin/matrixone/pkg/pb/metadata"
	"github.com/matrixorigin/matrixone/pkg/pb/pipeline"
	querypb "github.com/matrixorigin/matrixone/pkg/pb/query"
	"github.com/matrixorigin/matrixone/pkg/vm/engine"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
)

// ProtocolVersion returns the service's current negotiated MORPC protocol version -- the rollout
// floor, lowered before a rollback and raised only once every CN understands the newest contract.
func ProtocolVersion(service string) (int64, bool) {
	rt := moruntime.ServiceRuntime(service)
	if rt == nil {
		return 0, false
	}
	version, ok := rt.GetGlobalVariables(moruntime.MOProtocolVersion)
	if !ok {
		return 0, false
	}
	protocolVersion, ok := version.(int64)
	return protocolVersion, ok
}

// LocalAtLeast reports whether this service's current negotiated protocol version is at least the
// minimum. It is the plan-time / local gate: it consults only the local rollout floor, not any
// destination. Use it wherever a planner decides whether to EMIT a version-gated feature; the
// remote registry (RunAll) is the send-boundary backstop. Fails closed (false) when the version is
// unknown.
func LocalAtLeast(service string, minimum int64) bool {
	version, ok := ProtocolVersion(service)
	return ok && version >= minimum
}

// SupportProtocol reports whether the coordinator's rollout gate AND every selected worker support
// at least the given MORPC protocol version. It reads the CURRENT version (so a rollback between
// plan build and this call is caught) and probes each worker; capabilities are not cached across
// executions or sender checks.
func SupportProtocol(proc *process.Process, workers engine.Nodes, minimum int64) (bool, error) {
	if proc == nil {
		return false, nil
	}
	parent := proc.Ctx
	if parent == nil {
		parent = context.Background()
	}
	if err := parent.Err(); err != nil {
		return false, err
	}
	version, known := ProtocolVersion(proc.GetService())
	if !known || version < minimum {
		return false, nil
	}
	ctx, cancel := context.WithTimeoutCause(parent, 5*time.Second, moerr.NewInternalError(parent, "remote protocol capability probe timed out"))
	defer cancel()
	for _, worker := range workers {
		if worker.Addr == "" || proc.GetQueryClient() == nil {
			return false, nil
		}
		cluster, err := clusterservice.GetMOClusterWithContext(ctx, proc.GetService())
		if err != nil {
			return false, parent.Err()
		}
		var addr string
		var workerID string
		selector := clusterservice.NewSelector()
		if worker.Id != "" {
			selector = clusterservice.NewServiceIDSelector(worker.Id)
		}
		err = clusterservice.GetCNServiceWithoutWorkingStateWithContext(ctx, cluster,
			selector, func(cn metadata.CNService) bool {
				if cn.PipelineServiceAddress == worker.Addr && (worker.Id == "" || cn.ServiceID == worker.Id) {
					addr = cn.QueryAddress
					workerID = cn.ServiceID
					return false
				}
				return true
			})
		if err != nil {
			return false, parent.Err()
		}
		if addr == "" {
			return false, nil
		}
		if workerID != "" && workerID == proc.GetService() {
			continue
		}
		client := proc.GetQueryClient()
		req := client.NewRequest(querypb.CmdMethod_GetProtocolVersion)
		req.GetProtocolVersion = &querypb.GetProtocolVersionRequest{}
		resp, err := client.SendMessage(ctx, addr, req)
		if err != nil {
			if resp != nil {
				client.Release(resp)
			}
			return false, parent.Err()
		}
		if resp == nil {
			return false, nil
		}
		supported := resp.GetProtocolVersion != nil && resp.GetProtocolVersion.Version >= minimum
		client.Release(resp)
		if !supported {
			return false, nil
		}
	}
	return true, nil
}

// Check is one remote destination capability gate: a feature that puts something on the wire an
// older CN would mishandle registers one so the sender fences it.
type Check struct {
	// MinVer is the MORPC protocol version that introduced the feature. It is the ONLY place the
	// version number appears for a plain gate -- the error message is derived from it (see Require),
	// so a renumber on a merge collision changes just this one field.
	MinVer int64
	// Needs reports whether the pipeline about to be serialized actually uses the feature.
	Needs func(proc *process.Process, p *pipeline.Pipeline) bool
	// Name is the feature phrase used to build the NotSupported message ("remote destination does
	// not support <Name> (MORPC version <MinVer>)"). It carries no version number.
	Name string
	// Custom, when set, replaces the default Needs/MinVer/Name fail-closed gate for a feature whose
	// contract is not a plain "supported-or-error" (e.g. one that also degrades). RunAll invokes it
	// directly and ignores MinVer/Needs/Name.
	Custom func(proc *process.Process, p *pipeline.Pipeline) error
}

var registry []Check

// Register adds a destination check. Call from an init() in the feature's own package so the
// detector can reference that package's types without this package importing upward.
func Register(c Check) { registry = append(registry, c) }

// Registered returns a copy of the registry for tests/introspection.
func Registered() []Check { return append([]Check(nil), registry...) }

// Require fails closed when needsFeature is true and the destination does not support minVer.
// A no-op when the feature is not present. The message is derived from name and minVer, so the
// version number is not duplicated as literal text.
func Require(proc *process.Process, p *pipeline.Pipeline, needsFeature bool, minVer int64, name string) error {
	if !needsFeature {
		return nil
	}
	if p != nil && p.Node != nil {
		supported, err := SupportProtocol(proc, engine.Nodes{{Id: p.Node.Id, Addr: p.Node.Addr}}, minVer)
		if err != nil {
			return err
		}
		if supported {
			return nil
		}
	}
	return moerr.NewNotSupportedNoCtx(fmt.Sprintf("remote destination does not support %s (MORPC version %d)", name, minVer))
}

// RunAll runs every registered check against the pipeline about to be serialized to p.Node,
// returning the first failure. The sole enforcement point, called from every remote encode path.
func RunAll(proc *process.Process, p *pipeline.Pipeline) error {
	for _, c := range registry {
		if c.Custom != nil {
			if err := c.Custom(proc, p); err != nil {
				return err
			}
			continue
		}
		if err := Require(proc, p, c.Needs != nil && c.Needs(proc, p), c.MinVer, c.Name); err != nil {
			return err
		}
	}
	return nil
}
