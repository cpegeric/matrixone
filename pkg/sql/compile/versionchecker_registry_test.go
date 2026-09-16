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

package compile

import (
	"testing"

	"github.com/matrixorigin/matrixone/pkg/defines"
	"github.com/matrixorigin/matrixone/pkg/versionchecker"
	"github.com/stretchr/testify/require"
)

// TestDestinationChecksRegisteredAndWellFormed is the drift guard: every remote destination check
// this package's init()s register must be well-formed, so a new version-gated wire feature cannot
// land a malformed (or silently ineffective) fence. A plain gate needs a detector, a positive MinVer
// no greater than the latest protocol version, and a message; a Custom check owns its own logic. The
// compile package registers conv-bases, integer-domain, ip, string-numeric, strict-write, and
// group-concat, so the registry is at least those six.
func TestDestinationChecksRegisteredAndWellFormed(t *testing.T) {
	checks := versionchecker.Registered()
	require.GreaterOrEqual(t, len(checks), 6, "the migrated destination checks must be registered")
	for i, c := range checks {
		if c.Custom != nil {
			require.Nil(t, c.Needs, "check %d: Custom and Needs are mutually exclusive", i)
			continue
		}
		require.NotNil(t, c.Needs, "check %d: a plain gate needs a detector", i)
		require.NotEmpty(t, c.Name, "check %d: a plain gate needs a name", i)
		require.Greater(t, c.MinVer, int64(0), "check %d: MinVer must be positive", i)
		require.LessOrEqual(t, c.MinVer, defines.MORPCLatestVersion,
			"check %d: MinVer %d exceeds MORPCLatestVersion %d", i, c.MinVer, defines.MORPCLatestVersion)
	}
}
