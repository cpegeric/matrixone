// Copyright 2021 - 2022 Matrix Origin
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

package hakeeper

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTimeoutConfig(t *testing.T) {
	c := Config{}
	c.Fill()
	assert.Equal(t, DefaultTickPerSecond, c.TickPerSecond)
	assert.Equal(t, DefaultLogStoreTimeout, c.LogStoreTimeout)
	assert.Equal(t, DefaultTNStoreTimeout, c.TNStoreTimeout)
}

func TestCNMassFailureSuppressed(t *testing.T) {
	c := Config{}
	c.Fill()
	cases := []struct {
		name    string
		expired int
		total   int
		want    bool
	}{
		{"no stores", 0, 0, false},
		{"none expired", 0, 5, false},
		{"single expired in big cluster", 1, 5, false}, // below MinMassFailureStores
		{"single expired single cluster", 1, 1, false}, // real outage, act on it
		{"two of three expired", 2, 3, true},           // correlated dip -> suppress
		{"half expired", 3, 6, true},
		{"just under half", 2, 5, false},
		{"all expired", 5, 5, true}, // detector was blind
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, c.CNMassFailureSuppressed(tc.expired, tc.total))
		})
	}
}
