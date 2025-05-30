// Copyright 2023 Matrix Origin
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

package v2

import (
	"github.com/prometheus/client_golang/prometheus"
)

var (
	taskShortDurationHistogram = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "mo",
			Subsystem: "task",
			Name:      "short_duration_seconds",
			Help:      "Bucketed histogram of short tn task execute duration.",
			Buckets:   getDurationBuckets(),
		}, []string{"type"})

	TaskFlushTableTailDurationHistogram  = taskShortDurationHistogram.WithLabelValues("flush_table_tail")
	TaskCommitTableTailDurationHistogram = taskShortDurationHistogram.WithLabelValues("commit_table_tail")
	GetObjectStatsDurationHistogram      = taskShortDurationHistogram.WithLabelValues("get_object_stats")

	// storage usage / show accounts metrics
	TaskGCkpCollectUsageDurationHistogram          = taskShortDurationHistogram.WithLabelValues("gckp_collect_usage")
	TaskICkpCollectUsageDurationHistogram          = taskShortDurationHistogram.WithLabelValues("ickp_collect_usage")
	TaskCompactedCollectUsageDurationHistogram     = taskShortDurationHistogram.WithLabelValues("compacted_collect_usage")
	TaskStorageUsageReqDurationHistogram           = taskShortDurationHistogram.WithLabelValues("handle_usage_request")
	TaskShowAccountsGetTableStatsDurationHistogram = taskShortDurationHistogram.WithLabelValues("show_accounts_get_table_stats")
	TaskShowAccountsGetUsageDurationHistogram      = taskShortDurationHistogram.WithLabelValues("show_accounts_get_storage_usage")
	TaskShowAccountsTotalDurationHistogram         = taskShortDurationHistogram.WithLabelValues("show_accounts_total_duration")
	TaskSnapshotReadReqDurationHistogram           = taskShortDurationHistogram.WithLabelValues("handle_snapshot_read")

	taskLongDurationHistogram = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "mo",
			Subsystem: "task",
			Name:      "long_duration_seconds",
			Help:      "Bucketed histogram of long tn task execute duration.",
			Buckets:   prometheus.ExponentialBuckets(1, 2.0, 13),
		}, []string{"type"})

	taskBytesHistogram = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "mo",
			Subsystem: "task",
			Name:      "hist_bytes",
			Help:      "Bucketed histogram of task result bytes.",
			Buckets:   prometheus.ExponentialBuckets(1, 2.0, 30),
		}, []string{"type"})

	taskCountHistogram = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "mo",
			Subsystem: "task",
			Name:      "hist_total",
			Help:      "Bucketed histogram of task result count.",
			Buckets:   prometheus.ExponentialBuckets(1, 2.0, 30),
		}, []string{"type"})

	TaskCkpEntryPendingDurationHistogram = taskLongDurationHistogram.WithLabelValues("ckp_entry_pending")
)

var (
	taskDNMergeStuffCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "mo",
			Subsystem: "task",
			Name:      "merge_stuff_total",
			Help:      "Total number of stuff a merge task have generated",
		}, []string{"type", "source"})

	TaskDataInputSizeCounter      = taskDNMergeStuffCounter.WithLabelValues("input_size", "data")
	TaskTombstoneInputSizeCounter = taskDNMergeStuffCounter.WithLabelValues("input_size", "tombstone")
	TaskDataMergeSizeCounter      = taskDNMergeStuffCounter.WithLabelValues("merged_size", "data")
	TaskTombstoneMergeSizeCounter = taskDNMergeStuffCounter.WithLabelValues("merged_size", "tombstone")

	taskDNMergeDurationHistogram = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "mo",
			Subsystem: "task",
			Name:      "merge_duration_seconds",
			Help:      "Bucketed histogram of merge duration.",
			Buckets:   getDurationBuckets(),
		}, []string{"type", "source"})

	TaskCommitDataMergeDurationHistogram      = taskDNMergeDurationHistogram.WithLabelValues("commit_merge", "data")
	TaskCommitTombstoneMergeDurationHistogram = taskDNMergeDurationHistogram.WithLabelValues("commit_merge", "tombstone")
	TaskDataMergeDurationHistogram            = taskDNMergeDurationHistogram.WithLabelValues("merge", "data")
	TaskTombstoneMergeDurationHistogram       = taskDNMergeDurationHistogram.WithLabelValues("merge", "tombstone")
)

// selectivity metrics
var (
	taskSelectivityCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "mo",
			Subsystem: "task",
			Name:      "selectivity",
			Help:      "Selectivity counter for read filter, block etc.",
		}, []string{"type"})

	TaskSelReadFilterTotal = taskSelectivityCounter.WithLabelValues("readfilter_total")
	TaskSelReadFilterHit   = taskSelectivityCounter.WithLabelValues("readfilter_hit")
	TaskSelBlockTotal      = taskSelectivityCounter.WithLabelValues("block_total")
	TaskSelBlockHit        = taskSelectivityCounter.WithLabelValues("block_hit")
	TaskSelColumnTotal     = taskSelectivityCounter.WithLabelValues("column_total")
	TaskSelColumnHit       = taskSelectivityCounter.WithLabelValues("column_hit")
)

var (
	TaskMergeTransferPageLengthGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "mo",
			Subsystem: "task",
			Name:      "merge_transfer_page_size",
			Help:      "Size of merge generated transfer page",
		})

	TaskStorageUsageCacheMemUsedGauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "mo",
			Subsystem: "task",
			Name:      "storage_usage_cache_size",
			Help:      "Size of the storage usage cache used",
		})
)

// transfer page metrics
var (
	transferPageHitHistogram = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "mo",
		Subsystem: "task",
		Name:      "transfer_page_hit_count",
		Help:      "The total number of transfer hit counter.",
	}, []string{"type"})

	TransferPageTotalHitHistogram = transferPageHitHistogram.WithLabelValues("total")

	TransferPageRowHistogram = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "mo",
		Subsystem: "task",
		Name:      "transfer_page_row",
		Help:      "The total number of transfer row.",
	})

	transferDurationHistogram = prometheus.NewHistogramVec(prometheus.HistogramOpts{

		Namespace: "mo",
		Subsystem: "task",
		Name:      "transfer_duration",
		Buckets:   getDurationBuckets(),
	}, []string{"type"})

	TransferDiskLatencyHistogram           = transferDurationHistogram.WithLabelValues("disk_latency")
	TransferPageSinceBornDurationHistogram = transferDurationHistogram.WithLabelValues("page_since_born_duration")
	TransferTableRunTTLDurationHistogram   = transferDurationHistogram.WithLabelValues("table_run_ttl_duration")
	TransferPageFlushLatencyHistogram      = transferDurationHistogram.WithLabelValues("page_flush_latency")
	TransferPageMergeLatencyHistogram      = transferDurationHistogram.WithLabelValues("page_merge_latency")

	transferShortDurationHistogram = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "mo",
		Subsystem: "task",
		Name:      "transfer_short_duration",
		Buckets:   getDurationBuckets(),
	}, []string{"type"})

	TransferMemLatencyHistogram = transferShortDurationHistogram.WithLabelValues("mem_latency")
)

// Mo table stats metrics
var (
	moTableStatsDurHistogram = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "mo",
		Subsystem: "task",
		Name:      "mo_table_stats_duration",
		Buckets:   getDurationBuckets(),
	}, []string{"type"})

	AlphaTaskDurationHistogram        = moTableStatsDurHistogram.WithLabelValues("alpha_task_duration")
	GamaTaskDurationHistogram         = moTableStatsDurHistogram.WithLabelValues("gama_task_duration")
	BulkUpdateOnlyTSDurationHistogram = moTableStatsDurHistogram.WithLabelValues("bulk_update_only_ts_duration")
	BulkUpdateStatsDurationHistogram  = moTableStatsDurHistogram.WithLabelValues("bulk_update_stats_duration")
	CalculateStatsDurationHistogram   = moTableStatsDurHistogram.WithLabelValues("calculate_stats_duration")

	MoTableSizeRowsNormalDurationHistogram          = moTableStatsDurHistogram.WithLabelValues("mo_table_size_rows_normal_duration")
	MoTableSizeRowsForceUpdateDurationHistogram     = moTableStatsDurHistogram.WithLabelValues("mo_table_size_rows_force_update_duration")
	MoTableSizeRowsResetUpdateTimeDurationHistogram = moTableStatsDurHistogram.WithLabelValues("mo_table_size_rows_reset_update_duration")

	moTableStatsCountingHistogram = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "mo",
		Subsystem: "task",
		Name:      "mo_table_stats_total",
		Buckets:   prometheus.ExponentialBuckets(1, 2, 12),
	}, []string{"type"})

	AlphaTaskCountingHistogram        = moTableStatsCountingHistogram.WithLabelValues("alpha_task_counting")
	GamaTaskCountingHistogram         = moTableStatsCountingHistogram.WithLabelValues("gama_task_counting")
	BulkUpdateOnlyTSCountingHistogram = moTableStatsCountingHistogram.WithLabelValues("bulk_update_only_ts_counting")
	BulkUpdateStatsCountingHistogram  = moTableStatsCountingHistogram.WithLabelValues("bulk_update_stats_counting")

	MoTableSizeRowsNormalCountingHistogram          = moTableStatsCountingHistogram.WithLabelValues("mo_table_size_rows_normal_counting")
	MoTableSizeRowsForceUpdateCountingHistogram     = moTableStatsCountingHistogram.WithLabelValues("mo_table_size_rows_force_update_counting")
	MoTableSizeRowsResetUpdateTimeCountingHistogram = moTableStatsCountingHistogram.WithLabelValues("mo_table_size_rows_reset_update_counting")
)
