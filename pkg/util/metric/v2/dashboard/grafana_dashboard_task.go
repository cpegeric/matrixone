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

package dashboard

import (
	"context"
	"fmt"

	"github.com/K-Phoen/grabana/axis"
	"github.com/K-Phoen/grabana/dashboard"
	"github.com/K-Phoen/grabana/row"
	"github.com/K-Phoen/grabana/timeseries"
	tsaxis "github.com/K-Phoen/grabana/timeseries/axis"
)

func (c *DashboardCreator) initTaskDashboard() error {
	folder, err := c.createFolder(c.folderName)
	if err != nil {
		return err
	}

	build, err := dashboard.New(
		"Task Metrics",
		c.withRowOptions(
			c.initTaskFlushTableTailRow(),
			c.initTaskMergeRow(),
			c.initTaskMergeTransferPageRow(),
			c.initTaskCheckpointRow(),
			c.initTaskSelectivityRow(),
			c.initTaskStorageUsageRow(),
			c.initMoTableStatsTaskDurationRow(),
			c.initMoTableStatsTaskCountingRow(),
		)...)

	if err != nil {
		return err
	}
	_, err = c.cli.UpsertDashboard(context.Background(), folder, build)
	return err
}

func (c *DashboardCreator) initTaskMergeTransferPageRow() dashboard.Option {
	return dashboard.Row(
		"Task Merge Transfer Page",
		c.withGraph(
			"Transfer Page size",
			12,
			`sum(`+c.getMetricWithFilter("mo_task_merge_transfer_page_size", ``)+`)`+`* 28 * 1.3`,
			"{{ "+c.by+" }}", axis.Unit("decbytes"),
		),
		c.getTimeSeries(
			"Transfer page row count",
			[]string{fmt.Sprintf(
				"sum (increase(%s[$interval]))",
				c.getMetricWithFilter(`mo_task_transfer_page_row_sum`, ""),
			)},
			[]string{"count"},
			timeseries.Span(3),
		),
		c.getTimeSeries(
			"Transfer page hit count",
			[]string{fmt.Sprintf(
				"sum (increase(%s[$interval]))",
				c.getMetricWithFilter(`mo_task_transfer_page_hit_count_sum`, `type="total"`),
			)},
			[]string{"count"},
			timeseries.Span(3),
		),
		c.getPercentHist(
			"Transfer run ttl duration",
			c.getMetricWithFilter(`mo_task_transfer_duration_bucket`, `type="table_run_ttl_duration"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(3),
		),
		c.getPercentHist(
			"Transfer duration since born",
			c.getMetricWithFilter(`mo_task_transfer_duration_bucket`, `type="page_since_born_duration"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(3),
		),
		c.getPercentHist(
			"Transfer memory latency",
			c.getMetricWithFilter(`mo_task_transfer_short_duration_bucket`, `type="mem_latency"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(3),
		),
		c.getPercentHist(
			"Transfer disk latency",
			c.getMetricWithFilter(`mo_task_transfer_duration_bucket`, `type="disk_latency"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(3),
		),
		c.getPercentHist(
			"Transfer page write latency in flush",
			c.getMetricWithFilter(`mo_task_transfer_duration_bucket`, `type="page_flush_latency"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(3),
		),
		c.getPercentHist(
			"Transfer page write latency in merge",
			c.getMetricWithFilter(`mo_task_transfer_duration_bucket`, `type="page_merge_latency"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(3),
		),
	)
}

func (c *DashboardCreator) initTaskFlushTableTailRow() dashboard.Option {
	return dashboard.Row(
		"Flush Table Tail",
		c.getTimeSeries(
			"Flush table tail Count",
			[]string{fmt.Sprintf(
				"sum (increase(%s[$interval]))",
				c.getMetricWithFilter(`mo_task_short_duration_seconds_count`, `type="flush_table_tail"`),
			)},
			[]string{"Count"},
			timeseries.Span(4),
		),
		c.getPercentHist(
			"Flush table tail Duration",
			c.getMetricWithFilter(`mo_task_short_duration_seconds_bucket`, `type="flush_table_tail"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(4),
		),
		c.getPercentHist(
			"Flush table tail Commit Time",
			c.getMetricWithFilter(`mo_task_short_duration_seconds_bucket`, `type="commit_table_tail"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(4),
		),
	)
}

func (c *DashboardCreator) initTaskMergeRow() dashboard.Option {
	return dashboard.Row(
		"Merge",

		c.getTimeSeries(
			"Merge Count",
			[]string{
				fmt.Sprintf(
					"sum (increase(%s[$interval]))",
					c.getMetricWithFilter(`mo_task_merge_duration_seconds_count`, `type="merge",source="data"`)),
				fmt.Sprintf(
					"sum (increase(%s[$interval]))",
					c.getMetricWithFilter(`mo_task_merge_duration_seconds_count`, `type="merge",source="tombstone"`)),
			},
			[]string{
				"Data Count",
				"Tombstone Count",
			},
			timeseries.Span(4),
		),

		c.getPercentHist(
			"Merge Duration",
			c.getMetricWithFilter(`mo_task_merge_duration_seconds_bucket`, `type="merge"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(4),
		),

		c.getPercentHist(
			"Merge Commit Duration",
			c.getMetricWithFilter(`mo_task_merge_duration_seconds_bucket`, `type="commit_merge"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(4),
		),

		// data size
		c.getTimeSeries(
			"Merge Data Size",
			[]string{
				fmt.Sprintf(
					"sum (increase(%s[$interval]))",
					c.getMetricWithFilter(`mo_task_merge_stuff_total`, `type="merged_size",source="data"`)),
				fmt.Sprintf(
					"sum (increase(%s[$interval]))",
					c.getMetricWithFilter(`mo_task_merge_stuff_total`, `type="input_size",source="data"`)),
			},
			[]string{"Merged Size", "Input Size"},
			timeseries.Axis(tsaxis.Unit("decbytes")),
			timeseries.Span(6),
		),

		// tombstone size
		c.getTimeSeries(
			"Merge Tombstone Size",
			[]string{
				fmt.Sprintf(
					"sum (increase(%s[$interval]))",
					c.getMetricWithFilter(`mo_task_merge_stuff_total`, `type="merged_size",source="tombstone"`)),
				fmt.Sprintf(
					"sum (increase(%s[$interval]))",
					c.getMetricWithFilter(`mo_task_merge_stuff_total`, `type="input_size",source="tombstone"`)),
			},
			[]string{"Merged Size", "Input Size"},
			timeseries.Axis(tsaxis.Unit("decbytes")),
			timeseries.Span(6),
		),

		c.getTimeSeries(
			"Merge Ampilification",
			[]string{
				fmt.Sprintf(
					"sum (increase(%s[$interval])) / sum (increase(%s[$interval]))",
					c.getMetricWithFilter(`mo_task_merge_stuff_total`, `type="merged_size",source="data"`),
					c.getMetricWithFilter(`mo_task_merge_stuff_total`, `type="input_size",source="data"`),
				),
				fmt.Sprintf(
					"sum (increase(%s[$interval])) / sum (increase(%s[$interval]))",
					c.getMetricWithFilter(`mo_task_merge_stuff_total`, `type="merged_size",source="tombstone"`),
					c.getMetricWithFilter(`mo_task_merge_stuff_total`, `type="input_size",source="tombstone"`),
				),
			},
			[]string{"Data", "Tombstone"},
			timeseries.Span(6),
		),
	)
}

func (c *DashboardCreator) initTaskCheckpointRow() dashboard.Option {
	return dashboard.Row(
		"Checkpoint",
		c.getPercentHist(
			"Checkpoint Entry Pending",
			c.getMetricWithFilter(`mo_task_long_duration_seconds_bucket`, `type="ckp_entry_pending"`),
			[]float64{0.50, 0.8, 0.90, 0.99},
			SpanNulls(true),
			timeseries.Span(12),
		),
	)
}

func (c *DashboardCreator) initTaskStorageUsageRow() dashboard.Option {
	rows := c.getMultiHistogram(
		[]string{
			c.getMetricWithFilter(`mo_task_short_duration_seconds_bucket`, `type="gckp_collect_usage"`),
			c.getMetricWithFilter(`mo_task_short_duration_seconds_bucket`, `type="ickp_collect_usage"`),
			c.getMetricWithFilter(`mo_task_short_duration_seconds_bucket`, `type="compacted_collect_usage"`),
			c.getMetricWithFilter(`mo_task_short_duration_seconds_bucket`, `type="handle_usage_request"`),
			c.getMetricWithFilter(`mo_task_short_duration_seconds_bucket`, `type="show_accounts_get_table_stats"`),
			c.getMetricWithFilter(`mo_task_short_duration_seconds_bucket`, `type="show_accounts_get_storage_usage"`),
			c.getMetricWithFilter(`mo_task_short_duration_seconds_bucket`, `type="show_accounts_total_duration"`),
		},
		[]string{
			"gckp_collect_usage",
			"ickp_collect_usage",
			"compacted_collect_usage",
			"handle_usage_request",
			"show_accounts_get_table_stats",
			"show_accounts_get_storage_usage",
			"show_accounts_total_duration",
		},
		[]float64{0.50, 0.8, 0.90, 0.99},
		[]float32{3, 3, 3, 3},
		axis.Unit("s"),
		axis.Min(0))

	rows = append(rows, c.withGraph(
		"tn storage usage cache mem used",
		12,
		`sum(`+c.getMetricWithFilter("mo_task_storage_usage_cache_size", ``)+`)`,
		"cache mem used",
		axis.Unit("mb")))

	return dashboard.Row(
		"Storage Usage Overview",
		rows...,
	)

}

func (c *DashboardCreator) initTaskSelectivityRow() dashboard.Option {

	hitRateFunc := func(title, metricType string) row.Option {
		return c.getTimeSeries(
			title,
			[]string{
				fmt.Sprintf(
					"sum(%s) by (%s) / on(%s) sum(%s) by (%s)",
					c.getMetricWithFilter(`mo_task_selectivity`, `type="`+metricType+`_hit"`), c.by, c.by,
					c.getMetricWithFilter(`mo_task_selectivity`, `type="`+metricType+`_total"`), c.by),
			},
			[]string{fmt.Sprintf("filterout-{{ %s }}", c.by)},
			timeseries.Span(4),
		)
	}
	counterRateFunc := func(title, metricType string) row.Option {
		return c.getTimeSeries(
			title,
			[]string{
				fmt.Sprintf(
					"sum(rate(%s[$interval])) by (%s)",
					c.getMetricWithFilter(`mo_task_selectivity`, `type="`+metricType+`_total"`), c.by),
			},
			[]string{fmt.Sprintf("req-{{ %s }}", c.by)},
			timeseries.Span(4),
		)
	}
	return dashboard.Row(
		"Read Selectivity",
		hitRateFunc("Read filter rate", "readfilter"),
		hitRateFunc("Block range filter rate", "block"),
		hitRateFunc("Column update filter rate", "column"),
		counterRateFunc("Read filter request", "readfilter"),
		counterRateFunc("Block range request", "block"),
		counterRateFunc("Column update request", "column"),
		c.getPercentHist(
			"Iterate deletes rows count per block",
			c.getMetricWithFilter(`mo_task_hist_total_bucket`, `type="load_mem_deletes_per_block"`),
			[]float64{0.5, 0.7, 0.8, 0.9},
			timeseries.Axis(tsaxis.Unit("")),
			timeseries.Span(4),
			SpanNulls(true),
		),
	)
}

func (c *DashboardCreator) initMoTableStatsTaskDurationRow() dashboard.Option {
	return dashboard.Row(
		"Mo Table Stats Task Duration",
		c.getMultiHistogram(
			[]string{
				c.getMetricWithFilter(`mo_task_mo_table_stats_duration_bucket`, `type="alpha_task_duration"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_duration_bucket`, `type="gama_task_duration"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_duration_bucket`, `type="bulk_update_only_ts_duration"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_duration_bucket`, `type="bulk_update_stats_duration"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_duration_bucket`, `type="calculate_stats_duration"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_duration_bucket`, `type="mo_table_size_rows_normal_duration"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_duration_bucket`, `type="mo_table_size_rows_force_update_duration"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_duration_bucket`, `type="mo_table_size_rows_reset_update_duration"`),
			},

			[]string{
				"alpha_task_duration",
				"gama_task_duration",
				"bulk_update_only_ts_duration",
				"bulk_update_stats_duration",
				"calculate_stats_duration",
				"mo_table_size_rows_normal_duration",
				"mo_table_size_rows_force_update_duration",
				"mo_table_size_rows_reset_update_duration",
			},

			[]float64{0.50, 0.8, 0.90, 0.99},
			[]float32{3, 3, 3, 3},
			axis.Unit("s"),
			axis.Min(0))...)
}

func (c *DashboardCreator) initMoTableStatsTaskCountingRow() dashboard.Option {
	return dashboard.Row(
		"Mo Table Stats Task Counting",
		c.getMultiHistogram(
			[]string{
				c.getMetricWithFilter(`mo_task_mo_table_stats_total_bucket`, `type="alpha_task_counting"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_total_bucket`, `type="gama_task_counting"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_total_bucket`, `type="bulk_update_only_ts_counting"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_total_bucket`, `type="bulk_update_stats_counting"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_total_bucket`, `type="mo_table_size_rows_normal_counting"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_total_bucket`, `type="mo_table_size_rows_force_update_counting"`),
				c.getMetricWithFilter(`mo_task_mo_table_stats_total_bucket`, `type="mo_table_size_rows_reset_update_counting"`),
			},

			[]string{
				"alpha_task_counting",
				"gama_task_counting",
				"bulk_update_only_ts_counting",
				"bulk_update_stats_counting",
				"mo_table_size_rows_normal_counting",
				"mo_table_size_rows_force_update_counting",
				"mo_table_size_rows_reset_update_counting",
			},

			[]float64{0.50, 0.8, 0.90, 0.99},
			[]float32{3, 3, 3, 3},
			axis.Min(0))...)
}
