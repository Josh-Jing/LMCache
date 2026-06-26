# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import importlib.util
import sys
import types

# Third Party
import pandas as pd


def load_multi_round_qa_module():
    """Load the multi-round QA benchmark module from its hyphenated filename."""
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "multi_round_qa"
    sys.path.insert(0, str(benchmark_dir))
    sys.modules.setdefault("openai", types.ModuleType("openai"))
    spec = importlib.util.spec_from_file_location(
        "multi_round_qa", benchmark_dir / "multi-round-qa.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_process_summary_writes_performance_metrics_csv(tmp_path, capsys) -> None:
    module = load_multi_round_qa_module()
    output_path = tmp_path / "performance.csv"
    requests = pd.DataFrame(
        {
            "prompt_tokens": [10, 30],
            "generation_tokens": [5, 15],
            "ttft": [0.25, 0.75],
            "generation_time": [1.0, 3.0],
            "launch_time": [10.0, 12.0],
            "finish_time": [11.0, 14.0],
        }
    )

    returned = module.UserSessionManager.ProcessSummary(
        requests,
        start_time=10.0,
        end_time=14.0,
        pending_queries=1,
        config_qps=2.0,
        performance_output=output_path,
    )

    assert returned.equals(requests)
    terminal_output = capsys.readouterr().out
    assert "Performance summary" in terminal_output
    assert output_path.exists()

    metrics = pd.read_csv(output_path)
    assert list(metrics.columns) == [
        "config_qps",
        "actual_qps",
        "processing_speed",
        "requests_on_the_fly",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "average_generation_throughput_per_request",
        "average_ttft",
        "time_range_start",
        "time_range_end",
        "total_time",
        "launched_queries",
        "finished_queries",
        "pending_queries",
        "total_prompt_tokens",
        "total_generation_tokens",
    ]
    row = metrics.iloc[0]
    assert row["config_qps"] == 2.0
    assert row["actual_qps"] == 0.75
    assert row["processing_speed"] == 0.5
    assert row["requests_on_the_fly"] == 1
    assert row["input_tokens_per_second"] == 10.0
    assert row["output_tokens_per_second"] == 5.0
    assert row["average_generation_throughput_per_request"] == 5.0
    assert row["average_ttft"] == 0.5
    assert row["time_range_start"] == 10.0
    assert row["time_range_end"] == 14.0
    assert row["total_time"] == 4.0
    assert row["launched_queries"] == 2
    assert row["finished_queries"] == 2
    assert row["pending_queries"] == 1
    assert row["total_prompt_tokens"] == 40
    assert row["total_generation_tokens"] == 20


def test_default_performance_output_is_derived_from_request_output() -> None:
    module = load_multi_round_qa_module()

    assert module.default_performance_output("summary.csv") == "summary_performance.csv"
    assert module.default_performance_output("/tmp/run.details.csv") == (
        "/tmp/run.details_performance.csv"
    )
    assert module.default_performance_output("summary") == "summary_performance.csv"
