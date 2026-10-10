"""Small, reproducible MLOps smoke workflow for LectureMind.

The default mode is safe and offline: validate the benchmark and render a
dashboard from an existing report.  Pass ``--run`` explicitly to execute the
live benchmark (which may use the configured LLM provider).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow invocation as ``python scripts/run_mlops_smoke.py`` as well as
# ``python -m scripts.run_mlops_smoke``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.validate_benchmark import validate_benchmark
from evaluation.dashboard.dashboard_generator import generate_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate, optionally run, and report a LectureMind smoke benchmark")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/datasets/multilecture_smoke_45.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/reports/smoke_groq_20261010_fixed"))
    parser.add_argument("--dashboard", type=Path, default=Path("evaluation/dashboard/index.html"))
    parser.add_argument("--run", action="store_true", help="Run live retrieval and generation; may consume provider quota")
    args = parser.parse_args()

    validation = validate_benchmark(args.dataset, profile="smoke")
    if validation["status"] != "PASS":
        print("Benchmark validation failed:")
        for issue in validation.get("issues", []):
            print(f"- {issue}")
        return 1
    print(f"Validated {validation['sample_count']} samples across {len(validation['lecture_distribution'])} lectures")

    if args.run:
        from evaluation.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner(str(args.dataset), str(args.output_dir), use_provider_registry_llm=True)
        runner.setup()
        results = runner.run()
        print(f"Completed {len(results)} samples; failures={sum(bool(r.get('error')) for r in results)}")
    else:
        print("Validation-only mode; use --run to execute the live benchmark")

    # The dashboard is a pure renderer over existing artifacts.  Point both
    # inputs at the selected run directory so it cannot silently display an
    # unrelated historical report from evaluation/outputs/.
    dashboard_path = generate_dashboard(
        outputs_dir=args.output_dir,
        reports_dir=args.output_dir,
        out_html=args.dashboard,
    )
    print(f"Dashboard written to {dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
