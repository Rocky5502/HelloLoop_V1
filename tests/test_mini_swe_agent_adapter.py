from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from helloloop.adapters import (
    EXPECTED_MINI_SWE_AGENT_VERSION,
    HelloLoopAgent,
    check_upstream_version,
    run_deterministic_smoke,
)
from helloloop.adapters.mini_swe_agent import _is_visible_test_command, _model_response_evidence
from helloloop.enums import LoopVariant
from helloloop.schema import TaskSpec, read_trajectory

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("minisweagent") is None,
    reason="optional mini-SWE-agent dependency is not installed",
)


def test_upstream_version_is_pinned() -> None:
    assert check_upstream_version() == EXPECTED_MINI_SWE_AGENT_VERSION


def test_assertion_and_combined_reproduction_commands_count_as_visible_tests() -> None:
    assertion = "python - <<'PY'\nassert 2 + 2 == 4\nPY"
    combined = (
        "python - <<'PY'\nprint('patched')\nPY\npython - <<'PY'\nprint(2 + 2)\nPY\ngit diff --check"
    )

    assert _is_visible_test_command(assertion)
    assert _is_visible_test_command(combined)
    assert not _is_visible_test_command("python - <<'PY'\nprint('inspect only')\nPY")


def test_action_only_model_response_has_valid_neutral_evidence() -> None:
    actions = [{"command": "cd /testbed && git status --short"}]

    assert _model_response_evidence("", actions) == (
        "model proposed 1 action without textual rationale"
    )
    assert _model_response_evidence("THOUGHT: inspect first", actions) == "THOUGHT: inspect first"


def test_real_upstream_agent_dry_run_exports_common_artifacts(tmp_path: Path) -> None:
    result = run_deterministic_smoke(tmp_path)

    assert result["exit_status"] == "Submitted"
    patch = (tmp_path / "patch.diff").read_text(encoding="utf-8")
    assert "return a + b" in patch
    records = read_trajectory(tmp_path / "trajectory.jsonl")
    assert records
    assert [record.step_index for record in records] == list(range(len(records)))
    assert sum(record.cost.tool_calls for record in records) == 2
    assert records[-1].decision.value == "stop_success"
    assert all(record.run_id == "mini-swe-agent-smoke-linear-r0" for record in records)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"


def test_integration_does_not_modify_upstream_checkout(tmp_path: Path) -> None:
    upstream = Path(__file__).parents[1] / "third_party" / "mini-swe-agent"
    before = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    run_deterministic_smoke(tmp_path)
    after = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert before == after == ""


def _run_submission_policy(out_dir: Path, variant: LoopVariant) -> tuple[dict, HelloLoopAgent]:
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.test_models import DeterministicModel, make_output

    temp_dir = TemporaryDirectory(prefix="helloloop-policy-test-")
    repo = Path(temp_dir.name)
    (repo / "calculator.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "test_calculator.py").write_text(
        "import unittest\n"
        "from calculator import add\n\n"
        "class AddTest(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=HelloLoop CI",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    model = DeterministicModel(
        outputs=[
            make_output(
                "Apply the fix. Confidence: 0.95",
                [{"command": "printf 'def add(a, b):\\n    return a + b\\n' > calculator.py"}],
                cost=0.01,
            ),
            make_output(
                "Submit immediately.",
                [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"}],
                cost=0.01,
            ),
            make_output(
                "Run the required checks. Confidence: 0.99",
                [{"command": "python -m unittest -q && git diff --check"}],
                cost=0.01,
            ),
            make_output(
                "Submit after evidence.",
                [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"}],
                cost=0.01,
            ),
        ]
    )
    agent = HelloLoopAgent(
        model=model,
        env=LocalEnvironment(cwd=str(repo), timeout=10),
        task_spec=TaskSpec(
            task_id=f"policy-{variant.value}",
            benchmark="local_smoke",
            problem_statement="Fix add().",
        ),
        run_id=f"policy-{variant.value}",
        run_dir=out_dir,
        loop_variant=variant,
        system_template="You are a deterministic test agent.",
        instance_template="{{ task }}",
        step_limit=8,
        cost_limit=1.0,
    )
    agent._test_temp_dir = temp_dir  # type: ignore[attr-defined]
    return agent.run(), agent


def test_fixed_verification_defers_submission_until_evidence(tmp_path: Path) -> None:
    result, agent = _run_submission_policy(tmp_path, LoopVariant.VERIFICATION)
    records = read_trajectory(tmp_path / "trajectory.jsonl")

    assert result["exit_status"] == "Submitted"
    assert agent._submission_deferrals == 1
    assert agent._visible_tests_passed is True
    assert agent._static_checks_passed is True
    deferred = [
        record
        for record in records
        if record.stage.value == "termination" and record.decision.value == "verify_more"
    ]
    assert deferred
    assert records[-1].decision.value == "stop_success"


def test_linear_accepts_same_early_submission_without_controller_gate(tmp_path: Path) -> None:
    result, agent = _run_submission_policy(tmp_path, LoopVariant.LINEAR)

    assert result["exit_status"] == "Submitted"
    assert agent._submission_deferrals == 0
    assert agent.n_calls == 2


def test_reflection_injects_one_fixed_replan_without_submission_gate(tmp_path: Path) -> None:
    result, agent = _run_submission_policy(tmp_path, LoopVariant.REFLECTION)
    records = read_trajectory(tmp_path / "trajectory.jsonl")

    assert result["exit_status"] == "Submitted"
    assert agent._reflection_used is True
    assert agent._submission_deferrals == 0
    assert sum(record.decision.value == "replan" for record in records) == 1


def test_uncertainty_aware_gate_uses_evidence_and_residual_uncertainty(tmp_path: Path) -> None:
    result, agent = _run_submission_policy(tmp_path, LoopVariant.UNCERTAINTY_AWARE)
    records = read_trajectory(tmp_path / "trajectory.jsonl")

    assert result["exit_status"] == "Submitted"
    assert agent._submission_deferrals == 1
    assert agent._current_uncertainty.composite <= 0.20
    assert any(record.decision.value == "verify_more" for record in records)
    assert records[-1].decision.value == "stop_success"
