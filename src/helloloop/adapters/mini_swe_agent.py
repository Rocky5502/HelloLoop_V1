from __future__ import annotations

import os
import re
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from helloloop.controller import ControllerDecision, ControllerState, decide
from helloloop.enums import Decision, LoopVariant, Stage
from helloloop.schema import (
    CostRecord,
    JsonlTrajectoryWriter,
    StepEvidence,
    StepRecord,
    TaskSpec,
    UncertaintyVector,
    redact_credentials,
    redact_text,
    summarize_trajectory,
    utc_now_iso,
    write_summary_json,
)
from helloloop.uncertainty import composite_uncertainty, execution_uncertainty, progress_uncertainty

os.environ.setdefault(
    "MSWEA_GLOBAL_CONFIG_DIR",
    str(Path(tempfile.gettempdir()) / "helloloop-mini-swe-agent-config"),
)
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

try:
    import minisweagent
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.exceptions import InterruptAgentFlow, Submitted
except ImportError as error:  # pragma: no cover - exercised only without optional dependency
    minisweagent = None
    DefaultAgent = object  # type: ignore[assignment,misc]
    InterruptAgentFlow = Exception  # type: ignore[assignment,misc]
    Submitted = Exception  # type: ignore[assignment,misc]
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


EXPECTED_MINI_SWE_AGENT_VERSION = "2.4.5"
CONFIDENCE_RE = re.compile(r"(?i)\bconfidence\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)\b")
TEST_COMMAND_RE = re.compile(
    r"(?i)(?:^|[;&|]\s*)(?:python\s+-m\s+(?:pytest|unittest)|pytest|tox|nox|"
    r"npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|yarn\s+test|cargo\s+test|"
    r"go\s+test|mvn\s+test|gradle\s+test)(?:\s|$)"
)
STATIC_COMMAND_RE = re.compile(
    r"(?i)(?:git\s+diff\s+--check|ruff(?:\s+check)?|mypy|pyright|eslint|"
    r"python\s+-m\s+compileall)(?:\s|$)"
)
PYTHON_HEREDOC_RE = re.compile(r"(?i)python\s+-\s+<<")
RETRIEVAL_COMMAND_RE = re.compile(
    r"(?i)^\s*(?:pwd|ls|find|rg|grep|cat|head|tail|wc|tree|git\s+(?:status|log|show|diff))(?:\s|$)"
)


class MiniSWEAgentUnavailable(RuntimeError):
    """Raised when the optional pinned agent dependency cannot be imported."""


def check_upstream_version(strict: bool = True) -> str:
    if minisweagent is None:
        raise MiniSWEAgentUnavailable(
            "mini-SWE-agent is unavailable; install the pinned agent dependencies"
        ) from _IMPORT_ERROR
    installed = str(minisweagent.__version__)
    if strict and installed != EXPECTED_MINI_SWE_AGENT_VERSION:
        raise RuntimeError(
            "mini-SWE-agent version drift: "
            f"expected {EXPECTED_MINI_SWE_AGENT_VERSION}, found {installed}"
        )
    return installed


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values = []
        for item in content:
            if isinstance(item, dict):
                values.append(str(item.get("text", item.get("content", ""))))
        return "\n".join(value for value in values if value)
    return str(content or "")


def _usage_from_message(message: dict[str, Any]) -> tuple[int, int]:
    response = message.get("extra", {}).get("response", {})
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    return int(input_tokens), int(output_tokens)


def _confidence_from_text(text: str) -> float | None:
    match = CONFIDENCE_RE.search(text)
    return float(match.group(1)) if match else None


def _model_response_evidence(content: str, actions: list[dict[str, Any]]) -> str:
    if content.strip():
        return content
    count = len(actions)
    return f"model proposed {count} action{'s' if count != 1 else ''} without textual rationale"


def _is_visible_test_command(command: str) -> bool:
    if TEST_COMMAND_RE.search(command):
        return True
    python_blocks = len(PYTHON_HEREDOC_RE.findall(command))
    return bool(
        python_blocks
        and (
            re.search(r"(?m)^\s*assert\b", command)
            or (python_blocks >= 2 and STATIC_COMMAND_RE.search(command))
        )
    )


class HelloLoopAgent(DefaultAgent):  # type: ignore[misc]
    """Version-pinned mini-SWE-agent with privacy-safe HelloLoop event hooks."""

    def __init__(
        self,
        model: Any,
        env: Any,
        *,
        task_spec: TaskSpec,
        run_id: str,
        loop_variant: LoopVariant = LoopVariant.LINEAR,
        seed: int = 20260711,
        repetition: int = 0,
        run_dir: str | Path,
        strict_upstream_version: bool = True,
        **kwargs: Any,
    ) -> None:
        check_upstream_version(strict=strict_upstream_version)
        task_spec.validate()
        self.task_spec = task_spec
        self.run_id = run_id
        self.loop_variant = LoopVariant(loop_variant)
        self.seed = int(seed)
        self.repetition = int(repetition)
        self.run_dir = Path(run_dir)
        self.trajectory_path = self.run_dir / "trajectory.jsonl"
        self.patch_path = self.run_dir / "patch.diff"
        self.upstream_trajectory_path = self.run_dir / "upstream_trajectory.json"
        self.summary_path = self.run_dir / "summary.json"
        self._writer = JsonlTrajectoryWriter(self.trajectory_path)
        self._step_index = 0
        self._action_history: list[str] = []
        self._no_new_evidence_steps = 0
        self._current_uncertainty = UncertaintyVector()
        self._visible_tests_passed = False
        self._static_checks_passed = False
        self._evidence_count = 0
        self._reflection_used = False
        self._verification_attempts = 0
        self._submission_deferrals = 0
        self._returned_model_ids: list[str] = []
        super().__init__(model=model, env=env, output_path=self.upstream_trajectory_path, **kwargs)

    @property
    def model_name(self) -> str:
        config = getattr(self.model, "config", None)
        return str(getattr(config, "model_name", type(self.model).__name__))

    def initialize_task(self, task: str | None = None) -> TaskSpec:
        problem_statement = task or self.task_spec.problem_statement
        if problem_statement != self.task_spec.problem_statement:
            self.task_spec = TaskSpec(
                task_id=self.task_spec.task_id,
                benchmark=self.task_spec.benchmark,
                problem_statement=problem_statement,
                repo=self.task_spec.repo,
                base_commit=self.task_spec.base_commit,
                metadata=self.task_spec.metadata,
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_path.write_text("", encoding="utf-8")
        self._step_index = 0
        self._action_history = []
        self._no_new_evidence_steps = 0
        self._current_uncertainty = UncertaintyVector()
        self._visible_tests_passed = False
        self._static_checks_passed = False
        self._evidence_count = 0
        self._reflection_used = False
        self._verification_attempts = 0
        self._submission_deferrals = 0
        self._returned_model_ids = []
        return self.task_spec

    def collect_history(self) -> list[dict[str, Any]]:
        return deepcopy(self.messages)

    def propose_plan(self) -> dict[str, Any]:
        return self.query()

    def retrieve_context(self, command: str) -> dict[str, Any]:
        return self.execute_action({"command": command}, stage=Stage.RETRIEVAL)

    def verify(self, command: str) -> dict[str, Any]:
        return self.execute_action({"command": command}, stage=Stage.VERIFICATION)

    def execute_action(
        self,
        action: dict[str, Any],
        *,
        stage: Stage = Stage.EXECUTION,
    ) -> dict[str, Any]:
        command = str(action.get("command", ""))
        try:
            output = self.env.execute(action)
        except Submitted as error:
            return self._handle_submission(command, stage, error)
        except InterruptAgentFlow as error:
            evidence = "\n".join(_message_text(message) for message in error.messages)
            self._record_action(command, stage, None, evidence, submitted=True)
            raise
        self._record_action(
            command,
            stage,
            int(output.get("returncode", -1)),
            str(output.get("output", "")) + str(output.get("exception_info", "")),
        )
        return output

    def export_patch(self, path: str | Path | None = None) -> Path:
        output_path = Path(path) if path is not None else self.patch_path
        result = self.env.execute({"command": "git diff --binary --no-ext-diff"})
        patch, _ = redact_credentials(str(result.get("output", "")), max_chars=2_000_000)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(patch, encoding="utf-8")
        return output_path

    def query(self) -> dict[str, Any]:
        message = super().query()
        response = message.get("extra", {}).get("response", {})
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        if isinstance(response, dict) and response.get("model"):
            self._returned_model_ids.append(str(response["model"]))
        content = _message_text(message)
        actions = message.get("extra", {}).get("actions", [])
        proposes_submission = any(
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in str(action.get("command", ""))
            for action in actions
        )
        confidence = _confidence_from_text(content)
        if confidence is None:
            planning = 0.1 if proposes_submission else 0.5
        else:
            planning = 1.0 - confidence
        task = min(self._current_uncertainty.task, min(1.0, planning + 0.1))
        termination = (
            min(self._current_uncertainty.termination, planning)
            if proposes_submission
            else self._current_uncertainty.termination
        )
        components = {
            "task": task,
            "planning": planning,
            "retrieval": self._current_uncertainty.retrieval,
            "execution": self._current_uncertainty.execution,
            "verification": self._current_uncertainty.verification,
            "memory": self._current_uncertainty.memory,
            "progress": self._current_uncertainty.progress,
            "termination": termination,
        }
        self._current_uncertainty = UncertaintyVector(
            **components,
            composite=composite_uncertainty(components),
        )
        input_tokens, output_tokens = _usage_from_message(message)
        safe_content, _ = redact_text(content)
        evidence_content = _model_response_evidence(content, actions)
        self._append_record(
            stage=Stage.PLANNING,
            action="model_query",
            decision=Decision.EXECUTE,
            rationale=safe_content or "model proposed one or more actions",
            evidence=[
                StepEvidence.from_text(
                    "model_response",
                    evidence_content,
                    source="mini-swe-agent",
                )
            ],
            cost=CostRecord(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=float(message.get("extra", {}).get("cost", 0.0)),
            ),
        )
        return message

    def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        outputs = []
        stages = []
        for action in message.get("extra", {}).get("actions", []):
            stage = self._classify_stage(str(action.get("command", "")))
            stages.append(stage)
            outputs.append(self.execute_action(action, stage=stage))
        observations = self.model.format_observation_messages(
            message,
            outputs,
            self.get_template_vars(),
        )
        added = self.add_messages(*observations)
        for stage, output in zip(stages, outputs):
            if output.get("extra", {}).get("helloloop_submission_deferred"):
                added.extend(
                    self.add_messages(
                        self.model.format_message(role="user", content=str(output["output"]))
                    )
                )
                continue
            control_message = self._control_after_action(stage)
            if control_message is not None:
                added.extend(self.add_messages(control_message))
        return added

    def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        self.initialize_task(task or self.task_spec.problem_statement)
        result: dict[str, Any] | None = None
        try:
            result = super().run(self.task_spec.problem_statement, **kwargs)
            self._record_termination(result)
            return result
        except Exception as error:
            self._record_termination(
                {"exit_status": type(error).__name__, "submission": ""},
                infrastructure_error=True,
            )
            raise
        finally:
            try:
                self.export_patch()
            except Exception as patch_error:
                self.patch_path.write_text(
                    f"Patch export failed: {type(patch_error).__name__}\n",
                    encoding="utf-8",
                )
            if self.trajectory_path.exists() and self.trajectory_path.stat().st_size:
                status = (
                    "complete" if result and result.get("exit_status") == "Submitted" else "failed"
                )
                write_summary_json(
                    summarize_trajectory(self.trajectory_path, status=status),
                    self.summary_path,
                )

    def _record_action(
        self,
        command: str,
        stage: Stage,
        returncode: int | None,
        output: str,
        *,
        submitted: bool = False,
        decision: Decision = Decision.EXECUTE,
        rationale: str | None = None,
        count_as_evidence: bool = True,
    ) -> None:
        repetitions = sum(previous == command for previous in self._action_history)
        self._action_history.append(command)
        if output.strip() and count_as_evidence:
            self._no_new_evidence_steps = 0
            self._evidence_count += 1
        else:
            self._no_new_evidence_steps += 1
        execution = execution_uncertainty(
            exit_code=returncode,
            timeout="timed out" in output.lower(),
            repeated_command_failures=repetitions if returncode else 0,
            malformed_output=returncode is None and not submitted,
        )
        if submitted:
            execution = self._current_uncertainty.execution
        progress = progress_uncertainty(
            repeated_actions=repetitions,
            no_new_evidence_steps=self._no_new_evidence_steps,
            cycle_detected=repetitions >= 2,
        )
        retrieval = self._current_uncertainty.retrieval
        verification = self._current_uncertainty.verification
        if stage == Stage.RETRIEVAL:
            retrieval = 0.15 if returncode == 0 and output.strip() else 0.8
        memory = self._current_uncertainty.memory
        if stage == Stage.RETRIEVAL and returncode == 0 and output.strip():
            memory = min(memory, 0.15)
        if stage == Stage.VERIFICATION:
            verification = 0.1 if returncode == 0 else 0.9
            self._verification_attempts += 1
            if _is_visible_test_command(command):
                self._visible_tests_passed = returncode == 0
            if STATIC_COMMAND_RE.search(command):
                self._static_checks_passed = returncode == 0
        termination = 0.05 if submitted else self._current_uncertainty.termination
        components = {
            "task": self._current_uncertainty.task,
            "planning": self._current_uncertainty.planning,
            "retrieval": retrieval,
            "execution": execution,
            "verification": verification,
            "memory": memory,
            "progress": progress,
            "termination": termination,
        }
        self._current_uncertainty = UncertaintyVector(
            **components,
            composite=composite_uncertainty(components),
        )
        safe_command, _ = redact_text(command)
        self._append_record(
            stage=stage,
            action=safe_command or "empty command",
            decision=decision,
            rationale=rationale
            or ("submitted patch" if submitted else "executed upstream shell action"),
            evidence=[
                StepEvidence.from_text(
                    "shell_output",
                    output or "no output",
                    source="mini-swe-agent environment",
                    returncode=returncode,
                )
            ],
            cost=CostRecord(tool_calls=1, estimated_cost_usd=0.0),
            termination_state="submitted" if submitted else None,
        )

    @staticmethod
    def _classify_stage(command: str) -> Stage:
        if _is_visible_test_command(command) or STATIC_COMMAND_RE.search(command):
            return Stage.VERIFICATION
        if RETRIEVAL_COMMAND_RE.search(command):
            return Stage.RETRIEVAL
        return Stage.EXECUTION

    def _controller_state(self, stage: Stage) -> ControllerState:
        return ControllerState(
            loop_variant=self.loop_variant,
            stage=stage,
            uncertainty=self._current_uncertainty,
            step_index=self.n_calls,
            max_steps=max(1, int(self.config.step_limit or 20)),
            visible_tests_passed=self._visible_tests_passed,
            static_checks_passed=self._static_checks_passed,
            evidence_count=self._evidence_count,
            reflection_used=self._reflection_used,
            verification_attempts=self._verification_attempts,
            stagnation_steps=self._no_new_evidence_steps,
            cost_exhausted=0 < self.config.cost_limit <= self.cost,
        )

    def _control_after_action(self, stage: Stage) -> dict[str, Any] | None:
        if self.loop_variant == LoopVariant.LINEAR:
            return None
        if self.loop_variant == LoopVariant.REFLECTION:
            if self._reflection_used or stage != Stage.EXECUTION:
                return None
            controller_decision = decide(self._controller_state(stage))
            self._reflection_used = True
        else:
            controller_decision = decide(self._controller_state(stage))
        self._record_controller_decision(controller_decision, stage)
        return self.model.format_message(
            role="user",
            content=self._controller_instruction(controller_decision),
        )

    def _record_controller_decision(
        self,
        controller_decision: ControllerDecision,
        stage: Stage,
    ) -> None:
        self._append_record(
            stage=stage,
            action="controller_decision",
            decision=controller_decision.decision,
            rationale=controller_decision.rationale,
            evidence=[
                StepEvidence.from_text(
                    "controller_state",
                    (
                        f"tests_passed={self._visible_tests_passed}; "
                        f"static_checks_passed={self._static_checks_passed}; "
                        f"evidence_count={self._evidence_count}; "
                        f"verification_attempts={self._verification_attempts}"
                    ),
                    source="helloloop",
                )
            ],
            cost=CostRecord(estimated_cost_usd=0.0),
        )

    @staticmethod
    def _controller_instruction(controller_decision: ControllerDecision) -> str:
        action = {
            Decision.REPLAN: "Reflect on the current evidence and state a revised plan before acting.",
            Decision.RETRIEVE: "Retrieve only the repository context needed to reduce the identified uncertainty.",
            Decision.VERIFY_MORE: (
                "Run a targeted test or assertion-based reproduction and `git diff --check` "
                "before submitting."
            ),
            Decision.REVISE: "Revise the implementation in response to the failed or uncertain evidence.",
            Decision.STOP_SUCCESS: "Verification gates are satisfied; submit the patch when ready.",
            Decision.STOP_UNRESOLVED: "Do not claim success; report the unresolved evidence.",
            Decision.STOP_BUDGET: "Stop because the configured budget is exhausted.",
            Decision.ESCALATE: "Stop and record a transparent escalation rationale.",
            Decision.ABSTAIN: "Stop and record why abstention is appropriate.",
            Decision.EXECUTE: "Continue with the next evidence-producing action.",
        }[controller_decision.decision]
        return (
            f'<helloloop_control decision="{controller_decision.decision.value}">'
            f"{controller_decision.rationale} {action}</helloloop_control>"
        )

    def _handle_submission(
        self,
        command: str,
        stage: Stage,
        error: Any,
    ) -> dict[str, Any]:
        submission = "\n".join(_message_text(message) for message in error.messages)
        if self.loop_variant in {LoopVariant.LINEAR, LoopVariant.REFLECTION}:
            self._record_action(
                command,
                stage,
                None,
                submission,
                submitted=True,
                decision=Decision.STOP_SUCCESS,
            )
            raise error

        controller_decision = decide(self._controller_state(Stage.TERMINATION))
        if controller_decision.decision == Decision.STOP_SUCCESS:
            self._record_action(
                command,
                Stage.TERMINATION,
                None,
                submission,
                submitted=True,
                decision=Decision.STOP_SUCCESS,
                rationale=controller_decision.rationale,
            )
            raise error

        self._submission_deferrals += 1
        rationale = (
            f"submission deferred ({self._submission_deferrals}): {controller_decision.rationale}"
        )
        self._record_action(
            command,
            Stage.TERMINATION,
            0,
            rationale,
            decision=controller_decision.decision,
            rationale=rationale,
            count_as_evidence=False,
        )
        self._record_controller_decision(controller_decision, Stage.TERMINATION)
        return {
            "output": self._controller_instruction(controller_decision),
            "returncode": 0,
            "exception_info": "",
            "extra": {"helloloop_submission_deferred": True},
        }

    def _record_termination(
        self,
        result: dict[str, Any],
        *,
        infrastructure_error: bool = False,
    ) -> None:
        exit_status = str(result.get("exit_status", "Unknown"))
        success = exit_status == "Submitted" and not infrastructure_error
        decision = Decision.STOP_SUCCESS if success else Decision.STOP_UNRESOLVED
        termination = 0.05 if success else 0.9
        components = {
            "task": self._current_uncertainty.task,
            "planning": self._current_uncertainty.planning,
            "retrieval": self._current_uncertainty.retrieval,
            "execution": self._current_uncertainty.execution,
            "verification": self._current_uncertainty.verification,
            "memory": self._current_uncertainty.memory,
            "progress": self._current_uncertainty.progress,
            "termination": termination,
        }
        self._current_uncertainty = UncertaintyVector(
            **components,
            composite=composite_uncertainty(components),
        )
        self._append_record(
            stage=Stage.TERMINATION,
            action="terminate",
            decision=decision,
            rationale=f"mini-SWE-agent exit status: {exit_status}",
            evidence=[
                StepEvidence.from_text(
                    "submission",
                    str(result.get("submission", "")) or "no submission",
                    source="mini-swe-agent",
                )
            ],
            cost=CostRecord(estimated_cost_usd=0.0),
            termination_state=exit_status,
        )

    def _append_record(
        self,
        *,
        stage: Stage,
        action: str,
        decision: Decision,
        rationale: str,
        evidence: list[StepEvidence],
        cost: CostRecord,
        termination_state: str | None = None,
    ) -> None:
        safe_rationale, _ = redact_text(rationale)
        self._writer.append(
            StepRecord(
                run_id=self.run_id,
                task_id=self.task_spec.task_id,
                benchmark=self.task_spec.benchmark,
                loop_variant=self.loop_variant,
                model_name=self.model_name,
                seed=self.seed,
                repetition=self.repetition,
                step_index=self._step_index,
                stage=stage,
                action=action,
                decision=decision,
                rationale=safe_rationale,
                uncertainty=self._current_uncertainty,
                evidence=evidence,
                cost=cost,
                ended_at=utc_now_iso(),
                termination_state=termination_state,
            )
        )
        self._step_index += 1


def run_deterministic_smoke(out_dir: str | Path) -> dict[str, Any]:
    """Run the real pinned agent class against a disposable local Git repository."""
    check_upstream_version()
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.test_models import DeterministicModel, make_output

    output = Path(out_dir)
    with TemporaryDirectory(prefix="helloloop-mini-smoke-") as temp_dir:
        repo = Path(temp_dir)
        source = repo / "calculator.py"
        source.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "calculator.py"], cwd=repo, check=True)
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
                    "Fix the arithmetic implementation. Confidence: 0.95",
                    [
                        {
                            "command": (
                                "printf 'def add(a, b):\\n    return a + b\\n' > calculator.py"
                            )
                        }
                    ],
                    cost=0.01,
                ),
                make_output(
                    "The implementation is fixed and ready to submit. Confidence: 0.99",
                    [
                        {
                            "command": (
                                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff -- calculator.py"
                            )
                        }
                    ],
                    cost=0.01,
                ),
            ],
            cost_per_call=0.01,
        )
        task = TaskSpec(
            task_id="mini-swe-agent-smoke",
            benchmark="local_smoke",
            problem_statement="Fix add() so it returns the sum of its arguments.",
            repo="fixture/calculator",
        )
        agent = HelloLoopAgent(
            model=model,
            env=LocalEnvironment(cwd=str(repo), timeout=10),
            task_spec=task,
            run_id="mini-swe-agent-smoke-linear-r0",
            run_dir=output,
            system_template="You are a deterministic test agent.",
            instance_template="{{ task }}",
            step_limit=4,
            cost_limit=1.0,
        )
        result = agent.run()
        serialized = agent.serialize()
    return {
        "exit_status": result.get("exit_status"),
        "run_dir": str(output),
        "patch_path": str(output / "patch.diff"),
        "trajectory_path": str(output / "trajectory.jsonl"),
        "upstream_version": serialized["info"]["mini_version"],
    }
