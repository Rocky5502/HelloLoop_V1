from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .enums import Decision, LoopVariant, Stage

SCHEMA_ID = "helloloop.step.v1"
SUMMARY_SCHEMA_ID = "helloloop.summary.v1"


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s]+"),
]
CREDENTIAL_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"gh[oprsu]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def redact_text(value: str, max_chars: int = 8000) -> tuple[str, bool]:
    redacted = value
    changed = False
    for pattern in SECRET_PATTERNS:
        redacted, count = pattern.subn("[REDACTED_SECRET]", redacted)
        changed = changed or count > 0
    if len(redacted) > max_chars:
        redacted = redacted[:max_chars] + "[TRUNCATED]"
        changed = True
    return redacted, changed


def redact_credentials(value: str, max_chars: int = 8000) -> tuple[str, bool]:
    redacted = value
    changed = False
    for pattern in CREDENTIAL_PATTERNS:
        redacted, count = pattern.subn("[REDACTED_SECRET]", redacted)
        changed = changed or count > 0
    if len(redacted) > max_chars:
        redacted = redacted[:max_chars] + "[TRUNCATED]"
        changed = True
    return redacted, changed


def _validate_id(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} is required")


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    benchmark: str
    problem_statement: str
    repo: str | None = None
    base_commit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_id("task_id", self.task_id)
        _validate_id("benchmark", self.benchmark)
        _validate_id("problem_statement", self.problem_statement)


@dataclass(frozen=True)
class StepEvidence:
    kind: str
    content: str
    source: str | None = None
    confidence: float | None = None
    redacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_id("kind", self.kind)
        _validate_id("content", self.content)
        if self.confidence is not None:
            _validate_unit_interval("confidence", self.confidence)

    @classmethod
    def from_text(
        cls,
        kind: str,
        content: str,
        source: str | None = None,
        max_chars: int = 8000,
        **metadata: Any,
    ) -> StepEvidence:
        safe_content, redacted = redact_text(content, max_chars=max_chars)
        return cls(
            kind=kind, content=safe_content, source=source, redacted=redacted, metadata=metadata
        )


@dataclass(frozen=True)
class UncertaintyVector:
    task: float = 0.5
    planning: float = 0.5
    retrieval: float = 0.5
    execution: float = 0.5
    verification: float = 0.5
    memory: float = 0.5
    progress: float = 0.5
    termination: float = 0.5
    composite: float = 0.5

    def validate(self) -> None:
        for key, value in asdict(self).items():
            _validate_unit_interval(key, float(value))


@dataclass(frozen=True)
class CostRecord:
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    estimated_cost_usd: float | None = None

    def validate(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0 or self.tool_calls < 0:
            raise ValueError("token and tool-call counts must be non-negative")
        if self.estimated_cost_usd is not None and self.estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd must be non-negative")


@dataclass(frozen=True)
class EscalationRecord:
    reason: str
    unresolved_issue: str
    required_human_decision: str
    consequences: str

    def validate(self) -> None:
        _validate_id("reason", self.reason)
        _validate_id("unresolved_issue", self.unresolved_issue)
        _validate_id("required_human_decision", self.required_human_decision)
        _validate_id("consequences", self.consequences)


@dataclass(frozen=True)
class EthicsOutcomeRecord:
    false_assurance: bool = False
    unsafe_persistence: bool = False
    premature_stop: bool = False
    evidence_free_continuation: bool = False
    opaque_escalation: bool = False
    inappropriate_abstention: bool = False
    computational_overuse: bool = False


@dataclass(frozen=True)
class StepRecord:
    run_id: str
    task_id: str
    benchmark: str
    loop_variant: LoopVariant
    model_name: str
    seed: int
    repetition: int
    step_index: int
    stage: Stage
    action: str
    decision: Decision
    rationale: str
    uncertainty: UncertaintyVector
    evidence: list[StepEvidence] = field(default_factory=list)
    cost: CostRecord = field(default_factory=CostRecord)
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None
    termination_state: str | None = None
    schema_id: str = SCHEMA_ID

    def validate(self) -> None:
        for name in ("run_id", "task_id", "benchmark", "model_name", "action", "rationale"):
            _validate_id(name, str(getattr(self, name)))
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        self.uncertainty.validate()
        self.cost.validate()
        for item in self.evidence:
            item.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRecord:
        uncertainty = UncertaintyVector(**data["uncertainty"])
        evidence = [StepEvidence(**item) for item in data.get("evidence", [])]
        cost = CostRecord(**data.get("cost", {}))
        record = cls(
            run_id=data["run_id"],
            task_id=data["task_id"],
            benchmark=data["benchmark"],
            loop_variant=LoopVariant(data["loop_variant"]),
            model_name=data["model_name"],
            seed=int(data["seed"]),
            repetition=int(data["repetition"]),
            step_index=int(data["step_index"]),
            stage=Stage(data["stage"]),
            action=data["action"],
            decision=Decision(data["decision"]),
            rationale=data["rationale"],
            uncertainty=uncertainty,
            evidence=evidence,
            cost=cost,
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            termination_state=data.get("termination_state"),
            schema_id=data.get("schema_id", SCHEMA_ID),
        )
        record.validate()
        return record


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    task: TaskSpec
    loop_variant: LoopVariant
    model_name: str
    seed: int
    repetition: int
    status: str = "created"

    def validate(self) -> None:
        _validate_id("run_id", self.run_id)
        _validate_id("model_name", self.model_name)
        self.task.validate()


@dataclass(frozen=True)
class SummaryRecord:
    run_id: str
    benchmark: str
    task_id: str
    loop_variant: str
    model_name: str
    repetition: int
    seed: int
    status: str
    final_decision: str
    steps: int
    total_input_tokens: int
    total_output_tokens: int
    total_tool_calls: int
    estimated_cost_usd: float | None
    residual_uncertainty: float
    trajectory_path: str
    schema_id: str = SUMMARY_SCHEMA_ID

    def validate(self) -> None:
        _validate_id("run_id", self.run_id)
        _validate_id("benchmark", self.benchmark)
        _validate_id("task_id", self.task_id)
        _validate_unit_interval("residual_uncertainty", self.residual_uncertainty)


class JsonlTrajectoryWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: StepRecord) -> None:
        record.validate()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")


def read_trajectory(path: str | Path) -> list[StepRecord]:
    records: list[StepRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(StepRecord.from_dict(json.loads(line)))
    return records


def summarize_trajectory(
    path: str | Path,
    status: str = "complete",
    trajectory_path: str | None = None,
) -> SummaryRecord:
    records = read_trajectory(path)
    if not records:
        raise ValueError("cannot summarize empty trajectory")
    last = records[-1]
    total_input = sum(record.cost.input_tokens for record in records)
    total_output = sum(record.cost.output_tokens for record in records)
    total_tools = sum(record.cost.tool_calls for record in records)
    known_costs = [record.cost.estimated_cost_usd for record in records]
    total_cost = None
    if all(cost is not None for cost in known_costs):
        total_cost = float(sum(cost for cost in known_costs if cost is not None))
    summary = SummaryRecord(
        run_id=last.run_id,
        benchmark=last.benchmark,
        task_id=last.task_id,
        loop_variant=str(last.loop_variant),
        model_name=last.model_name,
        repetition=last.repetition,
        seed=last.seed,
        status=status,
        final_decision=str(last.decision),
        steps=len(records),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tool_calls=total_tools,
        estimated_cost_usd=total_cost,
        residual_uncertainty=last.uncertainty.composite,
        trajectory_path=trajectory_path or str(path),
    )
    summary.validate()
    return summary


def write_summary_csv(summaries: list[SummaryRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(asdict(summaries[0]).keys()) if summaries else list(SummaryRecord.__dataclass_fields__)
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            summary.validate()
            writer.writerow(asdict(summary))


def write_summary_json(summary: SummaryRecord, path: str | Path) -> None:
    summary.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
