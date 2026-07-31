from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from helloloop.schema import redact_credentials

GIB = 1024**3
REPORT_REQUIRED_KEYS = frozenset(
    {
        "total_instances",
        "submitted_instances",
        "completed_instances",
        "resolved_instances",
        "unresolved_instances",
        "empty_patch_instances",
        "error_instances",
        "completed_ids",
        "incomplete_ids",
        "empty_patch_ids",
        "submitted_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
        "schema_version",
    }
)


@dataclass(frozen=True)
class SWEbenchPrediction:
    instance_id: str
    model_patch: str
    model_name_or_path: str

    def validate(self) -> None:
        if not self.instance_id.strip():
            raise ValueError("instance_id is required")
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path is required")
        _, changed = redact_credentials(
            self.model_patch,
            max_chars=max(1, len(self.model_patch) + 1),
        )
        if changed:
            raise ValueError(f"prediction for {self.instance_id} contains secret-like material")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class DockerPreflight:
    system: str
    architecture: str
    architecture_supported: bool
    architecture_tier: str
    docker_path: str | None
    docker_server_available: bool
    docker_detail: str
    memory_gib: float
    memory_sufficient: bool
    disk_free_gib: float
    disk_sufficient: bool
    required_memory_gib: float
    required_disk_gib: float
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ready": self.ready}


@dataclass(frozen=True)
class HarnessConfig:
    dataset_name: str
    split: str
    predictions_path: str
    run_id: str
    instance_ids: tuple[str, ...] = ()
    max_workers: int = 1
    timeout_seconds: int = 1_800
    cache_level: str = "env"
    namespace: str = "swebench"
    report_dir: str = "."

    def validate(self) -> None:
        for name in ("dataset_name", "split", "predictions_path", "run_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if self.cache_level not in {"none", "base", "env", "instance"}:
            raise ValueError(f"unsupported cache_level: {self.cache_level}")


@dataclass(frozen=True)
class EvaluationOutcome:
    total_instances: int
    submitted_instances: int
    completed_instances: int
    resolved_instances: int
    unresolved_instances: int
    empty_patch_instances: int
    error_instances: int
    completed_ids: tuple[str, ...]
    incomplete_ids: tuple[str, ...]
    empty_patch_ids: tuple[str, ...]
    submitted_ids: tuple[str, ...]
    resolved_ids: tuple[str, ...]
    unresolved_ids: tuple[str, ...]
    error_ids: tuple[str, ...]
    schema_version: int

    @property
    def success_rate(self) -> float | None:
        if self.completed_instances == 0:
            return None
        return self.resolved_instances / self.completed_instances


def write_predictions_jsonl(
    predictions: list[SWEbenchPrediction],
    path: str | Path,
) -> Path:
    instance_ids = [prediction.instance_id for prediction in predictions]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("duplicate prediction instance IDs")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for prediction in sorted(predictions, key=lambda item: item.instance_id):
            handle.write(json.dumps(prediction.to_dict(), sort_keys=True) + "\n")
    return target


def read_predictions_jsonl(path: str | Path) -> list[SWEbenchPrediction]:
    predictions = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                prediction = SWEbenchPrediction(**json.loads(line))
                prediction.validate()
                predictions.append(prediction)
    if len({item.instance_id for item in predictions}) != len(predictions):
        raise ValueError("duplicate prediction instance IDs")
    return predictions


def _memory_bytes() -> int:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 0


def docker_preflight(
    path: str | Path = ".",
    *,
    required_memory_gib: float = 16.0,
    required_disk_gib: float = 120.0,
) -> DockerPreflight:
    system = platform.system()
    architecture = platform.machine().lower()
    if architecture in {"x86_64", "amd64"}:
        architecture_supported = True
        architecture_tier = "recommended"
    elif architecture in {"arm64", "aarch64"}:
        architecture_supported = True
        architecture_tier = "experimental"
    else:
        architecture_supported = False
        architecture_tier = "unsupported"
    docker_path = shutil.which("docker")
    docker_server_available = False
    docker_detail = "docker executable not found"
    if docker_path:
        try:
            result = subprocess.run(
                [docker_path, "info", "--format", "{{.ServerVersion}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            docker_server_available = result.returncode == 0
            docker_detail = (result.stdout if result.returncode == 0 else result.stderr).strip()
        except subprocess.TimeoutExpired:
            docker_detail = "docker info timed out"
    memory_gib = _memory_bytes() / GIB
    disk_free_gib = shutil.disk_usage(Path(path)).free / GIB
    memory_sufficient = memory_gib >= required_memory_gib
    disk_sufficient = disk_free_gib >= required_disk_gib
    blockers = []
    warnings = []
    if not architecture_supported:
        blockers.append(f"unsupported architecture: {architecture}")
    elif architecture_tier == "experimental":
        warnings.append("official SWE-bench ARM64 support is experimental")
    if not docker_path:
        blockers.append("docker executable unavailable")
    elif not docker_server_available:
        blockers.append("docker server unavailable")
    if not memory_sufficient:
        blockers.append(f"memory below {required_memory_gib:.0f} GiB")
    if not disk_sufficient:
        blockers.append(f"free disk below {required_disk_gib:.0f} GiB")
    return DockerPreflight(
        system=system,
        architecture=architecture,
        architecture_supported=architecture_supported,
        architecture_tier=architecture_tier,
        docker_path=docker_path,
        docker_server_available=docker_server_available,
        docker_detail=docker_detail,
        memory_gib=round(memory_gib, 2),
        memory_sufficient=memory_sufficient,
        disk_free_gib=round(disk_free_gib, 2),
        disk_sufficient=disk_sufficient,
        required_memory_gib=required_memory_gib,
        required_disk_gib=required_disk_gib,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def build_harness_command(config: HarnessConfig) -> list[str]:
    config.validate()
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        config.dataset_name,
        "--split",
        config.split,
        "--predictions_path",
        config.predictions_path,
        "--max_workers",
        str(config.max_workers),
        "--timeout",
        str(config.timeout_seconds),
        "--cache_level",
        config.cache_level,
        "--run_id",
        config.run_id,
        "--namespace",
        config.namespace,
        "--report_dir",
        config.report_dir,
    ]
    if config.instance_ids:
        command.extend(["--instance_ids", *config.instance_ids])
    return command


def parse_run_report(path: str | Path) -> EvaluationOutcome:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = REPORT_REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"official report missing fields: {sorted(missing)}")
    if int(data["schema_version"]) != 2:
        raise ValueError(f"unsupported official report schema: {data['schema_version']}")
    return EvaluationOutcome(
        total_instances=int(data["total_instances"]),
        submitted_instances=int(data["submitted_instances"]),
        completed_instances=int(data["completed_instances"]),
        resolved_instances=int(data["resolved_instances"]),
        unresolved_instances=int(data["unresolved_instances"]),
        empty_patch_instances=int(data["empty_patch_instances"]),
        error_instances=int(data["error_instances"]),
        completed_ids=tuple(data["completed_ids"]),
        incomplete_ids=tuple(data["incomplete_ids"]),
        empty_patch_ids=tuple(data["empty_patch_ids"]),
        submitted_ids=tuple(data["submitted_ids"]),
        resolved_ids=tuple(data["resolved_ids"]),
        unresolved_ids=tuple(data["unresolved_ids"]),
        error_ids=tuple(data["error_ids"]),
        schema_version=int(data["schema_version"]),
    )


def parse_instance_report(path: str | Path, instance_id: str) -> bool:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return bool(data[instance_id]["resolved"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid instance report for {instance_id}") from error


def classify_evaluator_log(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    markers = {
        "patch_apply_failed": ">>>>> Patch Apply Failed" in text,
        "initialization_failed": ">>>>> Init Failed" in text,
        "initialization_timeout": ">>>>> Init Timed Out" in text,
        "tests_errored": ">>>>> Tests Errored" in text,
        "tests_failed": ">>>>> Some Tests Failed" in text,
        "tests_passed": ">>>>> All Tests Passed" in text,
        "tests_timeout": ">>>>> Tests Timed Out" in text or "Timeout error:" in text,
    }
    infrastructure_failure = any(
        markers[name]
        for name in ("initialization_failed", "initialization_timeout", "tests_errored")
    )
    agent_failure = markers["patch_apply_failed"] or markers["tests_failed"]
    return {
        **markers,
        "infrastructure_failure": infrastructure_failure,
        "agent_failure": agent_failure,
    }


def evaluation_complete(report_path: str | Path, instance_id: str) -> bool:
    path = Path(report_path)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if "completed_ids" in data:
        return instance_id in data["completed_ids"]
    return instance_id in data and isinstance(data[instance_id].get("resolved"), bool)


def copy_evaluator_artifacts(
    *,
    harness_root: str | Path,
    run_id: str,
    model_name: str,
    instance_id: str,
    destination: str | Path,
) -> list[str]:
    source = (
        Path(harness_root)
        / "logs"
        / "run_evaluation"
        / run_id
        / model_name.replace("/", "__")
        / instance_id
    )
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for filename in ("report.json", "test_output.txt", "run_instance.log", "patch.diff"):
        candidate = source / filename
        if candidate.exists():
            shutil.copy2(candidate, target / filename)
            copied.append(filename)
    return copied


def prepare_gold_smoke_plan(
    *,
    out_dir: str | Path,
    instance_id: str,
    dataset_name: str = "SWE-bench/SWE-bench_Lite",
    split: str = "test",
    harness_root: str | Path = "third_party/SWE-bench",
) -> dict[str, Any]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    preflight = docker_preflight(harness_root)
    config = HarnessConfig(
        dataset_name=dataset_name,
        split=split,
        predictions_path="gold",
        run_id="helloloop-gold-smoke",
        instance_ids=(instance_id,),
        max_workers=1,
        report_dir=str(output.resolve()),
    )
    command = build_harness_command(config)
    (output / "preflight.json").write_text(
        json.dumps(preflight.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "command.json").write_text(
        json.dumps(
            {
                "cwd": str(Path(harness_root).resolve()),
                "argv": command,
                "official_evaluator": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    status = "ready" if preflight.ready else "blocked"
    if not preflight.ready:
        (output / "BLOCKED.txt").write_text(
            "Official gold smoke was not executed.\n"
            + "\n".join(f"- {blocker}" for blocker in preflight.blockers)
            + "\n",
            encoding="utf-8",
        )
    return {
        "status": status,
        "instance_id": instance_id,
        "preflight": preflight.to_dict(),
        "command_path": str(output / "command.json"),
    }


def run_official_evaluation(
    config: HarnessConfig,
    *,
    harness_root: str | Path,
    output_dir: str | Path,
    required_memory_gib: float = 16.0,
    required_disk_gib: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    preflight = docker_preflight(
        harness_root,
        required_memory_gib=required_memory_gib,
        required_disk_gib=required_disk_gib,
    )
    if not preflight.ready:
        raise RuntimeError("official evaluator preflight failed: " + "; ".join(preflight.blockers))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = build_harness_command(config)
    result = subprocess.run(
        command,
        cwd=Path(harness_root),
        check=False,
        capture_output=True,
        text=True,
    )
    (output / "evaluator_stdout.log").write_text(result.stdout, encoding="utf-8")
    (output / "evaluator_stderr.log").write_text(result.stderr, encoding="utf-8")
    (output / "evaluator_exit_code.txt").write_text(f"{result.returncode}\n", encoding="utf-8")
    return result
