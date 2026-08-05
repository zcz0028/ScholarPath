from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_FILES = (
    "predictions_top20.jsonl",
    "predictions_top50.jsonl",
    "predictions_top100.jsonl",
    "run_summary.json",
)


def get_project_root() -> Path:
    """
    获取 ScholarPath 项目根目录。

    当前脚本位于：
        ScholarPath/scripts/freeze_week1_baselines.py

    因此 scripts 的上一级目录就是项目根目录。
    """
    return Path(__file__).resolve().parents[1]


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    分块计算文件的 SHA256，避免一次性读取大型预测文件。
    """
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def get_modified_time(file_path: Path) -> str:
    """
    获取文件最后修改时间，输出带时区的 ISO 格式。
    """
    timestamp = file_path.stat().st_mtime

    return (
        datetime.fromtimestamp(timestamp)
        .astimezone()
        .isoformat(timespec="seconds")
    )


def inspect_jsonl(
    file_path: Path,
    expected_query_count: int | None,
) -> dict[str, Any]:
    """
    检查 JSONL 文件。

    检查内容：
    1. 每一条非空行是否为合法 JSON；
    2. 记录总数是否为预期的 50；
    3. 是否存在缺少 qid 的记录；
    4. 是否存在重复 qid。
    """
    record_count = 0
    malformed_lines: list[int] = []
    missing_qid_lines: list[int] = []
    qids: list[str] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            record_count += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines.append(line_number)
                continue

            if not isinstance(item, dict):
                missing_qid_lines.append(line_number)
                continue

            qid = item.get("qid")

            if qid is None or str(qid).strip() == "":
                missing_qid_lines.append(line_number)
            else:
                qids.append(str(qid))

    qid_counter = Counter(qids)

    duplicate_qids = sorted(
        qid
        for qid, count in qid_counter.items()
        if count > 1
    )

    expected_count_ok = (
        expected_query_count is None
        or record_count == expected_query_count
    )

    valid = (
        not malformed_lines
        and not missing_qid_lines
        and not duplicate_qids
        and expected_count_ok
    )

    return {
        "format": "jsonl",
        "record_count": record_count,
        "expected_record_count": expected_query_count,
        "expected_record_count_ok": expected_count_ok,
        "qid_count": len(qids),
        "unique_qid_count": len(set(qids)),
        "duplicate_qids": duplicate_qids,
        "missing_qid_lines": missing_qid_lines,
        "malformed_lines": malformed_lines,
        "valid": valid,
    }


def inspect_json(file_path: Path) -> dict[str, Any]:
    """
    检查普通 JSON 文件是否可以正常解析。
    """
    try:
        with file_path.open("r", encoding="utf-8") as file:
            value = json.load(file)

    except json.JSONDecodeError as error:
        return {
            "format": "json",
            "json_type": None,
            "parse_error": str(error),
            "valid": False,
        }

    return {
        "format": "json",
        "json_type": type(value).__name__,
        "parse_error": None,
        "valid": True,
    }


def inspect_file(
    file_path: Path,
    project_root: Path,
    expected_query_count: int | None,
) -> dict[str, Any]:
    """
    检查单个基线文件，并返回文件元数据。
    """
    relative_path = file_path.relative_to(project_root).as_posix()

    if not file_path.is_file():
        return {
            "path": relative_path,
            "exists": False,
            "valid": False,
            "error": "file_not_found",
        }

    if file_path.suffix.lower() == ".jsonl":
        content_information = inspect_jsonl(
            file_path=file_path,
            expected_query_count=expected_query_count,
        )

    elif file_path.suffix.lower() == ".json":
        content_information = inspect_json(file_path)

    else:
        content_information = {
            "format": file_path.suffix.lstrip("."),
            "valid": True,
        }

    return {
        "path": relative_path,
        "exists": True,
        "size_bytes": file_path.stat().st_size,
        "modified_at": get_modified_time(file_path),
        "sha256": calculate_sha256(file_path),
        **content_information,
    }


def load_config(config_path: Path) -> dict[str, Any]:
    """
    读取冻结配置文件。
    """
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Baseline config not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    baselines = config.get("baselines")

    if not isinstance(baselines, list) or not baselines:
        raise ValueError(
            "Config field 'baselines' must be a non-empty list."
        )

    for index, baseline in enumerate(baselines):
        if not isinstance(baseline, dict):
            raise ValueError(
                f"Baseline at index {index} must be an object."
            )

        if "name" not in baseline or "path" not in baseline:
            raise ValueError(
                f"Baseline at index {index} must contain "
                "'name' and 'path'."
            )

    return config


def get_git_commit(project_root: Path) -> str | None:
    """
    获取当前 Git commit。

    如果项目尚未初始化 Git，则返回 None，不影响脚本运行。
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )

    except (OSError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()

    return commit or None


def build_manifest(
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    根据配置检查所有第一周基线，并构建 manifest。
    """
    expected_query_count = config.get(
        "expected_query_count",
        50,
    )

    required_files = tuple(
        config.get(
            "required_files",
            DEFAULT_REQUIRED_FILES,
        )
    )

    baseline_entries: list[dict[str, Any]] = []
    missing_files: list[str] = []
    invalid_files: list[str] = []

    resolved_project_root = project_root.resolve()

    for baseline in config["baselines"]:
        baseline_name = str(baseline["name"])
        baseline_role = str(
            baseline.get("role", "baseline")
        )

        relative_directory = Path(str(baseline["path"]))
        baseline_directory = (
            resolved_project_root / relative_directory
        ).resolve()

        # 防止配置路径跳出项目目录。
        if not baseline_directory.is_relative_to(
            resolved_project_root
        ):
            raise ValueError(
                "Baseline path escapes project root: "
                f"{relative_directory}"
            )

        baseline_required_files = tuple(
            baseline.get(
                "required_files",
                required_files,
            )
        )

        file_entries: list[dict[str, Any]] = []

        for file_name in baseline_required_files:
            file_path = baseline_directory / str(file_name)

            entry = inspect_file(
                file_path=file_path,
                project_root=resolved_project_root,
                expected_query_count=expected_query_count,
            )

            file_entries.append(entry)

            if not entry["exists"]:
                missing_files.append(entry["path"])

            elif not entry["valid"]:
                invalid_files.append(entry["path"])

        baseline_entries.append(
            {
                "name": baseline_name,
                "role": baseline_role,
                "path": relative_directory.as_posix(),
                "files": file_entries,
                "valid": all(
                    entry["valid"]
                    for entry in file_entries
                ),
            }
        )

    file_count = sum(
        len(baseline["files"])
        for baseline in baseline_entries
    )

    return {
        "schema_version": 1,
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "project_root": resolved_project_root.as_posix(),
        "git_commit": get_git_commit(resolved_project_root),
        "expected_query_count": expected_query_count,
        "required_files": list(required_files),
        "baselines": baseline_entries,
        "validation": {
            "all_valid": (
                not missing_files
                and not invalid_files
            ),
            "baseline_count": len(baseline_entries),
            "file_count": file_count,
            "missing_files": missing_files,
            "invalid_files": invalid_files,
        },
    }


def write_json_atomic(
    output_path: Path,
    value: dict[str, Any],
) -> None:
    """
    原子写入 JSON。

    先写入临时文件，再替换正式文件，避免脚本中途中断导致
    baseline_manifest.json 只写入一半。
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(output_path)


def verify_manifest(
    project_root: Path,
    manifest_path: Path,
) -> int:
    """
    使用已经生成的 manifest 验证基线文件是否被修改。

    返回值：
        0：没有变化；
        1：存在丢失或发生变化的文件。
    """
    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    missing_files: list[str] = []
    changed_files: list[str] = []

    for baseline in manifest.get("baselines", []):
        for file_entry in baseline.get("files", []):
            relative_path = str(file_entry["path"])
            current_path = project_root / relative_path

            if not current_path.is_file():
                missing_files.append(relative_path)
                continue

            current_sha256 = calculate_sha256(current_path)
            frozen_sha256 = file_entry.get("sha256")

            if current_sha256 != frozen_sha256:
                changed_files.append(relative_path)

    print(f"Manifest: {manifest_path}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Changed files: {len(changed_files)}")

    for file_path in missing_files:
        print(f"  [MISSING] {file_path}")

    for file_path in changed_files:
        print(f"  [CHANGED] {file_path}")

    if missing_files or changed_files:
        return 1

    print("All frozen baseline files match the manifest.")

    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or verify ScholarPath Week-1 "
            "baseline outputs."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/week2/frozen_baselines.json",
        help=(
            "Baseline config path relative to "
            "the project root."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/week2_day1_diagnostics/"
            "baseline_manifest.json"
        ),
        help=(
            "Manifest output path relative to "
            "the project root."
        ),
    )

    parser.add_argument(
        "--verify",
        metavar="MANIFEST",
        help=(
            "Verify current files against an existing "
            "manifest instead of generating a new one."
        ),
    )

    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help=(
            "Write the manifest and exit successfully "
            "even when files are missing or invalid."
        ),
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    project_root = get_project_root()

    # 验证模式：检查文件是否相较于冻结时发生变化。
    if arguments.verify:
        manifest_path = (
            project_root / arguments.verify
        ).resolve()

        if not manifest_path.is_file():
            print(
                f"Manifest not found: {manifest_path}",
                file=sys.stderr,
            )
            return 2

        return verify_manifest(
            project_root=project_root,
            manifest_path=manifest_path,
        )

    config_path = (
        project_root / arguments.config
    ).resolve()

    output_path = (
        project_root / arguments.output
    ).resolve()

    try:
        config = load_config(config_path)

        manifest = build_manifest(
            project_root=project_root,
            config=config,
        )

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        return 2

    write_json_atomic(
        output_path=output_path,
        value=manifest,
    )

    validation = manifest["validation"]

    print(f"Manifest written to: {output_path}")
    print(
        f"Baselines checked: "
        f"{validation['baseline_count']}"
    )
    print(
        f"Files checked: "
        f"{validation['file_count']}"
    )
    print(
        f"Missing files: "
        f"{len(validation['missing_files'])}"
    )
    print(
        f"Invalid files: "
        f"{len(validation['invalid_files'])}"
    )
    print(
        f"All valid: "
        f"{validation['all_valid']}"
    )

    if (
        not validation["all_valid"]
        and not arguments.allow_invalid
    ):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())