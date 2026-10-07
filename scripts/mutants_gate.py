#!/usr/bin/env python3
"""Gate cargo-mutants results on a minimum viable kill rate.

Kill rate = caught / (caught + missed + timeout). Unviable mutants are
excluded because they cannot be killed by any test.

`--cargo-mutants-exit-code` 接收 `cargo mutants` 自己的退出码，用来把「结果」
与「工具故障」分开（cargo-mutants 27.1.0 的取值）：

  0            全部可杀灭变异体都被杀灭
  2            存在 MISSED
  3            存在 TIMEOUT
  1/4/5/6/70   用法错误 / 基线失败 / --in-diff 不匹配 / diff 非法 / 内部错误

只有 0/2/3 允许走到杀灭率判定；其余一律按工具故障硬失败（exit 2），因为此时
mutants.out 可能缺失或残缺。反过来，退出码 0 且报告缺失或为空时判 PASS：
那是「本次没有变异体可测」（例如 PR 只往 .rs 里加文档注释），不是故障。

Exit codes:
  0 - kill rate meets the threshold, or no viable mutants were generated
  1 - kill rate is below the threshold
  2 - cargo-mutants reported a tool failure, or its output is missing or malformed
"""

import argparse
import json
import sys
from pathlib import Path

# cargo-mutants 的「结果」类退出码；其余取值都意味着 mutants.out 不可信。
RESULT_EXIT_CODES = (0, 2, 3)


def load_summary(out_dir: Path):
    outcomes = out_dir / "outcomes.json"
    mutants = out_dir / "mutants.json"

    if outcomes.exists():
        data = json.loads(outcomes.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
        raise ValueError("outcomes.json has an unexpected shape")

    if mutants.exists():
        data = json.loads(mutants.read_text(encoding="utf-8"))
        if data == []:
            return None  # no mutants generated (e.g. doc-only PR)
        raise ValueError("mutants.json exists but outcomes.json is missing")

    raise FileNotFoundError("neither outcomes.json nor mutants.json was found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("mutants.out"))
    parser.add_argument("--threshold", type=float, default=90.0)
    parser.add_argument(
        "--cargo-mutants-exit-code",
        type=int,
        default=None,
        help=(
            "cargo mutants 自己的退出码；给出后，0/2/3 之外的取值直接判为工具故障"
            "（默认不检查，只按报告判定）"
        ),
    )
    args = parser.parse_args()

    code = args.cargo_mutants_exit_code
    if code is not None and code not in RESULT_EXIT_CODES:
        print(
            f"ERROR: cargo-mutants exited with {code}, which is not a mutation result "
            f"(expected one of {RESULT_EXIT_CODES}); treating it as a tool failure",
            file=sys.stderr,
        )
        return 2

    try:
        summary = load_summary(args.out_dir)
    except FileNotFoundError as exc:
        # 没有变异体可测时 cargo-mutants 既不创建 mutants.out/ 也返回 0。
        if code == 0:
            print("cargo-mutants reported no mutants to test (exit 0); nothing to gate. PASS")
            return 0
        print(f"ERROR: cannot read mutants results: {exc}", file=sys.stderr)
        return 2
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read mutants results: {exc}", file=sys.stderr)
        return 2

    if summary is None:
        print("No mutants generated; nothing to gate. PASS")
        return 0

    total = int(summary.get("total_mutants", 0))
    caught = int(summary.get("caught", 0))
    missed = int(summary.get("missed", 0))
    timeout = int(summary.get("timeout", 0))
    unviable = int(summary.get("unviable", 0))

    viable = caught + missed + timeout
    if viable == 0:
        print(f"No viable mutants (total={total}, unviable={unviable}). PASS")
        return 0

    rate = caught / viable * 100.0
    print(
        f"Mutants: total={total} caught={caught} missed={missed} "
        f"timeout={timeout} unviable={unviable} viable={viable} "
        f"kill_rate={rate:.1f}%"
    )
    if rate + 1e-9 >= args.threshold:
        print(f"Kill rate {rate:.1f}% >= {args.threshold:g}%. PASS")
        return 0

    print(f"Kill rate {rate:.1f}% < {args.threshold:g}%. FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
