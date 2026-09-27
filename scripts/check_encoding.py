#!/usr/bin/env python3
"""Fail on encoding corruption in tracked text files.

SoulMem writes comments and docs in Chinese. If an editor saves a file as
GBK/CP936 instead of UTF-8, every Chinese comment in it becomes unreadable
mojibake and part of the original bytes are lost for good. That happened to
`crates/soul-mem-runtime/src/cluster/memory_cluster.rs`, which lost 16 comment
lines -- including the only in-repo explanation of the cluster's index-reuse
invariants. Most of it was recoverable only by reversing the mangling
(`text.encode("cp936").decode("utf-8")`), and several characters were gone for
good.

Why a script and not `.gitattributes`: git stores bytes, so it cannot detect or
prevent this class of damage. `.editorconfig` sets `charset = utf-8` for editors
that honour it; this check is the enforcement that does not depend on the editor.

Four independent signals, each verified to have zero false positives on the
current tree (240 tracked text files):

  1. The file does not decode as UTF-8 at all.
  2. U+FFFD REPLACEMENT CHARACTER -- an already-corrupted decode.
  3. Private Use Area characters (U+E000-U+F8FF) -- a CP936 decoder maps the
     unmappable UTF-8 bytes there. Legitimate uses exist (icon-font glyphs in
     strings), so an allowlist escape hatch is provided.
  4. A curated set of CJK characters that UTF-8 Chinese produces when decoded as
     CP936 (e.g. U+9428 "鐨", U+93C4 "鏄", U+951B "锛"). These are vanishingly
     rare in real Chinese text, which is what makes them a usable signal.

Usage:
    python3 scripts/check_encoding.py                 # check every tracked text file
    python3 scripts/check_encoding.py path [path ...] # check specific files
    python3 scripts/check_encoding.py --allow FILE    # stop reporting PUA in FILE

Exit codes:
    0 - clean
    1 - findings reported
    2 - usage or IO error
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Text file types that carry prose or identifiers. Binary assets (models,
# databases, images) are deliberately absent.
TEXT_SUFFIXES = (
    ".rs",
    ".toml",
    ".md",
    ".yml",
    ".yaml",
    ".json",
    ".py",
    ".surql",
    ".in",
    ".txt",
)

EXTRA_NAMES = (".gitignore", ".gitattributes", ".editorconfig")

# Characters produced by decoding UTF-8 Chinese bytes as CP936.
#
# 这份签名**必须**是精选字面量，不能从码点推导：按"UTF-8 中文字节两两被 CP936 消费"
# 穷举会得到近 4000 个字符，覆盖 `不`、`到`、`本`、`存` 这类常用字，
# 结果是全仓库 201/259 个文件全部误报。只有逐一核对过、在当前代码树中零误报的字面量才可用。
MOJIBAKE_CHARS = frozenset(
    "鐨鏄锛鑺鍒閾婧娓杩娣瓨澶绱瀵缂鍥鎿妫缁閿欒闃闇闈闁闄鍑鏂鐢浣鎬鎰鎵鎯鏈璁娴婢鐩爣熷敤鍌"
)

# 按信号豁免：路径 -> (关闭的信号集合, 原因)。
#
# 本文件以数据形式内嵌了上面这份乱码签名，因此**必然**被自己的 `mojibake` 检查命中。
# 只关闭这一路：非法 UTF-8、U+FFFD、私用区三个信号对本文件仍然生效，
# 所以真正的编码损坏依然会被拦住，不会被这个豁免放过去。
SIGNAL_EXEMPTIONS: dict[str, tuple[frozenset[str], str]] = {
    "scripts/check_encoding.py": (
        frozenset({"mojibake"}),
        "内嵌乱码签名（MOJIBAKE_CHARS）必然自命中；其余信号仍然生效",
    ),
}

PUA_START, PUA_END = 0xE000, 0xF8FF


def tracked_files() -> list[Path]:
    """Every git-tracked file, so caches and build output are never scanned."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"ERROR: cannot list tracked files: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    return [Path(line) for line in result.stdout.splitlines() if line]


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in EXTRA_NAMES


def check(path: Path, allow_pua: set[str]) -> list[str]:
    """Return human-readable findings for one file (empty list means clean)."""
    # git ls-files 也会列出"已从工作区删除但尚未暂存删除"的文件。那种文件不是
    # 编码问题，直接跳过——否则重构（拆分/移动模块）期间门禁会误报。
    if not path.exists():
        return []

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"{path}: cannot read: {exc}"]

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [
            f"{path}: not valid UTF-8 at byte {exc.start} "
            f"(likely saved as GBK/CP936): {exc.reason}"
        ]

    findings: list[str] = []
    posix = path.as_posix()
    disabled, _reason = SIGNAL_EXEMPTIONS.get(posix, (frozenset(), ""))

    for lineno, line in enumerate(text.splitlines(), start=1):
        if "\ufffd" in line:
            findings.append(
                f"{path}:{lineno}: contains U+FFFD REPLACEMENT CHARACTER "
                "(text was already corrupted before being saved)"
            )
        if "mojibake" not in disabled:
            hits = sorted({c for c in line if c in MOJIBAKE_CHARS})
            if hits:
                findings.append(
                    f"{path}:{lineno}: mojibake {''.join(hits)} "
                    "(UTF-8 Chinese decoded as CP936)"
                )
        if posix not in allow_pua:
            pua = sorted({c for c in line if PUA_START <= ord(c) <= PUA_END})
            if pua:
                codes = ", ".join(f"U+{ord(c):04X}" for c in pua)
                findings.append(
                    f"{path}:{lineno}: Private Use Area characters {codes} "
                    "(CP936 mapped unmappable bytes here; add --allow to permit)"
                )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files to check; defaults to every tracked text file",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="FILE",
        help="path (as git reports it) where Private Use Area characters are expected",
    )
    args = parser.parse_args()

    allow_pua = {Path(a).as_posix() for a in args.allow}

    if args.paths:
        targets = [p for p in args.paths if is_text(p)]
    else:
        targets = [p for p in tracked_files() if is_text(p)]

    if not targets:
        print("No text files to check. PASS")
        return 0

    findings: list[str] = []
    for path in targets:
        findings.extend(check(path, allow_pua))

    if not findings:
        print(f"Encoding check PASS ({len(targets)} files).")
        return 0

    print(
        f"Encoding check FAILED: {len(findings)} finding(s) in {len(targets)} files.",
        file=sys.stderr,
    )
    for finding in findings:
        print(f"  {finding}", file=sys.stderr)
    print(
        "\nFix: the file was saved with a non-UTF-8 encoding. Recover the affected\n"
        "lines with `text.encode('cp936').decode('utf-8')`, or rewrite them from\n"
        "context. Ensure your editor is set to UTF-8 (.editorconfig asks for it).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
