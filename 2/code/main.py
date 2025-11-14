"""Runner script to execute all part_*.py scripts for the assignment.

This script discovers a fixed list of modules (part_1_1 .. part_2_4)
and runs them as separate Python processes. Each script is executed with the
working directory set to the parent `2` folder so relative paths inside the
scripts continue to work (they expect to find `data/` relative to that folder).

Features:
 - run all scripts in sequence
 - run a subset by name
 - list available scripts
 - continue on error and summarize results

Usage examples (from repository root):
  python -m code.main --all
  python -m code.main part_1_1 part_2_3
  python -m code.main --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


HERE = Path(__file__).resolve().parent  # c:/.../2/code
PROJECT_ROOT = HERE.parent  # c:/.../2

# ordered list of scripts to run (relative to `code` folder)
SCRIPTS: List[str] = [
	"part_1_1.py",
	"part_1_2.py",
	"part_1_3.py",
	"part_2_1.py",
	"part_2_2/part_2_2.py",
	"part_2_3.py",
	"part_2_4.py",
]


def run_script(script_path: Path) -> tuple[int, str, str]:
	"""Run a script as a separate Python process.

	Returns (returncode, stdout, stderr).
	The working directory is PROJECT_ROOT (so scripts with relative paths work).
	"""
	cmd = [sys.executable, str(script_path)]
	proc = subprocess.run(
		cmd,
		cwd=PROJECT_ROOT,
		capture_output=True,
		text=True,
		check=False,
	)
	return proc.returncode, proc.stdout, proc.stderr


def main(argv: List[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Run part_* scripts for the assignment")
	parser.add_argument("scripts", nargs="*", help="scripts to run (by filename, e.g. part_1_1). If omitted and --all not provided, nothing runs.")
	parser.add_argument("--all", action="store_true", help="run all known scripts in order")
	parser.add_argument("--list", action="store_true", help="list available scripts and exit")
	args = parser.parse_args(argv)

	if args.list:
		print("Available scripts:")
		for s in SCRIPTS:
			print(" -", s)
		return 0

	to_run: List[Path]
	if args.all:
		to_run = [HERE / s for s in SCRIPTS]
	elif args.scripts:
		# allow user to pass either bare names (part_1_1) or file paths
		picked: List[Path] = []
		for name in args.scripts:
			# try exact match first
			candidate = HERE / (name if name.endswith('.py') else name + '.py')
			if candidate.exists():
				picked.append(candidate)
				continue
			# try subpath lookup
			matches = [HERE / s for s in SCRIPTS if s.endswith(name) or s.endswith(name + '.py')]
			if matches:
				picked.extend(matches)
				continue
			print(f"Warning: could not find script '{name}', skipping.")
		to_run = picked
	else:
		print("No scripts selected. Use --all or pass script names, or --list to see available scripts.")
		return 2

	results = []
	for script in to_run:
		print(f"\n=== Running {script.relative_to(HERE)} ===")
		if not script.exists():
			print(f"  Skipped: {script} not found")
			results.append((str(script), -1, "", "not found"))
			continue
		code, out, err = run_script(script)
		print(f"  Exit code: {code}")
		if out:
			print("--- stdout ---")
			print(out.strip())
		if err:
			print("--- stderr ---")
			print(err.strip())
		results.append((str(script), code, out, err))

	# summary
	print("\n=== Summary ===")
	ok = [r for r in results if r[1] == 0]
	failed = [r for r in results if r[1] != 0]
	print(f"Ran {len(results)} scripts: {len(ok)} succeeded, {len(failed)} failed")
	if failed:
		print("Failed scripts:")
		for s, code, _, err in failed:
			print(f" - {s}: exit {code}")
	return 0 if len(failed) == 0 else 3


if __name__ == "__main__":
	raise SystemExit(main())

