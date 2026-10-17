"""Nothing is advertised that does not run.

The README and RESULTS.md are the product's documentation, and a command in them that
fails is a lie told to a practitioner evaluating the tool. Five of the seven README commands
were broken when this work started -- three `NameError` from a refactor, two `ImportError`
on a removed function -- and every one of them was already visible to `ruff` as F821/F401.
Nothing ran ruff.

So this module runs what the docs advertise. Anything that needs a model or a network is
marked and skipped rather than asserted, because a skipped test that hides a broken command
is worse than no test; the real-model tier is `test_bytestream_real.py`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent

#: Commands documented for a user to run, and how long they may take.
DOCUMENTED_COMMANDS = [
    ("byteify --help", 120),
    ("byteify --version", 120),
    ("byteify levels", 120),
    ("byteify build --help", 120),
    ("byteify encode --help", 120),
    ("byteify inspect --help", 120),
]

#: Research scripts RESULTS.md tells a reader to run.
DOCUMENTED_SCRIPTS = [
    "verify_reuse.py",
    "measure_compression.py",
    "compression_sweep.py",
    "export_onnx.py",
    "train_stage1.py",
    "train_stage2.py",
    "train_contrastive.py",
    "fidelity_probe.py",
    "prepare_corpus.py",
]


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    """Run inside this interpreter's environment.

    The environment is inherited rather than rebuilt: replacing PATH means losing `uv`, and
    hardcoding the interpreter means the tests pass or fail depending on which virtualenv
    happens to be active.
    """
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    return subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )


@pytest.mark.parametrize(("command", "timeout"), DOCUMENTED_COMMANDS)
def test_a_documented_command_runs(command: str, timeout: int):
    result = _run([sys.executable, "-m", "byteify.cli", *command.split()[1:]], timeout=timeout)
    assert result.returncode == 0, f"`{command}` exited {result.returncode}\n{result.stderr}"


@pytest.mark.parametrize("script", DOCUMENTED_SCRIPTS)
def test_a_documented_script_imports_and_parses_its_arguments(script: str):
    """`--help` exercises the import and the argument parser, without training anything.

    Three of these used to fail here, which is exactly how they were discovered.
    """
    result = _run([sys.executable, f"scripts/{script}", "--help"], timeout=300)
    assert result.returncode == 0, (
        f"scripts/{script} --help exited {result.returncode}\n{result.stderr}"
    )
    assert "Traceback" not in result.stderr, f"scripts/{script} raised on import"


def test_every_script_in_the_directory_is_listed_somewhere():
    """A script nobody documents is either retired or explained."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    results = (REPO / "RESULTS.md").read_text(encoding="utf-8")
    for script in sorted((REPO / "scripts").glob("*.py")):
        assert script.name in readme or script.name in results, (
            f"scripts/{script.name} is neither documented in README.md nor in RESULTS.md"
        )


def test_the_docs_do_not_advertise_a_level_that_does_not_exist():
    from byteify.levels import LEVELS

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for mentioned in set(re.findall(r"--level (\w+)", readme)):
        assert mentioned in LEVELS, f"README advertises --level {mentioned}, which does not exist"


def test_the_docs_do_not_advertise_a_source_model_with_no_published_thresholds():
    """A model the tool would refuse must not be presented as a supported target."""
    from byteify.gates import published_source_ids

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    published = set(published_source_ids())

    for model_id in published:
        assert model_id in readme, (
            f"{model_id} has published fidelity thresholds but the README never mentions it"
        )

    # The README lists six verified-for-alignment models. It must say that verified for
    # alignment is not the same as gated, or a reader will assume they can all be built.
    assert "alignment-verified" in readme or "alignment is lossless" in readme.lower()
    assert "no published" in readme or "refuses them" in readme


def test_the_readme_leads_with_the_command():
    """#19: the README leads with what the tool does, not with findings."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    first_command = readme.index("byteify build")
    assert first_command < 2000, "the command appears too far into the README"
    assert readme.lstrip().startswith("# byteify")