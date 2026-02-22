from __future__ import annotations

import contextlib
import errno
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import venv
import zipfile

import pytest
import setuptools.build_meta as build_meta


@contextlib.contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _copy_repo_tree(src_repo: Path, dst_repo: Path) -> None:
    shutil.copytree(
        src_repo,
        dst_repo,
        ignore=shutil.ignore_patterns(
            ".git",
            ".idea",
            ".vscode",
            ".pytest_cache",
            "__pycache__",
            "*.pyc",
            ".mypy_cache",
            ".tmp_build",
            "build",
            "dist",
        ),
    )


def _build_with_exdev_fallback(repo_dir: Path, dist_dir: Path) -> tuple[Path, Path]:
    dist_dir.mkdir(parents=True, exist_ok=True)
    original_rename = os.rename

    def _safe_rename(src: str, dst: str):
        try:
            original_rename(src, dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            # Handle cross-device rename failures on some mounted filesystems.
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                shutil.rmtree(src)
            else:
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                shutil.copy2(src, dst)
                os.unlink(src)

    os.rename = _safe_rename
    try:
        with _chdir(repo_dir):
            wheel_name = build_meta.build_wheel(str(dist_dir))
            sdist_name = build_meta.build_sdist(str(dist_dir))
    finally:
        os.rename = original_rename

    wheel_path = dist_dir / wheel_name
    sdist_path = dist_dir / sdist_name
    return wheel_path, sdist_path


def _python_in_venv(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def test_pennylane_q_layer_rejects_unsupported_ansatz():
    pytest.importorskip("pennylane")
    pytest.importorskip("torq")

    from torq_bench.pennylane_backend import PennyLaneQLayer

    with pytest.raises(ValueError, match="basic_entangling"):
        PennyLaneQLayer(n_qubits=2, n_layers=1, ansatz_name="strongly_entangling")


def test_pennylane_q_layer_rejects_data_reupload():
    pytest.importorskip("pennylane")
    pytest.importorskip("torq")

    from torq.simple import CircuitConfig
    from torq_bench.pennylane_backend import PennyLaneQLayer

    with pytest.raises(ValueError, match="data_reupload_every"):
        PennyLaneQLayer(
            n_qubits=2,
            n_layers=1,
            ansatz_name="basic_entangling",
            config=CircuitConfig(data_reupload_every=1),
        )


def test_missing_pennylane_reports_install_hint(tmp_path: Path):
    script = tmp_path / "check_missing_pennylane.py"
    src_dir = Path(__file__).resolve().parents[1] / "src"
    script.write_text(
        """
import builtins
import sys

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "pennylane":
        raise ImportError("blocked by test")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import

try:
    import torq_bench.PennyLaneComparison  # noqa: F401
except ImportError as exc:
    msg = str(exc)
    print(msg)
    raise SystemExit(0 if "pip install torq-bench[pennylane]" in msg else 2)

raise SystemExit(1)
""".strip(),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, (
        f"returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_build_smoke_produces_wheel_and_sdist(tmp_path: Path):
    repo_src = Path(__file__).resolve().parents[1]
    repo_copy = tmp_path / "repo"
    _copy_repo_tree(repo_src, repo_copy)

    dist_dir = tmp_path / "dist"
    wheel_path, sdist_path = _build_with_exdev_fallback(repo_copy, dist_dir)

    assert wheel_path.exists()
    assert sdist_path.exists()
    assert wheel_path.suffix == ".whl"
    assert sdist_path.suffix == ".gz"

    with zipfile.ZipFile(wheel_path) as zf:
        metadata_file = next(name for name in zf.namelist() if name.endswith("METADATA"))
        metadata = zf.read(metadata_file).decode("utf-8")
    assert "Requires-Dist: torq-quantum>=0.1.2" in metadata


def test_built_wheel_installs_in_clean_venv(tmp_path: Path):
    repo_src = Path(__file__).resolve().parents[1]
    repo_copy = tmp_path / "repo"
    _copy_repo_tree(repo_src, repo_copy)
    pyproject_text = (repo_copy / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"\s*$', pyproject_text, flags=re.MULTILINE)
    assert match is not None
    expected_version = match.group(1)

    dist_dir = tmp_path / "dist"
    wheel_path, _ = _build_with_exdev_fallback(repo_copy, dist_dir)

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python_exe = _python_in_venv(venv_dir)
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)

    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "--no-deps", str(wheel_path)],
        check=True,
        capture_output=True,
        text=True,
        env=clean_env,
    )

    result = subprocess.run(
        [
            str(python_exe),
            "-c",
            (
                "import importlib.metadata as im; "
                "import torq_bench; "
                "print(im.version('TorQ-bench')); "
                "print(torq_bench.__version__)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=clean_env,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0] == expected_version
    assert lines[1] == expected_version
