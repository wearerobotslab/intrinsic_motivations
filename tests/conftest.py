"""Shared fixtures: load each homeokinesis implementation in isolation.

Both Python ports live in files called homeokinesis.py, so they are loaded by
path under distinct module names. The torch port imports Isaac Lab and
gymnasium at module level although it does not use them; those are stubbed
when they are not installed so the controller can be tested on its own.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def numpy_impl():
    return _load("hk_numpy", ROOT / "python/homeokinesis/numpy/homeokinesis.py")


def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules.setdefault(name, mod)


@pytest.fixture(scope="session")
def torch_impl():
    pytest.importorskip("torch")
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        _stub_module("gymnasium", Env=object, Wrapper=object, error=None, logger=None)
    try:
        import omni.isaac.lab.envs  # noqa: F401
    except Exception:
        for n in ("omni", "omni.isaac", "omni.isaac.lab"):
            _stub_module(n)
        _stub_module("omni.isaac.lab.envs", ManagerBasedRLEnv=object)
    return _load("hk_torch", ROOT / "python/homeokinesis/torch/homeokinesis.py")


def _find_eigen():
    candidates = [os.environ.get("EIGEN_INCLUDE_DIR"), "/usr/include/eigen3", "/usr/local/include/eigen3"]
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        candidates.append(os.path.join(conda, "include", "eigen3"))
    home = Path.home()
    candidates += [str(p) for p in home.glob("*conda*/envs/*/include/eigen3")]
    candidates += [str(p) for p in home.glob("*conda*/include/eigen3")]
    for c in candidates:
        if c and (Path(c) / "Eigen" / "Core").exists():
            return Path(c)
    return None


LPZ_SOURCES = ["controller/sox.cpp", "controller/abstractcontroller.cpp", "utils/configurable.cpp",
               "utils/inspectable.cpp", "utils/storeable.cpp", "utils/controller_misc.cpp",
               "utils/parametrizable.cpp", "utils/stl_adds.cpp", "utils/plotoption.cpp",
               "utils/randomgenerator.cpp", "utils/backcaller.cpp", "matrix/matrix.cpp"]


@pytest.fixture(scope="session")
def lpz_sox_driver(tmp_path_factory):
    """Build the original lpzrobots Sox from a checkout of lpzrobots/selforg.

    Uses LPZROBOTS_SELFORG if set, else external/lpzrobots/selforg in this repo
    (clone of https://github.com/georgmartius/lpzrobots); skipped if neither exists.
    """
    src = os.environ.get("LPZROBOTS_SELFORG")
    if not src and (ROOT / "external" / "lpzrobots" / "selforg").is_dir():
        src = str(ROOT / "external" / "lpzrobots" / "selforg")
    if not src:
        pytest.skip("lpzrobots not found: clone it into external/lpzrobots (see README) "
                    "or set LPZROBOTS_SELFORG=/path/to/lpzrobots/selforg")
    src = Path(src).expanduser().resolve()
    if not (src / "controller" / "sox.cpp").exists():
        pytest.fail(f"{src} does not look like lpzrobots/selforg")
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        pytest.skip("no C++ compiler found")
    work = tmp_path_factory.mktemp("lpz")
    # selforg is normally built with include/selforg/*.h links made by its Makefile; recreate them.
    inc = work / "include" / "selforg"
    inc.mkdir(parents=True)
    skip = ("old_stuff", "simulations", "examples", "tests", "include")
    for h in src.rglob("*.h"):
        rel = h.relative_to(src).parts
        if any(p in skip for p in rel[:-1]):
            continue
        link = inc / h.name
        if not link.exists():
            link.symlink_to(h)
    flags = ["-std=c++14", "-O2", "-w", "-I", str(inc.parent), "-I", str(inc)]
    out = work / "lpz_sox_driver"
    cmd = [cxx, *flags, str(TESTS / "cpp" / "lpz_sox_driver.cpp"),
           *[str(src / f) for f in LPZ_SOURCES], "-o", str(out), "-lpthread"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail("building lpzrobots Sox failed:\n" + res.stderr[-3000:])
    return out


@pytest.fixture(scope="session")
def cpp_driver(tmp_path_factory):
    """Compile tests/cpp/driver.cpp against the Eigen controller; return the binary path."""
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        pytest.skip("no C++ compiler found")
    eigen = _find_eigen()
    if eigen is None:
        pytest.skip("Eigen headers not found (set EIGEN_INCLUDE_DIR)")
    # The controller includes <eigen3/Eigen/Core>, so we need the parent dir of eigen3/.
    out = tmp_path_factory.mktemp("cpp") / "hk_driver"
    cmd = [cxx, "-std=c++17", "-O1",
           "-I", str(eigen.parent),
           "-I", str(TESTS / "cpp" / "stubs"),
           "-I", str(ROOT / "cpp/homeokinesis/eigen"),
           str(TESTS / "cpp" / "driver.cpp"),
           str(ROOT / "cpp/homeokinesis/eigen/homeokinesis_controller.cpp"),
           "-o", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail("C++ build failed:\n" + " ".join(cmd) + "\n" + res.stderr)
    return out
