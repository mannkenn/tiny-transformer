"""Places the repo root on sys.path so tests can import model/train/utils."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _yaml_files(directory):
    return sorted(directory.rglob("*.yaml")) if directory.is_dir() else []


def pytest_generate_tests(metafunc):
    """Parameterise over the real config files shipped in the repo.

    Done here rather than with a glob inside a test so that each config shows up
    as its own test id, and so an empty directory fails loudly instead of
    vacuously passing.
    """
    if "repo_config_path" in metafunc.fixturenames:
        paths = _yaml_files(ROOT / "configs")
        assert paths, "no configs found under configs/"
        metafunc.parametrize(
            "repo_config_path", paths, ids=[str(p.relative_to(ROOT)) for p in paths]
        )

    if "repo_result_config_path" in metafunc.fixturenames:
        paths = _yaml_files(ROOT / "results")
        assert paths, "no committed run configs found under results/"
        metafunc.parametrize(
            "repo_result_config_path", paths, ids=[str(p.relative_to(ROOT)) for p in paths]
        )


@pytest.fixture
def minimal_config():
    return {
        "run_name": "unit",
        "learning_rate": 1e-3,
        "batch_size": 4,
        "block_size": 16,
        "n_embd": 32,
        "n_layers": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "eval_interval": 5,
        "eval_iters": 2,
        "max_iters": 10,
    }
