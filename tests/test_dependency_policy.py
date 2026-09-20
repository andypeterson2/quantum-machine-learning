"""Two environments, two files, and the rules that keep them apart.

The dev machine is an Intel Mac, where PyTorch's last wheel is 2.2.2; that caps
numpy below 2 and pennylane below 0.45. Production has no such cap and runs
torch 2.14 on numpy 2. Both facts are true at once, which is why there are two
pinned files rather than one — and why four Dependabot PRs (#15, #18, #23, #26)
could never have merged: each bumped one half of the torch/torchvision pair in
the file the dev machine has to install.

These tests pin the arrangement: each file stays a lock, the image installs from
the linux one, and Dependabot leaves the pins that cannot move alone.
"""

from __future__ import annotations

import re

import pytest

from classifiers.web_export import REPO_ROOT

PARITY = REPO_ROOT / "requirements.txt"
LINUX_LOCK = REPO_ROOT / "requirements" / "linux" / "requirements.txt"
LINUX_TORCH = REPO_ROOT / "requirements" / "linux" / "torch.txt"
DOCKERFILE = REPO_ROOT / "Dockerfile"
DEPENDABOT = REPO_ROOT / ".github" / "dependabot.yml"

PIN = re.compile(r"^[A-Za-z0-9_.\-]+==[A-Za-z0-9_.\-+]+$")


def _requirements(path):
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith(("#", "-"))
    ]


pytestmark = pytest.mark.skipif(
    not DOCKERFILE.is_file(), reason="repo shape; the image ships no Dockerfile"
)


class TestBothFilesArePinned:
    @pytest.mark.parametrize("path", [PARITY, LINUX_LOCK, LINUX_TORCH], ids=lambda p: p.name)
    def test_every_requirement_is_an_exact_pin(self, path) -> None:
        """A range here would make two builds of one commit differ."""
        loose = [line for line in _requirements(path) if not PIN.fullmatch(line)]
        assert not loose, f"{path.name} has unpinned entries: {loose}"

    def test_the_parity_file_states_the_intel_ceiling(self) -> None:
        assert "torch==2.2.2" in PARITY.read_text()

    def test_the_linux_lock_is_free_of_that_ceiling(self) -> None:
        """Production is the reason the ceiling is not the code's."""
        lock = LINUX_LOCK.read_text()
        assert re.search(r"^numpy==2\.", lock, re.MULTILINE), "linux should run numpy 2"
        assert "torch==" not in lock, "torch belongs in torch.txt, from the CPU index"


class TestTheImageInstallsFromTheLock:
    def test_dockerfile_uses_both_lock_files(self) -> None:
        """Without this the lock is decoration: Dependabot PRs against it would
        pass while the image resolved something else entirely."""
        dockerfile = DOCKERFILE.read_text()
        assert "-r requirements/linux/torch.txt" in dockerfile
        assert "-r requirements/linux/requirements.txt" in dockerfile

    def test_dockerfile_installs_nothing_unpinned(self) -> None:
        installs = re.findall(r"pip install[^\n]*", DOCKERFILE.read_text())
        for install in installs:
            assert "-r requirements/linux/" in install or "--no-deps ." in install, install

    def test_the_package_cannot_override_the_lock(self) -> None:
        """`pip install .` would re-resolve pyproject's ranges over the lock."""
        assert "pip install --no-cache-dir --no-deps ." in DOCKERFILE.read_text()

    def test_torch_comes_from_the_cpu_index(self) -> None:
        assert "--index-url https://download.pytorch.org/whl/cpu" in LINUX_TORCH.read_text()


class TestDependabotLeavesTheImmovablePinsAlone:
    """Parsed by hand rather than with PyYAML, which is not a dependency here."""

    @staticmethod
    def _ignored(directory: str) -> set[str]:
        """Dependency names ignored for one pip entry of the Dependabot config."""
        blocks = DEPENDABOT.read_text().split("- package-ecosystem:")
        for block in blocks[1:]:
            if not block.lstrip().startswith("pip"):
                continue
            if re.search(rf"^\s*directory:\s*{re.escape(directory)}\s*$", block, re.MULTILINE):
                return set(re.findall(r"dependency-name:\s*([A-Za-z0-9_.\-]+)", block))
        raise AssertionError(f"no pip entry for {directory}")

    def test_ruff_minors_are_held(self) -> None:
        """Dependabot leaves ruff's minor alone, so the range moves by hand."""
        assert "ruff" in self._ignored("/")

    def test_the_parity_file_holds_its_whole_chain(self) -> None:
        """torch caps numpy, and numpy caps pennylane; a bump to any of them
        breaks the dev machine, so all three are ignored together."""
        assert {"torch", "torchvision", "numpy", "pennylane"} <= self._ignored("/")

    def test_the_linux_lock_is_updated_but_not_for_torch(self) -> None:
        ignored = self._ignored("/requirements/linux")
        assert {"torch", "torchvision"} <= ignored
        assert "numpy" not in ignored, "production numpy should keep moving"


class TestTheLintJobAndTheDevExtraAgree:
    """One ruff range between them.

    The enforced rule set moves with ruff's minors, so a tree clean under one
    minor is not clean under the next. The lint job installs its own ruff and
    the [dev] extra pins another, and both files carry a comment asking a reader
    to keep them in step — this is what holds them there.
    """

    SPEC = re.compile(r"ruff(>=[^\"']+)")

    def _spec(self, path) -> str:
        match = self.SPEC.search(path.read_text())
        assert match, f"no ruff specifier in {path.name}"
        return match.group(1).strip()

    def test_the_ranges_are_identical(self) -> None:
        workflow = self._spec(REPO_ROOT / ".github" / "workflows" / "ci.yml")
        pyproject = self._spec(REPO_ROOT / "pyproject.toml")
        assert workflow == pyproject, (
            f"lint job pins ruff{workflow}, [dev] pins ruff{pyproject}"
        )
