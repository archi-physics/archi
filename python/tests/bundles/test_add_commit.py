"""`archi add` commits archi's own paths: the deployment, archi.lock and profiles/.

okg refuses to publish while they are uncommitted ("repository provenance inputs are not
committed"). The commit is limited to those paths: other staged work stays staged, an instance
nested in a larger repository is handled by paths relative to the top level, and a run that changed
nothing makes no commit.
"""
import subprocess
from pathlib import Path

from archi.bundles.lock import _git_commit

FILES = ("archi.yaml", "archi.lock", "profiles/p/x.yaml", "deployments/gp/deployment.yaml")


def git(top: Path, *args: str) -> str:
    cmd = ["git", "-C", str(top), "-c", "user.name=t", "-c", "user.email=t@e.invalid", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def make_repo(top: Path, instance: Path) -> None:
    for name in FILES:
        (instance / name).parent.mkdir(parents=True, exist_ok=True)
        (instance / name).write_text("1\n")
    (top / "other.txt").write_text("1\n")
    subprocess.run(["git", "init", "-q", str(top)], check=True)
    git(top, "add", "-A")
    git(top, "commit", "-q", "-m", "base")


def change_all(instance: Path) -> None:
    for name in FILES:
        (instance / name).write_text("2\n")
    (instance / "profiles/p/new.yaml").write_text("n\n")


def test_archi_add_commits_the_lock_and_profiles_beside_the_deployment(tmp_path):
    make_repo(tmp_path, tmp_path)
    change_all(tmp_path)
    assert _git_commit(tmp_path / "deployments" / "gp", "archi add x", tmp_path) is not None
    assert git(tmp_path, "status", "--short").split() == ["M", "archi.yaml"]  # the user's file stays


def test_other_staged_work_is_not_swept_into_the_commit(tmp_path):
    make_repo(tmp_path, tmp_path)
    change_all(tmp_path)
    (tmp_path / "other.txt").write_text("staged by someone else\n")
    (tmp_path / "brand-new.txt").write_text("new\n")
    git(tmp_path, "add", "other.txt", "brand-new.txt", "archi.yaml")
    assert _git_commit(tmp_path / "deployments" / "gp", "archi add x", tmp_path) is not None
    committed = git(tmp_path, "show", "--name-only", "--format=", "HEAD").split()
    assert sorted(committed) == ["archi.lock", "deployments/gp/deployment.yaml", "profiles/p/new.yaml",
                                 "profiles/p/x.yaml"]
    staged = git(tmp_path, "diff", "--cached", "--name-only").split()
    assert sorted(staged) == ["archi.yaml", "brand-new.txt", "other.txt"]


def test_an_instance_nested_in_a_larger_repository(tmp_path):
    instance = tmp_path / "work" / "inst"
    make_repo(tmp_path, instance)
    change_all(instance)
    assert _git_commit(instance / "deployments" / "gp", "archi add x", instance) is not None
    committed = git(tmp_path, "show", "--name-only", "--format=", "HEAD").split()
    assert sorted(committed) == ["work/inst/archi.lock", "work/inst/deployments/gp/deployment.yaml",
                                 "work/inst/profiles/p/new.yaml", "work/inst/profiles/p/x.yaml"]
    assert git(tmp_path, "status", "--short").split() == ["M", "work/inst/archi.yaml"]


def test_nothing_changed_makes_no_commit(tmp_path):
    make_repo(tmp_path, tmp_path)
    head = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "archi.yaml").write_text("user edit\n")  # not archi's path
    assert _git_commit(tmp_path / "deployments" / "gp", "archi add x", tmp_path) is None
    assert git(tmp_path, "rev-parse", "HEAD") == head


def test_no_repository_means_no_commit(tmp_path):
    (tmp_path / "deployments" / "gp").mkdir(parents=True)
    assert _git_commit(tmp_path / "deployments" / "gp", "archi add x", tmp_path) is None
