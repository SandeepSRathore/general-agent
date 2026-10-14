from pathlib import Path

import pytest

from general_agent.tools.files import FileSandbox, SandboxError


@pytest.fixture
def sandbox(tmp_path: Path) -> FileSandbox:
    workspace = tmp_path / "workspace"
    knowledge = tmp_path / "knowledge"
    workspace.mkdir()
    knowledge.mkdir()
    (knowledge / "notes.md").write_text("# Notes\nhello")
    (workspace / "a.txt").write_text("alpha")
    (tmp_path / "secret.txt").write_text("top secret")
    return FileSandbox(
        read_roots=[workspace, knowledge], write_root=workspace, project_root=tmp_path, max_bytes=50
    )


def test_read_relative_to_workspace(sandbox: FileSandbox) -> None:
    assert sandbox.read("a.txt") == "alpha"


def test_read_with_root_prefix(sandbox: FileSandbox) -> None:
    assert "hello" in sandbox.read("knowledge/notes.md")


def test_read_by_bare_name_in_second_root(sandbox: FileSandbox) -> None:
    assert "hello" in sandbox.read("notes.md")


@pytest.mark.parametrize("path", ["../secret.txt", "secret.txt", "/etc/passwd"])
def test_read_outside_sandbox_is_blocked(sandbox: FileSandbox, path: str) -> None:
    with pytest.raises(SandboxError):
        sandbox.read(path)


def test_symlink_escape_is_blocked(sandbox: FileSandbox, tmp_path: Path) -> None:
    (tmp_path / "workspace" / "link.txt").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(SandboxError):
        sandbox.read("link.txt")


def test_read_truncates_large_files(sandbox: FileSandbox, tmp_path: Path) -> None:
    (tmp_path / "workspace" / "big.txt").write_text("x" * 500)
    content = sandbox.read("big.txt")
    assert content.startswith("x" * 50)
    assert "truncated" in content


def test_write_only_inside_workspace(sandbox: FileSandbox, tmp_path: Path) -> None:
    written = sandbox.write("out/report.txt", "data")
    assert written == tmp_path / "workspace" / "out" / "report.txt"
    assert written.read_text() == "data"
    with pytest.raises(SandboxError):
        sandbox.write("../escape.txt", "nope")
    with pytest.raises(SandboxError):
        sandbox.write("knowledge/notes.md", "overwrite")


def test_list_directory(sandbox: FileSandbox) -> None:
    listing = sandbox.list_dir(".")
    assert "a.txt" in listing
    assert "workspace" in sandbox.list_dir("")  # empty -> overview of roots
