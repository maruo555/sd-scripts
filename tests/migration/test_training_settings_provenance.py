from library import training_settings


def test_revision_fallback_in_clone_and_packed_worktree(tmp_path, monkeypatch):
    def broken_git(*args, **kwargs):
        raise OSError("git unavailable")
    monkeypatch.setattr(training_settings.subprocess, "run", broken_git)
    sha = "0123456789" * 4
    repo = tmp_path / "clone"
    git = repo / ".git"
    (git / "refs/heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main", "ascii")
    loose = git / "refs/heads/main"
    loose.write_text(sha + "\n", "ascii")
    assert training_settings.source_identity(repo)["source_sha"] == sha
    loose.unlink()
    (git / "packed-refs").write_text(f"{sha} refs/heads/main\n", "ascii")
    checkout = tmp_path / "worktree"
    checkout.mkdir()
    workgit = git / "worktrees" / "test"
    workgit.mkdir(parents=True)
    (checkout / ".git").write_text("gitdir: " + str(workgit), "utf-8")
    (workgit / "commondir").write_text("../..", "utf-8")
    (workgit / "HEAD").write_text("ref: refs/heads/main", "ascii")
    assert training_settings.source_identity(checkout)["source_sha"] == sha
    (workgit / "HEAD").write_text(sha, "ascii")
    assert training_settings.source_identity(checkout)["source_sha"] == sha
    (workgit / "HEAD").write_text("ref: refs/../../../secrets", "ascii")
    assert training_settings.source_identity(checkout)["source_sha"] is None
