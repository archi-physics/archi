"""The code-text pack stores function bodies.

The code-symbols pack gives a function's name, signature, docstring and line
numbers, never its body, so "what does this function do when the name
contains a slash" could not be answered from the graph. okg's
`DocCorpusSource` with `PythonSymbolChunker` stores the code itself, one chunk
per function or class. These tests hold the pack's wiring for it:

- the install keeps `${deployment_dir}` for okg to expand at run time, and
  installs the extractors.yaml that the source reads from there;
- the shipped parameters, run over a synthetic repository, store a function's
  body with its line range and name;
- the shipped excludes keep credential-named code files out (a control run
  without them stores those files);
- every matched extension reaches a parser.

Synthetic fixtures only; no real repository content.
"""

import inspect
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

PACK = Path(__file__).resolve().parents[2] / "bundles" / "packs" / "code-text"
EXAMPLES = {"code_text": PACK / "sources" / "code_text.yaml"}

#: Only this function's body contains the marker, not its name, signature or
#: docstring: finding the marker proves the body was stored.
_MODULE = '''"""Bucket helpers."""

PREFIX = "bkt_"


def bucket_label(name=None):
    """Return the label for a bucket."""
    if not name:
        return PREFIX + "default"
    if "/" in name:
        raise ValueError("zanzibar: a slash is not allowed in a bucket label")
    return PREFIX + name


class Store:
    """A tiny store."""

    def put(self, key, value):
        self.data[key] = value
'''


def _entry(name):
    return yaml.safe_load(EXAMPLES[name].read_text(encoding="utf-8"))[name]


def _plan(tmp_path):
    """okg's install plan for the code-text pack rendered alone (it pulls in
    git-code, whose checkout it reads)."""
    from packs.pack_support import okg_plan, render_alone

    rendered = render_alone(tmp_path, "code-text")
    return okg_plan(
        rendered,
        tmp_path,
        {
            "git_code_repo_name": "demo",
            "git_code_repo_url": "https://example.org/demo.git",
            "archi_data_root": str(tmp_path / "data"),
        },
    )


def test_install_keeps_the_runtime_anchor_and_installs_extractors(tmp_path):
    """`${deployment_dir}` survives the install and okg expands it later.

    The installer refuses any `${name}` it has no answer for, so the source
    cannot name `${deployment_dir}` directly. The `extractors_dir` answer
    defaults to that literal text, and the installer does not re-expand an
    answer's value. okg's registry loader then replaces it with the directory
    the registry sits in, which is where `files:` put extractors.yaml.
    """
    from okg.substrate.deployment_bootstrap.profile_init import planned_writes
    from okg.substrate.sources.registry import load_source_registry

    plan = _plan(tmp_path)
    sid = "code-text.code_text"
    params = yaml.safe_load(plan.source_defaults[sid])[sid]["params"]
    assert params["extractors_dir"] == "${deployment_dir}"
    assert params["repo_root"] == str(tmp_path / "data") + "/repos/demo"
    assert params["predecessor_source"] == "qt.git-repo"

    for write in planned_writes(plan):
        write.path.parent.mkdir(parents=True, exist_ok=True)
        if getattr(write, "data", None) is not None:
            write.path.write_bytes(write.data)
        else:
            write.path.write_text(write.text, encoding="utf-8")
    deployment = plan.deployment_dir
    assert (deployment / "extractors.yaml").read_bytes() == (
        PACK / "extractors.yaml"
    ).read_bytes()

    entries = load_source_registry(deployment / "source_registry.yaml")
    loaded = entries[sid].raw["params"]
    assert Path(loaded["extractors_dir"]) == deployment.resolve()
    assert (Path(loaded["extractors_dir"]) / "extractors.yaml").is_file()


def test_a_bare_cern_team_render_ships_neither_the_source_nor_extractors(tmp_path):
    """Selecting the pack is what installs extractors.yaml; a bare cern-team
    render carries neither the source nor the file."""
    from archi.bundles.render import render
    from archi.bundles.resolve import resolve
    from packs.pack_support import shipped_catalog

    rendered = render(resolve(shipped_catalog(), ["cern-team"]), tmp_path)
    assert not [s for s in rendered.composition.sources if s.startswith("code-text.")]
    assert not (rendered.profile_dir / "extractors.yaml").exists()


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_the_entry_binds_against_the_screened_doc_corpus_source(name):
    """The source is okg's DocCorpusSource behind archi's secret screen, with
    DocCorpusSource's parameters unchanged."""
    from okg.substrate.library.sources.doc_corpus import DocCorpusSource

    from archi.sources.code_text import SecretScreenedDocCorpusSource

    entry = _entry(name)
    assert entry["module"] == "archi.sources.code_text"
    assert entry["class"] == "SecretScreenedDocCorpusSource"
    assert issubclass(SecretScreenedDocCorpusSource, DocCorpusSource)
    inspect.signature(SecretScreenedDocCorpusSource).bind(**entry["params"])


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_every_text_format_reaches_a_parser(name):
    """The source selects every file, so every format okg can route a
    selected file to must have a parser here, or that file becomes a
    document that stores nothing. Binary formats are excluded instead."""
    from okg.substrate.library.extraction.chunk.python_symbols import (
        PythonSymbolChunker,
    )
    from okg.substrate.library.extraction.registry import load_extractors
    from okg.substrate.library.sources.doc_corpus import DEFAULT_EXTENSION_TO_FORMAT

    params = _entry(name)["params"]
    assert params["include_globs"] == ["**/*"]
    assert params["text_fallback"] is True
    routes = dict(DEFAULT_EXTENSION_TO_FORMAT)
    routes.update(params["extension_to_format"])
    excluded = {g[len("**/*"):] for g in params["exclude_globs"] if g.startswith("**/*.")}
    bundle = load_extractors(PACK)
    for ext, fmt in sorted(routes.items()):
        if ext in excluded:
            continue
        assert fmt in bundle.parsers, (
            f"{ext} routes to {fmt!r}, which extractors.yaml has no parser for"
        )
        assert bundle.chunker_for_format(fmt) is not None
    assert isinstance(bundle.chunker_for_format("code"), PythonSymbolChunker)


#: Code files whose NAMES mark them as credentials. Only the shipped
#: `exclude_globs` can keep them out: their extension matches `include_globs`.
#: Invented values only.
_CREDENTIAL_CODE = {
    "pkg/secrets/keys.py": 'API_KEY = "not-a-real-key"\n',
    "pkg/aws_credentials.py": 'SECRET = "not-a-real-secret"\n',
}


def _git_repo(root: Path) -> Path:
    repo = root / "repos" / "demo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "buckets.py").write_text(_MODULE, encoding="utf-8")
    for rel, body in _CREDENTIAL_CODE.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(body, encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    for cmd in (["init", "-q"], ["add", "-A", "-f"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env)
    return repo


def _run(repo: Path, cls=None, **overrides):
    """Run the reader with the shipped parameters over `repo`.

    `predecessor_source` is dropped because it needs a database; the live
    install proves it (pact evidence).
    """
    return list(_source(repo, cls, **overrides).run("probe", mode="scope_complete").facts)


def _source(repo: Path, cls=None, **overrides):
    """The shipped source over `repo` (archi's screened class unless `cls`)."""
    from archi.sources.code_text import SecretScreenedDocCorpusSource

    params = dict(_entry("code_text")["params"])
    params.pop("predecessor_source")
    params.update(repo="demo", repo_root=str(repo), extractors_dir=str(PACK))
    params.update(overrides)
    return (cls or SecretScreenedDocCorpusSource)(**params)


def _documents(facts):
    return sorted(f.node_id for f in facts if getattr(f, "subtype", None) == "document")


def test_the_excludes_keep_credential_named_code_out(tmp_path):
    """The shipped excludes, not the include globs, stop these files.

    The include globs match both files, and their invented values are too
    plain for the secret screen. The control run, okg's own reader with no
    excludes, stores them, which proves the shipped list is what keeps them
    out.
    """
    from okg.substrate.library.sources.doc_corpus import DocCorpusSource

    repo = _git_repo(tmp_path)
    credential_docs = sorted(f"doc:demo:{rel}" for rel in _CREDENTIAL_CODE)

    assert _documents(_run(repo, DocCorpusSource, exclude_globs=[])) == sorted(
        ["doc:demo:pkg/buckets.py", *credential_docs]
    )
    shipped = _run(repo)
    assert _documents(shipped) == ["doc:demo:pkg/buckets.py"]
    stored = "\n".join(
        str(f.attrs.get("text", ""))
        for f in shipped
        if getattr(f, "subtype", None) == "document_chunk"
    )
    assert "not-a-real" not in stored


def test_the_shipped_parameters_store_a_function_body(tmp_path):
    """End to end over okg's reader: a body-only word comes back with its lines."""
    facts = _run(_git_repo(tmp_path))
    chunks = [f for f in facts if getattr(f, "subtype", None) == "document_chunk"]
    body = [c for c in chunks if "zanzibar" in str(c.attrs.get("text", ""))]
    assert len(body) == 1, "the function body was not stored as one chunk"
    attrs = body[0].attrs
    text = attrs["text"]
    assert "def bucket_label" in text
    assert 'raise ValueError("zanzibar' in text
    assert "class Store" not in text, "the chunk is not cut at the function"
    lines = _MODULE.splitlines()
    start = int(attrs["start_line"])
    end = int(attrs["end_line"])
    # The range may start at the blank lines above the `def`;
    # it must still cover the whole function.
    assert "def bucket_label" in "\n".join(lines[start - 1:end])
    assert "zanzibar" in "\n".join(lines[start - 1:end])
    assert attrs["symbol"] == "bucket_label"
    methods = [c for c in chunks if c.attrs.get("symbol") == "Store"]
    assert methods and "self.data[key] = value" in methods[0].attrs["text"]


def test_the_guide_tells_an_existing_deployment_how_to_add_the_pack():
    """An installed instance adds code-text with `archi add`, which does the
    registry edit, the extractors.yaml copy and the ownership claim the guide
    once listed as manual steps (a bare `github_repo` predecessor and a
    registry edit without a fresh claim were both refused on a staging
    deployment). The template's predecessor is an ownership id, so it
    survives the pack-qualified source ids."""
    guide = (PACK.parents[2] / "docs" / "cern-team-quickstart.md").read_text(encoding="utf-8")
    section = guide.split("### Function bodies, not only signatures", 1)[1]
    section = section.split("\n## ", 1)[0]
    assert "archi add code-text" in section
    assert "--code-text-extractors-dir" in section
    assert _entry("code_text")["params"]["predecessor_source"] == "${deployment_name}.git-repo"


#: Text files that are not code, each holding a word found nowhere else. The
#: Helm template does not parse as YAML; plain text still stores it.
_TEXT_FILES = {
    "README.md": ("# Demo\n\nRestart with the quokka procedure.\n", "quokka"),
    "deploy/values.yaml": ("replicas: {{ .Values.replicas }}\nmode: wombat\n", "wombat"),
    "deploy/config.json": ('{"queue": "platypus"}\n', "platypus"),
    "scripts/rotate.sh": ("#!/bin/bash\necho numbat\n", "numbat"),
    "scripts/cleanup": ("#!/bin/sh\necho bilby\n", "bilby"),
    "Makefile": ("all:\n\techo dingo\n", "dingo"),
    "setup.cfg": ("[tool]\nname = echidna\n", "echidna"),
    "docs/index.html": ("<p>wallaby</p>\n", "wallaby"),
}
_BINARY_FILES = {"docs/logo.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "dist/tool.jar": b"PK\x03\x04\x00"}


def _commit(repo: Path, files: dict, message: str = "x") -> None:
    for rel, body in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            path.write_bytes(body)
        else:
            path.write_text(body, encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    for cmd in (["add", "-A", "-f"], ["commit", "-q", "-m", message]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env)


def test_every_text_file_is_stored_as_text(tmp_path):
    """Markdown, YAML (templated), JSON, shell, extensionless scripts, a
    Makefile, config and HTML each become a document whose chunks hold the
    file's text; binary files become no document at all."""
    repo = _git_repo(tmp_path)
    _commit(repo, {**{rel: body for rel, (body, _) in _TEXT_FILES.items()}, **_BINARY_FILES})
    facts = _run(repo)
    docs = _documents(facts)
    stored = "\n".join(
        str(f.attrs.get("text", "")) for f in facts if getattr(f, "subtype", None) == "document_chunk"
    )
    for rel, (_, marker) in _TEXT_FILES.items():
        assert f"doc:demo:{rel}" in docs, rel
        assert marker in stored, (rel, marker)
    for rel in _BINARY_FILES:
        assert f"doc:demo:{rel}" not in docs, rel
    assert "doc:demo:pkg/buckets.py" in docs


#: An invented password, assembled from pieces so this file holds no
#: complete secret-shaped string.
_PW = "Zx9v" + "Q2mT7rLp"


def _secret_files() -> dict:
    """Files holding an invented literal secret, assembled at run time."""
    return {
        "deploy/db-values.yaml": "db:\n  user: archi\n  password: " + _PW + "\n",
        "scripts/upload.sh": "curl -H 'X-Token: " + "gh" + "p_" + "Q7x" * 12 + "' https://example.org\n",
        "notes/server.txt": "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIabc\n",
        "app/settings.py": 'DATABASE_URL = "postgres://archi:' + _PW + '@db01:5432/archi"\n',
    }


#: Files that only name or pass a secret around: kept.
_REFERENCE_FILES = {
    "deploy/app-values.yaml": "db:\n  password: ${DB_PASSWORD}\n  secretName: archi-db\n",
    "app/auth.py": 'import os\n\nTOKEN = os.environ["ARCHI_TOKEN"]\nheaders = {"Authorization": f"Bearer {TOKEN}"}\n',
}


def test_the_screen_leaves_out_files_that_hold_a_literal_secret(tmp_path):
    """No document and no stored text for a file holding a literal secret;
    files that only reference a secret are kept. The control run, okg's own
    reader with the same parameters, stores the secret files, so the screen
    is what keeps them out."""
    from okg.substrate.library.sources.doc_corpus import DocCorpusSource

    repo = _git_repo(tmp_path)
    secrets = _secret_files()
    _commit(repo, {**secrets, **_REFERENCE_FILES})

    control = _documents(_run(repo, DocCorpusSource))
    for rel in secrets:
        assert f"doc:demo:{rel}" in control, rel

    source = _source(repo)
    facts = list(source.run("probe", mode="scope_complete").facts)
    docs = _documents(facts)
    for rel in secrets:
        assert f"doc:demo:{rel}" not in docs, rel
    for rel in _REFERENCE_FILES:
        assert f"doc:demo:{rel}" in docs, rel
    assert set(secrets) <= set(source.screened_paths)
    stored = "\n".join(str(getattr(f, "attrs", {}).get("text", "")) for f in facts)
    assert _PW not in stored
    assert "PRIVATE KEY" not in stored


def test_a_file_that_gains_a_secret_drops_out_and_returns_when_it_is_removed(tmp_path):
    """The screen reads the file on every listing, so a later commit that adds
    a secret leaves the file out of the next run, and a commit that removes
    the secret brings it back. Same source object, so the screen's cache must
    notice the change. (The run is proven here; retracting the document from
    a published graph is okg's completed-scope deletion.)"""
    repo = _git_repo(tmp_path)
    clean = "db:\n  password: ${DB_PASSWORD}\n"
    _commit(repo, {"deploy/values.yaml": clean})
    source = _source(repo)

    def docs():
        return _documents(list(source.run("probe", mode="scope_complete").facts))

    assert "doc:demo:deploy/values.yaml" in docs()
    _commit(repo, {"deploy/values.yaml": "db:\n  password: " + _PW + "\n"}, "leak")
    assert "doc:demo:deploy/values.yaml" not in docs()
    assert source.screened_paths == ("deploy/values.yaml",)
    _commit(repo, {"deploy/values.yaml": clean + "# rotated\n"}, "fix")
    assert "doc:demo:deploy/values.yaml" in docs()
    assert source.screened_paths == ()


def test_a_same_size_edit_with_the_same_timestamp_is_screened(tmp_path):
    """The cache is keyed by content, not size and timestamp: a clean file
    rewritten with a same-length secret and its old modification time is
    still left out."""
    repo = _git_repo(tmp_path)
    path = repo / "deploy" / "values.yaml"
    clean = "db:\n  password: ${DB_PASSWORD_X}\n"
    leak = "db:\n  password: " + _PW + "abcd\n"
    assert len(clean) == len(leak)
    _commit(repo, {"deploy/values.yaml": clean})
    stat = path.stat()
    source = _source(repo)
    assert "doc:demo:deploy/values.yaml" in _documents(list(source.run("probe", mode="scope_complete").facts))
    path.write_text(leak, encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
    assert "doc:demo:deploy/values.yaml" not in _documents(list(source.run("probe", mode="scope_complete").facts))
    assert source.screened_paths == ("deploy/values.yaml",)
