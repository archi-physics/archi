"""okg's file-text source with a secret screen, for the code-text pack.

The code-text pack stores the text of every text file in a repository. Once a
file's text is in the graph, anyone who can search the graph can read it, so a
credential committed to the repository would become findable. Excluding file
names (`.env`, `secrets/`, `*credentials*`) is not enough: a password in a
Helm values file or a token in a shell script has an ordinary name.

`SecretScreenedDocCorpusSource` is okg's `DocCorpusSource` with one change: it
reads each file the globs select and leaves out any file whose text holds a
literal secret (see `secret_classes`). A file it leaves out gets no document
and no chunks; its git-code `file:` node still exists, so a text read of it
returns no text. The screen runs on every listing, so a file that gains a
secret in a later commit is not emitted by the next run (okg's
`missing_from_completed_scope` deletion should then retract its document; the
tests prove the run, not the retraction), and a file whose secret is removed
comes back.

okg's change probe and its read both list files through `_walk_files`, which
is the method overridden here, so the two always see the same set.

What counts as a secret. A file is left out when its text holds any of:

- a private key block (`-----BEGIN ... PRIVATE KEY-----`, PGP's `PRIVATE KEY
  BLOCK` too);
- a token in a known vendor format (GitHub, GitLab, OpenAI-style `sk-`, AWS
  access key id, Slack, Google API key) or a JWT;
- a base64 run of 80 or more characters that is not plain hexadecimal
  (embedded keys), unless it is public material: an SSH public key, a
  package integrity hash, an inline image, a cluster CA certificate;
- a URL with a password that is not a template (`scheme://user:pass@`,
  even when the password is a plain word or URL-encoded);
- an `Authorization:` or `Bearer` header, or `curl -u user:pass`, with a
  literal value;
- a key ending in password, passwd, passphrase, pwd, pass, secret, token,
  credential(s), api key, access key, secret key, private key or auth key,
  assigned a literal value on the same line (`=` or `:`, type annotations
  skipped), and a bare `auth` key with a literal of 16 or more characters;
- `--password VALUE` style flags, a .netrc `password VALUE` line and XML
  elements such as `<password>VALUE</password>`.

A value is a reference, not a literal, when it is shorter than 8 characters,
has no letter, reads as prose (three or more words), or reads as a variable,
template, call or path: `$DB_PASS`, `${DB_PASS}`, `{{ .Values.x }}`,
`{password}`, `%(password)s`, `<token>`, a YAML alias, `os.environ[...]`,
`config["password"]`, `get_token()`, `self.token`, `/etc/app/secret.pem`,
`None`, `changeme` and the like. A value with no digit,
no capital after its first letter and no symbol (`required`, `Password`) is a
word, not a secret. Files that only name or pass secrets around are kept.

This is a pattern screen, not a proof that no secret remains. Known misses: a
password under a key with no secret word (`x`, a Kubernetes `name:` /
`value:` pair), values under 8 characters or plain words, values that start
like a template (`$2b$...` bcrypt hashes, `{...}`), keys longer than 64
characters, and .netrc `default` lines. URLs to documentation or
placeholder hosts (loopback, `example.com/.org/.net`, `*.example`,
`*.invalid`, `*.test`, `host`, `server`) are not counted. It covers the documents this source
writes only; git-code's history source stores per-file diffs, which okg
redacts by its own patterns, not by this screen. Narrow `include_globs` for a repository
nobody has reviewed. The screen reports the paths it left out (never the
values) in the `screened_paths` attribute and in a log line per listing.

The screen reads each selected file in full on every listing, through okg's
contained reader; its verdict is cached by the content's SHA-256 within one
source object. Files larger than `max_bytes_per_file` are not read: okg
stores them as `oversize` documents with no text, so nothing of theirs
reaches the graph either way. Every pattern is bounded so that one long line
costs linear time.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from okg.substrate.library.sources.doc_corpus import DocCorpusSource

log = logging.getLogger(__name__)

__all__ = ["SecretScreenedDocCorpusSource", "secret_classes"]

# Every pattern below is anchored or bounded so that one long line costs
# linear time: a match may only start at the beginning of a run of key
# characters, and every repeated prefix has an upper bound.

# --- secrets that need no context -------------------------------------------

_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY(?: BLOCK)?-----")
_TOKENS = (
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_BASE64_RUN = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9+/])")
_HEX = re.compile(r"[0-9A-Fa-f]+")
# Public base64 that is not a secret: SSH public keys, package integrity
# hashes, inline images, a cluster's CA certificate.
_PUBLIC_BASE64_BEFORE = re.compile(
    r"(?:ssh-(?:rsa|ed25519|dss)|ecdsa-sha2-nistp\d+|sk-ssh-ed25519@openssh\.com|sha(?:1|256|384|512)-"
    r"|data:image/[\w+.-]{1,20};base64,|\"image/[a-z+]{1,10}\":[ \t]*\[?[ \t]*\"|certificate-authority-data:[ \t]*)[ \t]*$"
)

# --- secrets that are only secrets when the value is literal ----------------

_CREDENTIAL_URL = re.compile(
    r"\b[a-z][a-z0-9+.-]{0,32}://[^/\s:@'\"]{0,256}:(?P<value>[^@\s/'\"]{1,256})@(?P<host>[^/\s:'\"?#]{0,256})",
    re.I,
)
# Documentation and placeholder hosts: `postgres://u:okg@127.0.0.1` in a
# quickstart, `scheme://user:pass@host` in prose.
_DOC_HOST = re.compile(
    r"^(?:localhost|127\.\d+\.\d+\.\d+|\[?::1\]?|(?:[\w-]+\.)*example(?:\.(?:com|org|net))?"
    r"|.*\.(?:invalid|test|example)|host|hostname|server|myhost|your-?host|<[^>]*>)$",
    re.I,
)
_AUTHORIZATION = re.compile(
    r"\bAuthorization['\"]?[ \t]*[:=,][ \t]*['\"]?(?:(?:Bearer|Basic|Token)[ \t]+)?(?P<value>[^\s,;'\"}]+)", re.I
)
_BEARER = re.compile(r"\bBearer[ \t]+(?P<value>[A-Za-z0-9._~+/=-]{8,})")
_CURL_USER = re.compile(r"(?<!\S)(?:-u|--user)[ \t]+['\"]?[^\s:'\"/]{1,128}:(?P<value>[^\s'\"/][^\s'\"]*)")
_SECRET_KEY_WORDS = (
    r"password|passwd|passphrase|pwd|pass|secret|token|credentials?|auth"
    r"|api[_-]?key|access[_-]?key|secret[_-]?key|private[_-]?key|auth[_-]?key"
)
# The secret word must END the key: `client_secret` and `DB_PASSWORD` count,
# `secretName` and `token_url` do not. A match starts only where a run of key
# characters starts, and the key is at most 64 characters before its secret
# word. Key and value are on one line (a YAML `secret:` that opens a block has
# no value); a type annotation between them is skipped
# (`API_TOKEN: str = "..."`).
_ASSIGNMENT = re.compile(
    r"(?<![A-Za-z0-9_.-])(?P<key>[A-Za-z0-9_.-]{0,64}?(?:" + _SECRET_KEY_WORDS + r"))['\"]?"
    r"(?:[ \t]*:[ \t]*[A-Za-z_][\w.\[\], |]{0,40}?[ \t]*(?==))?"
    r"[ \t]*(?:=>?|:(?!:))[ \t]*"
    r"(?P<value>['\"][^'\"\n]*['\"]|[^\s,;)}\]#]+)",
    re.I,
)
# `--password VALUE`, a .netrc `password VALUE`, and XML elements.
_FLAG = re.compile(
    r"(?<!\S)--?[\w-]{0,40}?(?:password|passwd|pass|token|secret|api-key)[ \t]+(?P<value>[^\s'\"(),;\\-][^\s'\"(),;\\]*)", re.I
)
_MYSQL_P = re.compile(r"\b(?:mysql|mysqldump|mysqladmin)\b[^\n]{0,200}?[ \t]-p(?P<value>[^\s'\"]+)")
_NETRC = re.compile(r"(?m)^[ \t]*(?:machine[ \t]+\S+[ \t]+)?(?:login[ \t]+\S+[ \t]+)?password[ \t]+(?P<value>\S+)")
_XML = re.compile(
    r"<(?P<tag>password|passwd|secret|token|apikey|api_key|accesskey|secretkey|clientsecret)>"
    r"(?P<value>[^<\n]{1,256})</(?P=tag)>",
    re.I,
)

_TEMPLATE = re.compile(
    r"^\$|\$\{|\$\(|\{\{|\{[A-Za-z_]|%\([^)]{0,64}\)[sdr]|%[sdr](?![0-9A-Fa-f])|^<[^>]*>$"
)
_REFERENCE = re.compile(
    r"""^(?:
        [<\[({=&]                      # <x> [..] (..) {..} == &anchor
      | \*[A-Za-z_][\w-]*$             # a YAML alias
      | ~/ | \.\.?/ | /[\w.-]+/        # a path: ~/x ./x ../x /etc/x
      | (?:os|env|environ|self|cls|args|kwargs|config|conf|settings|options|opts|params|request|req|ctx|context
          |secrets|vault|values|var|vars)\b
      | (?:getenv|get_|read_|load_|fetch_|lookup|input|getpass)
      | (?:none|null|nil|true|false|undefined|optional|str|string|bytes|int|bool|dict|list|required)\b
      | (?:x{3,}|changeme|change_me|example|dummy|fake|placeholder|redacted|your[_-])
      | https?://
    )""",
    re.I | re.X,
)
# An unquoted identifier, dotted name or call: a variable, not a literal.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(.*)?")


def _literal(raw: str, min_len: int = 8, phrase: bool = False) -> bool:
    """True when an assigned value reads as a literal secret, not a reference.

    `phrase` is for passphrase keys: a quoted value of several words counts.
    """
    raw = raw.strip()
    quoted = len(raw) >= 2 and raw[0] in "'\"" and raw[-1] == raw[0]
    value = raw.strip("'\"` ").rstrip(",;")
    if len(value) < min_len or not re.search(r"[A-Za-z]", value):
        return False
    if len(value.split()) > 2 and not (phrase and quoted):
        return False  # prose
    if _TEMPLATE.search(value) or _REFERENCE.match(value):
        return False  # filled in elsewhere
    if not quoted and re.match(r"[A-Za-z_][A-Za-z0-9_.]*[\[(]", value):
        return False  # an expression: config["password"], get_token()
    digit = re.search(r"[0-9]", value)
    if not quoted and not digit and _IDENTIFIER.fullmatch(value) and re.search(r"[_.]", value):
        return False  # a variable or dotted name (`db_password`, `config.token`)
    if re.fullmatch(r"[A-Z][A-Z0-9_]+", value) and ("_" in value or not digit):
        return False  # an environment variable's name: os.environ["DB_PASSWORD"]
    if quoted and "(" in value:
        return False  # a quote that spans code: "password", getpass("
    # A secret has a digit, a capital after the first letter, or a symbol; a
    # bare word (`required`, `Password`, `readonly`) does not, and neither
    # does an unquoted camelCase name (`passwordValue`).
    if not re.search(r"[0-9]|.[A-Z]|[^A-Za-z0-9_.-]", value):
        return False
    if not quoted and re.fullmatch(r"[a-z]+(?:[A-Z][a-z]+)+", value):
        return False
    return True


def _url_password(m: re.Match) -> bool:
    """True for any password written into a URL that is not filled in from
    elsewhere: `user:Zx9@db` and `user:p%40ss@db` count, `user:${PASS}@` and
    `{user}:{pw}@` do not. Stricter than `_literal`, because a URL with a
    password is a working credential even when the password is a plain word.
    A URL to a documentation or placeholder host (loopback,
    `example.com/.org/.net`, `*.example`, `*.invalid`, `*.test`, `host`,
    `server`) is not counted. A placeholder password word (`password`,
    `secret`, `changeme`) on any other host is: it is a working default
    credential."""
    value = m.group("value")
    if _TEMPLATE.search(value) or value.startswith(("*", "<")):
        return False
    return not _DOC_HOST.match(m.group("host") or "")


def _assignment_secret(m: re.Match) -> bool:
    # A bare `auth` key holds a secret only as a long opaque value (a docker
    # config's base64 `auth`); `auth: kerberos5` is a setting.
    key = m.group("key").lower()
    return _literal(m.group("value"), 16 if key.endswith("auth") else 8, phrase=key.endswith("passphrase"))


def _public_base64(text: str, start: int) -> bool:
    return bool(_PUBLIC_BASE64_BEFORE.search(text, max(0, start - 64), start))


def secret_classes(text: str) -> list[str]:
    """The kinds of literal secret in `text`, sorted; empty when there is none.

    Returns class names only, never a value.
    """
    if not text:
        return []
    found: set[str] = set()
    if _PRIVATE_KEY.search(text):
        found.add("private_key")
    if any(p.search(text) for p in _TOKENS):
        found.add("token")
    if _JWT.search(text):
        found.add("jwt")
    if any(not _HEX.fullmatch(m.group(0).rstrip("=")) and not _public_base64(text, m.start())
           for m in _BASE64_RUN.finditer(text)):
        found.add("base64_blob")
    if any(_url_password(m) for m in _CREDENTIAL_URL.finditer(text)):
        found.add("credential_url")
    if any(_literal(m.group("value")) for rx in (_AUTHORIZATION, _BEARER, _CURL_USER) for m in rx.finditer(text)):
        found.add("authorization_header")
    if (any(_assignment_secret(m) for m in _ASSIGNMENT.finditer(text))
            or any(_literal(m.group("value")) for rx in (_FLAG, _MYSQL_P, _NETRC, _XML) for m in rx.finditer(text))):
        found.add("secret_assignment")
    return sorted(found)


@dataclass
class SecretScreenedDocCorpusSource(DocCorpusSource):
    """okg's `DocCorpusSource`, minus every file whose text holds a literal
    secret. Parameters are DocCorpusSource's; see the module docstring."""

    # okg's deployment lint reads both off this file's class body and accepts
    # only string literals, so they are restated here (same values as
    # DocCorpusSource's; a test keeps them equal).
    profile = "discovery_crawl"
    change_probe_kind = "content_hash"
    #: Repository-relative paths the last listing left out, sorted.
    screened_paths: tuple[str, ...] = field(default=(), init=False)
    _screen_cache: dict = field(default_factory=dict, init=False, repr=False)
    _previous_screen: dict = field(default_factory=dict, init=False, repr=False)

    def _walk_files(self, scope_keys: Optional[Sequence[str]] = None) -> list[Path]:
        files = super()._walk_files(scope_keys)
        # okg builds every listed path as the resolved root joined with the
        # file's repository path, so relative_to gives that path back.
        root = Path(self.repo_root).resolve()
        kept: list[Path] = []
        screened: list[str] = []
        # Keep only the verdicts this listing uses, so the cache is bounded by
        # the repository, not by the life of the source object.
        previous, self._screen_cache = self._screen_cache, {}
        self._previous_screen = previous
        for path in files:
            if self._screen(path):
                screened.append(path.relative_to(root).as_posix())
            else:
                kept.append(path)
        self.screened_paths = tuple(sorted(screened))
        if screened:
            log.warning(
                "code-text %s: left out %d of %d files that hold a literal secret: %s",
                self.repo, len(screened), len(files), ", ".join(self.screened_paths),
            )
        return kept

    def _screen(self, path: Path) -> list[str]:
        """Secret classes in one file, read through okg's contained reader.

        A file over `max_bytes_per_file` comes back as okg's size-and-digest
        token, which holds no text, so it screens clean (okg stores no text
        for it either). The cache is keyed by the content's SHA-256, so an
        edit is seen whatever the file's size and timestamp.
        """
        content = self._contained_probe_bytes(path)
        key = hashlib.sha256(content).digest()
        if key not in self._screen_cache:
            cached = self._previous_screen.get(key)
            self._screen_cache[key] = (
                cached if cached is not None else secret_classes(content.decode("utf-8", "replace"))
            )
        return self._screen_cache[key]
