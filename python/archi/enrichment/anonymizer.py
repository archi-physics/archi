"""Text anonymizer for connector emission hooks (anonymize_data).

Provenance: ported from archi v2 ``dev@28b977d1``,
``src/data_manager/collectors/utils/anonymizer.py`` (authors: Pietro
Lugato, Hasan Ozturk). This closes the ``anonymize_data`` cutover gate
noted in the :mod:`archi.sources.jira` docstring: the v2 JIRA collector
optionally ran ``Anonymizer().anonymize(issue_text)`` before storage;
in v3 the same text->text pass is applied at emission over the
connector text surface — ``jira_issue.attrs`` text fields and
``document_chunk.attrs["text"]`` — before embedding.

Changes from the v2 original:

- No v2 config plumbing: the ``data_manager.utils.anonymizer`` config
  block became constructor parameters, with the v2 base-config
  template's defaults inlined (nlp model ``en_core_web_sm``; JIRA
  ``[~user]`` mention pattern as the username default).
- spaCy is imported lazily on first use instead of at module import,
  so importing this module (and the enrichment package) never requires
  spaCy. ``nlp_model=None`` disables NER entirely — the regex passes
  (emails, usernames, greetings/sign-offs, markup author elements)
  still run, plus any caller-supplied ``known_names``.
- ``known_names``: connectors usually know author/assignee names from
  record metadata; these are always redacted, with or without NER.
- ``download_missing_model`` (default True, matching v2's
  download-on-missing behavior) can be set False to fail fast instead
  of downloading a model mid-ingest.

Hardening on top of the v2 behavior (circleback adversarial review,
see ``pact/changes/circleback-fixes/notes-enrichment.md``):

- Email redaction covers RFC-5322 local parts (``+`` tags, apostrophes,
  quoted local parts) and the percent-encoded ``%40`` form seen in
  URLs, so no fragment of the local part survives.
- NBSP (U+00A0 / ``&nbsp;``) and text-level HTML character references
  (``&#64;``, ``&amp;``, ...) are normalized before both discovery and
  replacement, so encoded occurrences of names/emails are redacted too.
  ``&lt;``/``&gt;`` are deliberately left encoded so markup structure
  is unchanged for the markup pass.
- NER-disabled mode additionally strips non-CDATA ``<dc:creator>``,
  ``mailto:`` anchor text, and TWiki ``-- Main.WikiWord`` signature
  lines.
- The greeting/sign-off line filters are tightened: greetings need an
  actual greeting word (the v2 ``^\\w+,`` rule deleted operational
  lines like "However, run 381000 ...") and must be short greeting
  lines (greeting word + at most four trailing words), and sign-offs
  must be the whole line (optionally followed by a short name), not a
  prefix.

Name replacement and text extraction for NER are kept verbatim.
"""
from __future__ import annotations

import bisect
import re
import unicodedata
from html import unescape
from collections.abc import Iterable, Sequence

# Generic markup patterns
_TAG_RE = re.compile(r"<[^>]+>")
_CDATA_RE = re.compile(r"<!\[CDATA\[|\]\]>")
_DC_CREATOR_RE = re.compile(
    r'(<dc:creator><!\[CDATA\[)[^\]]*(\]\]></dc:creator>)',
    re.IGNORECASE,
)
# <dc:creator>Name</dc:creator> without a CDATA wrapper. Disjoint from
# _DC_CREATOR_RE: CDATA content starts with "<", which [^<]* rejects.
_DC_CREATOR_PLAIN_RE = re.compile(
    r'(<dc:creator(?:\s[^>]*)?>)[^<]*(</dc:creator>)',
    re.IGNORECASE,
)
# <a href="mailto:jdoe@cern.ch">John Doe</a> → (removed): the email
# pass empties the href, but the anchor text is a name and must go too.
_DEFAULT_MARKUP_MAILTO_LINK_RE = re.compile(
    r'<a[^>]*href=["\']mailto:[^"\']*["\'][^>]*>.*?</a>',
    re.IGNORECASE | re.DOTALL,
)
# TWiki signature lines: "-- Main.JohnDoe - 2024-01-15" (also plain
# "-- Main.JohnDoe" and "-- TWiki.JohnDoe - 15 Jan 2024"). Applied in
# both the text and markup passes; the emptied line is dropped by the
# final blank-line filter.
_TWIKI_SIGNATURE_RE = re.compile(
    r'^[ \t]*-{2,}[ \t]*(?:Main|TWiki)\.[A-Z]\w*(?:[ \t]*-[ \t]*[^\n]*)?[ \t]*$',
    re.MULTILINE,
)
_ATTR_TEXT_RE = re.compile(r'(?:title|alt|creator|author)=["\']([^"\']+)["\']', re.IGNORECASE)
_CONTENT_TAG_RE = re.compile(
    r'<(?:p|li|td|description|title|dc:creator)[^>]*>(.*?)</(?:p|li|td|description|title|dc:creator)>',
    re.DOTALL | re.IGNORECASE,
)
# <a href="/author/Albert-Einstein">Albert-Einstein</a> → (removed)
_DEFAULT_GENERIC_MARKUP_USER_LINK_RE = re.compile(
    r'<a[^>]*href="[^"]*?/(?:Main|author|user|profile|members)/[^"]*"[^>]*>[^<]*</a>',
    re.IGNORECASE,
)
# Generic author link, like <a href="/author/Albert-Einstein">Albert-Einstein</a>
# <small itemprop="author">Stephenie Meyer</small>
# <span class="author">John Doe</span>
# <a rel="author" href="...">Jane Smith</a>
# <div class="post-author meta">Bob</div>
_DEFAULT_GENERIC_MARKUP_AUTHOR_ELEMENT_RE = re.compile(
    r'<[^>]*(?:itemprop=["\']author["\']|class=["\'][^"\']*\bauthor\b[^"\']*["\']|rel=["\']author["\'])[^>]*>[^<]*</[^>]+>',
    re.IGNORECASE,
)
# <a class="twikiLink" href="/twiki/bin//Main/JohnDoe">JohnDoe</a> → (removed)
_DEFAULT_MARKUP_TWIKI_USER_LINK_RE = re.compile(
    r'<a[^>]*href="[^"]*?/twiki/bin/\w+/Main/\w+"[^>]*>\w+</a>',
    re.IGNORECASE,
)
# <p>John</p> → (removed)
# <p><br>John Doe</p> → (removed)
_DEFAULT_MARKUP_SIGNOFF_TAG_RE = re.compile(
    r'<p>\s*(?:<br\s*/?>)?\s*[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,2}\s*</p>',
    re.IGNORECASE,
)
# ..atm<br>\nJohn</p> → ..atm</p>
# Thanks\John</description> → </description>
# Yours sincerely,\nJ.D.Doe]]> → ]]>
_DEFAULT_MARKUP_TRAILING_SIGNOFF_TAG_RE = re.compile(
    r'(?:'
        r'<br\s*/?>\s*\n?\s*'
        r'|(?:Thanks|Cheers|Best|Regards|HTH|Yours\s+sincerely)\s*,?\s*[\n\s]*'
    r')'
    r'[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,2}'
    r'\s*(?=</p>|</description>|\]\]>)',
    re.IGNORECASE,
)

# Defaults lifted from the v2 base-config template
# (src/cli/templates/base-config.yaml, dev@28b977d1), tightened per the
# circleback review: the v2 greeting rule ``^\w+,`` and the prefix-match
# sign-off rule deleted operational lines whole ("However, run 381000
# was affected badly.", "Regards to whoever fixed run 381000").
_DEFAULT_NLP_MODEL = "en_core_web_sm"
_DEFAULT_EXCLUDED_WORDS = ("John", "Jane", "Doe")
# Greeting lines must start with an actual greeting word (the bare
# ``^\w+,`` rule is gone) AND be short: greeting word plus at most four
# trailing words (mirroring the sign-off tail bound), so
# greeting-prefixed operational sentences ("Good morning update:
# transfers to T2_US_MIT stuck") survive.
_DEFAULT_GREETING_PATTERNS = (
    r"^(?:hi|hello|hey|greetings|dear|ciao|salut|hiya|howdy"
    r"|good\s+(?:morning|afternoon|evening|day))"
    r"(?:[\s,!]+[A-Za-z][\w'.-]*){0,4}[\s,.!]*$",
)
# Sign-off lines must be ONLY the sign-off phrase, optionally followed
# by punctuation and a short (<= 4 word) trailing name. A phrase that
# runs straight into more words ("Thank you note was filed as ...",
# "Best effort reprocessing ...") is content and survives.
_DEFAULT_SIGNOFF_PATTERNS = (
    r"^(?:yours\s+(?:sincerely|truly|faithfully)|sincerely(?:\s+yours)?"
    r"|(?:best|kind|warm)\s+regards|regards|best\s+wishes|best|cheers"
    r"|many\s+thanks|thanks(?:\s+(?:a\s+lot|in\s+advance|again))?"
    r"|thank\s+you|thx|hth|take\s+care|all\s+the\s+best)"
    r"(?:\s*[,.!;:-]+\s*(?:[A-Za-z][\w'.-]*(?:[ \t]+[A-Za-z][\w'.-]*){0,3})?)?"
    r"\s*[,.!]*\s*$",
    r"^\s*[-~]+\s*$",
)
# Local part: quoted form ("john doe"@...) or RFC-5322 atext plus dots
# and %-encoded octets — but not the URL-structural chars ``/ = ?`` so
# a surrounding URL's path/query is not swallowed. The separator also
# accepts the percent-encoded ``%40`` form (mail=john.doe%40cern.ch).
_DEFAULT_EMAIL_PATTERN = (
    r"(?:\"[^\"\n]+\"|[A-Za-z0-9.!#$%&'*+^_`{|}~-]+)(?:@|%40)[\w.-]+\.\w+"
)
_DEFAULT_USERNAME_PATTERN = r"\[~[^\]]+\]"

# Email-only redaction for source text, ported from okg-deployments
# ``cms/cms_sources/anonymizer.py`` (commit b25e36f06c), where the cms
# JIRA source applied it to every string it read.
#
# It decodes nothing: the encoded forms are recognized in place, and the
# only change to a string is the removal of each address-shaped token.
# Text with no such token comes back byte-identical, and entities around
# an address (``&lt;``, ``&amp;``) stay. A decode-first pass would rewrite
# ``AT&amp;T`` to ``AT&T`` in every string and so change the text, content
# hash and chunk id of chunks that hold no address.
#
# An address-shaped token is LOCAL SEP DOMAIN:
#
# - SEP: ``@``, fullwidth or small ``@`` (U+FF20, U+FE6B), URL ``%40`` or
#   ``%2540``, or ``&commat;`` / ``&#64;`` / ``&#x40;`` (any leading zeros,
#   ``;`` optional) behind any number of ``&amp;`` layers. A backslash
#   before ``@`` belongs to the separator: ``jdoe\@cern.ch`` is how Perl,
#   Doxygen and shell text escape an address.
# - LOCAL: a quoted string on one line, or a run of Unicode word
#   characters, combining marks, invisible characters (see
#   :func:`_is_invisible`), ``.!#$%&'*+^`{|}~-=``, ``&amp;`` layers and
#   encoded dots. ``/`` and ``?`` are excluded so a URL path is not
#   swallowed; ``=`` is included, so ``mail=`` before an address in a
#   query string goes with it. A quoted local part is found even when an
#   earlier match sits inside the quotes (``"x jdoe@a.b y"@cern.ch``): the
#   whole quoted string goes, together with the earlier match.
# - DOMAIN: word characters, marks, invisible characters, ``-``, and dots
#   (``.``, fullwidth ``.`` U+FF0E, a backslash-escaped ``\.`` as in Perl
#   regex text, or an encoded ``&#46;`` / ``&#x2e;`` / ``&period;``), with
#   at least one dot, ending in a label that contains
#   a letter. So ``numpy@1.26.4`` is a version pin, not an address.
#
# Matching runs in linear time. Separators are found by one regex pass;
# each local part is scanned leftwards only back to the previous
# separator or match, and each domain rightwards only up to the next
# character that cannot be in a domain, so every character is looked at
# a bounded number of times. (A single regex with a leftmost-start search
# is quadratic on a long run with no separator, and a lookbehind that
# avoids that misses an address that directly follows another one.)
_SEP_CORE_RE = re.compile(
    r"[@\uff20\ufe6b]|%(?:25)*40|commat;|#0*64;?|#x0*40;?", re.IGNORECASE
)
_DOT_TOKEN_RE = re.compile(
    r"&(?:amp;)*(?:#0*46;?|#x0*2e;?|period;)", re.IGNORECASE
)
_LOCAL_TOKEN_RE = re.compile(
    r"&(?:amp;)+|&(?:amp;)*(?:#0*46;?|#x0*2e;?|period;)", re.IGNORECASE
)
_LOCAL_PUNCT = frozenset(".!#$%&'*+^`{|}~-=")
_DOTS = frozenset(".\uff0e")
# ASCII control characters other than whitespace: C0 without tab, line
# feed, vertical tab, form feed, carriage return and the four information
# separators U+001C-U+001F (Python counts those as whitespace), plus DEL.
_ASCII_INVISIBLE = frozenset(
    chr(code)
    for code in range(0x80)
    if unicodedata.category(chr(code)) == "Cc" and not chr(code).isspace()
)


def _is_invisible(char: str) -> bool:
    """Whether ``char`` is a character a reader does not see.

    Every format character (Unicode category Cf: soft hyphen, zero-width
    space and joiners, word joiner, U+2061-U+2064, the bidi marks U+200E,
    U+200F and U+202A-U+202E, U+FEFF ...), every control character that is
    not whitespace (C0, DEL and C1), and U+FFFD, which a decoder writes in
    place of a byte it could not read. The address scans read these as part
    of an address, so ``jdoe<U+FEFF>x@cern.ch`` and ``jdoe<DEL>x@cern.ch``
    are removed whole instead of leaving ``jdoe`` behind.
    """
    if char < "\x80":
        return char in _ASCII_INVISIBLE
    if char == "\ufffd":
        return True
    category = unicodedata.category(char)
    return category == "Cf" or (category == "Cc" and not char.isspace())


# Every character that may be invisible: ASCII controls other than
# whitespace, and everything outside ASCII (checked with _is_invisible).
_MAYBE_INVISIBLE_RE = re.compile("[\x00-\x08\x0e-\x1b\x7f-\U0010ffff]")


def _is_word(char: str) -> bool:
    return (
        char.isalnum() or char == "_" or unicodedata.category(char)[0] == "M"
    )


def _is_local(char: str) -> bool:
    return _is_word(char) or char in _LOCAL_PUNCT or _is_invisible(char)


def _separator_start(text: str, match: re.Match, bound: int) -> int | None:
    """Start of the separator whose core ``match`` found, or None."""
    start = match.start()
    if text[start] == "@" and start - 1 >= bound and text[start - 1] == "\\":
        return start - 1
    if text[start] in "@\uff20\ufe6b%":
        return start
    # An entity core (``commat;``, ``#64;``) needs its ``&``, possibly
    # behind ``&amp;`` layers: ``&amp;amp;#64;``.
    while start - 4 >= bound and text.startswith("amp;", start - 4):
        start -= 4
    if start - 1 >= bound and text[start - 1] == "&":
        return start - 1
    return None


# How many backslash-escaped quotes (``\"``) a quoted local part is read
# across. Bounded so every quote is looked at a bounded number of times;
# past the bound the quote reached so far is used, which removes at least
# what the nearest quote alone would.
_QUOTE_HOPS = 16


def _escaped(text: str, index: int) -> bool:
    """Whether the character at ``index`` follows an odd run of backslashes."""
    run = 0
    while index - run - 1 >= 0 and text[index - run - 1] == "\\":
        run += 1
    return run % 2 == 1


def _quoted_local_start(text: str, sep: int) -> int | None:
    """Opening quote of a quoted local part that ends at ``sep``, or None.

    The nearest ``"`` before the closing one that is not escaped as ``\\"``,
    on the same line, with at least one character between the quotes. It
    is looked for without a lower bound: an earlier match inside the quotes
    (``"x jdoe@a.b y"@cern.ch``) does not hide the quoted local part that
    holds it. Linear over a whole text: the search from each closing quote
    stops at the quote before it, or at most :data:`_QUOTE_HOPS` escaped
    quotes further.
    """
    if sep < 2 or text[sep - 1] != '"':
        return None
    quote = text.rfind('"', 0, sep - 1)
    for _ in range(_QUOTE_HOPS):
        if quote <= 0 or not _escaped(text, quote):
            break
        earlier = text.rfind('"', 0, quote)
        if earlier == -1 or "\n" in text[earlier:quote]:
            break
        quote = earlier
    if quote != -1 and quote < sep - 2 and "\n" not in text[quote:sep]:
        return quote
    return None


def _local_start(text: str, sep: int, bound: int) -> int:
    """Leftmost start of the local part that ends at ``sep`` (``sep`` if none).

    A quoted local part may start before ``bound``; the caller merges it
    with the earlier matches it covers (see :func:`_add_span`). Local-part
    characters directly before the opening quote go with it
    (``jdoe"x"@cern.ch``), so no part of the name is left.
    """
    quote = _quoted_local_start(text, sep)
    if quote is not None and quote <= bound:
        return quote
    i = sep if quote is None else quote
    while i > bound:
        char = text[i - 1]
        if _is_local(char):
            i -= 1
        elif char == ";":
            amp = text.rfind("&", max(bound, i - 40), i)
            if amp == -1 or not _LOCAL_TOKEN_RE.fullmatch(text, amp, i):
                break
            i = amp
        else:
            break
    return i


def _domain_end(text: str, start: int) -> int | None:
    """End of the domain that starts at ``start``, or None if it is not one."""
    dots: list[tuple[int, int]] = []
    i, size = start, len(text)
    while i < size:
        char = text[i]
        if char in _DOTS:
            dots.append((i, i + 1))
            i += 1
        elif char == "\\" and i + 1 < size and text[i + 1] in _DOTS and i > start:
            dots.append((i, i + 2))
            i += 2
        elif _is_word(char) or char == "-" or _is_invisible(char):
            i += 1
        elif char == "&":
            token = _DOT_TOKEN_RE.match(text, i)
            if token is None:
                break
            dots.append((i, token.end()))
            i = token.end()
        else:
            break
    # The last label is the one after the right-most dot that still has a
    # letter; a trailing version-like label (``.4``) or dash is left out.
    end = i
    for dot_start, dot_end in reversed(dots):
        label_end = end
        while label_end > dot_end and not _is_word(text[label_end - 1]):
            label_end -= 1
        if dot_start > start and any(
            c.isalpha() for c in text[dot_end:label_end]
        ):
            return label_end
        end = dot_start
    return None


# Wrapping characters that may sit directly before a kept ``git@`` inside
# the scanned local part (```git@host```, ``'git@host'``, ``|git@host|``).
_GIT_WRAPPERS = frozenset("`'|{")
# A whole separator (core plus any leading ``&`` / ``&amp;`` layers)
# starting at a given position.
_SEP_AT_RE = re.compile(
    r"\\?@|[＠﹫]|%(?:25)*40|&(?:amp;)*(?:commat;|#0*64;?|#x0*40;?)",
    re.IGNORECASE,
)


def _runs_into_separator(text: str, start: int) -> bool:
    """Whether a local part could run from ``start`` into a separator.

    Scans rightwards over everything :func:`_local_start` would scan
    leftwards over (local-part characters and ``&amp;`` / encoded-dot
    tokens), plus the domain dots, and over a quoted string that ends
    directly at a separator. True if that scan reaches any separator
    ``_SEP_CORE_RE`` recognises, whether or not a domain follows it.
    """
    i, size = start, len(text)
    while i < size:
        if _SEP_AT_RE.match(text, i):
            return True
        char = text[i]
        if char == "&":
            # The longest token, as _local_start's fullmatch sees it:
            # ``&amp;#46;`` is one encoded dot, not ``&amp;`` then ``#46;``.
            ends = [
                token.end()
                for token in (
                    _DOT_TOKEN_RE.match(text, i),
                    _LOCAL_TOKEN_RE.match(text, i),
                )
                if token
            ]
            i = max(ends) if ends else i + 1
        elif char == '"':
            # ``git@host.jdoe"@cern.ch`` and ``git@host.jdoe"x y"@cern.ch``:
            # the quote may close or open a quoted local part.
            if _SEP_AT_RE.match(text, i + 1):
                return True
            # The closing quote may follow escaped ones: ``"x\"y"@cern.ch``.
            close = text.find('"', i + 1)
            for _ in range(_QUOTE_HOPS + 1):
                if close == -1 or "\n" in text[i:close]:
                    return False
                if _SEP_AT_RE.match(text, close + 1):
                    return True
                if not _escaped(text, close):
                    return False
                close = text.find('"', close + 1)
            return False
        elif _is_local(char) or char in _DOTS:
            i += 1
        else:
            return False
    return False


def _in_quoted_local(
    quotes: Sequence[int],
    text: str,
    local: int,
    end: int,
    openings: dict[int, int | None] | None = None,
) -> bool:
    """Whether ``text[local:end]`` sits in a quoted string that ends at a separator.

    ``quotes`` holds the position of every ``"`` in ``text``, in order.
    ``"x git@host y"@cern.ch`` is a quoted local part that holds the token.
    Two readings, either of which counts (fail closed):

    - the nearest quotes around the token, the second one followed by a
      separator; line breaks are not checked, so a quote pair across lines
      also counts;
    - the quoted local part the scan without the exemption finds
      (:func:`_quoted_local_start`), read across escaped quotes:
      ``"x git@host \\" y"@cern.ch``.

    A binary search keeps the first O(log n) per token; ``openings``
    caches the second per closing quote, so a long quoted string holding
    many tokens is read once.
    """
    k = bisect.bisect_left(quotes, end)
    if k == 0 or k == len(quotes):
        return False
    if quotes[k - 1] < local and _SEP_AT_RE.match(text, quotes[k] + 1):
        return True
    for close in quotes[k : k + _QUOTE_HOPS + 1]:
        if _SEP_AT_RE.match(text, close + 1):
            if openings is None:
                openings = {}
            if close not in openings:
                openings[close] = _quoted_local_start(text, close + 1)
            start = openings[close]
            return start is not None and start <= local
        if not _escaped(text, close):
            return False
    return False


def _is_git_account(
    text: str,
    sep: int,
    local: int,
    end: int,
    quotes: Sequence[int],
    openings: dict[int, int | None] | None = None,
) -> bool:
    """Whether the token ``text[local:end]`` is the literal ``git@`` account.

    The separator at ``sep`` must be a literal ``@``; the local part must be
    exactly ``git``, or ``git`` behind only wrapping characters (a backtick,
    ``'``, ``|`` or ``{``). So ``git@github.com``, ``ssh://git@host`` and
    ```git@host``` qualify, while ``john.git@cern.ch``, ``my-git@cern.ch``,
    ``jdoe'git@cern.ch``, ``url=git@host``, ``GIT@host`` and ``git%40host``
    do not.

    The token must also not run into another address: if a local part
    could continue from its end to any separator (``git@host.jdoe@cern.ch``,
    ``git@host.jdoe.1+x@cern.ch``, ``git@host.o'brien@cern.ch``,
    ``'git@host.jdoe'@cern.ch``, ``git@host.jdoe"x"@cern.ch``), its tail may
    be part of that address's local part, so it is not kept (fail closed).
    ``git@host:path``, ``git@host/path`` and ``git@host`` before a space
    stop the scan and are kept. Nor may the token sit inside a quoted local
    part (``"x git@host:y z"@cern.ch``); ``quotes`` lists every ``"`` in
    ``text`` and ``openings`` caches quoted local parts by closing quote.
    """
    if text[sep] != "@" or sep - 3 < local or text[sep - 3:sep] != "git":
        return False
    if any(char not in _GIT_WRAPPERS for char in text[local:sep - 3]):
        return False
    if _in_quoted_local(quotes, text, local, end, openings):
        return False
    return not _runs_into_separator(text, end)


def redact_email_addresses(text: str) -> str:
    """Remove whole email addresses, including tagged and encoded forms.

    ``john.doe+ops@cern.ch``, ``"john doe"@cern.ch``, ``über.müller@cern.ch``,
    ``jdoe&#64;cern.ch`` (and ``&#064;``, ``&#x0040;``, ``&#64`` without
    ``;``, ``&amp;#64;``), ``jdoe＠cern．ch``, ``bob&#64;cern&#46;ch``
    and the URL forms ``jdoe%40cern.ch`` and ``jdoe%2540cern.ch`` are
    removed outright (replaced by nothing, as in the cms source). A domain
    must end in a label with a letter, so ``numpy@1.26.4`` stays.

    Nothing is decoded: text that holds no address-shaped token (see the
    comment above for the exact shape) comes back byte-identical. Tokens
    that are address-shaped but not mail addresses, such as
    ``image@2x.png`` or the ``jdoe@lxplus.cern.ch`` of ``ssh
    jdoe@lxplus.cern.ch``, are removed too (fail closed).

    One exception (operator decision, 2026-09-28): the literal ``git@``
    account of a git host is kept whole, so ``git clone
    git@github.com:org/x.git`` and ``git@gitlab.cern.ch:group/y.git`` stay
    intact (see :func:`_is_git_account` for the exact form). A kept ``git@``
    token neither hides nor absorbs an address after it.
    """
    return redact_email_addresses_with_count(text)[0]


def redact_email_addresses_with_count(text: str) -> tuple[str, int]:
    """:func:`redact_email_addresses`, plus how many addresses it removed."""
    spans = email_address_spans(text)
    if not spans:
        return text, 0
    pieces: list[str] = []
    kept = 0  # text[kept:] is not yet copied to pieces
    for start, end in spans:
        pieces.append(text[kept:start])
        kept = end
    pieces.append(text[kept:])
    return "".join(pieces), len(spans)


def email_address_spans(text: str) -> list[tuple[int, int]]:
    """The ``(start, end)`` of every address :func:`redact_email_addresses`
    removes, in order and non-overlapping. Kept ``git@`` tokens are not
    spans (see :func:`_is_git_account`)."""
    spans: list[tuple[int, int]] = []
    bound = 0  # no local part may start before this
    quotes: list[int] | None = None  # every '"' position, built on first use
    openings: dict[int, int | None] = {}  # quoted local start by closing quote
    for core in _SEP_CORE_RE.finditer(text):
        if core.start() < bound:
            continue
        sep = _separator_start(text, core, bound)
        if sep is None:
            continue
        domain_start = core.end()
        end = _domain_end(text, domain_start)
        local = _local_start(text, sep, bound)
        if end is None or local == sep:
            bound = domain_start
            continue
        if quotes is None and text[sep] == "@":
            quotes = [match.start() for match in re.finditer('"', text)]
        if _is_git_account(text, sep, local, end, quotes or (), openings):
            # Kept in place; no later local part may start inside it.
            bound = end
            continue
        _add_span(spans, local, end)
        bound = end
    return spans


def _add_span(spans: list[tuple[int, int]], start: int, end: int) -> None:
    """Append ``(start, end)``, merging the earlier spans it overlaps.

    Only a quoted local part starts before the end of an earlier span
    (``"x jdoe@a.b y"@cern.ch``); the merged span covers both. Each span is
    merged away at most once, so this stays linear.
    """
    while spans and spans[-1][1] > start:
        start = min(start, spans.pop()[0])
    spans.append((start, end))


# Spelled-out ("anti-spam") address forms, which have no ``@`` at all and
# so are outside redact_email_addresses. Operator rule (Jason,
# 2026-09-28): remove the bracketed and parenthesised forms, and leave the
# free-prose "at ... dot" forms alone. Measured on the cms-kb TWiki
# snapshot (2026-09-28, 1,261 candidate topics of 43,888): bracketed
# ``jdoe[AT]cern.ch`` / ``jdoe(at)cern(dot)ch`` (37), glued
# ``john.doe_at_cern.ch`` (6) and ``NOSPAM`` insertions (8) are removed;
# spaced ``john.doe at cern.ch`` / ``jdoe AT cern DOT ch`` (123) are kept.
#
# - Bracketed ``(at)`` ``[at]`` ``{at}`` ``<at>`` (any case, optional
#   spaces) between a user name and a mail domain. The domain has a dot
#   (``.``, a bracketed ``(dot)`` / ``[DOT]``, or a spaced ``dot``), labels
#   of letters, digits and ``-`` only (no ``_``), and a last label that is
#   two letters (a country code) or a common generic top-level domain
#   (``com``, ``org``, ``edu``, ``gov``, ``net`` ...; see
#   :data:`_OBF_GENERIC_TLDS`), with nothing domain- or path-like after it.
#   The user name is not a CMS site name (``T2_CH_CERN``). So
#   ``Run2 [at] 13.6TeV``, ``site T2_CH_CERN [at] cern.ch``, ``dataset (at)
#   T2_US_MIT.mit.edu``, ``WebHome [AT] Twiki.Main`` and code such as
#   ``read[at]self.buf.data`` or ``cfg(at)process.source.fileNames`` stay.
# - A spaced ``" at "`` (any case) before a domain of the same shape with
#   at least one bracketed ``(dot)`` / ``[DOT]``: ``jdoe at cern(dot)ch``.
# - Free-prose word separators ``" at "`` / ``" AT "`` with only ``.`` or
#   a spaced ``" dot "`` / ``" DOT "`` as the dot are never removed:
#   ``john.doe at cern.ch``, ``jdoe AT cern DOT ch`` and "based at
#   cern.ch" stay byte for byte.
# - Glued separators ``_at_``, ``-at-``, ``_AT_`` and ``_NOSPAM_AT_`` (not
#   covered by the operator rule; removed, and flagged for review) count
#   only before exactly ``cern.ch``, ``fnal.gov`` or ``gmail.com`` (``.``,
#   ``(dot)``, ``_dot_`` or a spaced ``DOT`` as the dot) or a plain-dotted
#   ``.edu`` or ``.gov`` domain, with nothing domain- or path-like after it.
#   ``x_2016_at_13TeV.root`` and ``x-at-2.3`` stay.
# - A token that carries ``NOSPAM`` (any case) and, with it taken out,
#   ends like a mail domain (``.ch``, ``.edu``, ``.gov``, ``.org``,
#   ``.com``, ``DOTch`` ... or ``cernch``) is removed whole, with a
#   preceding ``name AT`` / ``name_at_`` / ``name@`` part (``jdoe@cernNOSPAMch``,
#   whose domain has no dot for the @ pass): the ``NOSPAM`` marker says
#   the token is an address. ``NOSPAM`` alone, in a name such as
#   ``NoSpamFilter``, or with nothing before the domain ending
#   (``nospam.ch``, ``NOSPAM.org``) stays: it must sit inside an address.
#
# Invisible characters (see :func:`_is_invisible`) are read as absent: the
# patterns match the text with them taken out, and a match removes the
# invisible characters inside it too. So ``jdoe<DEL>[at]cern.ch`` and
# ``jd<U+FEFF>oe[at]cern.ch`` go whole.
#
# Every repetition is bounded or unambiguous, so matching is linear.
# Like redact_email_addresses this decodes nothing and only removes the
# matched token, so text with no match comes back byte-identical.
# Accepted over-removal: code shaped like ``f(at)obj.ch`` or
# ``f(at)obj.attr.org`` (a last label that is a top-level domain) goes.
# A user name is not a CMS site name (T0_CH_CERN, T2_US_MIT, T3_IT_Trieste).
_OBF_LOCAL = r"(?<![\w.+-])(?!(?-i:T\d_[A-Z]{2}_))[\w.+-]{1,64}"
_OBF_BRACKET_DOT = r"\s?[(\[{<]\s?dot\s?[)\]}>]\s?"
_OBF_END = r"(?![\w-]|[./:@(][\w-])"
# After a bracketed ``[at]`` any dot counts, bracketed or spelled out
# (``jdoe[at]cern dot ch``): the brackets already mark the token.
_OBF_ANY_DOT = r"(?:\.|" + _OBF_BRACKET_DOT + r"| dot )"
_OBF_BRACKET_AT = r"\s?[(\[{<]\s?at\s?[)\]}>]\s?"
# A mail-domain label: letters, digits and "-", never "_" (T2_US_MIT is a
# site name, self.buf_x code).
_OBF_LABEL = r"(?:[^\W_]|-)+"
#: Generic top-level domains a spelled-out address may end in, besides any
#: two-letter country code. Code attribute chains (``self.buf.data``,
#: ``process.source.fileNames``) and wiki names (``Twiki.Main``) end in
#: words that are not on this list.
_OBF_GENERIC_TLDS = (
    "com", "org", "net", "edu", "gov", "mil", "int", "info", "biz", "name",
    "pro", "aero", "coop", "museum", "mobi", "asia", "cat", "jobs", "tel",
    "travel", "eus", "xyz", "dev", "app", "cloud", "online", "site", "tech",
    "email",
)
_OBF_TLD = r"(?:[^\W\d_]{2}|" + "|".join(_OBF_GENERIC_TLDS) + r")"
_OBF_STRONG_RE = re.compile(
    _OBF_LOCAL
    + _OBF_BRACKET_AT
    + _OBF_LABEL + r"(?:" + _OBF_ANY_DOT + _OBF_LABEL + r")*"
    + _OBF_ANY_DOT + _OBF_TLD
    + _OBF_END,
    re.IGNORECASE,
)
# A spaced " at " is prose unless the domain after it has a bracketed
# ``(dot)`` / ``[DOT]``: ``jdoe at cern(dot)ch`` goes, ``jdoe at cern.ch``
# and ``jdoe AT cern DOT ch`` stay.
_OBF_SPACED_BRACKET_DOT_RE = re.compile(
    _OBF_LOCAL
    + r" at "
    + _OBF_LABEL + r"(?:\." + _OBF_LABEL + r")*" + _OBF_BRACKET_DOT
    + r"(?:" + _OBF_LABEL + _OBF_ANY_DOT + r")*"
    + _OBF_TLD
    + _OBF_END,
    re.IGNORECASE,
)
_OBF_GLUED_SEP = r"(?:_at_|-at-|_AT_|_NOSPAM_AT_)"
_OBF_WORD_DOT = r"(?:\.|" + _OBF_BRACKET_DOT + r"| (?:DOT|dot) |_(?:DOT|dot)_)"
_OBF_GLUED_MAIL_RE = re.compile(
    _OBF_LOCAL
    + _OBF_GLUED_SEP
    + r"(?i:cern" + _OBF_WORD_DOT + r"ch|fnal" + _OBF_WORD_DOT + r"gov"
    + r"|gmail" + _OBF_WORD_DOT + r"com)"
    + _OBF_END
)
# Glued separators also count before any plain-dotted domain ending in
# .edu or .gov (john.doe_at_physics.ucsd.edu).
_OBF_GLUED_RE = re.compile(
    _OBF_LOCAL
    + _OBF_GLUED_SEP
    + r"[\w-]{1,63}(?:\.[\w-]{1,63}){0,5}\.(?i:edu|gov)"
    + _OBF_END
)
_OBF_NOSPAM_RE = re.compile(
    r"(?<![\w.+-])(?:[\w.+-]{1,64}(?: at | AT |_at_|_AT_|@))?"
    r"[\w.+-]*NOSPAM[\w.+-]*",
    re.IGNORECASE,
)
_OBF_NOSPAM_DOMAIN_END_RE = re.compile(
    r"(?:(?:\.|dot|_)(?:ch|edu|gov|org|com)|cernch)\.?$", re.IGNORECASE
)


def _nospam_token(match: re.Match) -> str:
    """``""`` if the NOSPAM token is an address (removed), else the token."""
    rest = re.sub("nospam", "", match.group(0), flags=re.IGNORECASE)
    ending = _OBF_NOSPAM_DOMAIN_END_RE.search(rest)
    # NOSPAM must sit inside an address: something name-like has to come
    # before the domain ending, so a bare "nospam.ch" is not one.
    if ending and any(c.isalnum() for c in rest[: ending.start()]):
        return ""
    return match.group(0)


def redact_obfuscated_email_addresses(text: str) -> str:
    """Remove spelled-out addresses: ``jdoe[AT]cern.ch``, ``jdoe(at)cern(dot)ch``,
    ``john.doe_at_cern.ch`` and ``NOSPAM`` forms. Free-prose forms such as
    ``john.doe at cern.ch`` and ``jdoe AT cern DOT ch`` are kept. See the
    comment above for the exact rules.

    Run it after :func:`redact_email_addresses`; it only removes matched
    tokens, so text without one is returned byte-identical.
    """
    return redact_obfuscated_email_addresses_with_count(text)[0]


def redact_obfuscated_email_addresses_with_count(text: str) -> tuple[str, int]:
    """:func:`redact_obfuscated_email_addresses`, plus how many tokens it removed."""
    spans, removed = _obfuscated_spans_and_count(text)
    if not spans:
        return text, 0
    pieces: list[str] = []
    kept = 0
    for start, end in spans:
        pieces.append(text[kept:start])
        kept = end
    pieces.append(text[kept:])
    return "".join(pieces), removed


def obfuscated_email_address_spans(text: str) -> list[tuple[int, int]]:
    """The ``(start, end)`` ranges of ``text`` that
    :func:`redact_obfuscated_email_addresses` removes, in order and merged
    where they touch."""
    return _obfuscated_spans_and_count(text)[0]


def _obfuscated_spans_and_count(text: str) -> tuple[list[tuple[int, int]], int]:
    """Run the obfuscated-address passes in order, each over the text the
    previous ones left, and map every removed range back to ``text``.

    ``positions[i]`` is the index in ``text`` of character ``i`` of the
    current text, so a later match that closes over an earlier removal maps
    to one range covering both.

    The passes start from ``text`` with its invisible characters taken out
    (:func:`_is_invisible`), so one inside a token does not split it; the
    range a match maps back to covers the invisible characters inside it.
    """
    hidden = {
        match.start()
        for match in _MAYBE_INVISIBLE_RE.finditer(text)
        if _is_invisible(match.group())
    }
    if hidden:
        positions = [i for i in range(len(text)) if i not in hidden]
        current = "".join(text[i] for i in positions)
    else:
        current = text
        positions = list(range(len(text)))
    removed_mask = bytearray(len(text))
    count = 0
    passes: tuple[tuple[re.Pattern, bool], ...] = (
        (_OBF_STRONG_RE, False),
        (_OBF_SPACED_BRACKET_DOT_RE, False),
        (_OBF_GLUED_MAIL_RE, False),
        (_OBF_GLUED_RE, False),
        (_OBF_NOSPAM_RE, True),
    )
    for pattern, conditional in passes:
        matches = [
            match
            for match in pattern.finditer(current)
            if not conditional or not _nospam_token(match)
        ]
        if not matches:
            continue
        count += len(matches)
        drop = bytearray(len(current))
        for match in matches:
            if match.end() > match.start():
                first = positions[match.start()]
                last = positions[match.end() - 1]
                removed_mask[first : last + 1] = b"\x01" * (last + 1 - first)
                drop[match.start() : match.end()] = b"\x01" * (match.end() - match.start())
        current = "".join(c for c, d in zip(current, drop, strict=True) if not d)
        positions = [p for p, d in zip(positions, drop, strict=True) if not d]
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(removed_mask):
        if removed_mask[index]:
            start = index
            while index < len(removed_mask) and removed_mask[index]:
                index += 1
            spans.append((start, index))
        else:
            index += 1
    return spans, count


# Text-level HTML character references decoded before redaction. The
# numeric-reference decoder below deliberately keeps &lt;/&gt; (and any
# reference that would decode to "<" or ">") encoded, so decoding never
# creates or breaks markup structure for the markup pass.
_NUMERIC_ENTITY_RE = re.compile(r"&#(x[0-9a-fA-F]{1,6}|[0-9]{1,7});")
_SAFE_NAMED_ENTITIES = {
    "&nbsp;": "\u00a0",
    "&amp;": "&",
    "&apos;": "'",
    "&quot;": '"',
    "&commat;": "@",
}


def _normalize_encodings(text: str) -> str:
    """Decode NBSP and text-level entities so encoded PII is caught.

    ``John&nbsp;Doe`` / ``John\\xa0Doe`` become ``John Doe`` and
    ``jdoe&#64;cern.ch`` becomes ``jdoe@cern.ch`` before the discovery,
    email, and replacement passes — which then all see the same string.

    Normalization iterates to a fixpoint (bounded at 3 passes) so
    double-encoded forms like ``jdoe&amp;#64;cern.ch`` — which a single
    pass only peels to ``jdoe&#64;cern.ch`` — are fully decoded too.
    References that would decode to ``<`` or ``>`` stay encoded on
    every pass, so markup structure is never created or broken.
    """

    def _decode(match: re.Match) -> str:
        ref = match.group(1)
        code = int(ref[1:], 16) if ref[0] in "xX" else int(ref)
        try:
            char = chr(code)
        except (ValueError, OverflowError):
            return match.group(0)
        if char in "<>":
            return match.group(0)
        return char

    for _ in range(3):
        previous = text
        text = _NUMERIC_ENTITY_RE.sub(_decode, text)
        for entity, char in _SAFE_NAMED_ENTITIES.items():
            text = text.replace(entity, char)
        text = text.replace("\u00a0", " ")
        if text == previous:
            break
    return text


class Anonymizer:
    """Redact names, emails, usernames, greetings, and sign-offs."""

    def __init__(
        self,
        *,
        nlp_model: str | None = _DEFAULT_NLP_MODEL,
        excluded_words: Iterable[str] = _DEFAULT_EXCLUDED_WORDS,
        greeting_patterns: Sequence[str] = _DEFAULT_GREETING_PATTERNS,
        signoff_patterns: Sequence[str] = _DEFAULT_SIGNOFF_PATTERNS,
        email_pattern: str = _DEFAULT_EMAIL_PATTERN,
        username_pattern: str = _DEFAULT_USERNAME_PATTERN,
        known_names: Iterable[str] = (),
        download_missing_model: bool = True,
    ) -> None:
        self._nlp_model = nlp_model
        self._download_missing_model = download_missing_model
        self._nlp = None
        self._nlp_loaded = nlp_model is None

        self.EXCLUDED_WORDS = set(excluded_words)
        self.GREETING_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in greeting_patterns]
        self.SIGNOFF_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in signoff_patterns]
        self.EMAIL_PATTERN = re.compile(email_pattern)
        self.USERNAME_PATTERN = re.compile(username_pattern)
        self.KNOWN_NAMES = {
            name.strip() for name in known_names if name and name.strip()
        }

    def _load_nlp(self):
        if self._nlp_loaded:
            return self._nlp
        import spacy

        try:
            self._nlp = spacy.load(self._nlp_model)
        except OSError:
            if not self._download_missing_model:
                raise
            spacy.cli.download(self._nlp_model)
            self._nlp = spacy.load(self._nlp_model)
        self._nlp_loaded = True
        return self._nlp

    def _discover_names(self, text: str) -> set:
        """NER (when enabled) plus known names present in the text."""
        names = {name for name in self.KNOWN_NAMES if name}
        nlp = self._load_nlp()
        if nlp is None:
            return names
        doc = nlp(text)
        names |= {
            ent.text for ent in doc.ents
            if ent.label_ == "PERSON" and ent.text not in self.EXCLUDED_WORDS
        }
        return names

    def _discover_names_markup(self, markup: str) -> set:
        # Full document: names with surrounding context (catches CDATA)
        full_text = self._extract_text(markup)
        names = self._discover_names(full_text)
        # Per-chunk: focused paragraphs (catches standalone names in <p>)
        for chunk in self._extract_text_chunks(markup):
            names |= self._discover_names(chunk)
        return names

    def anonymize(self, text: str) -> str:
        """
        Anonymize names, emails, usernames, greetings, and sign-offs from the text.
        """
        # Normalize NBSP/entity encodings first so discovery, the email
        # pass, and replacement all see the same decoded string.
        text = _normalize_encodings(text)
        names_to_replace = self._discover_names(text)

        # Remove email addresses and usernames
        text = self.EMAIL_PATTERN.sub("", text)
        text = self.USERNAME_PATTERN.sub("", text)

        text = _TWIKI_SIGNATURE_RE.sub("", text)
        text = self._strip_greetings_signoffs(text)
        return self._replace_names(text, names_to_replace)

    def anonymize_markup(self, markup: str) -> str:
        """
        Anonymize names, emails, usernames, greetings, and sign-offs from the markup.
        including html, rss, and other markup formats. (especially twiki and discourse markup)
        """
        # Normalize NBSP/entity encodings first so discovery and every
        # sub below see the same decoded string ("John&nbsp;Doe" and
        # "jdoe&#64;cern.ch" are redacted like their plain forms).
        # &lt;/&gt; stay encoded, so tag structure is unchanged.
        markup = _normalize_encodings(markup)
        names_to_replace = self._discover_names_markup(markup)
        # Remove email addresses and usernames
        markup = self.EMAIL_PATTERN.sub("", markup)
        markup = self.USERNAME_PATTERN.sub("", markup)
        markup = _DC_CREATOR_RE.sub(r'\1\2', markup)
        markup = _DC_CREATOR_PLAIN_RE.sub(r'\1\2', markup)
        markup = _DEFAULT_MARKUP_MAILTO_LINK_RE.sub("", markup)
        markup = _DEFAULT_GENERIC_MARKUP_AUTHOR_ELEMENT_RE.sub("", markup)
        markup = _DEFAULT_GENERIC_MARKUP_USER_LINK_RE.sub("", markup)
        markup = _DEFAULT_MARKUP_SIGNOFF_TAG_RE.sub("", markup)
        markup = _DEFAULT_MARKUP_TRAILING_SIGNOFF_TAG_RE.sub("", markup)
        markup = _DEFAULT_MARKUP_TWIKI_USER_LINK_RE.sub("", markup)
        markup = _TWIKI_SIGNATURE_RE.sub("", markup)
        markup = self._strip_greetings_signoffs(markup)
        return self._replace_names(markup, names_to_replace)

    def _strip_greetings_signoffs(self, text: str) -> str:
        lines = text.splitlines()
        filtered = []
        for line in lines:
            stripped = line.strip()
            if any(p.match(stripped) for p in self.GREETING_PATTERNS):
                continue
            if any(p.match(stripped) for p in self.SIGNOFF_PATTERNS):
                continue
            filtered.append(line)
        return "\n".join(filtered)

    def _replace_names(self, text: str, names: set) -> str:
        for name in sorted(names, key=len, reverse=True):
            text = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE).sub("", text)
        return "\n".join(line for line in text.splitlines() if line.strip())

    def _extract_text(self, markup: str) -> str:
        """Strip markup to plain text for NER. Format-agnostic."""
        attrs = " ".join(_ATTR_TEXT_RE.findall(markup))
        clean = _CDATA_RE.sub(" ", markup)
        clean = _TAG_RE.sub(" ", clean)
        clean = unescape(clean)
        return re.sub(r"\s+", " ", f"{clean} {attrs}").strip()

    def _extract_text_chunks(self, markup: str) -> list:
        chunks = []
        # Text content from tags
        for match in _CONTENT_TAG_RE.finditer(markup):
            inner = _CDATA_RE.sub(" ", match.group(1))
            clean = _TAG_RE.sub(" ", inner)
            clean = unescape(clean).strip()
            if clean:
                chunks.append(clean)
        # Text from attributes
        attr_text = " ".join(_ATTR_TEXT_RE.findall(markup))
        if attr_text.strip():
            chunks.append(attr_text.strip())
        return chunks
