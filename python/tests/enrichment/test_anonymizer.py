"""task.w2.enrichment — anonymizer text->text pass, offline.

Runs with NER disabled (``nlp_model=None``) so no spaCy model is
needed: the regex passes plus ``known_names`` cover the connector
emission hook (jira/docs ``anonymize_data``) deterministically.
"""
import pytest

from archi.enrichment.anonymizer import (
    Anonymizer,
    redact_email_addresses,
    redact_obfuscated_email_addresses,
)


def _anonymizer(**kwargs):
    kwargs.setdefault("nlp_model", None)
    return Anonymizer(**kwargs)


def test_redacts_email_name_and_jira_mention():
    an = _anonymizer(known_names=["John Doe"])
    text = (
        "Hi all,\n"
        "John Doe saw transfer failures at T2_US_MIT.\n"
        "Contact jdoe@cern.ch or [~jdoe] about run 381000.\n"
        "Cheers,\n"
        "the ops team\n"
    )
    out = an.anonymize(text)
    assert "jdoe@cern.ch" not in out
    assert "John Doe" not in out
    assert "[~jdoe]" not in out
    # Greeting and sign-off lines are stripped.
    assert "Hi all" not in out
    assert "Cheers" not in out
    # Operational content survives.
    assert "T2_US_MIT" in out
    assert "run 381000" in out


def test_known_names_case_insensitive_word_bounded():
    an = _anonymizer(known_names=["Jane Roe"])
    out = an.anonymize("jane roe and JaneRoeography met.")
    assert "jane roe" not in out.lower().replace("janeroeography", "")
    assert "JaneRoeography" in out  # word boundary respected


def test_markup_pass_strips_author_constructs():
    # Content paragraphs must exceed three words: the ported v2 markup
    # signoff heuristic removes any <p> of up to three capitalized-ish
    # words (case-insensitive), by design.
    an = _anonymizer()
    markup = (
        '<p>Transfer failures were observed at T2_US_MIT overnight</p>'
        '<a class="twikiLink" href="/twiki/bin/view/Main/JohnDoe">JohnDoe</a>'
        '<span class="author">Jane Roe</span>'
        '<p>Please contact jdoe@cern.ch for any further details</p>'
    )
    out = an.anonymize_markup(markup)
    assert "JohnDoe" not in out
    assert "Jane Roe" not in out
    assert "jdoe@cern.ch" not in out
    assert "Transfer failures were observed at T2_US_MIT overnight" in out
    assert "for any further details" in out


def test_construction_never_imports_spacy_eagerly():
    # Default model configured, but nothing loads until first use.
    an = Anonymizer()
    assert an._nlp is None and an._nlp_loaded is False
    # NER-disabled instances never load a model at all.
    off = _anonymizer()
    assert off._nlp is None and off._nlp_loaded is True


# ---------------------------------------------------------------------------
# Leak corpus (circleback adversarial review). Each block pins one
# previously-leaking format plus the neighboring previously-passing
# behavior, so a regression in either direction fails loudly.
# ---------------------------------------------------------------------------


def test_email_local_part_never_leaks_fragments():
    # Finding 1: the RFC-simple default pattern left local-part
    # fragments behind ("john.doe+", "o'").
    an = _anonymizer()
    cases = {
        "Contact john.doe+ops@cern.ch about the transfer.": (
            "john",
            "doe",
            "+ops",
        ),
        "Contact o'brien@cern.ch about the transfer.": ("o'", "brien"),
        "See https://x.test/page?mail=john.doe%40cern.ch for run 381000": (
            "john",
            "doe",
            "%40",
        ),
        'Reach "john doe"@cern.ch if the drain stalls.': ("john", "doe"),
        "Plain jdoe@cern.ch still redacted.": ("jdoe",),
    }
    for text, fragments in cases.items():
        out = an.anonymize(text)
        for fragment in fragments:
            assert fragment not in out, (text, out)
    # Operational context around the address survives.
    out = an.anonymize("Contact john.doe+ops@cern.ch about run 381000.")
    assert "run 381000" in out
    out = an.anonymize("See https://x.test/page?mail=john.doe%40cern.ch now")
    assert "https://x.test/page?mail=" in out


def test_encoded_and_nbsp_variants_redacted():
    # Finding 2: discovery ran on unescaped text but replacement on the
    # raw input, so NBSP/entity-encoded occurrences survived.
    an_names = _anonymizer(known_names=["John Doe"])
    out = an_names.anonymize("Assigned to John\xa0Doe for run 381000.")
    assert "John" not in out and "Doe" not in out
    assert "run 381000" in out

    out = an_names.anonymize_markup(
        "<p>Report prepared by John&nbsp;Doe covering all transfer failures seen</p>"
    )
    assert "John" not in out and "Doe" not in out
    assert "transfer failures" in out

    an = _anonymizer()
    out = an.anonymize_markup(
        "<p>Contact jdoe&#64;cern.ch for details on the failed transfers</p>"
    )
    assert "jdoe" not in out
    assert "failed transfers" in out

    out = an.anonymize("Contact jdoe&#64;cern.ch about run 381000.")
    assert "jdoe" not in out
    out = an.anonymize("Contact jdoe&#x40;cern.ch about run 381000.")
    assert "jdoe" not in out

    # Escaped angle brackets stay escaped: normalization must not
    # create or break markup structure.
    out = an.anonymize_markup(
        "<p>Escaped &lt;tag&gt; text stays escaped in the output here</p>"
    )
    assert "&lt;tag&gt;" in out

    # Double-encoded forms: normalization iterates to a bounded
    # fixpoint, so one &amp;-wrapping layer cannot smuggle PII through.
    out = an.anonymize("Contact jdoe&amp;#64;cern.ch about run 381000.")
    assert "jdoe" not in out
    assert "run 381000" in out
    out = an_names.anonymize_markup(
        "<p>Report prepared by John&amp;nbsp;Doe covering all transfer failures seen</p>"
    )
    assert "John" not in out and "Doe" not in out
    assert "transfer failures" in out


def test_ner_off_author_shapes_redacted():
    # Finding 3: NER-disabled mode leaked authors outside the four
    # hardcoded markup shapes.
    an = _anonymizer()

    # dc:creator without CDATA.
    out = an.anonymize_markup(
        "<item><dc:creator>Hasan Ozturk</dc:creator>"
        "<description>Disk full at T2</description></item>"
    )
    assert "Hasan" not in out and "Ozturk" not in out
    assert "Disk full at T2" in out

    # dc:creator with CDATA still redacted (previously passing).
    out = an.anonymize_markup(
        "<item><dc:creator><![CDATA[Jane Roe]]></dc:creator>"
        "<description>Transfer backlog cleared at the T1 site</description></item>"
    )
    assert "Jane Roe" not in out
    assert "Transfer backlog cleared" in out

    # mailto anchor text.
    out = an.anonymize_markup(
        '<p>Ping <a href="mailto:jdoe@cern.ch">John Doe</a>'
        " when the drain of the pool completes</p>"
    )
    assert "John Doe" not in out and "jdoe" not in out
    assert "drain of the pool completes" in out

    # TWiki signature line, text pass.
    out = an.anonymize("Disk pool drained at T2_US_MIT.\n-- Main.JohnDoe - 2024-01-15")
    assert "JohnDoe" not in out
    assert "T2_US_MIT" in out

    # TWiki signature line, markup pass, non-ISO date.
    out = an.anonymize_markup(
        "Some twiki topic body mentioning run 381000 here\n-- Main.JaneRoe - 15 Jan 2024\n"
    )
    assert "JaneRoe" not in out
    assert "run 381000" in out


def test_operational_lines_survive_greeting_signoff_filters():
    # Finding 4: "^\w+," deleted any line whose first word had a
    # trailing comma; sign-off patterns prefix-matched content lines.
    an = _anonymizer()
    text = (
        "However, run 381000 was affected badly.\n"
        "Note, T2_US_MIT needs a re-run of the workflow.\n"
        "Thank you note was filed as CMSCOMPPR-1.\n"
        "Regards to whoever fixed run 381000.\n"
        "Best effort reprocessing is enabled.\n"
    )
    out = an.anonymize(text)
    assert "However, run 381000 was affected badly." in out
    assert "Note, T2_US_MIT needs a re-run of the workflow." in out
    assert "Thank you note was filed as CMSCOMPPR-1." in out
    assert "Regards to whoever fixed run 381000." in out
    assert "Best effort reprocessing is enabled." in out


def test_greeting_prefixed_operational_lines_survive():
    # Reviewer delta: the expanded greeting word-list must only delete
    # SHORT greeting lines (greeting word + at most four trailing
    # words); greeting-prefixed operational sentences survive.
    an = _anonymizer()
    text = (
        "Good morning update: transfers to T2_US_MIT stuck\n"
        "Hello world example output was attached to run 381000 report\n"
        "Good morning all,\n"
        "Hey folks,\n"
    )
    out = an.anonymize(text)
    assert "Good morning update: transfers to T2_US_MIT stuck" in out
    assert "run 381000 report" in out
    assert "Good morning all" not in out
    assert "Hey folks" not in out


def test_real_greetings_and_signoffs_still_stripped():
    # Finding 4, other direction: the tightened defaults must still
    # strip actual greeting/sign-off lines.
    an = _anonymizer()
    text = (
        "Hi all,\n"
        "Dear colleagues,\n"
        "Good morning team,\n"
        "The transfer failed overnight.\n"
        "Thanks in advance,\n"
        "Best regards, John\n"
        "Cheers,\n"
        "Yours sincerely,\n"
        "-- \n"
    )
    out = an.anonymize(text)
    assert out.strip() == "The transfer failed overnight."


# --- redact_email_addresses: the email-only pass sources call ---------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail john.doe@cern.ch now", "mail  now"),
        ("tagged john.doe+ops@cern.ch", "tagged "),
        ('quoted "john doe"@cern.ch.', "quoted ."),
        ("encoded jdoe&#64;cern.ch", "encoded "),
        ("hex jdoe&#x40;cern.ch", "hex "),
        ("named jdoe&commat;cern.ch", "named "),
        ("double jdoe&amp;#64;cern.ch", "double "),
        ("url ?mail=john.doe%40cern.ch&x=1", "url ?&x=1"),
        ("host cmsweb.cern.ch stays", "host cmsweb.cern.ch stays"),
    ],
)
def test_redact_email_addresses_forms(text, expected):
    assert redact_email_addresses(text) == expected


def test_redact_email_addresses_changes_nothing_else():
    # Unlike Anonymizer.anonymize: no greeting, sign-off, name, NBSP or
    # general entity pass, so text without an address is returned as is.
    text = "Hi,\nJohn Doe &lt;b&gt; run 381000\nThanks"
    assert redact_email_addresses(text) == text


# Address forms the first version let through whole or in part. Each one
# must be removed entirely: no prefix of the local part may survive.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("non-ascii \u00fcber.m\u00fcller@cern.ch end", "non-ascii  end"),
        # NFD: "u" followed by a combining diaeresis (U+0308).
        ("nfd u\u0308ber.mu\u0308ller@cern.ch end", "nfd  end"),
        ("nfd domain jdoe@ce\u0301rn.ch end", "nfd domain  end"),
        ("zero-padded jdoe&#064;cern.ch end", "zero-padded  end"),
        ("hex zero-padded jdoe&#x0040;cern.ch end", "hex zero-padded  end"),
        ("no semicolon jdoe&#64cern.ch end", "no semicolon  end"),
        ("hex no semicolon jdoe&#x40cern.ch end", "hex no semicolon  end"),
        ("url ?mail=john.doe%40cern.ch&x=1", "url ?&x=1"),
        ("double url ?mail=john.doe%2540cern.ch&x=1", "double url ?&x=1"),
        ("fullwidth jdoe\uff20cern.ch end", "fullwidth  end"),
        ("small at jdoe\ufe6bcern.ch end", "small at  end"),
        ("amp in local jdoe&amp;x@cern.ch end", "amp in local  end"),
        ("double commat jdoe&amp;commat;cern.ch end", "double commat  end"),
    ],
)
def test_redact_email_addresses_removes_whole_address(text, expected):
    assert redact_email_addresses(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "AT&amp;T ok",
        "R&amp;D &commat; CERN, ref &#64; and &#x40; alone",
        "double &amp;amp; stays, run 381000",
        "Hi,\nJohn Doe &lt;b&gt; 100% done \uff20 home",
    ],
)
def test_redact_email_addresses_leaves_address_free_text_byte_identical(text):
    # No address, no change: entities are never decoded, so the chunk
    # text (and its content hash and chunk id) is exactly the input.
    assert redact_email_addresses(text) == text


def test_redact_email_addresses_keeps_entities_around_an_address():
    # Only the address goes; the surrounding entities stay encoded.
    text = "R&amp;D &lt;jdoe&#64;cern.ch&gt; ok"
    assert redact_email_addresses(text) == "R&amp;D &lt;&gt; ok"


def test_redact_email_addresses_is_linear_on_long_runs():
    # Each local part is scanned back only to the previous separator or
    # match, and each domain forward only to the next non-domain
    # character, so long runs and runs full of separators stay linear.
    # A leftmost-start regex took seconds (tens of seconds at 50,000).
    import time

    run = "a" * 20000 + " " + "&amp;" * 4000 + " " + "b&" * 10000
    crowded = [
        "a%40" * 5000,
        "a@" * 10000,
        "&amp;" * 4000 + "#64",
        "x@1.2+" * 4000,
        "a@b-" * 5000,
        "&#64" * 5000,
    ]
    started = time.perf_counter()
    assert redact_email_addresses(run) == run
    assert redact_email_addresses(run + " jdoe@cern.ch") == run + " "
    for text in crowded:
        assert redact_email_addresses(text) == text
    assert time.perf_counter() - started < 1.0


# An address directly after another one (JIRA table cells, braces, "+",
# "&amp;", quotes). The first lookbehind version let the second through.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("|bob@cern.ch|alice@fnal.gov|", "|"),
        ("{bob@cern.ch}{alice@fnal.gov}", "}"),
        ("bob@cern.ch+alice@fnal.gov", ""),
        ("bob@cern.ch&amp;alice@fnal.gov", ""),
        ("bob@cern.ch'alice@fnal.gov'", "'"),
    ],
)
def test_redact_email_addresses_adjacent_addresses(text, expected):
    assert redact_email_addresses(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # "=" is a local-part character: no user name prefix survives.
        ("bounce list-bounces+bob=cern.ch@lists.cern.ch end", "bounce  end"),
        ("eq first=last@cern.ch end", "eq  end"),
        ("srs SRS0=HHH=TT=example.com=bob@fwd.org end", "srs  end"),
        # Encoded or fullwidth dot in the domain.
        ("encoded dot bob&#64;cern&#46;ch end", "encoded dot  end"),
        ("hex dot bob@cern&#x2e;ch end", "hex dot  end"),
        ("fullwidth dot bob\uff20cern\uff0ech end", "fullwidth dot  end"),
        # Invisible characters inside the address.
        ("soft hyphen b\u00adob@cern.ch end", "soft hyphen  end"),
        ("zero width b\u200bob@cern\u200d.ch end", "zero width  end"),
        ("word joiner bo\u2060b@cern.ch end", "word joiner  end"),
    ],
)
def test_redact_email_addresses_equals_dots_and_invisibles(text, expected):
    assert redact_email_addresses(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "pip install numpy@1.26.4",
        "py-numpy@1.26.4 %gcc@11.2.0",
        "npm i @types/node@18.0.1",
        "no dot bob@localhost",
        "known gap bob@[127.0.0.1]",
    ],
)
def test_redact_email_addresses_keeps_version_pins(text):
    # A domain must end in a label with a letter: version pins stay.
    assert redact_email_addresses(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Address-shaped but not mail: removed (fail closed).
        ("logo image@2x.png end", "logo  end"),
        ("see https://user@host.org/x", "see https:///x"),
        # A trailing numeric label is left; the address before it goes.
        ("v bob@cern.ch.123", "v .123"),
    ],
)
def test_redact_email_addresses_address_shaped_tokens(text, expected):
    assert redact_email_addresses(text) == expected


# The literal git@ account of a git host is kept (operator decision,
# 2026-09-28): copy-paste clone instructions must survive.
@pytest.mark.parametrize(
    "text",
    [
        "git clone git@github.com:org/x.git",
        "git clone git@gitlab.cern.ch:group/y.git",
        "git@gitlab.cern.ch:7999/group/y.git",
        "git remote add origin ssh://git@gitlab.cern.ch:7999/cms/z.git",
        "url = `git@github.com:cms-sw/cmssw.git`",
        "url: 'git@github.com:x/y.git' and |git@github.com:a/b|",
        "{git@github.com:x/y.git}",
    ],
)
def test_redact_email_addresses_keeps_the_git_account(text):
    assert redact_email_addresses(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A kept git@ token neither hides nor absorbs a real address after
        # it, on the same line or directly adjacent.
        (
            "git@github.com:x/y.git by jdoe@cern.ch",
            "git@github.com:x/y.git by ",
        ),
        (
            "git@github.com:x/y.git,jdoe@cern.ch",
            "git@github.com:x/y.git,",
        ),
        (
            "jdoe@cern.ch pushed to git@gitlab.cern.ch:g/y.git",
            " pushed to git@gitlab.cern.ch:g/y.git",
        ),
        # A git@ domain run that ends at another separator may hold a
        # local part, so it is not kept (the #5 result).
        ("git@github.com.jdoe@cern.ch", "@cern.ch"),
        # ... also when the domain's trimmed tail hides that separator
        # (review finding on 4ec37e51).
        ("git@github.com.jdoe.1@cern.ch", ""),
        ("git@github.com.jdoe-@cern.ch", ""),
        ("git@github.com.jdoe­@cern.ch", ""),
        ("git@github.com.jdoe.1&amp;#64;cern.ch", ""),
        ("git@github.com.jdoe.1%40cern.ch", ""),
        ("git@github.com.jdoe.1＠cern.ch", ""),
        # A name before "git" in the local part is not a wrapper.
        ("mail jdoe'git@cern.ch now", "mail  now"),
        ("mail jdoe=git@cern.ch now", "mail  now"),
        ("mail jdoe|git@cern.ch now", "mail  now"),
        ("mail jdoe`git@cern.ch now", "mail  now"),
        ("mail jdoe!git@cern.ch now", "mail  now"),
        ("mail jdoe&amp;amp;git@cern.ch now", "mail  now"),
        ("cfg url=git@github.com:x/y.git", "cfg :x/y.git"),
        # Only the literal "git" account with a literal "@" is kept.
        ("mail john.git@cern.ch now", "mail  now"),
        ("mail my-git@cern.ch now", "mail  now"),
        ("mail GIT@cern.ch now", "mail  now"),
        ("mail git&#64;cern.ch now", "mail  now"),
        ("mail git%40cern.ch now", "mail  now"),
        # ssh user@host logins are still removed (fail closed).
        ("ssh jdoe@lxplus.cern.ch", "ssh "),
        ("ssh -Y jdoe@lxplus.cern.ch -L 8080:host", "ssh -Y  -L 8080:host"),
        ("scp f jdoe@lxplus.cern.ch:~/x", "scp f :~/x"),
    ],
)
def test_redact_email_addresses_git_account_and_logins(text, expected):
    assert redact_email_addresses(text) == expected


def test_redact_email_addresses_git_account_is_linear():
    import time

    crowded = [
        "git@" * 10000,
        "git@a.b " * 5000,
        "git@a.b" * 5000,
        " git@x.org:y jdoe@cern.ch" * 2000,
    ]
    started = time.perf_counter()
    for text in crowded:
        redact_email_addresses(text)
    assert time.perf_counter() - started < 1.0


# Review finding on bfbadbf2: the git@ guard refused only a domain run that
# ran straight into a separator, so any other local-part character between
# the host and the next separator kept the git@ token and with it the first
# part of the next address's username. Every case below is removed whole
# without the exemption, and must be removed whole with it.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("clone git@github.com.jdoe.1+x@cern.ch now", "clone  now"),
        ("git@host.cern.ch.jdoe+ops@cern.ch", ""),
        ("git@gitlab.cern.ch.o'brien@cern.ch", ""),
        ("git@github.com.john_doe~x@cern.ch", ""),
        ("git@github.com.jdoé+x@cern.ch", ""),
        ('git@github.com.jdoe"x"@cern.ch', ""),
        ("'git@github.com.jdoe'@cern.ch", ""),
        # Quoted strings that close or open right after the host.
        ('git@github.com.jdoe" x y"@cern.ch', ""),
        ('"git@github.com.jdoe"@cern.ch', ""),
        ('git@github.com.jdoe"@cern.ch', '"@cern.ch'),
        # Wrappers around the kept token.
        ("`git@github.com.jdoe`x@cern.ch", ""),
        ("`git@github.com.jdoe+x`@cern.ch", ""),
        ("|git@github.com.jdoe|x@cern.ch", ""),
        ("{git@github.com.jdoe}x@cern.ch", ""),
        ("{git@github.com.jdoe+x}@cern.ch", ""),
        # Encoded separators after a +x.
        ("git@github.com.jdoe+x%40cern.ch", ""),
        ("git@github.com.jdoe+x%2540cern.ch", ""),
        ("git@github.com.jdoe+x&#64;cern.ch", ""),
        ("git@github.com.jdoe+x&#x40;cern.ch", ""),
        ("git@github.com.jdoe+x&commat;cern.ch", ""),
        ("git@github.com.jdoe+x&amp;#64;cern.ch", ""),
        ("git@github.com.jdoe+x＠cern.ch", ""),
        ("git@github.com.jdoe+x﹫cern.ch", ""),
        # Local-part tokens between the host and the separator.
        ("git@github.com.jdoe&amp;x@cern.ch", ""),
        # (An encoded dot is a domain dot, so the git@ domain runs up to
        # the "@" and the token goes the #5 way.)
        ("git@github.com.jdoe&#46;x@cern.ch", "@cern.ch"),
        ("git@github.com.jdoe%2ex@cern.ch", ""),
        # Double-escaped encoded dots are one token, as the leftward scan
        # reads them (review finding on b657568d).
        ("see git@john.doe+&amp;#46;x@cern.ch now", "see  now"),
        ("clone git@github.com&amp;#46;1@cern.ch now", "clone  now"),
        ("clone git@github.com+&amp;#46;x@cern.ch now", "clone  now"),
        ("clone git@github.com+&amp;#x2e;x@cern.ch now", "clone  now"),
        ("clone git@github.com+&amp;period;x@cern.ch now", "clone  now"),
        ("clone git@github.com+&amp;amp;#46;x@cern.ch now", "clone  now"),
        ("clone git@github.com+&AMP;#X2E;x@cern.ch now", "clone  now"),
        # A token inside a quoted local part goes as it would with no
        # exemption (found by fuzzing b657568d), and since #14 the whole
        # quoted local part and its domain go with it.
        ('"x git@github.com:y z"@cern.ch', ""),
        ('"git@github.com##]"@cern.ch', ""),
        ('"git@h.cern.cha:com"&commat;cern.ch', ""),
        ('say "hi git@github.com:x"@cern.ch', "say "),
        (
            'url: "git@github.com:x/y.git" and jdoe@cern.ch',
            'url: "git@github.com:x/y.git" and ',
        ),
        # Uppercase.
        ("git@GITHUB.COM.JDOE+X@CERN.CH", ""),
        ("git@github.com.JDOE+x&#X40;CERN.CH", ""),
        ("git@github.com.jdoe+x%40CERN.CH", ""),
    ],
)
def test_redact_email_addresses_git_token_never_keeps_a_local_part(
    text, expected
):
    assert redact_email_addresses(text) == expected


# Every character that can sit inside a local part, between the host and
# the next address's separator.
@pytest.mark.parametrize("char", list("+=!#$*^`{|}~'"))
def test_redact_email_addresses_git_token_local_part_characters(char):
    assert redact_email_addresses(f"git@github.com.jdoe{char}x@cern.ch") == ""
    assert redact_email_addresses(f"git@github.com.jdoe{char}@cern.ch") == ""
    assert redact_email_addresses(
        f"see git@github.com.jdoe.1{char}ops@cern.ch now"
    ) == "see  now"


@pytest.mark.parametrize(
    "text",
    [
        "git@github.com:org/x.git",
        "'git@host:x'",
        "`git@host:path`",
        "ssh://git@host/x/y.git",
        "git@github.com.jdoe",
        "git@github.com.jdoe now",
        'say "git@github.com:x/y.git" here',
        'say "hi" to git@github.com:x/y.git and "q" there',
    ],
)
def test_redact_email_addresses_git_token_positives_still_kept(text):
    assert redact_email_addresses(text) == text


def _address_spans(text):
    # The spans redact_email_addresses removes with no git@ exemption,
    # built from the module's own grammar helpers (not from the exemption
    # code under test).
    from archi.enrichment import anonymizer

    spans, bound = [], 0
    for core in anonymizer._SEP_CORE_RE.finditer(text):
        if core.start() < bound:
            continue
        sep = anonymizer._separator_start(text, core, bound)
        if sep is None:
            continue
        end = anonymizer._domain_end(text, core.end())
        local = anonymizer._local_start(text, sep, bound)
        if end is None or local == sep:
            bound = core.end()
            continue
        # A quoted local part may enclose earlier matches (#14): one span.
        while spans and spans[-1][1] > local:
            local = min(local, spans.pop()[0])
        spans.append((local, end))
        bound = end
    return spans


def _remove_spans(text, spans):
    out, kept = [], 0
    for start, end in spans:
        out.append(text[kept:start])
        kept = end
    out.append(text[kept:])
    return "".join(out)


def _git_exemption_inputs():
    import itertools
    import random

    rng = random.Random(20260928)
    prefixes = [
        "", "clone ", "'", "`", "|", "{", '"', "ssh://", "x ", '"x ', "x@",
        '"a" ', "&#64;", '"\n',
        # #14: quotes that enclose an earlier match, escaped quotes, and
        # invisible characters before the token.
        '"jdoe@cern.ch ', '"q\\"', '\\"', "﻿", "\x7f",
    ]
    hosts = [
        "github.com", "gitlab.cern.ch", "GITHUB.COM", "h.cern．ch",
        "git‎hub.com",
    ]
    tails = ["", ".jdoe", ".jdoe.1", ".jdoe-", ".jdoé", ".JDOE", ".o"]
    joiners = list("+=!#$*^`{|}~'\"") + [
        "", ".", "-", "_", "%", "&", ":", "/", " ", ";", ",", "\n",
        "&amp;", "&#46;", "%2e", "­", "​", "．", '"x y"', '"x"',
        "&#x2e;", "&period;", "&amp;#46;", "&amp;#x2e;", "&amp;period;",
        "&amp;amp;#46;", "&AMP;#X2E;", "&#0046", "⁠", "́",
        # #14: every kind of invisible character (Cf, C0, DEL, C1, U+FFFD)
        # and an escaped quote.
        "﻿", "‎", "⁡", "‮", "\x01", "\x7f", "\x81",
        "�", '\\"',
    ]
    locals_ = [
        "", "x", "ops", "1", "brien", '"q"', ' y"', ':y z"', '\\" y"',
        "x‎y",
    ]
    seps = [
        "@", "＠", "﹫", "%40", "%2540", "&#64;", "&#064", "&#x40;",
        "&commat;", "&amp;#64;", "&amp;amp;#x0040;", "",
    ]
    domains = ["cern.ch", "CERN.CH", "cern．ch", "1.2", "", " now"]
    # Every joiner meets every separator; the other parts are drawn.
    for joiner, sep in itertools.product(joiners, seps):
        for _ in range(3):
            yield "".join(
                (
                    rng.choice(prefixes),
                    "git@",
                    rng.choice(hosts),
                    rng.choice(tails),
                    joiner,
                    rng.choice(locals_),
                    sep,
                    rng.choice(domains),
                )
            )
    # Random fuzz over the same pieces, with more than one git@ token.
    pieces = (
        ["git@", "git", "a.b", ".c", "jdoe", "cern.ch", "x", "1", "é"]
        + joiners
        + seps
    )
    for _ in range(20000):
        yield "".join(rng.choice(pieces) for _ in range(rng.randint(1, 10)))


def test_redact_email_addresses_git_exemption_never_keeps_part_of_an_address(
    monkeypatch,
):
    """Differential property test: shipped function vs no exemption.

    Invariant: the only characters the exemption keeps that the function
    without it removes are standalone git@ tokens, whose wrappers and host
    are not part of any address-shaped span once "git@" is taken out.
    """
    import re
    import time

    from archi.enrichment import anonymizer

    state = {"off": False, "kept": []}
    original = anonymizer._is_git_account

    def patched(text, sep, local, end, *rest):
        if state["off"]:
            return False
        keep = original(text, sep, local, end, *rest)
        if keep:
            state["kept"].append((local, sep, end))
        return keep

    monkeypatch.setattr(anonymizer, "_is_git_account", patched)
    started = time.perf_counter()
    checked = kept_tokens = 0
    for text in _git_exemption_inputs():
        state["off"], state["kept"] = True, []
        without = redact_email_addresses(text)
        state["off"] = False
        shipped = redact_email_addresses(text)
        kept = state["kept"]
        spans = _address_spans(text)
        assert _remove_spans(text, spans) == without, text
        # The exemption only puts whole git@ tokens back, nothing else.
        kept_spans = {(local, end) for local, _, end in kept}
        assert kept_spans <= set(spans), text
        assert shipped == _remove_spans(
            text, [span for span in spans if span not in kept_spans]
        ), text
        for local, sep, end in kept:
            assert re.fullmatch(r"[`'|{]*git", text[local:sep]), text
            # Take "git@" out: the wrappers and host must not be part of
            # any address-shaped span of what remains. Of the text before
            # the token only quotes can join a local part (_local_start
            # stopped at everything else), so the prefix keeps its quotes
            # and line breaks and every other character becomes a space,
            # rather than gluing an earlier separator to the host.
            lead = "".join(
                char if char in '"\n' else " " for char in text[:local]
            )
            probe = lead + text[local:sep - 3] + text[sep + 1:]
            region = (local, local + (sep - 3 - local) + (end - sep - 1))
            for start, stop in _address_spans(probe):
                assert stop <= region[0] or start >= region[1], (
                    text, probe[start:stop]
                )
        checked += 1
        kept_tokens += len(kept)
    # The generated set is wide and does exercise the exemption.
    assert checked > 20000
    assert kept_tokens > 300
    assert time.perf_counter() - started < 2.0


def test_redact_email_addresses_git_scan_is_linear():
    import time

    crowded = [
        "git@a.b" + "+x" * 50000,
        "git@a.b." + "x" * 100000,
        ("git@a.b" + "+" * 50) * 2000,
        "git@a.b" + "&amp;" * 20000,
        "git@a.b" + "&" * 100000,
        "git@a.b" + "%25" * 30000,
        "git@a.b" + "&#000" * 20000,
        'git@a.b"' + "x " * 50000,
        ('git@a.b"' + "x" * 20 + " ") * 5000,
        ('git@a.b"x"') * 10000,
        "git@a.b\"" + "git@a.b " * 20000,
        '"' + "git@a.b " * 10000 + '"@x.yz',
        '"' + "git@a.b " * 10000,
        ('"q" git@a.b ') * 10000 + '"@x.yz',
    ]
    started = time.perf_counter()
    for text in crowded:
        redact_email_addresses(text)
    assert time.perf_counter() - started < 1.0


def _reference_redact(text):
    # Slow single-regex definition of the same grammar, restricted to
    # plain ASCII (no quotes, entities, fullwidth or invisible forms).
    import re

    pattern = (
        r"[\w.!#$%&'*+^`{|}~=-]+@"
        r"[\w.-]+\.(?=[\w-]*[^\W\d_])[\w-]*\w"
    )
    return re.sub(pattern, "", text)


def test_redact_email_addresses_matches_reference_on_random_text():
    import random

    rng = random.Random(20260928)
    alphabet = "ab1_.-+=|{}'&%# @@@"
    for _ in range(20000):
        text = "".join(
            rng.choice(alphabet) for _ in range(rng.randint(0, 18))
        )
        assert redact_email_addresses(text) == _reference_redact(text), text


# --- redact_obfuscated_email_addresses: spelled-out TWiki forms --------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Bracketed separators: any case, optional spaces, any domain.
        ("mail jdoe[AT]cern.ch now", "mail  now"),
        ("mail j-d-o-e(AT)cern.ch now", "mail  now"),
        ("mail jdoe+ops[at]phys.cern.ch now", "mail  now"),
        ("mail john.doe [at] cern.ch now", "mail  now"),
        ("mail jdoe (at) univ.edu now", "mail  now"),
        ("mail jdoe{at}fnal.gov now", "mail  now"),
        ("mail j-doe(at)cern(dot)ch now", "mail  now"),
        ("end jdoe[AT]cern.ch. Next", "end . Next"),
        # Bracketed dots, any case, with any kind of "at" before them
        # (operator rule, 2026-09-28: bracketed and parenthesised forms go).
        ("mail JDOE [At] CERN [DOT] CH now", "mail  now"),
        ("mail jdoe[at]cern dot ch now", "mail  now"),
        ("mail jdoe at cern(dot)ch now", "mail  now"),
        ("mail jdoe AT cern [DOT] ch now", "mail  now"),
        # Glued word separators (not covered by the rule; kept removed).
        ("mail john.doe_at_cern.ch now", "mail  now"),
        ("mail john-doe-at-cern.ch now", "mail  now"),
        ("mail john.doe_at_physics.ucsd.edu now", "mail  now"),
        ("mail jdoe-at-fnal.gov now", "mail  now"),
        ("mail jdoe.x_NOSPAM_AT_cern.ch now", "mail  now"),
        # NOSPAM insertions.
        ("mail jdoeNOSPAM.cern.ch now", "mail  now"),
        ("mail jdoe.cernNOSPAMch now", "mail  now"),
        ("mail jdoeATfnalDOTeduNOSPAM now", "mail  now"),
    ],
)
def test_redact_obfuscated_email_addresses_forms(text, expected):
    assert redact_obfuscated_email_addresses(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # Prose and file names that use "at" as a word, and hosts.
        "run at 13.6 TeV and at 8 TeV",
        "served at cmsweb.cern.ch and at lxplus.cern.ch",
        "file Zmm_2016_at_13TeV.root and x-at-2.3",
        "at cern.chip and at cern.ch.example",
        "remove NOSPAM and _NOSPAM_ markers",
        "Main.JohnDoe (John Doe) at CERN",
        "AT&amp;T &commat; ops, &#64; alone",
        "",
        # Hosts and paths after a word separator (review finding 3).
        "files are served at xrootd.t2.ucsd.edu for T2",
        "log in at login.hep.wisc.edu",
        "SE at srm.unl.edu:8443",
        "see the page at cern.ch/cms",
        "Tier2 at T2_US_UCSD.edu",
        # Bracketed separators before a number or a path.
        "Run2 [at] 13.6TeV",
        "mirror [at] host.org/path",
        # Names and words that merely contain NOSPAM.
        "NoSpamFilter enabled",
        "TWiki.NOSPAMPlugin and nospam-policy",
    ],
)
def test_redact_obfuscated_email_addresses_leaves_other_text_identical(text):
    assert redact_obfuscated_email_addresses(text) == text


@pytest.mark.parametrize(
    "text",
    [
        # Known gap (recorded in the PACT): a glued separator before a
        # domain that is not cern.ch / fnal.gov / gmail.com, .edu or .gov.
        "jdoe_at_infn.it",
    ],
)
def test_redact_obfuscated_email_addresses_known_gaps(text):
    assert redact_obfuscated_email_addresses(text) == text


# Operator rule (2026-09-28): free-prose "at ... dot" forms stay byte for
# byte, even when they spell out an address.
@pytest.mark.parametrize(
    "text",
    [
        "mail john.doe at cern.ch now",
        "mail john.doe AT CERN.CH now",
        "mail jdoe AT cern DOT ch now",
        "mail jdoe at cern dot ch now",
        "mail jdoe at fnal.gov now",
        "mail jdoe AT gmail.com now",
        "mail jdoe at cern.ch.",
        "jdoe at physics.ucsd.edu",
        "jdoe AT host.cern.ch",
        "based at cern.ch.",
        "Main.JohnDoe at cern.ch",
    ],
)
def test_redact_obfuscated_email_addresses_keeps_free_prose_forms(text):
    assert redact_obfuscated_email_addresses(text) == text
    assert redact_email_addresses(text) == text


def test_redact_obfuscated_email_addresses_accepted_over_removal():
    # Code-like text after a bracketed "(at)" goes with it when its last
    # label is a top-level domain (#14 keeps f(at)obj.attr and
    # f(at)x dot product: "attr" and "product" are not).
    assert redact_obfuscated_email_addresses("f(at)obj.ch") == ""
    assert redact_obfuscated_email_addresses("f(at)obj.attr.org") == ""
    assert redact_obfuscated_email_addresses("f(at)x dot py") == ""


def test_redact_obfuscated_email_addresses_is_linear_on_long_runs():
    import time

    crowded = [
        "a" * 50000,
        "a[at]" * 10000,
        "a[at]b." + "1" * 50000,
        "a at " * 10000,
        "a_at_" * 10000,
        "NOSPAM" * 10000,
        "a." * 25000 + "[at]",
        # Dot-word chains: the first version backtracked exponentially on
        # these (126 characters ran for minutes; review finding 1).
        "x_at_" + "a_dot_" * 5000 + "1",
        "x at " + "a dot " * 5000 + "1",
        "x AT " + "a DOT " * 5000 + "1",
        "x[at]" + "a(dot)" * 5000 + "1",
        "x[at]" + "a." * 20000 + "1",
        "x_NOSPAM_AT_" + "a_dot_" * 5000,
        "x_at_" + "a." * 20000 + "1",
        "x_at_" * 10000 + "a.edu",
        # The bracketed-dot and spelled-dot rules added for the
        # operator's rule.
        "x at " + "a(dot)" * 5000 + "1",
        "x at " + "a." * 20000 + "(dot)1",
        "x at a(dot)" + "a dot " * 5000 + "1",
        "x[at]" + "a dot " * 5000 + "1",
        "a at " * 10000 + "b(dot)",
    ]
    started = time.perf_counter()
    for text in crowded:
        redact_obfuscated_email_addresses(text)
    assert time.perf_counter() - started < 1.0


# --- #14: invisible characters, quoted local parts, tighter spelled-out forms --

#: One character of each invisible kind: format characters (Cf) outside the
#: five the scans listed before #14, C0 controls, DEL, C1 controls, and the
#: U+FFFD a decoder writes for an unreadable byte.
INVISIBLE_CHARS = [
    "﻿", "‎", "‏", "⁡", "⁢", "⁣", "⁤",
    "‪", "‮", "⁦", "⁩", "᠎", "؀", "\U000e0001",
    "\x01", "\x08", "\x1b", "\x7f", "\x81", "\x9b", "�",
]


def test_is_invisible_covers_every_format_and_control_character():
    import sys
    import unicodedata

    from archi.enrichment import anonymizer

    for code in range(sys.maxunicode + 1):
        char = chr(code)
        category = unicodedata.category(char)
        expected = (
            category == "Cf"
            or (category == "Cc" and not char.isspace())
            or char == "�"
        )
        assert anonymizer._is_invisible(char) is expected, hex(code)
        if expected:
            assert anonymizer._MAYBE_INVISIBLE_RE.match(char), hex(code)


@pytest.mark.parametrize("char", INVISIBLE_CHARS, ids=lambda c: f"U+{ord(c):04X}")
@pytest.mark.parametrize(
    "shape",
    [
        "jdoe{c}x@cern.ch",
        "{c}jdoe@cern.ch",
        "jdoe{c}@cern.ch",
        "jdoe@{c}cern.ch",
        "jdoe@ce{c}rn.ch",
        "jdoe@cern{c}.ch",
        "jdoe@cern.{c}ch",
        "jd{c}oe+o{c}ps@phys{c}.cern.ch",
    ],
)
def test_redact_email_addresses_invisible_characters_do_not_split_an_address(
    char, shape
):
    # Before #14 the scans stopped at any invisible character but five, so
    # "jdoe" (or the whole address, when one sat in the domain) survived.
    assert redact_email_addresses("mail " + shape.format(c=char) + " now") == (
        "mail  now"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Whitespace control characters still end an address.
        ("a\tjdoe@cern.ch", "a\t"),
        ("a\njdoe@cern.ch\nb", "a\n\nb"),
        ("a\rjdoe@cern.ch", "a\r"),
        ("a\x1cjdoe@cern.ch", "a\x1c"),
        ("a\x85jdoe@cern.ch", "a\x85"),
    ],
)
def test_redact_email_addresses_whitespace_controls_still_separate(text, expected):
    assert redact_email_addresses(text) == expected


def test_redact_email_addresses_invisible_characters_property():
    """Seeded property: invisible characters inserted anywhere inside an
    address never leave a letter or digit of it behind, and an address
    with them is removed exactly where the same address without them is."""
    import random

    rng = random.Random(20260928)
    locals_ = ["jdoe", "john.doe", "j.doe+ops", "o'brien", "x", "über.müller"]
    domains = ["cern.ch", "fnal.gov", "phys.ucsd.edu", "CERN.CH"]
    seps = ["@", "%40", "&#64;", "＠"]
    checked = 0
    def sprinkle(part, first, last):
        # Invisible characters inside ``part``; at its start only when
        # ``first`` and at its end only when ``last`` is False (an invisible
        # character before or after the whole address only touches it).
        chars = list(part)
        for _ in range(rng.randint(0, 2)):
            low = 1 if first else 0
            high = len(chars) - 1 if last else len(chars)
            chars.insert(rng.randint(low, high), rng.choice(INVISIBLE_CHARS))
        return "".join(chars)

    for _ in range(4000):
        # Not inside a multi-character separator: "%4<U+2063>0" is not a
        # separator to a reader or a URL decoder either.
        address = (
            sprinkle(rng.choice(locals_), True, False)
            + rng.choice(seps)
            + sprinkle(rng.choice(domains), False, True)
        )
        text = "mail " + address + " now"
        assert redact_email_addresses(text) == "mail  now", repr(text)
        checked += 1
    assert checked == 4000


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The #14 report: an earlier match inside the quotes left the rest
        # of the quoted local part and the domain behind.
        ('"x git@a.b y"@cern.ch', ""),
        ('"x jdoe@a.bc y"@cern.ch', ""),
        ('say "x git@a.b y"@cern.ch ok', "say  ok"),
        ('a "b jdoe@x.org c jane@y.org d"@cern.ch z', "a  z"),
        ('"jdoe@cern.ch"@fnal.gov', ""),
        # Escaped quotes inside the quoted local part.
        ('see "a\\"b"@cern.ch now', "see  now"),
        ('"q\\"x jdoe@a.bc"@cern.ch', ""),
        ('"q\\\\"x"@cern.ch', '"q\\\\'),
        # A name directly before the opening quote goes with it (it kept
        # "jdoe" before #14).
        ('mail jdoe"x"@cern.ch now', "mail  now"),
        ('a jdoe.x"q y"@cern.ch b', "a  b"),
        ('mail="john doe"@cern.ch x', " x"),
        # Quotes elsewhere are left alone.
        ('say "hi"@cern.ch', "say "),
        ('jdoe@cern.ch said "hi"', ' said "hi"'),
        ('"a" jdoe@cern.ch "b"@x.org', '"a"  '),
        # A quote pair across a line break is not a quoted local part.
        ('"a"\njdoe@cern.ch"@x.org', '"a"\n"@x.org'),
    ],
)
def test_redact_email_addresses_quoted_local_part_after_a_match(text, expected):
    assert redact_email_addresses(text) == expected


def test_redact_email_addresses_quoted_local_part_is_linear():
    import time

    crowded = [
        '"' + "x jdoe@a.bc " * 20000 + '"@cern.ch',
        ('"x jdoe@a.bc y"@cern.ch ') * 10000,
        ("\\\"" * 30 + 'x"@a.bc ') * 2000,
        ('x\\"@a.bc ') * 20000,
        '"' + ('x\\"@a.bc ') * 20000,
        ("\\" * 40 + '"@a.bc ') * 2000,
        "jdoe﻿" * 30000 + "@cern.ch",
    ]
    started = time.perf_counter()
    for text in crowded:
        redact_email_addresses(text)
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    "text",
    [
        "mail jdoe\x7f[at]cern.ch now",
        "mail jd﻿oe[at]cern.ch now",
        "mail jdoe[at]ce​rn.ch now",
        "mail jdoe\x81[at]cern.ch now",
        "mail jdoe�[at]cern.ch now",
        "mail jdoe[‎at]cern.ch now",
        "mail jdoe (at) cern⁡(dot)ch now",
        "mail jdoe at cern(‎dot)ch now",
        "mail john.doe\x7f_at_cern.ch now",
        "mail jdoe\x01NOSPAM.cern.ch now",
    ],
)
def test_redact_obfuscated_email_addresses_invisible_characters(text):
    # Before #14 an invisible character split the token: it was kept, or
    # the part of the name before the character survived.
    assert redact_obfuscated_email_addresses(text) == "mail  now"


@pytest.mark.parametrize(
    "text",
    [
        # CMS site names, wiki markup and code before a bracketed "at"
        # (lane B's review of #11); all were removed before #14.
        "site T2_CH_CERN [at] cern.ch",
        "site T1_US_FNAL_Disk(at)fnal.gov",
        "dataset (at) T2_US_MIT.mit.edu",
        "WebHome [AT] Twiki.Main",
        "cfg(at)process.source.fileNames",
        "read[at]self.buf.data",
        "data[at]i.root",
        "f(at)obj.attr",
        "f(at)x dot product",
        "x[at]cern.ch.example",
        # A bare NOSPAM domain is not an address.
        "nospam.ch config",
        "see NOSPAM.org",
        # Kept before and after.
        "T2_US_MIT (at) CERN",
        "Run2 [at] 13.6TeV",
    ],
)
def test_redact_obfuscated_email_addresses_keeps_sites_markup_and_code(text):
    assert redact_obfuscated_email_addresses(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "mail jdoe[at]desy.de now",
        "mail j.doe (AT) ox.ac.uk now",
        "mail jdoe[at]infn.it now",
        "mail JDOE{AT}CERN.CH now",
        "mail jdoe <at> mail.cern.ch now",
        "mail jdoe[at]a-b.org now",
        "mail T2admin[at]cern.ch now",
        "mail jdoe_x[at]cern.ch now",
        "mail jdoe[at]physics.ucsd.edu now",
        "mail jdoe[at]cern [dot] ch now",
        "mail jdoe at cern(dot)ch now",
        "mail jdoeNOSPAM.cern.ch now",
        "mail NOSPAMjdoe.cern.ch now",
        # Found on the TWiki snapshot: the @ pass skips a domain with no
        # dot, and before #14 only "cernNOSPAMch" went, leaving "jdoe@".
        "mail jdoe@cernNOSPAMch now",
    ],
)
def test_redact_obfuscated_email_addresses_still_removes_real_addresses(text):
    assert redact_obfuscated_email_addresses(text) == "mail  now"


# The bracketed pattern as #13 shipped it (6193edd0ca), for the differential
# test below.
_OBF_STRONG_13 = (
    r"(?<![\w.+-])[\w.+-]{1,64}"
    r"\s?[(\[{<]\s?at\s?[)\]}>]\s?"
    r"[\w-]+(?:(?:\.|\s?[(\[{<]\s?dot\s?[)\]}>]\s?| dot )[\w-]+)*"
    r"(?:\.|\s?[(\[{<]\s?dot\s?[)\]}>]\s?| dot )[^\W\d_]{2,}"
    r"(?![\w-]|[./:@(][\w-])"
)


def test_redact_obfuscated_email_addresses_differential_against_13():
    """Differential property test, the #14 bracketed rule vs #13's.

    Over a seeded grid of user names, bracketed separators and domains:
    a user name before a mail domain (labels of letters, digits and "-", a
    country-code or listed generic last label) is removed whole, also with
    invisible characters inside it; anything else that #13 removed is now
    kept byte for byte; and #14 removes nothing #13 did not.
    """
    import itertools
    import random
    import re

    old = re.compile(_OBF_STRONG_13, re.IGNORECASE)
    rng = random.Random(20260928)
    users = ["jdoe", "john.doe", "j-doe+ops", "x_y", "JDoe", "a1", "T2admin"]
    sites = ["T2_CH_CERN", "T1_US_FNAL_Disk", "T0_CH_CERN", "T3_IT_Trieste"]
    mail = [
        "cern.ch", "CERN.CH", "fnal.gov", "phys.ucsd.edu", "cern(dot)ch",
        "cern [DOT] ch", "cern dot ch", "gmail.com", "infn.it", "ox.ac.uk",
        "a-b.org", "univ.edu", "lab.gov", "x.net", "x.info",
    ]
    not_mail = [
        "T2_US_MIT.mit.edu", "Twiki.Main", "self.buf.data", "i.root",
        "process.source.fileNames", "obj.attr", "x dot product", "my_host.cern.ch",
        "a.b_c", "Main.WebHome",
    ]
    seps = ["[at]", "[AT]", " [at] ", "(at)", " (At) ", "{at}", "<at>", "[ at ]"]
    prefixes = ["", "mail ", "(", "site ", "| "]
    tails = ["", " now", ".", ", next", ")", " |"]
    removed = kept_now = 0
    for user, domain, sep in itertools.product(
        users + sites, mail + not_mail, seps
    ):
        for _ in range(2):
            prefix, tail = rng.choice(prefixes), rng.choice(tails)
            token = user + sep + domain
            text = prefix + token + tail
            shipped = redact_obfuscated_email_addresses(text)
            was = old.sub("", text)
            if user in users and domain in mail:
                assert shipped == prefix + tail, text
                # With invisible characters inside, still removed whole.
                chars = list(token)
                chars.insert(rng.randint(1, len(chars) - 1), rng.choice(INVISIBLE_CHARS))
                hidden = prefix + "".join(chars) + tail
                assert redact_obfuscated_email_addresses(hidden) == prefix + tail, hidden
                removed += 1
            else:
                assert shipped == text, text
                if was != text:
                    kept_now += 1
    # The grid exercises both sides, and #13 did remove the kept shapes.
    assert removed > 1000
    assert kept_now > 500


def test_redact_obfuscated_email_addresses_tighter_rules_are_linear():
    import time

    crowded = [
        "x[at]" + "a." * 20000 + "attr",
        "x[at]" + "a_b." * 10000 + "ch",
        "T2_CH_CERN[at]" * 5000,
        "x[at]" + "a dot " * 5000 + "product",
        "nospam.ch " * 10000,
        "\x7f" * 50000 + "[at]",
        ("jdoe﻿[at]" * 5000) + "cern.ch",
    ]
    started = time.perf_counter()
    for text in crowded:
        redact_obfuscated_email_addresses(text)
    assert time.perf_counter() - started < 1.0
