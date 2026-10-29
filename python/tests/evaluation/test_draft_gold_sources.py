"""The gold-source drafting script, on invented question files shaped like a benchmark item
(atoms with answer_source, a free-text sources list, verification evidence). Nothing here comes
from a real question set: every key, repository, path, URL and site name is made up."""
import yaml

from archi.evaluation.draft_gold_sources import draft_item, extract, main, resolve_document

ITEM = {
    "id": "q-demo-07",
    "question": "Why did the invented rule cleanup stop?",
    "atoms": [
        {"id": "a1", "text": "the cleanup job errored", "required": True, "answer_source": "ticket"},
        {"id": "a2", "text": "the function that deletes rules", "required": True, "answer_source": "source_code"},
        {"id": "a3", "text": "how many rules are stuck now", "required": False, "answer_source": "live_rucio"},
        {"id": "a4", "text": "the procedure", "required": True, "answer_source": "doc or ticket"},
        {"id": "a5", "text": "the current state", "required": False, "answer_source": "ticket+live_rucio"},
        {"id": "a6", "text": "the deployed config", "required": False, "answer_source": "repo_config"},
    ],
    "sources": [
        "repos/fakerepo @1a2b3c4d src/fakerepo/core/rules.py (delete_rule 10-42)",
        "jira:DEMOOPS-12",
        "T2_ZZ_Nowhere",
        "a conversation in a meeting",
    ],
    "verification": {"evidence": "See https://docs.example.org/ops/rules.html and "
                                 "https://tracker.example.org/browse/DEMOOPS-13; also repos/otherrepo/cfg/app.yaml."},
}


def test_references_of_every_kind_are_extracted():
    refs, found = extract("s", "repos/r @abcdef12 a/b.py (f 1-2), jira:AB-1, CD-22, T1_ZZ_Lab_Disk, "
                               "https://docs.example.org/x.html https://github.com/o/r HIG-19-001")
    by = {r["id"]: r for r in refs}
    assert found
    assert by["git_repo:r/a/b.py"]["key"] == "r/a/b.py" and "abcdef12" in by["git_repo:r/a/b.py"]["confidence"]
    assert by["jira:AB-1"]["confidence"].startswith("high") and by["jira:CD-22"]["confidence"].startswith("medium")
    assert by["site:T1_ZZ_Lab_Disk"]["source"] == "cric"
    assert by["docsite:https://docs.example.org/x.html"]["source"] == "docsite"
    assert not any("github.com" in r["id"] for r in refs)  # a repository URL is not a docsite page
    assert "jira:HIG-19" not in by  # an analysis id is not a Jira key


def test_a_question_is_drafted_with_atoms_notes_and_unresolved_items():
    draft = draft_item(ITEM["id"], ITEM)
    assert set(draft["gold"]) == {"git_repo:fakerepo/src/fakerepo/core/rules.py", "jira:DEMOOPS-12",
                                  "site:T2_ZZ_Nowhere", "docsite:https://docs.example.org/ops/rules.html",
                                  "jira:DEMOOPS-13", "git_repo:otherrepo/cfg/app.yaml"}
    assert draft["gold_sources"]["jira:DEMOOPS-12"] == {"source": "jira", "key": "DEMOOPS-12"}
    assert draft["gold_sources"]["git_repo:fakerepo/src/fakerepo/core/rules.py"] == {
        "source": "git_repo", "key": "fakerepo/src/fakerepo/core/rules.py"}
    atoms = {a["id"]: a for a in draft["atoms"]}
    assert set(atoms["a1"]["gold"]) == {"jira:DEMOOPS-12", "jira:DEMOOPS-13"}
    assert set(atoms["a2"]["gold"]) == {"git_repo:fakerepo/src/fakerepo/core/rules.py", "git_repo:otherrepo/cfg/app.yaml"}
    assert atoms["a3"]["gold"] == [] and "live system" in atoms["a3"]["note"]
    assert "docsite:https://docs.example.org/ops/rules.html" in atoms["a4"]["gold"]   # doc or ticket
    assert set(atoms["a5"]["gold"]) == {"jira:DEMOOPS-12", "jira:DEMOOPS-13"}         # counts through its ticket
    assert "delete_rule 10-42" in draft["notes"]["git_repo:fakerepo/src/fakerepo/core/rules.py"]
    assert draft["unresolved"] == ["sources[3]: no reference recognised"]


def test_an_atom_with_no_matching_reference_is_unresolved():
    draft = draft_item("q", {"atoms": [{"id": "a1", "answer_source": "source_code"},
                                       {"id": "a2", "answer_source": "carrier_pigeon"}],
                             "sources": ["jira:DEMOOPS-1"]})
    assert any("atom a1 (source_code): no git reference" in u for u in draft["unresolved"])
    assert any("atom a2: answer_source kind(s) ['carrier_pigeon']" in u for u in draft["unresolved"])


def test_the_command_writes_a_private_draft_beside_the_set(tmp_path, capsys):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "q-demo-07.yaml").write_text(yaml.safe_dump(ITEM))
    (golden / "q-demo-08.yaml").write_text(yaml.safe_dump({"atoms": [{"id": "a1", "answer_source": "monitoring"}]}))
    out = tmp_path / "private" / "gold_sources.draft.yaml"
    assert main(["--questions", str(golden), "--out", str(out)]) == 0
    text = out.read_text()
    assert text.startswith("# DRAFT gold_sources") and "Never commit it" in text
    data = yaml.safe_load(text)
    assert [q["id"] for q in data["questions"]] == ["q-demo-07", "q-demo-08"]
    assert "Why did the invented rule cleanup stop" not in text  # question text is not copied
    assert "unresolved items for review" in capsys.readouterr().out


def test_the_command_refuses_to_write_inside_the_repository(tmp_path, capsys):
    from pathlib import Path

    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "q.yaml").write_text(yaml.safe_dump(ITEM))
    inside = Path(__file__).resolve().parents[1] / "gold_sources.draft.yaml"
    assert main(["--questions", str(golden), "--out", str(inside)]) == 2
    assert "never committed" in capsys.readouterr().err and not inside.exists()


CHECKOUTS = {
    "demoops-Documentation": ["README.md", "docs/Policy/pile_rules.md", "docs/Policy/dm_policies.md",
                              "docs/Operators/Getting Started.md", "docs/Old/dup.md", "docs/New/dup.md"],
    "demopr-documentation": ["docs/roles/campaign_lead.md", "docs/roles/assistant_campaign_lead.md"],
    "demoops": ["notes.md"],
    "unrelated": ["docs/Policy/pile_rules.md"],
}


def test_a_loosely_named_document_is_left_for_review_without_checkouts():
    problems = []
    refs, found = extract("sources[0]", "demoprdocumentation/campaign_lead.md", problems=problems)
    assert found and refs == []
    assert problems == ["sources[0]: 'demoprdocumentation/campaign_lead.md' names a repository document "
                        "(<repo>/<path>); draft again with --repos to resolve it"]


def test_prose_and_neighbours_of_a_loose_document_are_not_misread():
    problems = []
    refs, _ = extract("s", "DEMOOPS-897 and docs/Policy/dm_policies.md; T2_ZZ_Nowhere: notes.md", CHECKOUTS, problems)
    assert {"jira:DEMOOPS-897", "site:T2_ZZ_Nowhere"} <= {r["id"] for r in refs}
    for text in ("Evidence: README.md", "policy in docs/Policy/dm_policies.md", "evidence/q009_findings.md",
                 "status/summary.html: 19 nodes", "docs_Operators_Update.md covers corpus_page and"):
        problems = []
        refs, _ = extract("s", text, CHECKOUTS, problems)
        assert not any(r["source"] == "git_repo" for r in refs), text
    refs, _ = extract("s", "demoops Documentation docs/Policy/dm_policies.md vs pile_rules.md", CHECKOUTS, [])
    assert [r["key"] for r in refs] == ["demoops-Documentation/docs/Policy/dm_policies.md"]  # stops at the first .md


def test_resolution_never_matches_inside_a_file_name_or_a_short_hint():
    assert resolve_document("demoops", "policies.md", CHECKOUTS) == []              # not dm_policies.md
    assert resolve_document("demoprdocumentation", "campaign_lead.md", CHECKOUTS) == [
        "demopr-documentation/docs/roles/campaign_lead.md"]                         # not assistant_campaign_lead.md
    assert resolve_document("demoops", "notes.md", CHECKOUTS) == ["demoops/notes.md"]   # the same name ranks first
    assert resolve_document("doc", "README.md", CHECKOUTS) == []                   # under four characters


def test_repos_references_with_parentheses_around_the_commit():
    refs, _ = extract("s", "repos/fakerepo (Sub) @065beb3b (main 2026-07-14) Sub/utils.py:4718-4775; "
                           "repos/other @2e14c417 (4.0rc1; prod 4.1) lib/core/rule.py:2034-2103 move_rule")
    assert {r["key"] for r in refs} == {"fakerepo/Sub/utils.py", "other/lib/core/rule.py"}


def test_loosely_named_documents_resolve_to_the_one_checkout_file():
    cases = {
        "demoops/pile_rules.md": "demoops-Documentation/docs/Policy/pile_rules.md",           # folders left out
        "demoprdocumentation/campaign_lead.md": "demopr-documentation/docs/roles/campaign_lead.md",
        "demoops Documentation: docs_Operators_Getting_Started.md":                            # flattened path
            "demoops-Documentation/docs/Operators/Getting Started.md",
        "documentation_page:demoops_Documentation:docs_Policy_pile_rules.md":
            "demoops-Documentation/docs/Policy/pile_rules.md",
        'demoops/Documentation docs/Operators/Getting Started.md (chunk:0a1b2c3d4e5f6a7b) "a quote"':
            "demoops-Documentation/docs/Operators/Getting Started.md",
    }
    for text, key in cases.items():
        problems = []
        refs, found = extract("sources[0]", text, CHECKOUTS, problems)
        assert found and problems == [], text
        assert [(r["id"], r["key"]) for r in refs] == [(f"git_repo:{key}", key)], text
        assert refs[0]["confidence"].startswith("high"), text


def test_a_document_that_does_not_resolve_to_one_file_is_left_for_review():
    draft = draft_item("q", {"atoms": [{"id": "a1", "answer_source": "doc or source_code"}],
                             "sources": ["demoops/dup.md", "demoops/missing.md", "jira:DEMOOPS-3"]}, CHECKOUTS)
    assert draft["gold"] == ["jira:DEMOOPS-3"]
    assert any(u.startswith("sources[0]: 'demoops/dup.md' matches 2 checkout files") for u in draft["unresolved"])
    assert any(u.startswith("sources[1]: 'demoops/missing.md' matches no file") for u in draft["unresolved"])
    assert not any("no reference recognised" in u for u in draft["unresolved"])


def test_the_command_resolves_documents_against_the_checkouts_given(tmp_path, capsys):
    golden, repos = tmp_path / "golden", tmp_path / "repos"
    golden.mkdir()
    (golden / "q.yaml").write_text(yaml.safe_dump({"id": "q", "atoms": [{"id": "a1", "answer_source": "doc"}],
                                                   "sources": ["demoops/pile_rules.md"]}))
    for name, files in CHECKOUTS.items():
        for rel in files + [".git/HEAD"]:
            (repos / name / rel).parent.mkdir(parents=True, exist_ok=True)
            (repos / name / rel).write_text("x")
    out = tmp_path / "private" / "draft.yaml"
    assert main(["--questions", str(golden), "--out", str(out), "--repos", str(repos)]) == 0
    question = yaml.safe_load(out.read_text())["questions"][0]
    assert question["gold_sources"] == {"git_repo:demoops-Documentation/docs/Policy/pile_rules.md": {
        "source": "git_repo", "key": "demoops-Documentation/docs/Policy/pile_rules.md"}}
    assert main(["--questions", str(golden), "--out", str(out), "--repos", str(tmp_path / "none")]) == 2
    assert "is not a directory" in capsys.readouterr().err
