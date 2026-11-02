End-to-end proof, 2026-10-22. okg at the head of mitdbg/okg PR 4020 (88e06a5f2a3), the cern-team profile installed twice
from this repo (before: schemas at origin/main 0cc3d97c; after: this commit), same invented data, a throwaway Postgres.
Records came from the real readers over small invented caches: TWiki EOS dump (6 topics), JIRA records.json (3 issues),
CDS OAI-PMH XML served by a local stub (4 records), CMSSW frozen releases.map (4 labels).
Each line: query, then the top four results as rank:id [match marker].
Data: TWiki SUS29991Review bears SUS-29-991; SurveyOfPendingReviews only mentions it five times; ZZQ-4472 quotes ZZQ-4471 five times.

== before
'SUS-29-991' 1:SurveyOfPendingReviews [lexical] | 2:SUS29991Review [lexical] | 3:ComparisonSUS29975andEXO29976 [lexical] | 4:SUS28991Notes [lexical]
'sus-29-991' 1:SurveyOfPendingReviews [lexical] | 2:SUS29991Review [lexical] | 3:ComparisonSUS29975andEXO29976 [lexical] | 4:SUS28991Notes [lexical]
'CMS-PAS-SUS-29-991' 1:SUS29991Review [lexical] | 2:SurveyOfPendingReviews [lexical] | 3:SUS28991Notes [lexical] | 4:SUS29981Review [lexical]
'EXO-29-976' 1:ComparisonSUS29975andEXO29976 [lexical] | 2:SurveyOfPendingReviews [lexical] | 3:SUS29981Review [lexical] | 4:SUS29991Review [lexical]
'CMSSW_99_1_0' 1:cmssw_release:CMSSW_99_1_0 [exact_name] | 2:cmssw_release:CMSSW_99_0_1 [lexical] | 3:cmssw_release_family:CMSSW_99_0 [lexical] | 4:cmssw_release_family:CMSSW_99_1 [lexical]
'CMSSW-99-1-0' 1:cmssw_release:CMSSW_99_1_0 [exact_name_variant] | 2:cmssw_release_family:CMSSW_99_0 [lexical] | 3:cmssw_release_family:CMSSW_99_1 [lexical] | 4:cmssw_release:CMSSW_99_0_1 [lexical]
-- subtype jira_issue
'ZZQ-4471' 1:jira:ZZQ-4472 [lexical] | 2:jira:ZZQ-4471 [lexical] | 3:jira:ZZQ-4417 [lexical]
'zzq-4471' 1:jira:ZZQ-4472 [lexical] | 2:jira:ZZQ-4471 [lexical] | 3:jira:ZZQ-4417 [lexical]
'ZZQ-4417' 1:jira:ZZQ-4417 [lexical] | 2:jira:ZZQ-4472 [lexical] | 3:jira:ZZQ-4471 [lexical]
-- subtype cds_record
'CMS-PAS-SUS-29-991' 1:cds:9000003 [lexical] | 2:cds:9000001 [lexical] | 3:cds:9000002 [lexical] | 4:cds:9000004 [lexical]
'SUS-29-991' 1:cds:9000003 [lexical] | 2:cds:9000004 [lexical] | 3:cds:9000001 [lexical] | 4:cds:9000002 [lexical]
'sus-29-991' 1:cds:9000003 [lexical] | 2:cds:9000004 [lexical] | 3:cds:9000001 [lexical] | 4:cds:9000002 [lexical]
'CMS-NOTE-2029-005' 1:cds:9000003 [lexical] | 2:cds:9000004 [lexical] | 3:cds:9000002 [lexical] | 4:cds:9000001 [lexical]
== after
'SUS-29-991' 1:SUS29991Review [exact_name] | 2:SurveyOfPendingReviews [lexical] | 3:ComparisonSUS29975andEXO29976 [lexical] | 4:SUS28991Notes [lexical]
'sus-29-991' 1:SUS29991Review [exact_name_variant] | 2:SurveyOfPendingReviews [lexical] | 3:ComparisonSUS29975andEXO29976 [lexical] | 4:SUS28991Notes [lexical]
'CMS-PAS-SUS-29-991' 1:SUS29991Review [lexical] | 2:SurveyOfPendingReviews [lexical] | 3:SUS28991Notes [lexical] | 4:SUS29981Review [lexical]
'EXO-29-976' 1:ComparisonSUS29975andEXO29976 [exact_name] | 2:SurveyOfPendingReviews [lexical] | 3:SUS29981Review [lexical] | 4:SUS29991Review [lexical]
'CMSSW_99_1_0' 1:cmssw_release:CMSSW_99_1_0 [exact_name] | 2:cmssw_release:CMSSW_99_0_1 [lexical] | 3:cmssw_release_family:CMSSW_99_0 [lexical] | 4:cmssw_release_family:CMSSW_99_1 [lexical]
'CMSSW-99-1-0' 1:cmssw_release:CMSSW_99_1_0 [exact_name_variant] | 2:cmssw_release_family:CMSSW_99_0 [lexical] | 3:cmssw_release_family:CMSSW_99_1 [lexical] | 4:cmssw_release:CMSSW_99_0_1 [lexical]
-- subtype jira_issue
'ZZQ-4471' 1:jira:ZZQ-4471 [exact_name] | 2:jira:ZZQ-4472 [lexical] | 3:jira:ZZQ-4417 [lexical]
'zzq-4471' 1:jira:ZZQ-4471 [exact_name_variant] | 2:jira:ZZQ-4472 [lexical] | 3:jira:ZZQ-4417 [lexical]
'ZZQ-4417' 1:jira:ZZQ-4417 [exact_name] | 2:jira:ZZQ-4472 [lexical] | 3:jira:ZZQ-4471 [lexical]
-- subtype cds_record
'CMS-PAS-SUS-29-991' 1:cds:9000001 [exact_name] | 2:cds:9000003 [lexical] | 3:cds:9000002 [lexical] | 4:cds:9000004 [lexical]
'SUS-29-991' 1:cds:9000001 [exact_name] | 2:cds:9000004 [exact_name] | 3:cds:9000003 [lexical] | 4:cds:9000002 [lexical]
'sus-29-991' 1:cds:9000001 [exact_name_variant] | 2:cds:9000004 [exact_name_variant] | 3:cds:9000003 [lexical] | 4:cds:9000002 [lexical]
'CMS-NOTE-2029-005' 1:cds:9000003 [exact_name] | 2:cds:9000004 [lexical] | 3:cds:9000002 [lexical] | 4:cds:9000001 [lexical]
