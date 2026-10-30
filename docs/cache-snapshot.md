# Cache snapshots

> **Private data. Never commit or publish a snapshot from this repository.**
> archi-okg is a public repository. Snapshot archives (`*.tar.zst`) and the
> `snapshot.lock.yaml` a build writes are private release assets of
> `mitdbg/okg-deployments` (a private repository). They must never be
> committed to, attached to a release of, or published from archi-okg or any
> other public repository, even though email addresses are removed: they still
> carry names, usernames, internal documents and ticket text. Build them
> outside any checkout of this repository. `.gitignore` here ignores
> `*.tar.zst`, `snapshot.lock.yaml` and `cms-cache-*/` as a backstop.

A cache snapshot packs the files the cache-backed readers read into one
checksummed archive per source group, so a fresh install needs the archives
and a lock file, not CERN credentials. `python -m archi.snapshot` builds and
verifies them. It never downloads anything: it packs caches that already exist
(from the downloaders in `archi.downloaders`, an operator, or an older host).

## Build

Write a config that maps each group to the directory holding its cache and to
the date that cache was collected. Groups may have different dates.

```yaml
snapshot: cms-cache-20260928        # letters, digits, '.', '_' or '-'
zstd_level: 19                      # optional, 1-19, default 19
groups:
  jira:
    path: /data/cms-20260831/jira   # relative paths resolve against this file
    collected: 2026-08-31           # required: YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD
  cric:
    path: /data/cms-202606/cric
    collected: 2026-04-09           # when the data was fetched
    note: files written 2026-06-12; fetched 2026-04-09   # optional, into the lock
  indico:
    path: /data/cms-202606/indico
    collected: 2026-06-12..2026-06-16
  wmstats:
    path: /data/cms-20260831/wmstats-workflows
    collected: 2026-08-31
    drop_fields: [SomeField]        # optional, top-level record fields
    keep_fields: [OtherField]       # optional, keep an unread field
    file_dates:                     # optional, only when one file's date differs
      records.json: 2026-08-30      # from the group's own `collected`
```

`collected` is when the data was fetched, not when the files were written.
`note` is free text of at most 300 characters, carried into the lock so the
snapshot describes itself; it must contain no email address.

```
python -m archi.snapshot build --config snapshot.yaml --out <new-empty-dir> [--built-by NAME] [--tmp-dir DIR]
```

Known groups and the files each one reads (from `archi/snapshot/groups.py`):

| Group | Files in its directory | Archived under |
|---|---|---|
| `cric` | `sites.json`, `storage_units.json`, `compute_units.json`, `facilities.json`, `responsibilities.json` | `data/cric/` |
| `cric-core` | `services.json`, `rcsites.json`, `federations.json` | `data/cric-core/` |
| `cmssw-releases` | `releases.map`, or `records.json` when there is no map | `data/cmssw-releases/` |
| `jira` | `records.json`, optional `meta.json` | `data/jira/` |
| `indico` | `records.json` | `data/indico/` |
| `dqm` | `records.json` | `data/dqm/` |
| `gocdb-downtimes` | `records.json` | `data/gocdb-downtimes/` |
| `gitlab-docs`, `docsite` | `records.json` | `data/gitlab-docs/`, `data/docsite/` |
| `twiki-eos` | every `*.txt` below the directory | `data/twiki-eos/` |
| `conddb-global-tags` | `records.json` | `data/conddb-global-tags/` |
| `wmstats` | `records.json` | `data/wmstats-workflows/` |
| `dbs` | `records.json` | `data/dbs-datasets/` |
| `cmssst-site-status` | `records.json` | `data/cmssst-site-status/` |
| `tickets`, `servicenow` | `records.json`, optional `meta.json` | `data/tickets/`, `data/servicenow/` |
| `node-health` | `records.json` | `data/node-health/` |
| `slurm`, `htcondor` | `records.json` (one object), optional `meta.json` | `data/slurm/`, `data/htcondor/` |

The five cluster-manager groups (the last three rows) keep their records whole:
there is no field schema, because their downloaders write only what the readers
read and the readers remove email addresses themselves. The reader check still
runs, and an item a reader would skip refuses the group. They were added so
`archi eval --raw-corpus` can pack a cluster-manager instance's caches.

For every group the build:

1. **Refuses input it cannot check.** Any symbolic link (file or directory)
   or any path with a component starting with `.` in the files a group reads
   (for TWiki, anywhere in its tree), any path that resolves outside the
   group's directory, any text file that contains a NUL byte (UTF-16 text
   would hide an address from the redactor) or is not UTF-8, any file path
   that contains an email address, and any file the builder cannot read
   (permission denied) or JSON nested too deeply to walk. The configured
   directory itself may be a link. TWiki pages are the one exception to
   UTF-8: a page is decoded as UTF-8, and only each invalid byte is decoded
   as cp1252 (the Windows superset of Latin-1; Latin-1 for the five bytes
   cp1252 leaves undefined), so valid UTF-8 around a stray byte stays
   intact. A printable fallback character next to or inside an address
   (`jean.dupont` + a stray `0x93` + `@cern.ch`) would hide it from the
   redactor, so the page is also read with its invalid bytes left out; if
   any character that redaction removes from that reading (an address or a
   spelled-out address such as `jdoe[at]cern.ch`) would survive redaction of
   the fallback reading, the page is stored without its invalid bytes
   instead. The same happens when redaction of the fallback reading removes
   a C1 control that an invalid byte became (`a.b@c` + `0x81` + `d.ch`):
   the redactor reads control characters as part of an address, but the
   check below would refuse an address holding one. The lock records `text_pages`, `pages_valid_utf8`,
   `pages_with_fallback_runs`, `pages_bytes_dropped`, `fallback_bytes` (per
   decoding). A text file with more than 1% control characters, and at
   least 16 of them, is refused as binary. Control characters here are C0
   controls, DEL and C1 controls (U+0080 to U+009F); tab, newline, vertical
   tab, form feed and carriage return do not count, and neither do the
   characters of ANSI colour sequences (`ESC[32m`).

   **TWiki file names that are not UTF-8.** Each invalid byte in a file or
   folder name is decoded the same way (cp1252, else Latin-1): a dump
   topic named `DsTotauTo` + three Latin-1 `0xB5` bytes is archived as
   `DsTotauToµµµ.txt`. The lock lists every such file in `renamed_files`:
   `source_name` (invalid bytes written as `\xNN`), `archive_name`, and
   `archived` (false for an empty topic, which is left out as below).
   Two folders whose names decode alike are merged; each file that came
   from the folder with invalid bytes is listed. The group is refused when two files, or a file and a
   folder, decode to the same name; when an invalid byte would become a
   control or invisible character (`0x81`, `0x9D`, `0xA0`, `0xAD` ...);
   and when the name, decoded, with its invalid bytes left out, or as
   listed, shows an address. A non-UTF-8 name in any other group refuses
   it.

   **Empty TWiki topics.** A TWiki topic file that is zero bytes or holds
   only whitespace and NUL bytes is left out of the archive (the reader
   skips it too) and listed in the lock as `empty_topic_files` (a count)
   and `empty_topic_names`. A topic that is empty upstream leaves one or
   two such files (the Aug 31 snapshot has two); an interrupted sync
   leaves many, so more than max(10, 0.1% of the group's text files)
   refuses the group, and so does a tree whose every text file is empty.

   Before anything is stripped or redacted, each text file, each JSON
   string and key, and each archive path is checked for addresses that
   hidden or look-alike characters keep from the redactor. A JSON string
   with an unpaired UTF-16 surrogate (which JSON can carry and UTF-8
   cannot store) refuses the group.

   Each refusal names the file (for JSON, also the record and field, such
   as `records.json [3].description`), the line within the text, and the
   rule that fired: `span rule`, `path rule`, `surrogate rule` or
   `stored-text rule`. It does not repeat the address.

   **Match rule.** The check reads the value twice, keeping a map from
   every character read back to the raw value:

   - as a terminal shows it: terminal sequences removed, hidden characters
     removed, and NFKC applied to each remaining character (NFKC turns
     full-width and other compatibility forms, such as `＠`, `．` or `ﬁ`,
     into their plain forms);
   - when the value has a terminal sequence, also as a program that ignores
     control characters sees it: only the hidden characters removed, NFKC
     applied.

   When the value has HTML character references that decode (`&#8203;`,
   `&#x200B;`, `&shy;`, `&lrm;`, `&#65312;`, `&#64;` ...), both readings
   are also made of the value as a browser shows it, with each decoded
   reference mapped back to its raw span. NFKC here also reads the
   ideographic full stops U+3002 and U+FF61 as `.`.

   Hidden characters are the control characters above (including `ESC`
   and U+009B) and every Unicode format character (category Cf, such as
   U+200E, U+2068, U+FEFF, soft hyphen and zero-width space). Terminal
   sequences are, tried in this order: CSI (`ESC [` or U+009B, parameters,
   a final byte); OSC, DCS, SOS, PM and APC strings (`ESC ]`, `ESC P`,
   `ESC X`, `ESC ^`, `ESC _` or their 8-bit forms, a body of at most 4,096
   characters with no control character in it, ended by BEL, `ESC \` or
   U+009C); and any other `ESC` + optional intermediate bytes + a final
   byte (`ESC ( B`, `ESC >`, `ESC \`). An opening character with no valid
   body and ending is a lone hidden character, which keeps the check linear
   on unterminated strings.

   Both redactors, with all their forms (`@`, `[at]`, `_at_`, `%40`,
   `&#64;`, `NOSPAM` and the rest), match each reading, and each match is
   mapped back to the raw value. Then:

   - **Refused:** the match contains, or sits directly next to, a control
     character, `ESC`, a character of a terminal sequence, or a Cf
     character other than five common invisible ones (soft hyphen,
     zero-width space, zero-width non-joiner, zero-width joiner, word
     joiner). The redactor itself reads every Cf and control character as
     part of an address, so these are refused here rather than stored
     partly redacted. So `jdoe<DEL>x@example.org`, `jdoe@example<ESC>.org`,
     `jdoe<U+200E>x@example.org`, `<U+2068>jdoe<U+2069>@example.org`, an
     address wrapped directly in colour codes (`ESC[31mjdoe@example.org
     ESC[0m`) and a
     string where a control character only joins a word to an address
     (`word<control>jdoe@example.org`) refuse the group.
   - **Removed whole:** otherwise, if the match contains one of those five
     invisible characters or a character NFKC changed, the raw text from
     the match's first character to its last is removed, and the lock
     counts it in `normalized_addresses_removed`. A decoded character
     reference inside the match counts as a character NFKC changed. So
     `jdoe＠example.org`, `jdoe [at] example．org`,
     `jdoe<U+200B> [at] example.org`, `jdoe&#8203;@example.org`,
     `jdoe&shy;@example.org`, `jdoe&#64;example.org`, `jdoe@example。org`,
     an address with a ligature (`ﬁ`) in it and one with a soft hyphen in
     its domain are removed with no fragment left. (`jdoe&lrm;@example.org`
     decodes to a bidi control and refuses.)
   - **Redacted normally:** otherwise the match is left to the normal
     redaction. A character NFKC changes that only sits next to a match
     (`：jdoe@example.org`, `jdoe@example.org…`) or is in text that is not
     an address (`@example™`, `user@host²`) has no effect.

   An archive path with any refused or removed match is refused, since a
   file cannot be renamed. The check runs on the raw value and again after
   the colour codes are stripped, since stripping can join text. Afterwards
   every stored string, and the configured `note` (which is checked but
   not redacted), must be left unchanged by both redactors in every
   reading, or the group is refused. Tab, newline, carriage return,
   vertical tab, form feed and the Unicode line and paragraph separators
   (U+2028, U+2029) are not hidden characters.

   Only then are ANSI CSI sequences (`ESC [` + parameters + a final byte)
   and two-character escapes (`ESC` + one byte from `0` to `~`, such as
   `ESC 7` and `ESC 8`) removed and the value redacted. A lone `ESC`, one
   followed by a space or other intermediate byte (`ESC<space>5`), or a lone
   U+009B is kept, with the text after it. The lock counts the removed
   sequences as `ansi_sequences_stripped` for every group.
2. **Refuses malformed input.** A missing file, invalid JSON, the wrong
   top-level shape, or one record the reader would skip or silently drop (no
   identity key, not an object, a non-numeric GOCDB `downtime_id`, a CRIC
   `responsibilities.json` without its `result` list) refuses the whole group.
   So does a file that has the right container but comes from another
   export: at least 95% of the records in every CRIC and CRIC-core file
   (and at least one record) must carry the key that marks that file (`sitedb_title`, `pledged-CMS`, `potential_max`, `cmssites`,
   `rcsite`, `sites`, `accounting_name`); CRIC `responsibilities.json` must
   have the columns `username, site_name, role` and exactly three fields per
   row (CRIC's `sites-compat` export has five); JIRA keys must look like
   `PROJECT-123`, CMSSW labels must start `CMSSW_`, Indico events must carry
   `_contributions_text` or `_pdf_texts`, CondDB tags `release` or
   `scenario`, and DBS datasets a `/primary/processed/tier` path, each for at
   least 95% of the records (the readers treat these keys as optional, so one
   odd record is allowed). On the real caches each CRIC and CRIC-core key is
   in all of its own file's records and in none of its six siblings'. The group is refused
   with a message naming the file and the record. A group is never packed in
   part.
3. **Removes every email address** from every string, JSON keys included, and
   from whole text files, with `archi.enrichment.anonymizer.redact_email_addresses`
   (which keeps a literal `git@` host account such as
   `git@gitlab.cern.ch:group/repo.git`, and reads a backslash-escaped
   `jdoe\@example.org` or `jdoe\@example\.org`, as Perl, Doxygen and
   shell text write it, as an address, backslash included), then every spelled-out address
   (`jdoe[AT]cern.ch`, `jdoe(at)cern(dot)ch`, `jdoe_at_cern.ch`, `NOSPAM`
   forms) with `redact_obfuscated_email_addresses`; free prose such as
   `john.doe at cern.ch` is kept. It repeats both until neither changes the
   text, checks that no string still changes under either, and records the
   counts (`addresses_removed`, `obfuscated_addresses_removed`, and
   `normalized_addresses_removed` for step 1's whole removals) in the lock.
4. **Drops every field the reader does not read**, plus the group's
   `drop_fields`. For example Indico chairs and speakers keep only their name
   fields, JIRA people keep only `displayName`/`name`/`key`, and a
   documentation record keeps title, URL, body, site, repository and path.
   The WMStats group keeps `Requestor` and removes every `RequestorDN` and
   `DN` key at any depth (a DN also sits in each `RequestTransition` entry);
   no `keep_fields` entry can bring them back.
5. **Runs the group's archi reader** over the result. The reader must report
   health `ok` and a completed scope, and the facts it emits must equal the
   facts it emits from the redacted but unpruned cache. If they differ, the
   field list in `groups.py` misses a field the reader reads (or a
   `drop_fields` entry names one), and the group is refused.

If any group is refused, the build writes nothing and exits 2, listing every
refused group.

## Output

`<group>.tar.zst` per group, and `snapshot.lock.yaml`. Archives are
deterministic: members sorted by path, mtime fixed at 2000-01-01, owner 0,
mode 0644, PAX tar, `zstd -T1` at the configured level. Two builds from the
same input with the same zstd version give byte-identical archives.

The lock records, per group: `collected`, `file_dates` and `note` (when
configured), `archive`, `sha256`, `bytes`,
`file_count`, `record_count` (records as that group's reader counts them),
`addresses_removed`, `obfuscated_addresses_removed`,
`normalized_addresses_removed`, `input_file` (for CMSSW, which input was used),
`text_pages`, `pages_valid_utf8`, `pages_with_fallback_runs`,
`fallback_bytes`, `renamed_files`, `empty_topic_files` and
`empty_topic_names` (TWiki),
`contents_sha256` (a digest of the unpacked files), `archive_dir`, and the
dropped and extra kept fields. It also records who built it, when, on which
host, the archi version and commit, and the zstd version and level.

## Verify

```
python -m archi.snapshot verify --lock snapshot.lock.yaml --archives <dir>
```

For every group in the lock it checks the archive's size and SHA-256, then
decompresses it and checks the member count, that every member is a plain
file under the group's directory, and the contents digest; for TWiki, also
that `empty_topic_files` counts `empty_topic_names` and that no named empty
topic is in the archive. It prints one
`ok` or `FAIL` line per group and exits 1 if any group fails. Unpack only
after it exits 0, for example `zstd -d -c <group>.tar.zst | tar -x -C "$ARCHI_DATA_ROOT"`.

The reader check stages two copies of each group in a temporary directory,
so it needs about twice the largest group's size (TWiki is about 1 GB).
`--tmp-dir` puts it somewhere with room (default: the system temp dir). It
is removed when each group finishes, when the group is refused, on Ctrl-C
and on SIGTERM (the command line turns SIGTERM into a clean exit, ignores a
second SIGTERM while it cleans up, and also
removes the output staging directory); only a SIGKILL leaves it behind. A
full `--tmp-dir` refuses the group with "cannot write the reader-check copy
... (disk full?)".

Both commands need the `zstd` command on `PATH`.
