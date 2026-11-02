# analysis-notes pack

CMS analysis notes, read from a local mirror of
[`gitlab.cern.ch/tdr/notes`](https://gitlab.cern.ch/tdr/notes). Each note is
a LaTeX tree named `<PAG>-YY-NNN` (for example `HIG-99-001`) or, for older
notes, `AN-YY-NNN`.

| Emitted | Id | What it is |
|---|---|---|
| `analysis_note` | `an:<note id>` | One note: id, title (`\title{}`), PAG |
| `section` | `an_section:<note id>:<file stem>` | One section `.tex` (or the main `.tex` of a single-file note) |
| `document_chunk` | `chunk:<hash>` | A 4000-character window of a section, so note text ranks against TWiki text |
| `note_figure` | `an_figure:<note id>:<label>` | A `figure` environment: caption, label, the image files it places |
| `reference` | `an_ref:<hash>` | A bibliography entry, shared across notes by bib key |
| `physics_analysis_group` | `pag:<code>` | The PAG hub a note belongs to |

Edges: a note contains its sections and figures and cites its references;
a section contains its chunks; a chunk that `\ref`s a figure label
references that figure; a note references its PAG. The
**bibliography-to-TWiki bridge** adds `analysis_note -> documentation_page`
when a bib entry's URL names a TWiki topic that the twiki pack ingested.

## Fetching the mirror

The source reads a clone from disk and never touches the network. Cloning
needs CERN credentials, which is why the pack says `needs_credentials: true`
and names `CERN_GITLAB_TOKEN`.

1. Create a personal access token at
   `https://gitlab.cern.ch/-/profile/personal_access_tokens` with the
   `read_api` and `read_repository` scopes. Your account must be able to read
   the `tdr/notes` group.
2. Export it as `CERN_GITLAB_TOKEN` in the shell that runs the commands
   below. Do not write it into a registry entry, a config file or a clone URL.
3. Clone every note project into one directory:

```bash
export MIRROR=/absolute/path/to/analysis-notes   # the answer to --analysis-notes-mirror-root
mkdir -p "$MIRROR"
page=1
while :; do
  body=$(curl -sSf --header "PRIVATE-TOKEN: $CERN_GITLAB_TOKEN" \
    "https://gitlab.cern.ch/api/v4/groups/tdr%2Fnotes/projects?per_page=100&include_subgroups=true&archived=false&page=$page")
  [ "$(echo "$body" | jq length)" = 0 ] && break
  echo "$body" | jq -r '.[] | [.path, .path_with_namespace] | @tsv' |
  while IFS=$'\t' read -r path full; do
    [[ "$path" =~ ^[A-Z0-9]{3}-[0-9]{2}-[0-9]{3}$|^AN-[0-9]{2,4}-[0-9]{3}$ ]] || continue
    if [ -d "$MIRROR/$path/.git" ]; then
      git -C "$MIRROR/$path" -c http.extraHeader="PRIVATE-TOKEN: $CERN_GITLAB_TOKEN" pull --ff-only --quiet
    else
      git -c http.extraHeader="PRIVATE-TOKEN: $CERN_GITLAB_TOKEN" \
        clone --quiet --depth 1 "https://gitlab.cern.ch/$full.git" "$MIRROR/$path"
    fi
  done
  page=$((page + 1))
done
```

The token travels in an HTTP header, so it is not stored in each clone's
`.git/config`. The loop is safe to repeat: existing clones are fast-forwarded.
Needs `git`, `curl` and `jq`.

Directories that do not match `<PAG>-YY-NNN` or `AN-YY-NNN` are ignored. The
PAG codes read are HIG TOP SUS SMP B2G BPH BTV EXO FTR HIN JME MUO TAU TRK EGM
PPS FSQ. A project the walker accepts but that holds no `.tex` is skipped.

## Installing

```bash
okg install --profile <bundle> --deployment-name <slug> \
  --analysis-notes-mirror-root "$MIRROR" --analysis-notes-min-notes <floor> ...
```

The twiki pack must be installed too (this pack requires it). To decide which
bibliography-to-TWiki bridge edges exist, this source reads the same dump the
twiki source reads (`--twiki-eos-root`), with the same physics filter
(`--twiki-physics-filter`), and keeps the pages that source would emit. A bib
entry that names a page the twiki source did not keep gets no edge, rather
than an edge to nothing. The cost is a second read of the dump.

## Coverage invariant and the note floor

`analysis_notes_coverage` (error) requires live `analysis_note` and `section`
records. It does not require `reference`, so a mirror without `.bib` files
still publishes.

The note-count floor is the `--analysis-notes-min-notes` answer (default 1,
which only catches an empty mirror). The source enforces it at preflight: a
mirror with fewer note directories fails and publishes nothing, so a walker
whose directory or section rules stop matching cannot silently publish a
fraction of the corpus. A static invariant cannot read an answer, which is why
the floor lives in the source. The cms-physics bundle sets a production value.

## What the walker does

- A top-level `.tex` other than the main file and macro files (`*defs.tex`) is
  a section. Numbered (`03-Selection.tex`), `App`/appendix and letter-prefixed
  files come first, then the rest.
- Below the top level, a file is a section only when the main `.tex` reaches
  it through `\input`, `\include`, `\subfile` or `\import`.
- A note with no section files becomes one section: its main `.tex`.
- Figures are found in sections and the main file. Image files are located
  with a `stat`, resolved through `\graphicspath`, and never opened here; okg's
  ingest renders them.
- Bibliography report numbers are split, and the `CMS-<PAG>-YY-NNN` code is
  kept as `analysis_code` so the cms-analyses pack can join a citation to its
  CDS record.
