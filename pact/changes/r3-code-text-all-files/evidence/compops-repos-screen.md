# The secret screen over the comp-ops instance's repositories

Evidence for `acc.r3-code-text-all-files.real-repos` (req.r3-code-text-all-files.secret-screen).
The repository paths, the instance's exclusion list and the CompOps-42 gold are private to the
instance; this file holds the method and the counts only. No file content or value was printed
during the measurement.

## Result

Over the 18 Git repositories of the comp-ops instance, the shipped code-text globs select 9,008
files. The screen leaves out 156 of them. 150 of those are also on the list of 303 files excluded
by hand on that instance (2026-10-31), and none of the 44 git files in the CompOps-42 gold is left
out.

| | files |
|---|---|
| screened (tracked, selected by the shipped globs, at most 1 MB) | 9,008 |
| left out by the screen | 156 |
| of those, also on the instance's hand-made list | 150 |
| left out by the screen only | 6 |
| on the hand-made list, kept by the screen | 153 |
| gold git files (CompOps-42) left out | 0 of 44 |

Left out, by the kind of secret found (a file can have more than one):

| kind | files |
|---|---|
| base64 run of 80+ characters, not hexadecimal and not public material | 100 |
| URL with a password that is not a template, a placeholder or a documentation host | 21 |
| secret-named key, flag, `mysql -p`, .netrc line or XML element with a literal value | 46 |
| Authorization or Bearer header, or `curl -u`, with a literal value | 3 |

## Method

- Code: `archi.sources.code_text.secret_classes` at the commit this evidence row names, run by a
  lane script over `git ls-files` of each checkout, filtered with the shipped `include_globs` and
  `exclude_globs` (okg's own glob matcher) and the shipped 1 MB cap; okg 0.2.0.
- The hand-made list is the one the instance used for its code-text sources: files flagged by a
  credential-pattern scan, then by okg's transcript leak patterns (every literal value or base64
  blob, and every authorization header, credential URL or token match regardless of value). That
  rule exempted gold files by hand; the screen has no such exemption.
- Every rule hit on the files the two lists disagree on was inspected by shape only: each value
  masked to letter, digit and punctuation classes, never printed.

## Reading the differences

- The 6 files only the screen leaves out hold password-shaped literals: a `--password` flag or a
  secret-named key followed by letters and digits (`aaaaaa9999`, `aaaaaaaaaaaa999` as masked), or
  a `user:password` pair under a token key (`aaaaa-aaaa:aaaaa-aaaaaaaa`).
- The 153 files only the hand-made list removed match on references and placeholders:
  environment-variable and template values, file paths, dictionary and function lookups, `%s`
  format strings, CLI help text, values under 8 characters or plain lower-case words, and URLs
  whose password is a placeholder word (`password`, `passwd`, `secret`, `pass`) or whose host is
  loopback. The short-value and plain-word group is where the screen is weaker than the hand list:
  a short literal password, such as a development default, is kept.

## How the rules got here

Each step was measured on the same 9,008 files; each fix has a unit test.

| version | left out | gold lost | change |
|---|---|---|---|
| 1 | 852 | 4 | first rules: a YAML `secret:` block's next line (`secretName: ...`) read as its value |
| 2 | 136 | 0 | key and value on one line; bare words, expressions, templates and prose are references |
| 3 | 170 | 0 | any password in a URL that is not a template counts; `.otf` fonts excluded as binary |
| 4 | 193 | 1 | adversarial review: more key words, flags, .netrc, XML, `curl -u`, PGP keys, public base64 exempt, linear-time patterns |
| 5 | 179 | 0 | a flag's value stops at quotes and brackets (`'--password password')`); `-u` followed by a URL is not a user; f-strings and quoted calls are references |
| 6 | 156 | 0 | second review: identifier-shaped values with a digit are literals; `=>`, any `--*-secret` flag, `mysql -p`, passphrases; only image `data:` URIs exempt; URLs to documentation hosts or with placeholder passwords not counted |
