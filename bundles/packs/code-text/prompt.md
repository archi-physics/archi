## Code text

The text of a repository file is a `document` (`doc:<repo>:<path>`) cut
into `document_chunk` records. Python is cut per function or class, each
chunk with `start_line`, `end_line` and `symbol`; every other text file
(Markdown, YAML, JSON, shell scripts, configuration, other code) is cut
into plain text windows. Reading the text of a `file:` node points to its
document. Quote from the chunk and cite path and lines. A file that holds a
literal secret is left out, so its `file:` node has no text.
