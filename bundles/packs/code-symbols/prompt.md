## Code symbols

Functions and classes are `code_symbol` records linked from their
`source_file` by `contains`; `calls` links a function to the ones it calls,
and `code_import` / `code_reference` record imports and uses. Find a
definition by symbol name, then read its body from the file's text.
