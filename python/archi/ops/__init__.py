"""The operations layer (ADR 0002 section 7).

`archi.ops.record` writes the operational memory: incidents, proposals,
decisions, actions and outcomes, admitted into okg as a declared producer.
`archi.ops.precedents` reads it back: past incidents on the same entity with
what was done and how it turned out.
The automation runtime that calls it is PACT r3-ops-runtime.
"""
