"""Bundle smoke evals (PACT r3-bundle-evals).

`archi eval` asks each question of a bundle's smoke set through a small
tool-calling agent that reads the instance only over its MCP endpoint, then
checks the answer with deterministic rules (cited node ids, subtypes or
sources, required strings). There is no LLM judge, and the model is a local
one, so a run costs nothing.

This harness is a stop-gap. The eval framework tracked in #619 replaces it;
its question files and report format are kept so they can move across.
"""
