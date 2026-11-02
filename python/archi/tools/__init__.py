"""Live agent tools an OKG deployment wires through ``agent_tools``.

Unlike :mod:`archi.sources` (generation-pinned graph producers), these
callables answer *live* questions at agent time and label every payload
``boundary: external_live`` so agents cannot confuse the readings with
OKG graph evidence. Wiring contract: each tool is a module-level
function referenced from a deployment's ``deployment.yaml``
``agent_tools`` block (see the module docstrings for adapted blocks).
"""
