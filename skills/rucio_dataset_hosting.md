# CMS Rucio Dataset Hosting

Use this skill for questions about RSEs, datasets, replicas, storage endpoint
impact, production output, tape/disk placement, or hosted data.

## OKG Mapping

- RSE or storage element names normally map to `storage_endpoint` nodes whose
  IDs start with `se:`.
- Datasets normally map to `dataset:*` nodes.
- Site names map to `site:T*_...` nodes and may contain or reference storage
  endpoints.
- Rucio overlay evidence may be exposed through storage endpoint to dataset
  edges, monitoring snapshots, or bounded `okg.v_*` views.

## Traversal Pattern

1. Search literal RSE, site, or dataset names.
2. Use `inspect` on the best `se:*`, `site:*`, or `dataset:*` candidate.
3. Use `aggregate` before listing many rows from high-cardinality
   hosting relationships.
4. Use `expand` or `expand` only after confirming the relevant
   edge direction and relation type.
5. For impact/count questions, prefer bounded `query` over row-by-row
   traversal when an `okg.v_*` impact view exists.

## Caveats

- Do not treat missing monitoring evidence as proof that an RSE hosts nothing.
- Separate current generation graph facts from historical tickets.
- For tape/disk policy questions, cite both storage endpoint evidence and
  ticket/document policy evidence when both are present.
