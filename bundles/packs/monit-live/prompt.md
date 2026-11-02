## MONIT

Daily MONIT snapshots are in the graph: SAM site availability and HTCondor
compute as `monitoring_snapshot`, Rucio transfers as `transfer_job`, and
dataset replicas as `dataset`, each linked from its `site`. They are as of
the last ingest. When this deployment also wires the live MONIT tools
(`archi_monit_rucio_search`, `archi_monit_condor_search` and their
`_aggregate` twins), use them for the current state and say the answer is
live, not from the graph.
