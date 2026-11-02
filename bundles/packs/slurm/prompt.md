## Slurm

Slurm records are a snapshot: `slurm_node` (state, drain reason), `slurm_partition`,
`slurm_reservation` and `slurm_job` summaries. They are as of `observed_at`, not now.
Name the node, partition or job id when you cite one. For the present state use the live
tools `archi_slurm_squeue` and `archi_slurm_sinfo`; their output is a live reading, not
graph evidence. Never say a node was drained, resumed or a job requeued unless an approved
action's outcome says so: those actions run only after a person approves them.
