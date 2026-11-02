## HTCondor

HTCondor records are a pool snapshot: `condor_pool` (CPU totals by slot state),
`condor_machine` and `condor_schedd` (running, idle and held jobs). They are as of
`observed_at`, not now. For the present state use the live tools `archi_condor_status` and
`archi_condor_q`; their output is a live reading, not graph evidence. archi changes nothing in
an HTCondor pool.
