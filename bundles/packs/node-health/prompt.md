## Node health

`compute_node` records say whether a node is healthy, degraded or unhealthy as of `observed_at`,
and `failing_checks` names what failed; the detail is on its `node_health_check` records. A
reading is as old as its `observed_at`: say so when you cite one, and do not call a node healthy
from a reading older than the freshness the pack declares.
