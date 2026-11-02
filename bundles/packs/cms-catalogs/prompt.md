## CMS catalogs

CRIC gives `site`, `facility`, `storage_endpoint`, `compute_endpoint`,
`operator`, `infrastructure_service` and `federation` records; SITECONF a
`site_config` per site; DBS and WMStats `dataset` and `workflow` records;
GOCDB `downtime` records; CondDB `global_tag` records. A site's facts are
spread over these: start from the `site` node and expand its links. Cite a
record by its name (site name, dataset path, workflow name, global tag).
