# C01 training review

Result: PASS; 119 checks, 0 blockers.

| Epoch | Examples / batches | AdamW step (all 104) | Training seconds |
|---|---|---|---|
| 1 | 2980 / 373 | 373 | 139.831 |
| 2 | 2980 / 373 | 746 | 136.409 |

Both committed snapshots and all checkpoint hashes match. Snapshot adapters match saved adapters tensor-for-tensor. Parameter order and frozen base fingerprint agree between epochs. Epoch2 restoration names exactly epoch1 manifest and reports RNG restoration; both epoch order hashes were independently recomputed from seed+epoch. All pinned code/config and installed dependency versions match.

Limits: no replay/model execution; no direct post-restore RAM dump. The epoch2 AdamW step of 746 supports preserved optimizer continuity. Formal train_epoch reseeds at each epoch start. Base weights and any market datasets were not read. Financial selection and May approval belong to other reviewers.
