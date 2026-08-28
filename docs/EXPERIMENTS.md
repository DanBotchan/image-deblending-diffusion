# Paper experiment map

This file is the release checklist connecting paper claims to executable code. "Packaged" means the code/config is present here; it does not mean the expensive GPU run was repeated in this extracted repository.

| Paper experiment | Protocol | Release status |
|---|---|---|
| Main known scalar | 300 pairs, 3 Stage-1 candidates, PickScore, one same-ID Stage-2 refinement | Packaged |
| Main unknown scalar | 300 pairs, alpha in [0.2, 0.8], 3 candidates, PickScore, Stage 2 | Packaged |
| Main spatial alpha | 300 pairs, polynomial alpha map, 3 candidates, PickScore, Stage 2 | Packaged |
| Domain generalization | 100 Vehicle--Flower pairs for all three scenarios | Core configs packaged; exact held-out manifest/hash pending |
| Two-stage ablation | Compare each selected Stage-1 candidate with its exact same-ID Stage-2 refinement | Evaluation code packaged; convenience command pending |
| Text guidance | 100 pairs; none, standard CFG, class-level, cross-negative; CFG 1, 2, 3 | Core overrides supported; frozen experiment configs pending |
| Timestep grid | Stage 1 start 20/40/60/80/100; Stage 2 start 50/200/350/500; 20 spatial pairs | Core overrides supported; sweep driver and exact subset manifest pending |
| Recurrence grid | Stage 1 4/6/8/10; Stage 2 2/4/6/8/10; same 20 pairs | Core overrides supported; sweep driver pending |
| PickScore vs CLIP | Same 3 Stage-1 candidates; freeze selection before matching Stage 2 | PickScore packaged; CLIP selector comparison pending |
| Candidate budget | 10 pairs/scenario; cumulative 1/2/3/4/6/8/12/16 candidates | Core sampling supports 16 candidates; cumulative selector analysis pending |
| Automated prompts | Qwen2-VL-7B-Instruct preliminary 20 and held-out end-to-end 80 | Revision-pinned VLM prompt generator and frozen instruction packaged; end-to-end convenience runner pending |
| DoubleDIP best-of-three | 300 comparison + 100 generalization pairs, all scenarios | External adapter pending; iteration-count discrepancy must be resolved |

## Canonical method settings

| Setting | Known scalar | Unknown scalar | Spatial |
|---|---:|---:|---:|
| Stage-1 steps/start | 80/64 | 100/80 | 100/80 |
| Stage-1 candidates | 3 | 3 | 3 |
| Stage-1 CFG | 2.0 | 2.0 | 2.0 |
| Stage-1 recurrence | 8 | 8 | 8 |
| Stage-1 LPIPS weight | 100 | 100 | 100 |
| Stage-2 steps/start | 600/350 | 600/350 | 600/350 |
| Stage-2 recurrence | 8 | 8 | 8 |
| Stage-2 LPIPS weight | 20 | 20 | 20 |
| Alpha update | fixed | Adam every 5 steps, lr 0.01 | Adam every 5 steps, lr 0.01 |

The configs under `configs/paper/` are sourced from the saved canonical run configs rather than the stale development defaults.
