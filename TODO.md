# TODO

## Close The Loop Gap

1. Wire failed-trial feedback into generation.
   The generator already accepts `negative_trials`, but the orchestrator currently always passes an empty list. Pull recent failed or low-signal trials from storage and feed compact failure summaries into prompt construction.

2. Upgrade parent sampling into parent-plus-inspirations sampling.
   Replace the current single-parent weighted sample with a sampler that selects one main parent plus additional inspiration trials from the database. Include lineage metadata so each child records exactly which trials influenced it.

3. Introduce an evaluator pool.
   The current runner path is effectively a single evaluator. Add an evaluator registry so tracks can run multiple evaluation passes or scoring functions and combine them into one persisted result.

4. Persist explicit lineage and mutation metadata.
   Provenance is recorded, but lineage is still too implicit. Store parent IDs, inspiration IDs, mutation summaries, and evaluator IDs in first-class structured fields or a normalized lineage table.

5. Move from reconcile polling to a long-running controller loop.
   `reconcile_track` already performs the core controller step, but it is still a one-shot call. Add a continuously running controller service that keeps the ready queue full, launches work, and sweeps stale trials automatically.

## Nice To Have

1. Support true patch or diff generation.
   The generator currently returns a full `train.py` and relies on post-generation validation. Add a diff-oriented mutation mode so the model proposes smaller, more traceable edits.

2. Make prompt templates configurable.
   The prompt contract is hardcoded today. Move prompt templates and sampling policy into versioned config so the scientist or engineer layer can iterate without code edits.

3. Expand beyond one evolvable block and one component.
   The current baseline uses a single evolve block inside one training script. Generalize the program representation so multiple components can evolve independently while still being evaluated as one candidate.
