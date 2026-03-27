OBJECTIVE:
- Propose a safe improvement to CURRENT PROGRAM.
- Optimize for higher val_acc.
- Use lower val_loss as a secondary signal when comparing similar edits.
- Prefer the smallest coherent change likely to improve results.
- Use REFERENCE PROGRAMS as inspiration only.
TASK CONTEXT:
{{task_context}}
REFERENCE PROGRAMS:
{{prior_programs}}
AVOID THESE RECENT NEGATIVE TRIALS:
{{negative_trials}}
CURRENT PROGRAM:
Patch this program. SEARCH blocks must match text from CURRENT PROGRAM, not from REFERENCE PROGRAMS.
{{current_program}}
REPLACEMENTS:
