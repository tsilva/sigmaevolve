OBJECTIVE:
- Improve CURRENT_PROGRAM for higher val_acc.
- Use lower val_loss as the tie-breaker between similar edits.
- Prefer the smallest coherent safe change.
- Use REFERENCE appendices as inspiration only.
- Do not reproduce any REFERENCE or NEGATIVE candidate verbatim.
- The candidate must be textually distinct from CURRENT_PROGRAM and every shown REFERENCE/NEGATIVE example.
TASK CONTEXT:
{{task_context}}
Only CURRENT_PROGRAM is editable.
Later appendices are ordered as REFERENCE, NEGATIVE, CURRENT_PROGRAM.
