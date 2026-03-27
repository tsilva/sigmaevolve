You may edit CURRENT_PROGRAM only inside code between
{{EVOLVE_BLOCK_START}} and {{EVOLVE_BLOCK_END}}.
Everything else, including the marker lines, is immutable.

Goal:
- Maximize val_acc. Use lower val_loss only to break ties between similar candidates.
- Randomly choose either a focused improvement or a broader revamp when it is safe and promising.

Return exactly:
1. TASK_DESCRIPTION: one or two short plain-text sentences.
2. Then either one or more SEARCH/REPLACE blocks or NO_CHANGES.

Rules:
- No markdown, code fences, or commentary.
- Appendix order: REFERENCE, NEGATIVE, CURRENT_PROGRAM.
- Use REFERENCE appendices as inspiration only.
- Never copy any shown REFERENCE or NEGATIVE example verbatim.
- The candidate must stay textually distinct from CURRENT_PROGRAM and every shown REFERENCE/NEGATIVE example.
- Use as few SEARCH/REPLACE blocks as needed.
- Each SEARCH/REPLACE block must stay fully inside evolvable code.
- SEARCH must match exactly once in CURRENT_PROGRAM after shared outer indentation is removed.
- Dedent SEARCH and REPLACE by that shared outer indentation.
- REPLACE must fully replace SEARCH.
- Keep the result coherent and runnable.
- If no complete safe improvement is available, output NO_CHANGES.

Format:
TASK_DESCRIPTION:
<brief reason>

<<<<<<< SEARCH
<old text>
=======
<new text>
>>>>>>> REPLACE
