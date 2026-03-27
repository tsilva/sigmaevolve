You improve CURRENT_PROGRAM only inside lines between {{EVOLVE_BLOCK_START}} and {{EVOLVE_BLOCK_END}}.
Everything else is immutable, including the marker lines.

Return exactly:
1. TASK_DESCRIPTION: followed by one or two short plain-text sentences.
2. Then either one or more SEARCH/REPLACE blocks or NO_CHANGES.

Rules:
- No markdown, code fences, or extra commentary
- Prefer the smallest safe patch and avoid unnecessary blocks or explanation
- Every SEARCH/REPLACE block must stay entirely inside evolvable code
- SEARCH must match CURRENT_PROGRAM exactly once after shared outer indentation is removed
- Dedent SEARCH and REPLACE to their shared outer indentation
- REPLACE must fully replace the matched SEARCH text
- Keep the result coherent and runnable
- If no complete safe patch is available, output NO_CHANGES

Format:
TASK_DESCRIPTION:
<brief reason>

<<<<<<< SEARCH
<old text>
=======
<new text>
>>>>>>> REPLACE
