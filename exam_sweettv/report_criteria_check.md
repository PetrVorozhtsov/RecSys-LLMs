# Report criteria audit

Sources checked: exam_quiz_GSB.md, LLM4Rec. Midterm Quiz.pdf, and the full
homework_guidelines.pdf. The PDF is entirely in English. Main content occupies
five pages; references occupy one additional page.

| Requirement | Evidence / status |
|---|---|
| Title, student, individual status, email, date, topic | Page 1 |
| Four-sentence, 50–150-word abstract | Page 1 |
| Introduction, motivation, example, contributions | Section 1 |
| Related work, valid sources and alternatives | Section 2; seven numbered references |
| Join, threshold rationale, missing IDs | Section 3.1 |
| Two implemented personalised methods plus popularity | Sections 3.2 and 4 |
| Split, candidate pool, repeats and output construction | Sections 3.3 and 4 |
| Parameters, environment, seeds, runtime | Sections 3.2 and 4.1; independent rerun measured 511.10 seconds; all four output artifacts byte-identical |
| Two identical users across methods, five ordered IDs, titles, categories | Section 5.1 |
| User histories and specific comparisons | Section 5.1 |
| Strategy, cost, association vs content/MF/PageRank | Sections 5 and 5.2 |
| Failure, cause, change and verification | Section 5 documents verified failure and report correction; model fix remains unimplemented |
| AI disclosure and personal/assistant verification distinction | Section 6 |
| Repository link and private-score screenshot | Page 1 and Section 6 |
| No fabricated business results, experiments or personal verification | Explicit limitations in Sections 4–6 |
| Original unmodified AI export | Separate delivery requirement; not generated or rewritten by this audit |

## Unresolved implementation and personal requirements

These cannot be repaired by wording alone:

- The content model does not share the popularity cold-start fallback.
- Validation-only users receive no predictions in the recorded evaluation.
- Content features merge attributes from later schedules; strict metadata
  timing requires a separate model change and a fresh evaluation.
- Final association pairs are not refitted on weeks 11–30; lift uses a
  nonstandard denominator. Both are documented.
- The selected association method is not the best measured overall method:
  popularity is stronger on the recorded local hold-out.
- The student must personally review the reasoning and verification example.
  The assistant did not claim the student performed an unconfirmed calculation.
- A fresh original native session export must accompany the PDF. A rewritten
  history.json is not claimed to be the original native export.

The report now answers the required topics, but this audit does not certify
full implementation compliance or guarantee a grade.
