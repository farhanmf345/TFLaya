# Hybrid Laya scoring implementation

## Factor ownership
- **Python scoring:** Financial Strength, Legal Standing, Geographic Alignment, and DEI Bonus. These rely primarily on arithmetic, dates, explicit matching, and fixed points.
- **Laya assessment:** Domain Experience, Technical Capability, and Certifications & Registrations. These require more interpretation of project relevance, capability evidence, and certificate relevance.
- **Python finalization:** Python combines all seven dimensions, applies existing subject-fail/hard-disqualifier rules, normalizes the score, and determines the final verdict.

## Important limitation of this Laya backend
The configured Laya `Router` is a yes/no classifier, not a generative model that can return a criterion-by-criterion score with a grounded explanation. For the three interpretive factors, this implementation therefore records Laya's probability as an **estimated factor score** and explicitly labels it as such in the notes. This is narrower than the previous all-seven-factor Laya scoring, but it is not equivalent to a fully rubric-grounded assessment. Do not use these probabilities as proof that a bidder meets a mandatory requirement.

A stronger next step is to use a generative model to extract criterion-level evidence into validated structured output (criterion, tender clause, company evidence, status, points, uncertainty), then have Python validate the result and calculate points from the tender's actual rubric. Missing/ambiguous evidence should be sent for review, not treated as a failure by default.

## Safety and fallback
- If Laya fails or returns invalid output, all seven existing Python factor scores are retained for that run.
- Formula-based dimensions are never replaced by Laya output.
- Python remains responsible for score aggregation and the final eligibility verdict.

## Tests
The offline Laya scoring, tender-requirement, and tender-document-reader tests are included. These tests verify the factor split and fallback-oriented Laya scoring path; they do not establish real-world scoring accuracy.


## Mandatory eligibility gate (2026-10-14)

- Added `server/core/mandatory_eligibility.py` to check whether the available tender text exposes mandatory qualification clauses.
- A high score alone no longer produces `ELIGIBLE` when the mandatory clauses are missing, ambiguous, or not verifiable from the available company records; the verdict becomes `CONDITIONAL` and the unresolved checks are stored in the result gaps/recommendations.
- Explicit certificate clauses are checked against the company's recorded certificates. A certificate absent from the profile is `UNVERIFIED`, not a confirmed failure. Other mandatory clauses that this layer cannot reliably validate are also routed for review.
- Existing confirmed hard-disqualifier logic remains responsible for `INELIGIBLE`; this gate does not reject a company merely because evidence is missing.
- Scoring version bumped to `hybrid_probability_v3_mandatory_gate`.
- Targeted tests: 14 passed. Full test suite: 32 passed, 1 unrelated import failure because `google.genai` is not installed in the test environment.
