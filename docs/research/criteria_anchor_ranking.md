# Offline criterion-anchor ranking research prototype

`scripts/criteria_anchor_ranking.py` exposes one pure standard-library function:

```python
from scripts.criteria_anchor_ranking import rank_saved_candidates

result = rank_saved_candidates(
    criteria_text="Measure Google Trends using total interest.",
    groups=[
        {"name": "measurement_source", "alternatives": ["Google Trends"]},
        {"name": "measurement_quantity", "alternatives": ["total interest"]},
    ],
    candidates=[
        {"title": "An unrelated report", "text": "Other information."},
        {"text": "Google Trends reports total interest."},
    ],
)
assert result["order"] == [1, 0]
```

This is an explicit, optional local research call over saved candidates. It has
no file access, network, model, embedding, provider or retrieval operation. It
does not alter `NewsPipeline`, its callers or live routing. An existing CLI
caller already supplies criteria in its query string; this prototype adds
caller-authored anchor roles rather than correcting universal criteria omission.

## Contract and matching

Criteria are text or `None`; groups are a sequence of mappings or `None`.
Each group has exactly `name` and a nonempty sequence of `alternatives`.
Names must be nonempty text; duplicate names are checked after casefolding and
collapsing whitespace. Alternative phrases must contain alphanumeric tokens and
must not duplicate another normalized alternative within that group. The same
phrase may appear in different groups because roles remain caller-authored.

Every alternative must bind to the supplied criteria as a contiguous sequence
of casefolded Unicode alphanumeric tokens (`[^\W_]+`). Binding returns the first
original criterion character span and quote. Punctuation and whitespace between
tokens are separators; underscores are separators too. There is no stemming,
synonym expansion, semantic contract extraction or automatic role validation.

Full candidate `title`, `summary` and `text` fields are scanned independently.
Absent and `None` fields are empty; other non-text fields raise `ValueError`.
Phrases cannot bridge fields. Each alternative records its first match in each
field, with original character offsets and quote. Text after character 512 is
included. Raw markup remains ordinary text; no markup semantics are inferred.

## Output and fallbacks

The returned dictionary contains:

- `order`: every original candidate index, stable-sorted by descending coverage.
- `assessments`: entries in original candidate order, with `candidate_index`,
  `coverage`, `matched_groups`, `missing_groups` and `matches`. Each match has
  `group`, `phrase`, `field`, `start`, `end` and `quote`.
- `source_anchor_bindings`: one criterion binding per supplied alternative,
  with `group`, `phrase`, `start`, `end` and `quote`.
- `contract_status`, `pool_status`, `unassessed_reason`, `advisory=True`, and
  `semantic_status`/`historical_status`, both always `unverified`.

Alternatives are OR within each group. Coverage is matched distinct groups
divided by total groups, with equal group weight. Multiple mentions or matched
alternatives cannot inflate a group's weight. Ties retain original order.
Candidates and their existing scores are never mutated or filtered.

Missing/blank/token-empty criteria or missing/empty groups return
`contract_status="unavailable"`, `pool_status="unassessed"`, unchanged order,
and `None` coverage/matched/missing groups. Unsupported types and nonempty
malformed groups still fail closed. A nonempty, usable contract whose anchor
does not bind to the criteria raises `ValueError`; aliases are not invented.
An assessed all-zero or empty candidate pool preserves order and returns
`pool_status="no_anchor_evidence"`. Otherwise the status is
`anchor_evidence_present`. These statuses describe lexical evidence only.

## Limits and verification

The synthetic negative control, “This is not Google Trends and does not measure
total interest,” receives full coverage. Negation, misleading mentions, entity
roles, conditions and semantic criterion satisfaction are not resolved by this
helper. Coverage is not a relevance probability, historical availability
judgment, outcome verification or permission to use a source in a forecast.

Run the focused synthetic contracts from the repository root:

```powershell
py -B -m unittest discover -s tests -p test_criteria_anchor_ranking.py -v
```

The saved-pool development check uses fixed contracts for two previously seen
questions and preserves every original saved candidate. Its baseline is saved
order, not reconstructed historical pipeline ranking. It supports descriptive
checks only: no held-out benchmark, human validation, retrieval-quality gain or
forecast improvement is established. A reranker cannot recover useful evidence
absent from its candidate pool. The prototype remains uncommitted research code
with no automatic integration or changes to frozen study outputs.
