"""Offline, caller-authored anchor coverage; no semantic or historical validation."""

import re
from collections.abc import Mapping, Sequence

_WORD = re.compile(r'[^\W_]+')
_FIELDS = ('title', 'summary', 'text')


def _tokens(text: str) -> list:
    return [(match.group().casefold(), match.start(), match.end()) for match in _WORD.finditer(text)]


def _first_span(tokens: list, phrase: tuple) -> tuple | None:
    for start in range(len(tokens) - len(phrase) + 1):
        if tuple(token[0] for token in tokens[start:start + len(phrase)]) == phrase:
            return tokens[start][1], tokens[start + len(phrase) - 1][2]
    return None


def _groups(groups: Sequence[Mapping] | None) -> list:
    if groups is None:
        return []
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise ValueError('Anchor groups must be a sequence of mappings or None')
    result, names = [], set()
    for group in groups:
        if not isinstance(group, Mapping) or set(group) != {'name', 'alternatives'}:
            raise ValueError('Each group requires exactly name and alternatives')
        name, alternatives = group['name'], group['alternatives']
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Group names must be nonempty text')
        key = ' '.join(name.casefold().split())
        if key in names:
            raise ValueError('Duplicate normalized group name')
        names.add(key)
        if (not isinstance(alternatives, Sequence) or isinstance(alternatives, (str, bytes))
                or not alternatives):
            raise ValueError('Every group requires a nonempty alternative sequence')
        phrases, seen = [], set()
        for phrase in alternatives:
            if not isinstance(phrase, str):
                raise ValueError('Anchor phrases must be text')
            normalized = tuple(token[0] for token in _tokens(phrase))
            if not normalized or normalized in seen:
                raise ValueError('Empty or duplicate normalized alternative')
            seen.add(normalized)
            phrases.append((phrase, normalized))
        result.append((name, phrases))
    return result


def rank_saved_candidates(
    criteria_text: str | None,
    groups: Sequence[Mapping] | None,
    candidates: Sequence[Mapping],
) -> dict:
    """Return new advisory assessments and an order of original candidate indices.

    Groups have exactly ``name`` and ``alternatives``. Every alternative must bind
    to the criteria. Coverage is equal-weight OR-group presence, never criterion
    satisfaction. Missing criteria/groups are unassessed; malformed inputs raise
    ValueError. No file access, model call, filtering or input mutation occurs.
    """
    if criteria_text is not None and not isinstance(criteria_text, str):
        raise ValueError('Criteria must be text or None')
    definitions = _groups(groups)
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError('Candidates must be a sequence of mappings')
    candidate_fields = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError('Each candidate must be a mapping')
        fields = {}
        for field in _FIELDS:
            value = candidate.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError('Candidate ' + field + ' must be text or None')
            fields[field] = (value or '', _tokens(value or ''))
        candidate_fields.append(fields)

    criteria_tokens = _tokens(criteria_text or '')
    reason = 'criteria_unavailable' if not criteria_tokens else (
        'anchor_groups_unavailable' if not definitions else None)
    result = {'advisory': True, 'semantic_status': 'unverified', 'historical_status': 'unverified',
              'contract_status': 'unavailable' if reason else 'available',
              'pool_status': 'unassessed' if reason else 'no_anchor_evidence',
              'unassessed_reason': reason, 'order': list(range(len(candidates))),
              'source_anchor_bindings': [], 'assessments': []}
    if not reason:
        for name, alternatives in definitions:
            for phrase, normalized in alternatives:
                span = _first_span(criteria_tokens, normalized)
                if span is None:
                    raise ValueError('Anchor does not bind to supplied criteria: ' + name + ': ' + phrase)
                start, end = span
                result['source_anchor_bindings'].append(
                    {'group': name, 'phrase': phrase, 'start': start, 'end': end,
                     'quote': criteria_text[start:end]})

    for index, fields in enumerate(candidate_fields):
        matches, matched = [], set()
        if not reason:
            for name, alternatives in definitions:
                for phrase, normalized in alternatives:
                    for field, (text, tokens) in fields.items():
                        span = _first_span(tokens, normalized)
                        if span is not None:
                            start, end = span
                            matched.add(name)
                            matches.append({'group': name, 'phrase': phrase, 'field': field,
                                            'start': start, 'end': end, 'quote': text[start:end]})
        result['assessments'].append(
            {'candidate_index': index, 'coverage': None if reason else len(matched) / len(definitions),
             'matched_groups': None if reason else [name for name, _ in definitions if name in matched],
             'missing_groups': None if reason else [name for name, _ in definitions if name not in matched],
             'matches': matches, 'semantic_status': 'unverified', 'historical_status': 'unverified'})
    if not reason and any(a['coverage'] > 0 for a in result['assessments']):
        result['pool_status'] = 'anchor_evidence_present'
        result['order'].sort(key=lambda index: -result['assessments'][index]['coverage'])
    return result
