"""Synthetic behavior contracts; lexical coverage never validates meaning."""

import copy
import unittest

from scripts.criteria_anchor_ranking import rank_saved_candidates

CRITERIA = 'Popularity is measured by Google Trends using total interest.'
GROUPS = [{'name': 'measurement_source', 'alternatives': ['Google Trends']},
          {'name': 'measurement_quantity', 'alternatives': ['total interest']}]


class CriteriaAnchorRankingTests(unittest.TestCase):
    def test_metric_anchors_reorder_ambiguous_legal_title_without_exclusion(self):
        candidates = [{'title': 'Musk appeal before the court'},
                      {'title': 'Musk appeal measured by Google Trends total interest'}]
        result = rank_saved_candidates(CRITERIA, GROUPS, candidates)
        self.assertEqual(result['order'], [1, 0])
        self.assertEqual([a['coverage'] for a in result['assessments']], [0.0, 1.0])
        self.assertEqual(result['assessments'][0]['missing_groups'],
                         ['measurement_source', 'measurement_quantity'])
        self.assertTrue(result['advisory'])
        self.assertEqual(result['pool_status'], 'anchor_evidence_present')

    def test_or_membership_equal_weight_and_unlisted_rival(self):
        criteria = 'Eligible alternatives: Minds, Voat. Compare Alexa Traffic Rank.'
        groups = [{'name': 'eligible_alternative', 'alternatives': ['minds', 'voat']},
                  {'name': 'comparison_metric', 'alternatives': ['Alexa Traffic Rank']}]
        candidates = [{'text': 'Digg Alexa Traffic Rank'}, {'text': 'Minds Alexa Traffic Rank'},
                      {'text': 'Minds and Voat Alexa Traffic Rank'}]
        result = rank_saved_candidates(criteria, groups, candidates)
        self.assertEqual(result['order'], [1, 2, 0])
        self.assertEqual([a['coverage'] for a in result['assessments']], [0.5, 1.0, 1.0])
        self.assertEqual(len(result['assessments'][2]['matched_groups']), 2)
        self.assertEqual(len(result['order']), len(candidates))

    def test_complete_text_after_character_512_has_original_offsets(self):
        body = 'A neutral prefix. ' * 80 + 'Google Trends and total interest.'
        result = rank_saved_candidates(CRITERIA, GROUPS, [{'text': body}])
        assessment = result['assessments'][0]
        self.assertEqual(assessment['coverage'], 1.0)
        for match in assessment['matches']:
            self.assertGreater(match['start'], 512)
            self.assertEqual(match['field'], 'text')
            self.assertEqual(body[match['start']:match['end']], match['quote'])

    def test_unicode_casefold_punctuation_and_source_bindings(self):
        criteria = 'Use Straße and GOOGLE—TRENDS for this criterion.'
        groups = [{'name': 'street', 'alternatives': ['STRASSE']},
                  {'name': 'source', 'alternatives': ['Google Trends']}]
        candidate = {'text': 'Straße; Google_Trends.'}
        result = rank_saved_candidates(criteria, groups, [candidate])
        self.assertEqual(result['assessments'][0]['coverage'], 1.0)
        self.assertEqual([b['quote'] for b in result['source_anchor_bindings']], ['Straße', 'GOOGLE—TRENDS'])
        self.assertEqual([m['quote'] for m in result['assessments'][0]['matches']], ['Straße', 'Google_Trends'])
        for binding in result['source_anchor_bindings']:
            self.assertEqual(criteria[binding['start']:binding['end']], binding['quote'])

    def test_phrases_do_not_bridge_fields_or_match_partial_tokens(self):
        result = rank_saved_candidates(CRITERIA, GROUPS,
                                       [{'title': 'Google', 'text': 'Trends'},
                                        {'text': 'Google Trendsish; subtotal interest'}])
        self.assertEqual(result['order'], [0, 1])
        self.assertEqual(result['pool_status'], 'no_anchor_evidence')
        self.assertEqual([a['coverage'] for a in result['assessments']], [0.0, 0.0])

    def test_first_match_for_each_alternative_in_each_field(self):
        criteria = 'Minds or Voat may be eligible.'
        groups = [{'name': 'eligible', 'alternatives': ['Minds', 'Voat']}]
        candidate = {'title': 'Minds minds Voat', 'summary': 'minds', 'text': None}
        result = rank_saved_candidates(criteria, groups, [candidate])
        matches = result['assessments'][0]['matches']
        self.assertEqual([(m['phrase'], m['field'], m['start']) for m in matches],
                         [('Minds', 'title', 0), ('Minds', 'summary', 0), ('Voat', 'title', 12)])
        self.assertEqual(result['assessments'][0]['coverage'], 1.0)

    def test_stable_ties_duplicate_candidates_and_empty_pool(self):
        duplicate = {'text': 'Google Trends'}
        result = rank_saved_candidates(CRITERIA, GROUPS,
                                       [{'text': 'none'}, duplicate, duplicate, {'text': 'total interest'}])
        self.assertEqual(result['order'], [1, 2, 3, 0])
        self.assertEqual([a['candidate_index'] for a in result['assessments']], [0, 1, 2, 3])
        empty = rank_saved_candidates(CRITERIA, GROUPS, [])
        self.assertEqual(empty['order'], [])
        self.assertEqual(empty['assessments'], [])
        self.assertEqual(empty['pool_status'], 'no_anchor_evidence')

    def test_unavailable_contract_has_unassessed_coverage_and_preserves_order(self):
        for criteria, groups in [(None, GROUPS), ('', GROUPS), ('  ', GROUPS),
                                  (CRITERIA, None), (CRITERIA, [])]:
            result = rank_saved_candidates(criteria, groups, [{'text': 'Google Trends'}, {'text': 'none'}])
            self.assertEqual(result['order'], [0, 1])
            self.assertEqual(result['contract_status'], 'unavailable')
            self.assertEqual(result['pool_status'], 'unassessed')
            for assessment in result['assessments']:
                self.assertIsNone(assessment['coverage'])
                self.assertIsNone(assessment['missing_groups'])
                self.assertEqual(assessment['matches'], [])

    def test_malformed_contracts_and_unbound_alternatives_fail_closed(self):
        invalid = [
            [{}], [{'name': '', 'alternatives': ['Google Trends']}],
            [{'name': 'source', 'alternatives': []}],
            [{'name': 'source', 'alternatives': 'Google Trends'}],
            [{'name': 'source', 'alternatives': [None]}],
            [{'name': 'source', 'alternatives': ['!!!']}],
            [{'name': 'source', 'alternatives': ['Google Trends', 'GOOGLE—TRENDS']}],
            [{'name': ' SOURCE ', 'alternatives': ['Google Trends']},
             {'name': 'source', 'alternatives': ['total interest']}],
            [{'name': 'source', 'alternatives': ['invented alias']}],
            [{'name': 'source', 'alternatives': ['Google Trends'], 'weight': 2}],
            [None], 'not groups',
        ]
        for groups in invalid:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                rank_saved_candidates(CRITERIA, groups, [])
        with self.assertRaises(ValueError):
            rank_saved_candidates(123, GROUPS, [])
        with self.assertRaises(ValueError):
            rank_saved_candidates(None, [{'name': 'source', 'alternatives': []}], [])

    def test_candidate_missing_none_and_unsupported_fields(self):
        result = rank_saved_candidates(CRITERIA, GROUPS, [{}, {'title': None, 'summary': None, 'text': None}])
        self.assertEqual([a['coverage'] for a in result['assessments']], [0.0, 0.0])
        for candidates in ([None], [{'text': 0}], [{'title': []}], [{'summary': {}}], 'not candidates'):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                rank_saved_candidates(CRITERIA, GROUPS, candidates)

    def test_inputs_and_existing_scores_are_not_mutated_or_returned_by_reference(self):
        candidates = [{'text': 'Google Trends', 'relevance_score': 0.123, 'nested': {'value': 1}}]
        groups = copy.deepcopy(GROUPS)
        before = copy.deepcopy((groups, candidates))
        result = rank_saved_candidates(CRITERIA, groups, candidates)
        result['assessments'][0]['matches'][0]['quote'] = 'changed output'
        result['source_anchor_bindings'][0]['phrase'] = 'changed output'
        self.assertEqual((groups, candidates), before)
        self.assertEqual(candidates[0]['relevance_score'], 0.123)

    def test_negated_full_coverage_remains_semantically_and_historically_unverified(self):
        candidates = [{'text': 'This is not Google Trends and does not measure total interest.'}]
        result = rank_saved_candidates(CRITERIA, GROUPS, candidates)
        self.assertEqual(result['assessments'][0]['coverage'], 1.0)
        self.assertEqual(result['semantic_status'], 'unverified')
        self.assertEqual(result['historical_status'], 'unverified')
        self.assertEqual(result['assessments'][0]['semantic_status'], 'unverified')
        self.assertEqual(result['assessments'][0]['historical_status'], 'unverified')


if __name__ == '__main__':
    unittest.main()
