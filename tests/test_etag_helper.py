import unittest

from starlette.requests import Request

from analyzing_llm_rationale.etag_helper import check_if_none_match, json_or_304, make_etag


def _dummy_request(headers: dict) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()],
    }
    return Request(scope)


class ETagHelperTests(unittest.TestCase):
    def test_make_etag_deterministic(self):
        e1 = make_etag({"a": 1, "b": 2})
        e2 = make_etag({"b": 2, "a": 1})
        self.assertEqual(e1, e2)
        self.assertTrue(e1.startswith('"') and e1.endswith('"'))

    def test_check_if_none_match_exact_and_wildcard(self):
        etag = '"abcdef1234567890"'
        self.assertTrue(check_if_none_match('"abcdef1234567890"', etag))
        self.assertTrue(check_if_none_match('W/"abcdef1234567890"', etag))
        self.assertTrue(check_if_none_match('*', etag))
        self.assertTrue(check_if_none_match('"other", "abcdef1234567890"', etag))
        self.assertFalse(check_if_none_match('"different"', etag))
        self.assertFalse(check_if_none_match(None, etag))

    def test_json_or_304_returns_304_when_matched(self):
        payload = {"data": "test"}
        etag = make_etag(payload)
        req = _dummy_request({"if-none-match": etag})
        resp = json_or_304(req, payload, etag=etag)
        self.assertEqual(resp.status_code, 304)
        self.assertEqual(resp.headers.get("etag"), etag)

    def test_json_or_304_returns_200_when_not_matched(self):
        payload = {"data": "test"}
        etag = make_etag(payload)
        req = _dummy_request({"if-none-match": '"outdated"'})
        resp = json_or_304(req, payload, etag=etag)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("etag"), etag)


if __name__ == "__main__":
    unittest.main()
