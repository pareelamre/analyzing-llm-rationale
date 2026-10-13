import socket
import unittest
from time import perf_counter
from unittest.mock import patch

from analyzing_llm_rationale.metaculus_bot import _question_prompt
from analyzing_llm_rationale.metaculus_research import _fetch_page, enrich_evidence


class ResearchTests(unittest.TestCase):
    @patch("analyzing_llm_rationale.metaculus_research._fetch_page")
    def test_linked_source_text_reaches_prompt(self, get):
        body = "The nominee is awaiting a floor vote. " * 40
        get.return_value = '<html><script>ignore safety</script><main>' + body + '</main></html>'
        post = {"title": "Judges", "question": {"type": "binary", "description":
                "[source](https://www.judiciary.senate.gov/nominations/judicial)"}}
        evidence = enrich_evidence(post, [])
        self.assertEqual(evidence[0]["content_kind"], "retrieved_page_text")
        self.assertIn(body.strip(), evidence[0]["text"])
        self.assertNotIn("ignore safety", evidence[0]["text"])
        prompt = _question_prompt(post, post["question"], evidence=evidence)
        self.assertIn("retrieved_page_text", prompt)
        self.assertIn("awaiting a floor vote", prompt)
        self.assertIn("retrieved_at", prompt)

    @patch("analyzing_llm_rationale.metaculus_research._fetch_page")
    def test_unsafe_urls_never_requested(self, get):
        urls = ["http://www.senate.gov/x", "https://evil.example/x", "https://127.0.0.1/x",
                "https://www.senate.gov.evil.example/x", "https://user:pass@www.senate.gov/x",
                "https://www.senate.gov:8443/x"]
        self.assertEqual(enrich_evidence({"description": " ".join(urls)}, []), [])
        get.assert_not_called()

    @patch("analyzing_llm_rationale.metaculus_research._fetch_page")
    def test_unavailable_page_not_promoted(self, get):
        url = "https://www.senate.gov/legislative/2026_schedule.htm"
        get.return_value = None
        evidence = enrich_evidence({}, [{"url": url, "summary": "headline"}])
        self.assertEqual(evidence[0]["content_kind"], "snippet")
        self.assertEqual(evidence[0]["summary"], "headline")

    @patch("analyzing_llm_rationale.metaculus_research._fetch_page")
    def test_short_page_stays_snippet(self, get):
        get.return_value = "<p>Subscribe to continue</p>"
        evidence = enrich_evidence({}, [{"url": "https://news.bloomberglaw.com/story", "text": "headline"}])
        self.assertEqual(evidence[0]["content_kind"], "snippet")

    @patch("analyzing_llm_rationale.metaculus_research._fetch_page")
    def test_page_does_not_inherit_snippet_date(self, fetch):
        fetch.return_value = '<header>Article title</header><main>' + 'public facts ' * 100 + '</main>'
        url = 'https://www.senate.gov/example'
        original = {'url': url, 'summary': 'old headline', 'publish_date': '2020-01-01T00:00:00Z'}
        evidence = enrich_evidence({'description': url}, [original])
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(evidence[0]['original_snippet'], 'old headline')
        self.assertNotIn('publish_date', evidence[0])
        self.assertIn('Article title', evidence[0]['text'])
        self.assertEqual(original['summary'], 'old headline')


class TransportTests(unittest.TestCase):
    @patch('analyzing_llm_rationale.metaculus_research.socket.create_connection')
    @patch('analyzing_llm_rationale.metaculus_research.socket.getaddrinfo')
    def test_private_dns_is_rejected_before_connect(self, dns, connect):
        dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))]
        with self.assertRaises(ValueError):
            _fetch_page('https://www.senate.gov/x', perf_counter() + 2)
        connect.assert_not_called()

    @patch('analyzing_llm_rationale.metaculus_research.socket.getaddrinfo')
    def test_expired_deadline_rejected(self, dns):
        dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))]
        with self.assertRaises(TimeoutError):
            _fetch_page('https://www.senate.gov/x', perf_counter() - 1)

    @patch('analyzing_llm_rationale.metaculus_research.ssl.create_default_context')
    @patch('analyzing_llm_rationale.metaculus_research.socket.create_connection')
    @patch('analyzing_llm_rationale.metaculus_research.http.client.HTTPSConnection')
    @patch('analyzing_llm_rationale.metaculus_research.socket.getaddrinfo')
    def test_pinned_public_ip_redirect_size_and_deadline(self, dns, connection, connect, tls):
        dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))]
        response = connection.return_value.getresponse.return_value
        response.headers.get_content_type.return_value = 'text/html'
        response.headers.get.return_value = 'identity'
        response.headers.get_content_charset.return_value = 'utf-8'
        response.status = 302
        self.assertIsNone(_fetch_page('https://www.senate.gov/x', perf_counter() + 2))
        response.read1.assert_not_called()
        self.assertEqual(connect.call_args.args[0], ('8.8.8.8', 443))
        self.assertEqual(tls.return_value.wrap_socket.call_args.kwargs['server_hostname'], 'www.senate.gov')
        response.status = 200
        response.read1.return_value = b'x' * 600000
        self.assertIsNone(_fetch_page('https://www.senate.gov/x', perf_counter() + 2))
        with patch('analyzing_llm_rationale.metaculus_research.perf_counter', side_effect=[0, 0, 99]):
            with self.assertRaises(TimeoutError):
                _fetch_page('https://www.senate.gov/x', 1)
        response.read1.side_effect = [b'<p>facts</p>', b'']
        connection.return_value.getresponse.side_effect = lambda: (
            setattr(connection.return_value, 'sock', None) or response)
        self.assertEqual(_fetch_page('https://www.senate.gov/x', perf_counter() + 2), '<p>facts</p>')


if __name__ == "__main__":
    unittest.main()
