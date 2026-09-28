import unittest

from analyzing_llm_rationale import webhooks


class TestWebhooks(unittest.TestCase):
    def setUp(self):
        self.mgr = webhooks.WebhookManager()

    def test_register_and_list_webhook(self):
        sub = self.mgr.register("https://example.com/agent-webhook", events=["edge_alert"], min_edge=0.10)
        self.assertTrue(sub.id.startswith("wh_"))
        self.assertTrue(sub.secret.startswith("whsec_"))
        self.assertEqual(sub.url, "https://example.com/agent-webhook")
        self.assertEqual(sub.min_edge, 0.10)

        subs = self.mgr.list_subscriptions()
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].id, sub.id)

    def test_delete_webhook(self):
        sub = self.mgr.register("https://example.com/webhook")
        deleted = self.mgr.delete_subscription(sub.id)
        self.assertTrue(deleted)
        self.assertEqual(len(self.mgr.list_subscriptions()), 0)

    def test_signature_generation_and_verification(self):
        secret = "whsec_testsecret12345"
        payload_bytes = b'{"event":"edge_alert","market_id":"kalshi-FED-26"}'
        sig_header = webhooks.generate_signature(secret, payload_bytes)
        self.assertIn("t=", sig_header)
        self.assertIn("v1=", sig_header)

        # Verification should succeed
        self.assertTrue(webhooks.verify_signature(secret, payload_bytes, sig_header))

        # Tampered payload should fail
        tampered_bytes = b'{"event":"edge_alert","market_id":"kalshi-FED-TAMPERED"}'
        self.assertFalse(webhooks.verify_signature(secret, tampered_bytes, sig_header))

        # Wrong secret should fail
        self.assertFalse(webhooks.verify_signature("wrong_secret", payload_bytes, sig_header))


if __name__ == "__main__":
    unittest.main()
