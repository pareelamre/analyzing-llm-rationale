import unittest

from analyzing_llm_rationale.twin.worker import WorkerAuthenticationError, require_worker_request


class TwinWorkerTests(unittest.TestCase):
    def test_private_worker_rejects_public_request(self):
        require_worker_request("valid", expected_token="valid")
        with self.assertRaises(WorkerAuthenticationError):
            require_worker_request(None, expected_token="valid")
