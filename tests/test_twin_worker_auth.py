"""Worker admission must be based on Google OIDC claims, never queue headers."""
import inspect
import unittest
from datetime import datetime, timedelta, timezone

from analyzing_llm_rationale.twin.worker import (
    WorkerAuthenticationError,
    verify_google_worker_oidc,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
AUDIENCE = "https://twin-maintenance-abc-uc.a.run.app"
DISPATCHER = "twin-task-dispatcher@example.iam.gserviceaccount.com"
RESEARCH = "twin-research@example.iam.gserviceaccount.com"


def claims(*, audience=AUDIENCE, email=DISPATCHER, expires_at=NOW + timedelta(minutes=5)):
    return {
        "iss": "https://accounts.google.com",
        "aud": audience,
        "email": email,
        "email_verified": True,
        "exp": expires_at.timestamp(),
    }


def verifier(payload):
    def verify(token, audience):
        if token != "google-id-token":
            raise ValueError("unrecognized test token")
        return payload

    return verify


class TwinWorkerOidcTests(unittest.TestCase):
    def test_valid_google_identity_for_exact_audience_is_accepted(self):
        principal = verify_google_worker_oidc(
            "google-id-token",
            expected_audience=AUDIENCE,
            allowed_service_accounts=frozenset({DISPATCHER}),
            now=NOW,
            verifier=verifier(claims()),
        )
        self.assertEqual(principal.service_account_email, DISPATCHER)

    def test_wrong_audience_is_rejected_even_when_the_service_account_matches(self):
        with self.assertRaisesRegex(WorkerAuthenticationError, "wrong audience"):
            verify_google_worker_oidc(
                "google-id-token",
                expected_audience=AUDIENCE,
                allowed_service_accounts=frozenset({DISPATCHER}),
                now=NOW,
                verifier=verifier(claims(audience="https://another-service.example")),
            )

    def test_expired_and_unauthorized_identities_are_rejected(self):
        with self.assertRaisesRegex(WorkerAuthenticationError, "expired"):
            verify_google_worker_oidc(
                "google-id-token",
                expected_audience=AUDIENCE,
                allowed_service_accounts=frozenset({DISPATCHER}),
                now=NOW,
                verifier=verifier(claims(expires_at=NOW)),
            )
        with self.assertRaisesRegex(WorkerAuthenticationError, "not authorized"):
            verify_google_worker_oidc(
                "google-id-token",
                expected_audience=AUDIENCE,
                allowed_service_accounts=frozenset({DISPATCHER}),
                now=NOW,
                verifier=verifier(claims(email=RESEARCH)),
            )

    def test_spoofed_queue_header_and_anonymous_request_do_not_authenticate(self):
        spoofed_headers = {"X-CloudTasks-QueueName": "maintenance"}
        self.assertNotIn("headers", inspect.signature(verify_google_worker_oidc).parameters)
        with self.assertRaisesRegex(WorkerAuthenticationError, "bearer token"):
            verify_google_worker_oidc(
                None,
                expected_audience=AUDIENCE,
                allowed_service_accounts=frozenset({DISPATCHER}),
                now=NOW,
                verifier=verifier(claims()),
            )
        self.assertEqual(spoofed_headers["X-CloudTasks-QueueName"], "maintenance")

    def test_unverified_or_untrusted_google_claims_fail_closed(self):
        unverified = claims()
        unverified["email_verified"] = False
        with self.assertRaisesRegex(WorkerAuthenticationError, "not verified"):
            verify_google_worker_oidc(
                "google-id-token",
                expected_audience=AUDIENCE,
                allowed_service_accounts=frozenset({DISPATCHER}),
                now=NOW,
                verifier=verifier(unverified),
            )
        untrusted = claims()
        untrusted["iss"] = "https://issuer.example"
        with self.assertRaisesRegex(WorkerAuthenticationError, "invalid issuer"):
            verify_google_worker_oidc(
                "google-id-token",
                expected_audience=AUDIENCE,
                allowed_service_accounts=frozenset({DISPATCHER}),
                now=NOW,
                verifier=verifier(untrusted),
            )


if __name__ == "__main__":
    unittest.main()
