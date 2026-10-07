import gc
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as healthpulse
from werkzeug.security import check_password_hash


VALID_MEASUREMENTS = {
    "Pregnancies": "2",
    "Glucose": "120",
    "BloodPressure": "70",
    "SkinThickness": "25",
    "Insulin": "80",
    "BMI": "27.5",
    "DiabetesPedigreeFunction": "0.4",
    "Age": "35",
}


class AuthHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path_patch = patch.object(
            healthpulse,
            "DATABASE_PATH",
            Path(self.temp_dir.name) / "healthpulse.db",
        )
        self.database_path_patch.start()
        healthpulse.init_db()
        self.app_config_patch = patch.dict(
            healthpulse.app.config,
            {"TESTING": True, "SECRET_KEY": "test-secret-key"},
        )
        self.app_config_patch.start()
        self.client = healthpulse.app.test_client()

    def tearDown(self):
        self.client = None
        self.app_config_patch.stop()
        self.database_path_patch.stop()
        gc.collect()
        self.temp_dir.cleanup()

    def csrf_token(self, client, url):
        response = client.get(url)
        self.assertEqual(response.status_code, 200)
        match = re.search(
            r'name="csrf_token" value="([^"]+)"',
            response.get_data(as_text=True),
        )
        self.assertIsNotNone(match)
        return match.group(1)

    def register(self, client, name, email):
        token = self.csrf_token(client, "/register")
        response = client.post(
            "/register",
            data={
                "csrf_token": token,
                "name": name,
                "email": email,
                "password": "secure-pass-123",
                "confirm_password": "secure-pass-123",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Enter Health Measurements", response.data)
        return response

    def submit_assessment(self, client, measurements=None):
        token = self.csrf_token(client, "/")
        response = client.post(
            "/",
            data={
                "csrf_token": token,
                **(measurements or VALID_MEASUREMENTS),
            },
        )
        return response

    def test_registration_login_and_logout(self):
        response = self.register(self.client, "Test User", "user@example.com")
        self.assertIn(b"Test User</strong>", response.data)

        connection = sqlite3.connect(healthpulse.DATABASE_PATH)
        try:
            password_hash = connection.execute(
                "SELECT password_hash FROM users WHERE email = ?",
                ("user@example.com",),
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertNotEqual(password_hash, "secure-pass-123")
        self.assertTrue(check_password_hash(password_hash, "secure-pass-123"))

        logout_token = re.search(
            rb'name="csrf_token" value="([^"]+)"',
            response.data,
        ).group(1).decode()
        logout_response = self.client.post(
            "/logout",
            data={"csrf_token": logout_token},
        )
        self.assertEqual(logout_response.status_code, 302)
        self.assertEqual(logout_response.location, "/login")
        self.assertEqual(self.client.get("/").status_code, 302)

        login_token = self.csrf_token(self.client, "/login")
        login_response = self.client.post(
            "/login",
            data={
                "csrf_token": login_token,
                "email": "user@example.com",
                "password": "secure-pass-123",
            },
            follow_redirects=True,
        )
        self.assertEqual(login_response.status_code, 200)
        self.assertIn(b"Test User</strong>", login_response.data)

    def test_invalid_assessment_does_not_predict_or_save(self):
        self.register(self.client, "Test User", "user@example.com")
        cases = [
            ({**VALID_MEASUREMENTS, "Age": "-1"}, b"Age must be between 1 and 120."),
            (
                {**VALID_MEASUREMENTS, "Glucose": "301"},
                b"Glucose must be between 0 and 300.",
            ),
            ({**VALID_MEASUREMENTS, "BMI": ""}, b"Please enter a value for BMI."),
        ]

        with patch.object(
            healthpulse.model,
            "predict",
            wraps=healthpulse.model.predict,
        ) as predict:
            for measurements, expected_message in cases:
                response = self.submit_assessment(self.client, measurements)
                self.assertEqual(response.status_code, 200)
                self.assertIn(expected_message, response.data)
                self.assertNotIn(b"Model Assessment", response.data)
                predict.assert_not_called()

        connection = sqlite3.connect(healthpulse.DATABASE_PATH)
        try:
            count = connection.execute(
                "SELECT COUNT(*) FROM assessments"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(count, 0)

    def test_assessment_history_and_detail_are_user_isolated(self):
        self.register(self.client, "User A", "a@example.com")
        response = self.submit_assessment(self.client)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Lower Risk Indicator", response.data)
        self.assertIn(b"41.0%", response.data)
        self.assertIn(b"Your Educational Health Report", response.data)
        self.assertEqual(response.data.count(b'class="importance-bar"'), 8)

        connection = sqlite3.connect(healthpulse.DATABASE_PATH)
        try:
            assessment_id = connection.execute(
                "SELECT id FROM assessments"
            ).fetchone()[0]
        finally:
            connection.close()

        history_response = self.client.get("/history")
        self.assertIn(b"Lower Risk Indicator", history_response.data)
        self.assertIn(b"View Details", history_response.data)
        detail_response = self.client.get(f"/history/{assessment_id}")
        self.assertEqual(detail_response.status_code, 200)
        self.assertIn(b"Important Disclaimer", detail_response.data)
        self.assertEqual(detail_response.data.count(b'class="importance-bar"'), 8)

        client_b = healthpulse.app.test_client()
        self.register(client_b, "User B", "b@example.com")
        empty_history = client_b.get("/history")
        self.assertIn(
            b"No assessments yet. Start your first health assessment.",
            empty_history.data,
        )
        self.assertEqual(client_b.get(f"/history/{assessment_id}").status_code, 404)
        self.assertNotIn(b"Glucose", empty_history.data)


if __name__ == "__main__":
    unittest.main()
