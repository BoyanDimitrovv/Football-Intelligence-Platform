import unittest

from app import create_app


class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_health_returns_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_index_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
