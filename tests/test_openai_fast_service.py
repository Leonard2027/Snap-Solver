import json
import unittest
from pathlib import Path
from unittest.mock import patch

from models.factory import ModelFactory
from models.openai import OpenAIModel


class OpenAIFastServiceTests(unittest.TestCase):
    def setUp(self):
        self.model = OpenAIModel(
            api_key="test-key",
            model_identifier="gpt-5.6-terra",
            reasoning_tier="deep",
            service_tier="priority",
        )

    def test_request_options_include_priority_and_medium_reasoning(self):
        self.assertEqual(
            self.model._request_kwargs(),
            {"reasoning_effort": "medium", "service_tier": "priority"},
        )

    @patch("models.openai.OpenAI")
    def test_priority_reaches_text_and_image_requests(self, openai_client):
        create = openai_client.return_value.chat.completions.create
        create.return_value = []

        list(self.model.analyze_text("test"))
        self.assertEqual(create.call_args.kwargs["service_tier"], "priority")

        list(self.model.analyze_image("base64-image"))
        self.assertEqual(create.call_args.kwargs["service_tier"], "priority")
        self.assertEqual(create.call_count, 2)

    @patch("models.openai.OpenAI")
    def test_priority_error_is_returned_without_retry(self, openai_client):
        create = openai_client.return_value.chat.completions.create
        create.side_effect = RuntimeError("priority is unsupported")

        events = list(self.model.analyze_text("test"))

        self.assertEqual(create.call_count, 1)
        self.assertEqual(events[-1]["status"], "error")
        self.assertIn("priority is unsupported", events[-1]["error"])

    def test_default_service_omits_service_tier(self):
        model = OpenAIModel(api_key="test-key", reasoning_tier="fast")
        self.assertEqual(model._request_kwargs(), {"reasoning_effort": "low"})


class TerraModelConfigTests(unittest.TestCase):
    def test_terra_is_a_medium_default_openai_model(self):
        config_path = Path(__file__).parents[1] / "config" / "models.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        terra = config["models"]["gpt-5.6-terra"]

        self.assertEqual(terra["provider"], "openai")
        self.assertEqual(terra["defaultTier"], "deep")
        self.assertEqual(terra["reasoningTiers"], ["fast", "deep", "max"])
        self.assertTrue(terra["supportsMultimodal"])

    def test_factory_passes_priority_to_terra_only_as_an_openai_option(self):
        ModelFactory.initialize()
        model = ModelFactory.create_model(
            model_name="gpt-5.6-terra",
            api_key="test-key",
            service_tier="priority",
        )

        self.assertIsInstance(model, OpenAIModel)
        self.assertEqual(model.get_model_identifier(), "gpt-5.6-terra")
        self.assertEqual(model.service_tier, "priority")

    @patch("app.load_proxy_api", return_value={"enabled": False})
    @patch("app.get_api_key", return_value="test-key")
    def test_fast_service_setting_maps_to_priority(
        self, _get_api_key, _load_proxy_api
    ):
        from app import create_model_instance

        model = create_model_instance(
            "gpt-5.6-terra",
            {"fastService": True, "reasoningTier": "deep"},
            is_reasoning=True,
        )

        self.assertEqual(model.service_tier, "priority")


if __name__ == "__main__":
    unittest.main()
