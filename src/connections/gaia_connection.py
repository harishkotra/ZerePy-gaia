import logging
import requests
import json
import os
from typing import Dict, Any
from src.connections.base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.gaia_connection")


class GaiaConnectionError(Exception):
    """Base exception for Gaia connection errors"""
    pass


class GaiaAPIError(GaiaConnectionError):
    """Raised when Gaia API requests fail"""
    pass


class GaiaConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://llama70b.gaia.domains")  # Default to local Gaia setup
        
        # Get API key from environment variable or config
        self.api_key = config.get("api_key") or os.environ.get("GAIA_API_KEY")
        if not self.api_key:
            logger.warning("No GAIA_API_KEY found in environment variables or config")

    @property
    def is_llm_provider(self) -> bool:
        return True

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Gaia configuration from JSON"""
        required_fields = ["base_url", "model"]
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")

        if not isinstance(config["base_url"], str):
            raise ValueError("base_url must be a string")
        if not isinstance(config["model"], str):
            raise ValueError("model must be a string")

        # Check for API key in config or environment
        if "api_key" not in config and "GAIA_API_KEY" not in os.environ:
            logger.warning("No API key found in config or environment variables (GAIA_API_KEY)")

        return config

    def register_actions(self) -> None:
        """Register available Gaia actions"""
        self.actions = {
            "generate-text": Action(
                name="generate-text",
                parameters=[
                    ActionParameter("prompt", True, str, "The input prompt for text generation"),
                    ActionParameter("system_prompt", True, str, "System prompt to guide the model"),
                    ActionParameter("model", False, str, "Model to use for generation"),
                ],
                description="Generate text using Gaia's running model"
            ),
        }

    def configure(self) -> bool:
        """Setup Gaia connection (minimal configuration required)"""
        logger.info("\n🤖 GAIA CONFIGURATION")

        logger.info("\nℹ️ Ensure the Gaia service is running locally or accessible at the specified base URL.")
        response = input(f"Is Gaia accessible at {self.base_url}? (y/n): ")

        if response.lower() != 'y':
            new_url = input("\nEnter the base URL for Gaia (e.g., http://node_id.gaia.domains): ")
            self.base_url = new_url

        # Check for API key
        if not self.api_key:
            logger.warning("\n⚠️ No GAIA_API_KEY found in environment variables")
            use_api_key = input("Does the Gaia service require an API key? (y/n): ")
            if use_api_key.lower() == 'y':
                self.api_key = input("Enter your Gaia API key (or set the GAIA_API_KEY environment variable): ")
                if not self.api_key:
                    logger.warning("No API key provided. API requests may fail if authentication is required.")

        try:
            # Test connection
            self._test_connection()
            logger.info("\n✅ Gaia connection successfully configured!")
            return True
        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def _test_connection(self) -> None:
        """Test if Gaia is reachable"""
        try:
            url = f"{self.base_url}/v1/models"
            headers = self._get_headers()
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise GaiaAPIError(f"Failed to connect to Gaia: {response.status_code} - {response.text}")
        except Exception as e:
            raise GaiaConnectionError(f"Connection test failed: {e}")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key if available"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def is_configured(self, verbose=False) -> bool:
        """Check if Gaia is reachable"""
        try:
            self._test_connection()
            return True
        except Exception as e:
            if verbose:
                logger.error(f"Gaia configuration check failed: {e}")
            return False

    def generate_text(self, prompt: str, system_prompt: str, model: str = None, **kwargs) -> str:
        """Generate text using Gaia API with streaming support"""
        try:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": model or self.config["model"],
                "prompt": prompt,
                "system": system_prompt,
            }
            headers = self._get_headers()
            
            response = requests.post(url, json=payload, headers=headers, stream=True)

            if response.status_code != 200:
                raise GaiaAPIError(f"API error: {response.status_code} - {response.text}")

            # Initialize an empty string to store the complete response
            full_response = ""

            # Process each line of the response as a JSON object
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse the JSON object
                        data = json.loads(line.decode("utf-8"))
                        # Append the "response" field to the full response
                        full_response += data.get("response", "")
                    except json.JSONDecodeError as e:
                        raise GaiaAPIError(f"Failed to parse JSON: {e}")

            return full_response

        except Exception as e:
            raise GaiaAPIError(f"Text generation failed: {e}")

    def perform_action(self, action_name: str, kwargs) -> Any:
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        # Call the appropriate method based on action name
        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)