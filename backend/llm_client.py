import json
import os
import random
import re
import time

import httpx
from dotenv import load_dotenv


# Load detective-ai/.env
load_dotenv()


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

FALLBACK_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash",
]

GEMINI_API_BASE = (
    "https://generativelanguage.googleapis.com/v1beta/models"
)


class LLMClient:
    """
    Native Google Gemini REST client.

    Required:
        GEMINI_API_KEY

    Optional:
        GEMINI_MODEL

    No OpenAI compatibility layer is used.
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        api_key=None,
        model=None,
    ):
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
        )

        self.model = (
            model
            or os.getenv("GEMINI_MODEL")
            or DEFAULT_GEMINI_MODEL
        )

        if not self.api_key:
            raise ValueError(
                "Gemini API key missing. "
                "Set GEMINI_API_KEY in the project .env file."
            )


    # =====================================================
    # PARSE JSON FROM GEMINI
    # =====================================================

    @staticmethod
    def _parse_json_content(content):

        if isinstance(content, dict):
            return content

        if not isinstance(content, str):
            raise ValueError(
                "Gemini returned an unsupported response format."
            )

        content = content.strip()

        # Defensive removal of markdown fences
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
        )

        try:
            return json.loads(content)

        except json.JSONDecodeError as error:
            raise ValueError(
                "Gemini did not return valid JSON.\n"
                f"Raw response:\n{content}"
            ) from error


    # =====================================================
    # EXTRACT TEXT FROM GEMINI RESPONSE
    # =====================================================

    @staticmethod
    def _extract_text(data):

        try:
            candidates = data["candidates"]

            if not candidates:
                raise ValueError(
                    "Gemini returned no candidates."
                )

            parts = (
                candidates[0]
                ["content"]
                ["parts"]
            )

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as error:

            raise ValueError(
                "Unexpected Gemini API response.\n"
                f"{json.dumps(data, indent=2)}"
            ) from error

        text_parts = []

        for part in parts:

            text = part.get("text")

            if text:
                text_parts.append(text)

        if not text_parts:
            raise ValueError(
                "Gemini returned no text content.\n"
                f"{json.dumps(data, indent=2)}"
            )

        return "\n".join(text_parts)


    # =====================================================
    # SEND REQUEST WITH RETRIES + MODEL FALLBACK
    # =====================================================

    def _send_request(
        self,
        payload,
    ):

        headers = {
            "x-goog-api-key":
                self.api_key,

            "Content-Type":
                "application/json",
        }

        timeout = httpx.Timeout(
            90.0,
            connect=20.0,
        )

        models_to_try = [
            self.model,
            *FALLBACK_MODELS,
        ]

        # Remove duplicates while preserving order
        models_to_try = list(
            dict.fromkeys(
                models_to_try
            )
        )

        last_error = None

        for model in models_to_try:

            api_url = (
                f"{GEMINI_API_BASE}/"
                f"{model}:generateContent"
            )

            for attempt in range(4):

                try:

                    with httpx.Client(
                        timeout=timeout
                    ) as client:

                        response = client.post(
                            api_url,
                            headers=headers,
                            json=payload,
                        )

                    # -----------------------------
                    # Success
                    # -----------------------------

                    if response.is_success:

                        if model != self.model:
                            print(
                                "Gemini fallback model used: "
                                f"{model}"
                            )

                        return response.json()

                    # -----------------------------
                    # Temporary errors
                    # -----------------------------

                    if response.status_code in {
                        408,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }:

                        last_error = (
                            httpx.HTTPStatusError(
                                (
                                    "Gemini temporary error "
                                    f"{response.status_code}"
                                ),
                                request=response.request,
                                response=response,
                            )
                        )

                        if attempt < 3:

                            delay = (
                                2 ** attempt
                                + random.uniform(
                                    0,
                                    1
                                )
                            )

                            print(
                                f"{model} returned "
                                f"{response.status_code}. "
                                f"Retrying in "
                                f"{delay:.1f}s..."
                            )

                            time.sleep(delay)

                            continue

                        # Try next Gemini model
                        break

                    # -----------------------------
                    # Permanent/client error
                    # -----------------------------

                    response.raise_for_status()

                except httpx.HTTPError as error:

                    last_error = error

                    if attempt < 3:

                        delay = (
                            2 ** attempt
                            + random.uniform(
                                0,
                                1
                            )
                        )

                        time.sleep(delay)

                        continue

                    break

        raise RuntimeError(
            "Gemini API remained unavailable "
            "after retries and fallback models."
        ) from last_error


    # =====================================================
    # PUBLIC INTERFACE USED BY INVESTIGATOR / FACT CHECKER
    # =====================================================

    def chat_json(
        self,
        system_prompt,
        user_prompt,
        temperature=0.2,
    ):
        """
        Ask Gemini to return structured JSON.
        """

        payload = {

            "systemInstruction": {
                "parts": [
                    {
                        "text":
                            system_prompt
                    }
                ]
            },

            "contents": [
                {
                    "role":
                        "user",

                    "parts": [
                        {
                            "text":
                                user_prompt
                        }
                    ],
                }
            ],

            "generationConfig": {

                "temperature":
                    temperature,

                "responseMimeType":
                    "application/json",
            },
        }

        data = self._send_request(
            payload
        )

        text = self._extract_text(
            data
        )

        return self._parse_json_content(
            text
        )


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    client = LLMClient()

    result = client.chat_json(
        system_prompt=(
            "You are a test assistant. "
            "Return only valid JSON."
        ),

        user_prompt=(
            "Return JSON with status set to ok "
            "and provider set to gemini."
        ),

        temperature=0.0,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )