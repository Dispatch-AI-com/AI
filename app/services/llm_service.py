from openai import AsyncOpenAI
from typing import Optional
from app.config import get_settings

settings = get_settings()


class LLMService:
    def __init__(self):
        if settings.llm_provider == "openai":
            # In development, allow service to start without OpenAI API key
            if not settings.openai_api_key and (settings.environment == "development" or settings.debug):
                print("⚠️  OpenAI API key not found. LLM features will be disabled.")
                self.client = None
            else:
                self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            self.client = None

    async def generate_response(
        self, message: str, context: Optional[str] = None
    ) -> str:
        if settings.llm_provider == "mock":
            return f"Mock response to: {message}"

        if self.client is None:
            # In development, return a mock response if client is not initialized
            if settings.environment == "development" or settings.debug:
                return f"[Mock LLM Response] This is a mock response to: {message}"
            raise Exception("LLM client not initialized")

        try:
            prompt = message
            if context:
                prompt = f"Context: {context}\n\nUser: {message}"

            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.openai_max_tokens,
                temperature=settings.openai_temperature,
            )
            content = response.choices[0].message.content
            if content is None:
                raise Exception("Empty response from LLM")
            return content
        except Exception as e:
            raise Exception(f"LLM service error: {str(e)}")


llm_service = LLMService()
