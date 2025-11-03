"""
Simple Chatbot Handler

Provides basic conversational responses for non-scam calls.
This handler does NOT collect booking information or use the appointment workflow.
It simply engages in friendly conversation and provides helpful responses.
"""

from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI
from config import get_settings

settings = get_settings()


class SimpleChatbot:
    """Simple conversational chatbot for non-scam interactions

    This chatbot provides friendly, helpful responses without attempting
    to collect booking information or schedule services.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize simple chatbot

        Args:
            api_key: Optional OpenAI API key (defaults to settings)
        """
        self.api_key = api_key
        self._client = None
        self.model = settings.openai_model

    @property
    def client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key or settings.openai_api_key)
        return self._client

    async def generate_response(
        self,
        current_message: str,
        message_history: Optional[List[Dict]] = None,
        intent_type: Optional[str] = None,
        company_name: Optional[str] = None,
    ) -> str:
        """Generate conversational response

        Args:
            current_message: Current user message
            message_history: Optional conversation history
            intent_type: Optional intent classification (opportunity/other)
            company_name: Optional company name for personalization

        Returns:
            AI-generated response string
        """
        # Build system prompt based on intent type
        system_prompt = self._build_system_prompt(intent_type, company_name)

        # Build conversation messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add message history if available
        if message_history:
            for msg in message_history[-6:]:  # Last 6 messages for context
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        # Add current message
        messages.append({
            "role": "user",
            "content": current_message
        })

        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,  # Balanced creativity
                max_tokens=200,   # Concise responses
            )

            ai_message = response.choices[0].message.content.strip()
            print(f"✅ [SIMPLE_CHATBOT] Generated response: {ai_message[:100]}...")
            return ai_message

        except Exception as e:
            print(f"❌ [SIMPLE_CHATBOT] Error generating response: {str(e)}")
            # Fallback response
            return self._get_fallback_response()

    def _build_system_prompt(
        self,
        intent_type: Optional[str],
        company_name: Optional[str]
    ) -> str:
        """Build system prompt based on intent type and context

        Args:
            intent_type: Intent classification (opportunity/other)
            company_name: Company name for personalization

        Returns:
            System prompt string
        """
        company = company_name or "our organization"

        base_prompt = f"""You are a friendly and helpful AI assistant for {company}.

Your role is to engage in natural, helpful conversation. You should:
- Be warm, professional, and courteous
- Provide helpful information when possible
- Keep responses concise (2-3 sentences)
- Be conversational and natural
- NOT collect personal information or schedule appointments
- NOT try to sell services or products

"""

        # Add intent-specific guidance
        if intent_type == "opportunity":
            base_prompt += """The caller has been identified as potentially having a legitimate opportunity or professional inquiry.
Be especially helpful and accommodating with their questions."""
        elif intent_type == "other":
            base_prompt += """The caller's intent is unclear. Be helpful but let them guide the conversation.
Offer to assist with their questions or concerns."""
        else:
            base_prompt += """Engage naturally with the caller and help with their questions."""

        return base_prompt

    def _get_fallback_response(self) -> str:
        """Get fallback response when AI generation fails

        Returns:
            Safe fallback response string
        """
        return "I appreciate you reaching out. How can I assist you today?"


# Global chatbot instance
simple_chatbot = SimpleChatbot()
