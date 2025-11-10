"""
Simple Chatbot Handler

Provides basic conversational responses for non-scam calls.
This handler does NOT collect booking information or use the appointment workflow.
It simply engages in friendly conversation and provides helpful responses.
"""

from typing import Optional, Dict, List
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
- Keep responses concise (2-3 sentences)
- Be conversational and natural

"""

        # Add intent-specific guidance
        if intent_type == "opportunity":
            base_prompt += """IMPORTANT: This caller has a VALUABLE OPPORTUNITY (job offer, research collaboration, academic partnership, etc.).

Your approach:
- Be enthusiastic and highly accommodating
- Ask probing questions to understand the opportunity details
- Try to gather: their name, organization, nature of opportunity, timeline, contact info
- Show genuine interest and appreciation
- Make them feel their opportunity is important and will be prioritized

Example responses:
- "That sounds like an exciting opportunity! Could you tell me more about [specific aspect]?"
- "I'd love to learn more about this. What organization are you with?"
- "This is definitely something our team would be interested in. Can you share your contact details so we can follow up properly?"
"""
        elif intent_type == "other":
            base_prompt += """IMPORTANT: This caller's intent is unclear or requires human handling.

Your approach:
- Keep conversation brief and focused
- Ask 1-2 direct questions to understand their core need
- Don't go too deep - this will be escalated to a human
- Reassure them their matter will be handled by the appropriate person
- Be professional but don't overpromise

Example responses:
- "I understand. To make sure you get the right help, could you briefly explain what you need assistance with?"
- "Thank you for explaining. This sounds like something our team should handle directly."
- "I've noted your concern. What's the best number to reach you at?"
"""
        else:
            # Unclassified - still gathering information
            base_prompt += """Your approach:
- Engage naturally to understand what the caller needs
- Ask open-ended questions to clarify their purpose
- Be helpful and patient
- Listen for signals about whether this is a valuable opportunity or a general inquiry

Example responses:
- "I'm here to help! What brings you to call us today?"
- "Could you tell me a bit more about what you're looking for?"
"""

        return base_prompt

    def _get_fallback_response(self) -> str:
        """Get fallback response when AI generation fails

        Returns:
            Safe fallback response string
        """
        return "I appreciate you reaching out. How can I assist you today?"


# Global chatbot instance
simple_chatbot = SimpleChatbot()
