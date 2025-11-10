"""
Simple Information Collector for Non-Scam Calls

Collects basic information in 3 steps:
1. Name
2. Background (organization, purpose)
3. Additional info (contact, supplementary details)
"""

from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()


class SimpleInfoCollector:
    """Simple 3-step information collector for legitimate calls"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize info collector

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

    async def collect_name(
        self,
        current_message: str,
        message_history: Optional[List[Dict]] = None,
        intent_type: Optional[str] = None,
        company_name: Optional[str] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Step 1: Collect caller's name

        Args:
            current_message: Current user message
            message_history: Conversation history
            intent_type: Intent classification (opportunity/other)
            company_name: Company name for personalization
            retry_count: Number of retries so far

        Returns:
            Dict with:
                - response: AI response to user
                - name_extracted: Whether name was extracted
                - name: Extracted name (or None)
                - should_retry: Whether to retry
        """
        company = company_name or "us"

        system_prompt = f"""You are a friendly AI assistant for {company}.

Your task: Extract the caller's NAME from their message.

Guidelines:
- If they provide a name, extract it and respond warmly
- If they don't provide a name, politely ask for it
- Keep responses brief (1-2 sentences)
- Be natural and conversational

Return JSON format:
{{
    "name": "extracted name or null",
    "name_extracted": true/false,
    "response": "your response to the user"
}}
"""

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        if message_history:
            for msg in message_history[-4:]:  # Last 4 messages for context
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        messages.append({
            "role": "user",
            "content": f"Current message: {current_message}"
        })

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)

            name_extracted = result.get("name_extracted", False)
            extracted_name = result.get("name")

            # Determine if we should retry
            should_retry = not name_extracted and retry_count < settings.max_retries_per_step

            return {
                "response": result.get("response", "May I have your name, please?"),
                "name_extracted": name_extracted,
                "name": extracted_name if name_extracted else None,
                "should_retry": should_retry,
                "step_complete": name_extracted
            }

        except Exception as e:
            print(f"❌ [NAME_COLLECTION] Error: {str(e)}")
            return {
                "response": "May I have your name, please?",
                "name_extracted": False,
                "name": None,
                "should_retry": retry_count < settings.max_retries_per_step,
                "step_complete": False
            }

    async def collect_background(
        self,
        current_message: str,
        caller_name: Optional[str],
        message_history: Optional[List[Dict]] = None,
        intent_type: Optional[str] = None,
        company_name: Optional[str] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Step 2: Collect background information

        Args:
            current_message: Current user message
            caller_name: Caller's name (from step 1)
            message_history: Conversation history
            intent_type: Intent classification (opportunity/other)
            company_name: Company name for personalization
            retry_count: Number of retries so far

        Returns:
            Dict with:
                - response: AI response to user
                - background_extracted: Whether background was extracted
                - background: Extracted background info
                - should_retry: Whether to retry
        """
        company = company_name or "us"

        # Customize prompt based on intent
        if intent_type == "opportunity":
            task_description = """Your task: Extract BACKGROUND information about the OPPORTUNITY.

Important information to collect:
- Organization/institution they represent
- Nature of the opportunity (job, research, collaboration, etc.)
- Brief description of what they're offering
- Any key details about the opportunity

If they provide background, extract and summarize it warmly.
If they don't provide enough background, ask specifically about their organization and the opportunity."""

        elif intent_type == "other":
            task_description = """Your task: Extract BACKGROUND information about their inquiry.

Important information to collect:
- What they need help with
- Any relevant context
- Their situation or issue

If they provide background, extract and summarize it.
If they don't provide enough context, ask directly what they need assistance with."""

        else:
            task_description = """Your task: Extract BACKGROUND information.

Ask them what brings them to call today and collect relevant context."""

        system_prompt = f"""You are a friendly AI assistant for {company}.

{task_description}

Guidelines:
- Be warm and show interest
- Keep responses brief (2-3 sentences)
- Be specific in your questions

Return JSON format:
{{
    "background": "extracted background info or null",
    "background_extracted": true/false,
    "response": "your response to the user"
}}
"""

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        if message_history:
            for msg in message_history[-4:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        messages.append({
            "role": "user",
            "content": f"Current message: {current_message}"
        })

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)

            background_extracted = result.get("background_extracted", False)
            extracted_background = result.get("background")

            should_retry = not background_extracted and retry_count < settings.max_retries_per_step

            return {
                "response": result.get("response", "Thank you. What brings you to call us today?"),
                "background_extracted": background_extracted,
                "background": extracted_background if background_extracted else None,
                "should_retry": should_retry,
                "step_complete": background_extracted
            }

        except Exception as e:
            print(f"❌ [BACKGROUND_COLLECTION] Error: {str(e)}")
            return {
                "response": "Thank you. What brings you to call us today?",
                "background_extracted": False,
                "background": None,
                "should_retry": retry_count < settings.max_retries_per_step,
                "step_complete": False
            }

    async def collect_additional_info(
        self,
        current_message: str,
        caller_name: Optional[str],
        background: Optional[str],
        message_history: Optional[List[Dict]] = None,
        intent_type: Optional[str] = None,
        company_name: Optional[str] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Step 3: Collect additional information

        Args:
            current_message: Current user message
            caller_name: Caller's name
            background: Background info from step 2
            message_history: Conversation history
            intent_type: Intent classification (opportunity/other)
            company_name: Company name for personalization
            retry_count: Number of retries so far

        Returns:
            Dict with:
                - response: AI response to user
                - additional_info_extracted: Whether additional info was extracted
                - additional_info: Extracted additional information
                - should_retry: Whether to retry
        """
        company = company_name or "us"

        # Customize prompt based on intent
        if intent_type == "opportunity":
            task_description = """Your task: Collect ADDITIONAL CONTACT & TIMING information.

Important information to collect:
- Best contact method (email, phone)
- Contact details
- Timeline or urgency
- Any other relevant details they want to share

This is the FINAL step - gather any remaining important information."""

        else:
            task_description = """Your task: Collect ADDITIONAL INFORMATION.

Important information to collect:
- Best way to reach them (phone, email)
- Any other details they want to share
- Anything else relevant to their inquiry

This is the FINAL step - gather any remaining important information."""

        system_prompt = f"""You are a friendly AI assistant for {company}.

{task_description}

Guidelines:
- Ask for contact information if not provided
- Check if there's anything else they'd like to share
- Keep responses brief (2-3 sentences)
- This is the last step before ending the call

Return JSON format:
{{
    "additional_info": "extracted additional info or null",
    "additional_info_extracted": true/false,
    "response": "your response to the user"
}}
"""

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        if message_history:
            for msg in message_history[-4:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        messages.append({
            "role": "user",
            "content": f"Current message: {current_message}"
        })

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)

            info_extracted = result.get("additional_info_extracted", False)
            extracted_info = result.get("additional_info")

            should_retry = not info_extracted and retry_count < settings.max_retries_per_step

            default_response = (
                "Is there anything else you'd like to share? "
                "What's the best way to contact you?"
            )

            return {
                "response": result.get("response", default_response),
                "additional_info_extracted": info_extracted,
                "additional_info": extracted_info if info_extracted else None,
                "should_retry": should_retry,
                "step_complete": info_extracted
            }

        except Exception as e:
            print(f"❌ [ADDITIONAL_INFO_COLLECTION] Error: {str(e)}")
            default_response = (
                "Is there anything else you'd like to share? "
                "What's the best way to contact you?"
            )

            return {
                "response": default_response,
                "additional_info_extracted": False,
                "additional_info": None,
                "should_retry": retry_count < settings.max_retries_per_step,
                "step_complete": False
            }

    def generate_closing_message(
        self,
        caller_name: Optional[str],
        intent_type: Optional[str]
    ) -> str:
        """Generate appropriate closing message based on intent

        Args:
            caller_name: Caller's name
            intent_type: Intent classification (opportunity/other)

        Returns:
            Closing message string
        """
        name_str = caller_name or ""
        name_part = f", {name_str}" if name_str else ""

        if intent_type == "opportunity":
            return (
                f"Thank you so much for reaching out{name_part}. "
                f"I've recorded all your information and our team will contact you shortly to discuss this opportunity further. "
                f"We appreciate your interest. Have a great day!"
            )
        elif intent_type == "other":
            return (
                f"Thank you for calling{name_part}. "
                f"I've recorded your inquiry and our team will review it carefully. "
                f"Someone from our office will get back to you as soon as possible. "
                f"Have a good day!"
            )
        else:
            return (
                f"Thank you for your call{name_part}. "
                f"I've noted your information and our team will follow up with you shortly. "
                f"Have a great day!"
            )


# Global instance
simple_info_collector = SimpleInfoCollector()
