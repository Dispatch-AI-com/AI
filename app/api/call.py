from fastapi import APIRouter, HTTPException
from typing import Any, Dict
from pydantic import BaseModel, Field, ValidationError
from models.call import Message, CallSkeleton
from services.redis_service import (
    get_call_skeleton,
    update_intent_classification,
    get_intent_classification,
)
from services.call_handler import CustomerServiceLangGraph
from services.simple_chatbot import simple_chatbot
from intent_classification.services.classifier import intent_classifier
from custom_types import CustomerServiceState
from datetime import datetime, timezone

router = APIRouter(
    prefix="/ai",
    tags=["AI"],
    responses={404: {"description": "Not found"}},
)


# AI conversation input model
class ConversationInput(BaseModel):
    callSid: str = Field(..., description="Twilio CallSid – unique call ID")
    customerMessage: Message = Field(..., description="Customer message object")


# Simple reply input model (for telephony service)
class ReplyInput(BaseModel):
    callSid: str = Field(..., description="Twilio CallSid – unique call ID")
    message: str = Field(..., description="User message text")


# Global customer service agent
cs_agent = CustomerServiceLangGraph()


@router.post("/conversation")
async def ai_conversation(data: ConversationInput):
    """AI conversation endpoint with intent classification

    Flow:
    1. Get CallSkeleton data from Redis
    2. Check if intent has been classified
    3. If not classified and sufficient messages (>=3), perform classification
    4. Based on intent:
       - SCAM: Return polite goodbye + hangup
       - NON-SCAM: Use simple chatbot for conversation
    5. Save intent classification to Redis (calllog)
    """
    # 1. Get CallSkeleton data
    try:
        callskeleton_dict = get_call_skeleton(data.callSid)
        callskeleton = CallSkeleton.model_validate(callskeleton_dict)
    except ValueError:
        raise HTTPException(status_code=422, detail="CallSkeleton not found")
    except ValidationError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid CallSkeleton data format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    # 2. Convert message history
    message_history = []
    if callskeleton.history:
        for msg in callskeleton.history[-8:]:  # Last 8 messages for context
            message_history.append(
                {
                    "role": "user" if msg.speaker == "customer" else "assistant",
                    "content": msg.message,
                }
            )

    print(f"🔍 [CONVERSATION] CallSid: {data.callSid}")
    print(f"🔍 [CONVERSATION] Message history length: {len(callskeleton.history)}")
    print(f"🔍 [CONVERSATION] Intent classified: {callskeleton.intentClassified}")

    # 3. Check if intent classification is needed
    intent_result = None
    should_classify = (
        not callskeleton.intentClassified
        and len(callskeleton.history) >= 3  # Need at least 3 messages for context
    )

    if should_classify:
        print("🎯 [INTENT] Performing intent classification...")
        try:
            intent_result = await intent_classifier.classify_intent(
                current_message=data.customerMessage.message,
                message_history=message_history,
                call_sid=data.callSid
            )

            # Save intent classification to Redis
            update_intent_classification(
                call_sid=data.callSid,
                intent=intent_result["intent"],
                confidence=intent_result["confidence"],
                reasoning=intent_result["reasoning"],
                timestamp=datetime.now(timezone.utc).isoformat()
            )

            print(f"✅ [INTENT] Classified as: {intent_result['intent']}")
            print(f"   Confidence: {intent_result['confidence']:.2f}")
            print(f"   Reasoning: {intent_result['reasoning']}")

        except Exception as e:
            print(f"❌ [INTENT] Classification failed: {str(e)}")
            # Continue with conversation on classification failure
            intent_result = None

    # 4. Determine current intent (from new classification or existing)
    current_intent = None
    if intent_result:
        current_intent = intent_result["intent"]
    elif callskeleton.intentClassified:
        current_intent = callskeleton.intent

    print(f"🔍 [CONVERSATION] Current intent: {current_intent}")

    # 5. Generate response based on intent
    should_hangup = False
    ai_message = ""

    if current_intent == "scam":
        # SCAM detected - polite but firm goodbye
        print("🚫 [SCAM] Scam detected, ending call...")
        ai_message = (
            "Thank you for calling. I'm sorry, but I'm unable to assist with this matter. "
            "If you have a legitimate inquiry, please contact us through our official channels. "
            "Have a good day. Goodbye."
        )
        should_hangup = True

    else:
        # NON-SCAM or not yet classified - use simple chatbot
        print(f"💬 [CHATBOT] Generating conversational response (intent: {current_intent})...")
        try:
            ai_message = await simple_chatbot.generate_response(
                current_message=data.customerMessage.message,
                message_history=message_history,
                intent_type=current_intent,
                company_name=callskeleton.company.name if callskeleton.company else None
            )
        except Exception as e:
            print(f"❌ [CHATBOT] Response generation failed: {str(e)}")
            ai_message = "I appreciate you reaching out. How can I assist you today?"

    # 6. Build response
    ai_response = {
        "speaker": "AI",
        "message": ai_message,
        "startedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    response_data: Dict[str, Any] = {"aiResponse": ai_response}

    if should_hangup:
        response_data["shouldHangup"] = True
        print("📞 [HANGUP] Call will be terminated")

    return response_data


@router.post("/reply")
async def ai_reply(data: ReplyInput):
    """Simple AI reply endpoint for telephony service - Updated for 8-step workflow

    This endpoint provides a simplified interface that matches
    what the telephony service expects:
    - Input: { callSid, message }
    - Output: { replyText }
    """
    # Convert simple input to our internal format
    customer_message = Message(
        speaker="customer",
        message=data.message,
        startedAt=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    # Use the existing conversation logic
    conversation_data = ConversationInput(
        callSid=data.callSid, customerMessage=customer_message
    )

    # Call the main conversation handler
    result = await ai_conversation(conversation_data)

    # Return in format expected by telephony service
    return {"replyText": result["aiResponse"]["message"]}
