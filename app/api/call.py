from fastapi import APIRouter, HTTPException
from typing import Any, Dict
from pydantic import BaseModel, Field, ValidationError
from app.models.call import Message, CallSkeleton
from app.services.redis_service import (
    get_call_skeleton,
    update_intent_classification,
)
from app.services.simple_info_collector import simple_info_collector
from app.intent_classification.services.classifier import intent_classifier
from datetime import datetime, timezone
import httpx
from app.config import get_settings

settings = get_settings()

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


async def create_calllog(
    call_sid: str,
    user_id: str,
    caller_number: str,
    caller_name: str,
    start_at: datetime,
    intent: str
):
    """Create calllog in backend

    Args:
        call_sid: Twilio CallSid
        user_id: User ID (owner of the phone number)
        caller_number: Caller's phone number
        caller_name: Caller's name (if available)
        start_at: Call start time
        intent: Intent classification (opportunity/other)
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{settings.backend_url}/internal/users/{user_id}/calllogs",
                json={
                    "callSid": call_sid,
                    "userId": user_id,
                    "callerNumber": caller_number,
                    "callerName": caller_name,
                    "startAt": start_at.isoformat(),
                    "intent": intent
                }
            )

            if response.status_code == 201:
                print(f"✅ [CALLLOG] Created calllog for {call_sid} with intent: {intent}")
                return True
            else:
                print(f"⚠️ [CALLLOG] Failed to create calllog: {response.status_code} - {response.text}")
                return False

    except Exception as e:
        print(f"❌ [CALLLOG] Error creating calllog: {str(e)}")
        return False


def get_collection_state(callskeleton: CallSkeleton) -> Dict[str, Any]:
    """Extract collection state from callskeleton

    Returns:
        Dict with current_step, collected_name, collected_background, collected_additional,
        retry_counts, and calllog_created flag
    """
    # Use user.userInfo to store collected information
    user_info = callskeleton.user.userInfo if callskeleton.user else {}

    # Default state
    state = {
        "current_step": "name",  # name | background | additional_info | completed
        "collected_name": user_info.name if user_info.name else None,
        "collected_background": None,  # Will be stored in address field temporarily
        "collected_additional": None,  # Will be stored in phone field temporarily
        "name_retry_count": 0,
        "background_retry_count": 0,
        "additional_retry_count": 0,
        "calllog_created": callskeleton.intentClassified  # Use intentClassified as flag
    }

    # Try to extract from address/phone fields (temporary storage)
    if user_info.address:
        state["collected_background"] = user_info.address

    if user_info.phone:
        state["collected_additional"] = user_info.phone

    # Determine current step based on what's been collected
    if state["collected_name"] and state["collected_background"] and state["collected_additional"]:
        state["current_step"] = "completed"
    elif state["collected_name"] and state["collected_background"]:
        state["current_step"] = "additional_info"
    elif state["collected_name"]:
        state["current_step"] = "background"
    else:
        state["current_step"] = "name"

    return state


@router.post("/conversation")
async def ai_conversation(data: ConversationInput):
    """AI conversation endpoint with real-time intent classification

    New Flow:
    1. Get CallSkeleton data from Redis
    2. Perform intent classification on EVERY turn
    3. If SCAM: Return polite goodbye + hangup immediately
    4. If NON-SCAM (opportunity/other):
       - Create calllog (if not already created)
       - Execute 3-step information collection:
         Step 1: Name
         Step 2: Background
         Step 3: Additional Info
       - After Step 3: Generate closing message + hangup
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

    # 3. ⚡ REAL-TIME INTENT CLASSIFICATION (Every Turn)
    print("🎯 [INTENT] Performing real-time intent classification...")
    intent_result = None
    try:
        intent_result = await intent_classifier.classify_intent(
            current_message=data.customerMessage.message,
            message_history=message_history,
            call_sid=data.callSid
        )

        print(f"✅ [INTENT] Classified as: {intent_result['intent']}")
        print(f"   Confidence: {intent_result['confidence']:.2f}")
        print(f"   Reasoning: {intent_result['reasoning']}")

    except Exception as e:
        print(f"❌ [INTENT] Classification failed: {str(e)}")
        # Default to "other" on failure - safer than blocking
        intent_result = {
            "intent": "other",
            "confidence": 0.0,
            "reasoning": f"Classification failed: {str(e)}"
        }

    current_intent = intent_result["intent"]

    # 4. Handle SCAM - Immediate hangup
    if current_intent == "scam":
        print("🚫 [SCAM] Scam detected, ending call immediately...")

        # Save intent to Redis
        update_intent_classification(
            call_sid=data.callSid,
            intent="scam",
            confidence=intent_result["confidence"],
            reasoning=intent_result["reasoning"],
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        ai_message = (
            "Thank you for calling. I'm sorry, but I'm unable to assist with this matter. "
            "If you have a legitimate inquiry, please contact us through our official channels. "
            "Have a good day. Goodbye."
        )

        ai_response = {
            "speaker": "AI",
            "message": ai_message,
            "startedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

        return {
            "aiResponse": ai_response,
            "shouldHangup": True
        }

    # 5. Handle NON-SCAM (opportunity/other)
    print(f"✅ [NON-SCAM] Legitimate call detected: {current_intent}")

    # Get collection state
    collection_state = get_collection_state(callskeleton)
    current_step = collection_state["current_step"]
    calllog_created = collection_state["calllog_created"]

    print(f"📊 [COLLECTION] Current step: {current_step}")
    print(f"📊 [COLLECTION] Calllog created: {calllog_created}")

    # 6. Create calllog if not already created
    if not calllog_created:
        print("📝 [CALLLOG] Creating calllog for non-scam call...")

        # Save intent to Redis first
        update_intent_classification(
            call_sid=data.callSid,
            intent=current_intent,
            confidence=intent_result["confidence"],
            reasoning=intent_result["reasoning"],
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Parse start time
        start_at = datetime.fromisoformat(callskeleton.callStartAt.replace('Z', '+00:00')) if callskeleton.callStartAt else datetime.now(timezone.utc)

        # Create calllog
        await create_calllog(
            call_sid=data.callSid,
            user_id=callskeleton.company.userId,
            caller_number=callskeleton.callerNumber or "Unknown",
            caller_name=collection_state["collected_name"] or "Unknown Caller",
            start_at=start_at,
            intent=current_intent
        )

        calllog_created = True

    # 7. Execute 3-Step Information Collection
    should_hangup = False
    ai_message = ""

    if current_step == "completed":
        # All steps completed - should not reach here normally
        print("⚠️ [COLLECTION] Unexpected: all steps already completed")
        ai_message = simple_info_collector.generate_closing_message(
            caller_name=collection_state["collected_name"],
            intent_type=current_intent
        )
        should_hangup = True

    elif current_step == "name":
        # Step 1: Collect Name
        print(f"📝 [STEP 1] Collecting name (retry: {collection_state['name_retry_count']})")

        result = await simple_info_collector.collect_name(
            current_message=data.customerMessage.message,
            message_history=message_history,
            intent_type=current_intent,
            company_name=callskeleton.company.name if callskeleton.company else None,
            retry_count=collection_state['name_retry_count']
        )

        ai_message = result["response"]

        # If name extracted, save to Redis and move to next step
        if result["step_complete"] and result["name"]:
            from app.services.redis_service import update_user_info_field
            update_user_info_field(
                call_sid=data.callSid,
                field_name="name",
                field_value=result["name"]
            )
            print(f"✅ [STEP 1] Name collected: {result['name']}")

    elif current_step == "background":
        # Step 2: Collect Background
        print(f"📝 [STEP 2] Collecting background (retry: {collection_state['background_retry_count']})")

        result = await simple_info_collector.collect_background(
            current_message=data.customerMessage.message,
            caller_name=collection_state["collected_name"],
            message_history=message_history,
            intent_type=current_intent,
            company_name=callskeleton.company.name if callskeleton.company else None,
            retry_count=collection_state['background_retry_count']
        )

        ai_message = result["response"]

        # If background extracted, save to Redis (using address field temporarily)
        if result["step_complete"] and result["background"]:
            from app.services.redis_service import update_user_info_field
            update_user_info_field(
                call_sid=data.callSid,
                field_name="address",
                field_value=result["background"]
            )
            print(f"✅ [STEP 2] Background collected: {result['background'][:100]}...")

    elif current_step == "additional_info":
        # Step 3: Collect Additional Info
        print(f"📝 [STEP 3] Collecting additional info (retry: {collection_state['additional_retry_count']})")

        result = await simple_info_collector.collect_additional_info(
            current_message=data.customerMessage.message,
            caller_name=collection_state["collected_name"],
            background=collection_state["collected_background"],
            message_history=message_history,
            intent_type=current_intent,
            company_name=callskeleton.company.name if callskeleton.company else None,
            retry_count=collection_state['additional_retry_count']
        )

        ai_message = result["response"]

        # If additional info extracted, save to Redis and prepare to close
        if result["step_complete"] and result["additional_info"]:
            from app.services.redis_service import update_user_info_field
            update_user_info_field(
                call_sid=data.callSid,
                field_name="phone",
                field_value=result["additional_info"]
            )
            print(f"✅ [STEP 3] Additional info collected: {result['additional_info'][:100]}...")

            # All 3 steps completed - generate closing message
            print("🎉 [COLLECTION] All 3 steps completed, preparing to end call")
            ai_message = simple_info_collector.generate_closing_message(
                caller_name=collection_state["collected_name"],
                intent_type=current_intent
            )
            should_hangup = True

    # 8. Build response
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
    """Simple AI reply endpoint for telephony service

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
