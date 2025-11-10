from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum


# Call Status and Processing Models
class CallStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CallRequest(BaseModel):
    phone_number: str
    customer_name: Optional[str] = None
    purpose: Optional[str] = None


class CallSummary(BaseModel):
    call_id: str
    status: CallStatus
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None
    duration: Optional[int] = None


# Call Data Structure Models
class Message(BaseModel):
    speaker: Literal["AI", "customer"]
    message: str
    startedAt: str


class Address(BaseModel):
    address: str = Field(default="", description="Complete address as a single string")


class UserInfo(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None  # Complete address string


class Company(BaseModel):
    id: str
    name: str
    email: str
    userId: str


class UserState(BaseModel):
    userInfo: UserInfo


class CallSkeleton(BaseModel):
    callSid: str
    company: Company
    user: UserState
    history: List[Message]
    createdAt: Optional[str] = None

    # Caller information
    callerNumber: Optional[str] = None
    callStartAt: Optional[str] = None

    # Intent classification fields
    intent: Optional[str] = None  # Classification result: scam/opportunity/other
    intentClassified: bool = False  # Whether intent has been classified
