# AI Service - DispatchAI Platform

**FastAPI-based AI Agent Service** for intelligent call handling, customer information extraction, and service scheduling.

## 🎯 Overview

The AI Service is the brain of the DispatchAI platform, handling all AI-powered interactions with customers during phone calls. It uses LangGraph-based multi-step agents to collect customer information, understand service requests, and schedule appointments.

## 🏗️ Architecture

### Core Components

```
apps/ai/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration & settings
│   ├── api/                    # API endpoints
│   │   ├── health.py          # Health check
│   │   ├── chat.py            # Chat endpoints
│   │   ├── call.py            # Call conversation handling
│   │   ├── summary.py         # Call summaries
│   │   ├── email.py           # Email sending
│   │   └── dispatch.py        # Service dispatch & calendar
│   ├── services/              # Core business logic
│   │   ├── call_handler.py   # Main conversation orchestrator
│   │   ├── dialog_manager.py # Multi-turn conversation state
│   │   ├── llm_service.py    # LLM integration
│   │   ├── llm_speech_corrector.py  # Speech-to-text correction
│   │   ├── call_summary.py   # Post-call summaries
│   │   ├── redis_service.py  # Redis interactions
│   │   ├── ses_email.py      # Email service (AWS SES)
│   │   ├── ics_lib.py        # ICS calendar file generation
│   │   └── retrieve/         # Information extractors
│   │       ├── customer_info_extractors.py
│   │       └── time_extractor.py
│   ├── models/                # Data models
│   │   ├── call.py           # CallSkeleton, Message, UserInfo
│   │   └── chat.py           # Chat models
│   ├── client/                # External service clients
│   │   └── mcp_client.py     # MCP (Model Context Protocol) client
│   ├── custom_types/         # Custom type definitions
│   │   └── customer_service_types.py
│   ├── infrastructure/       # Infrastructure clients
│   │   └── redis_client.py  # Redis async client
│   └── utils/                # Utilities
│       ├── mcp_parse.py     # MCP response parsing
│       ├── prompts/         # LLM prompts
│       └── validators/      # Data validators
├── tests/                    # Test suite
├── pyproject.toml           # Python dependencies (uv)
├── Dockerfile.dev          # Development Docker image
├── Dockerfile.uat          # UAT Docker image
└── Makefile                # Build automation
```

## 🤖 AI Workflow - 8-Step Conversation Agent

The AI service implements a sophisticated **8-step LangGraph workflow** to handle customer service conversations:

### Workflow Steps

1. **Name Collection** → Collect customer's full name
2. **Phone Collection** → Verify contact phone number
3. **Address Collection** → Get service address with address validation
4. **Service Selection** → Help customer choose from available services
5. **Time Selection** → Schedule preferred appointment time
6. **Booking Confirmation** → Confirm all details
7. **Dispatch** → Send email + calendar invite
8. **Completion** → Wrap up conversation

### State Management

The agent maintains conversation state in **Redis** using `CallSkeleton` format:

```python
{
    "callSid": "CAxxx...",           # Unique call ID
    "company": {...},                # Company info
    "user": {
        "userInfo": {                # Collected customer info
            "name": "...",
            "phone": "...",
            "email": "...",
            "address": "..."
        },
        "service": {...}             # Selected service
    },
    "services": [...],               # Available services list
    "history": [...]                 # Message history
}
```

## 🔌 API Endpoints

### Core Endpoints

**Base URL**: `http://localhost:8000/api`

#### `/ai/conversation` (POST)
Main conversation endpoint for call handling.

**Request**:
```json
{
    "callSid": "CA1234567890",
    "customerMessage": {
        "message": "Hi, I need a cleaning service",
        "speaker": "customer",
        "timestamp": "2024-01-01T12:00:00Z"
    }
}
```

**Response**:
```json
{
    "assistantMessage": {
        "message": "I'd be happy to help you with a cleaning service. May I get your name please?",
        "speaker": "assistant",
        "timestamp": "2024-01-01T12:00:01Z"
    },
    "userInfo": {...},              # Updated customer info
    "conversationComplete": false,  # Booking status
    "currentStep": "collect_name"   # Current workflow step
}
```

#### `/dispatch/send-email-and-calendar` (POST)
Send email with calendar integration (Google/Outlook/iCal).

**Request**:
```json
{
    "to": "customer@example.com",
    "subject": "Service Booking Confirmation",
    "body": "Your appointment is confirmed...",
    "summary": "Cleaning Service",
    "start": "2024-01-15T10:00:00+10:00",
    "end": "2024-01-15T12:00:00+10:00",
    "calendarapp": "google",       # "none", "google", "outlook"
    "access_token": "...",         # OAuth token
    "calendar_id": "primary"
}
```

#### `/summary/generate` (POST)
Generate AI-powered call summary after conversation ends.

#### `/email/send` (POST)
Send simple email without calendar.

## 🧠 Key Services

### CustomerServiceLangGraph (`call_handler.py`)

Main orchestrator for conversation workflow.

**Key Methods**:
- `process_conversation(state, message)` - Process user message through workflow
- `_collect_name()` - Name collection logic
- `_collect_phone()` - Phone validation & collection
- `_collect_address()` - Address extraction with validation
- `_select_service()` - Service recommendation & selection
- `_select_time()` - Time slot scheduling
- `_complete_booking()` - Final confirmation & dispatch

**Features**:
- Multi-attempt collection (3 max attempts per field)
- Speech correction for addresses
- Service price & description display
- Natural conversation flow
- Context-aware responses

### Information Extractors (`services/retrieve/`)

Specialized extractors for structured data:

- **`customer_info_extractors.py`**: Name, phone, email, address extraction
- **`time_extractor.py`**: Natural language time parsing with timezone handling

**Usage**:
```python
from services.retrieve.customer_info_extractors import (
    extract_name_from_conversation,
    extract_phone_from_conversation,
    extract_address_from_conversation,
)

name = extract_name_from_conversation(message_history)
phone = extract_phone_from_conversation(message_history)
address = extract_address_from_conversation(message_history)
```

### LLM Service (`llm_service.py`)

Abstraction layer for OpenAI LLM calls.

**Features**:
- OpenAI GPT-4 integration
- Streaming support
- Custom system prompts
- Context management
- Error handling & retries

### Speech Corrector (`llm_speech_corrector.py`)

Corrects common speech-to-text errors for addresses.

**Example**:
```
Input:  "1 twenty five Johnson street"
Output: "1/25 Johnson Street"
```

### Call Summary (`call_summary.py`)

Generates structured summaries of completed calls:
- Key information extracted
- Service requested
- Booking details
- Customer sentiment

### Email Service (`ses_email.py`)

AWS SES integration for sending emails:
- Plain text & HTML emails
- ICS calendar attachments
- OAuth-based calendar integration
- Email templates

## 🧪 Testing

### Run Tests

```bash
cd apps/ai

# Install dependencies
make sync

# Run all tests with coverage
make test

# Run tests in verbose mode
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_smoke.py -v

# Coverage report
uv run pytest --cov=app --cov-report=html
```

### Code Quality

```bash
# Lint with Ruff
make lint

# Auto-fix linting issues
make lint-fix

# Format code
make format

# Type check with MyPy
make typecheck

# Run all checks
make check-all
```

### Manual Testing

```bash
# Start AI service
docker compose up ai

# Test health endpoint
curl http://localhost:8000/api/health

# View API docs
open http://localhost:8000/docs
```

## 🔧 Configuration

### Environment Variables

**Required**:
```bash
OPENAI_API_KEY=sk-...              # OpenAI API key
OPENAI_MODEL=gpt-4o-mini           # LLM model name
REDIS_HOST=redis                   # Redis host
REDIS_PORT=6379                    # Redis port
```

**Optional**:
```bash
REDIS_URL=redis://localhost:6379   # Full Redis URL
API_PREFIX=/api                    # API prefix
DEBUG=true                         # Debug mode
MAX_ATTEMPTS=3                     # Max collection attempts
OPENAI_MAX_TOKENS=2500             # Max response tokens
OPENAI_TEMPERATURE=0.0             # Temperature (0=deterministic)
CORS_ORIGINS=["*"]                 # CORS allowed origins
```

### Supported Services

Services are configured via `config.py`:

```python
supported_services = [
    "clean", "cleaning",
    "garden", "gardening",
    "plumber", "plumbing",
    "electric", "electrical",
    "repair"
]
```

### Supported Time Keywords

Natural language time parsing:

```python
supported_time_keywords = [
    "tomorrow", "morning", "afternoon", "evening",
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday"
]
```

## 🚀 Development

### Local Development

1. **Install dependencies**:
   ```bash
   cd apps/ai
   make sync
   ```

2. **Run locally**:
   ```bash
   # With uv
   uv run uvicorn app.main:app --reload --port 8000

   # Or with Docker
   docker compose up ai
   ```

3. **Access service**:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/api/health

### Dependency Management

The AI service uses **uv** for fast Python package management:

```bash
# Install new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update dependencies
uv sync --upgrade

# Remove dependency
uv remove package-name
```

### Adding New Extractors

1. Create extractor in `app/services/retrieve/customer_info_extractors.py`:
   ```python
   def extract_new_field_from_conversation(message_history: list) -> str:
       # Extract logic
       return extracted_value
   ```

2. Import in `call_handler.py`:
   ```python
   from .retrieve.customer_info_extractors import extract_new_field_from_conversation
   ```

3. Use in workflow step:
   ```python
   value = extract_new_field_from_conversation(state["message_history"])
   ```

### Adding New Workflow Steps

1. Add step to `CustomerServiceState` in `custom_types/customer_service_types.py`
2. Implement step handler in `call_handler.py`
3. Add step transition logic in `process_conversation()`
4. Update prompts in `utils/prompts/`

## 📊 Monitoring & Debugging

### Logs

```bash
# View AI service logs
docker logs dispatchai-ai -f

# Filter logs
docker logs dispatchai-ai 2>&1 | grep "CONVERSATION"
```

### Redis Inspection

```bash
# Connect to Redis
docker exec -it dispatchai-redis redis-cli

# List all call skeletons
KEYS callskeleton:*

# Get specific call skeleton
GET callskeleton:CA1234567890

# Check message history
LRANGE history:CA1234567890 0 -1
```

### Debugging Conversation Flow

Add debug logs in `call_handler.py`:

```python
print(f"🔍 [STEP] {state['current_step']}")
print(f"🔍 [INFO] Name: {state['name']}, Phone: {state['phone']}")
print(f"🔍 [COMPLETE] All fields: {all_complete}")
```

## 🔗 Integration Points

### With Backend

**Redis**: CallSkeleton storage & message history
```python
from services.redis_service import get_call_skeleton, update_user_info
```

**MongoDB**: Not directly accessed, all via Backend API

### With Frontend

**HTTP**: Frontend calls `/api/ai/conversation` with user messages

**WebSocket/SSE**: Not currently used (can be added for real-time)

### With External Services

**OpenAI**: LLM inference
**AWS SES**: Email sending
**Google Calendar**: OAuth + Calendar API
**Outlook Calendar**: Microsoft Graph API

## 🐛 Common Issues

### Issue: "CallSkeleton not found"

**Cause**: Redis doesn't have call data
**Fix**: Ensure Backend created CallSkeleton before AI service called

### Issue: "Address validation failed"

**Cause**: Invalid address format or speech-to-text error
**Fix**: Check speech corrector, add more patterns

### Issue: "Service not in available list"

**Cause**: Customer requesting unsupported service
**Fix**: Update `supported_services` in config or add service mapping

### Issue: "LLM timeout"

**Cause**: Slow response from OpenAI or network issue
**Fix**: Increase timeout, add retry logic, check API key

## 📚 Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **OpenAI API**: https://platform.openai.com/docs
- **Redis Python**: https://redis.readthedocs.io
- **uv Package Manager**: https://github.com/astral-sh/uv

## 🤝 Contributing

When adding new features:
1. Update tests in `tests/`
2. Add type hints
3. Run `make check-all`
4. Update this README
5. Document new API endpoints