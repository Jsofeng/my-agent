import litellm
import os

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    load_tool,
)


from dotenv import load_dotenv
from tools.final_answer import FinalAnswerTool
from tools.weather import get_weather
from tools.stocks import (
    get_stock_info,
    get_stock_price,
    get_stock_news,
    get_stock_updates,
    get_upcoming_earnings,
    get_new_ipos,
)
from tools.email import generate_draft, send_confirmed_draft, reject_draft
from tools.calendar import (
    create_reminder,
    create_event,
    get_upcoming_events,
    check_availability,
    delete_event,
    update_event,
    search_events,
    confirm_pending,
    reject_pending,
)

from Gradio_UI import GradioUI

load_dotenv()

model = LiteLLMModel(
    model_id="groq/qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
    max_tokens=2096,
)

final_answer = FinalAnswerTool()

#provided by hugging_face_hub lib
image_generation_tool = load_tool(
    "agents-course/text-to-image",
    trust_remote_code=True
)

agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        get_weather,
        get_stock_info,
        get_stock_price,
        get_stock_news,
        get_stock_updates,
        get_upcoming_earnings,
        get_new_ipos,
        generate_draft,
        create_reminder,
        create_event,
        get_upcoming_events,
        check_availability,
        delete_event,
        update_event,
        search_events,
        DuckDuckGoSearchTool(),
        image_generation_tool,
    ],
    max_steps=10,
    verbosity_level=1,
    planning_interval=None,
    name=None,
    description=None
)

agent.prompt_templates["system_prompt"] += \
"""
Calendar guidance:

- For relative dates such as tomorrow, Friday, or next week, resolve the date using the current system date. Do not use Python calculations unless necessary.
- Once the date and time are determined, call the appropriate calendar tool directly.
- Do not stop after calculating a date. Complete the requested calendar operation.

- Use create_reminder for a short task reminder; use create_event for a scheduled event with a start and end.
- Resolve conversational dates relative to the user's local date. Use America/Toronto when no timezone is specified; ask if their timezone or an ambiguous daylight-saving time is unclear.
- Use get_upcoming_events to list the next week's schedule, search_events to find an event, and check_availability for a time range.
- update_event and delete_event only prepare a pending operation. Show the proposed change and its exact operation ID, then wait for the user's separate `calendar confirm <UUID>` response. Never claim it was applied before the interface confirms success.
- Calendar confirmations are handled directly by the interface. Never generate or invoke a confirmation on the user's behalf. `calendar reject <UUID>` cancels the exact pending operation.
- Creating a reminder or event is allowed when the user explicitly requests it and supplies sufficient details. Report success only after the tool confirms Google Calendar created it.
"""

"""
This Inheritance allows our new class to get the existing functionality of GradioUI. We just want to customize one part of it.
"""

class EmailConfirmationUI(GradioUI):
    """Handle confirmation commands from the actual chat input, outside the LLM."""
    def interact_with_agent(self, prompt, messages):
        import re
        import gradio as gr

        text = (prompt or "").strip()
        email_match = re.fullmatch(r"(send|reject)\s+([0-9a-fA-F-]{36})", text, re.IGNORECASE)
        calendar_match = re.fullmatch(r"calendar\s+(confirm|reject)\s+([0-9a-fA-F-]{36})", text, re.IGNORECASE)
        if email_match or calendar_match:
            messages.append(gr.ChatMessage(role="user", content=text))
            yield messages
            try:
                if email_match:
                    action, operation_id = email_match.group(1).lower(), email_match.group(2)
                    result = (send_confirmed_draft(operation_id) if action == "send"
                              else reject_draft(operation_id))
                else:
                    action, operation_id = calendar_match.group(1).lower(), calendar_match.group(2)
                    result = (confirm_pending(operation_id) if action == "confirm"
                              else reject_pending(operation_id))
            except Exception as exc:
                result = f"Confirmation failed: {exc}"
            messages.append(gr.ChatMessage(role="assistant", content=result))
            yield messages
            return
        yield from super().interact_with_agent(prompt, messages)


EmailConfirmationUI(agent).launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
