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
        DuckDuckGoSearchTool(),
        image_generation_tool,
    ],
    max_steps=10,
    verbosity_level=1,
    planning_interval=None,
    name=None,
    description=None
)

class EmailConfirmationUI(GradioUI):
    """Handle confirmation commands from the actual chat input, outside the LLM."""
    def interact_with_agent(self, prompt, messages):
        import re
        import gradio as gr

        text = (prompt or "").strip()
        match = re.fullmatch(r"(send|reject)\s+([0-9a-fA-F-]{36})", text, re.IGNORECASE)
        if match:
            messages.append(gr.ChatMessage(role="user", content=text))
            yield messages
            action, draft_id = match.group(1).lower(), match.group(2)
            try:
                result = (send_confirmed_draft(draft_id) if action == "send"
                          else reject_draft(draft_id))
            except Exception as exc:
                result = f"Email action failed: {exc}"
            messages.append(gr.ChatMessage(role="assistant", content=result))
            yield messages
            return
        yield from super().interact_with_agent(prompt, messages)


EmailConfirmationUI(agent).launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
