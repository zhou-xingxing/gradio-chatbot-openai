import os
import logging
from typing import Generator
from dotenv import load_dotenv
from openai import OpenAI

os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
import gradio as gr

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL", "https://api.openai.com/v1")
)

# Configuration
DEFAULT_MODEL = os.getenv("MODEL_ID", "gpt-4o")
DEFAULT_CONTEXT_SIZE = int(os.getenv("DEFAULT_CONTEXT_SIZE", "10"))
DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a helpful AI assistant.")


class ChatState:
    def __init__(self):
        self.context_size = DEFAULT_CONTEXT_SIZE
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.history = []
        self.show_thinking = True


# Global state
chat_state = ChatState()


def reset_chat() -> None:
    """Reset the chat history."""
    chat_state.history = []
    return None


def update_context_size(size: float | int) -> str:
    """Update the context size for conversation memory."""
    try:
        size = int(size)
        if size < 1:
            size = 1
        chat_state.context_size = size
        return f"上下文记忆已设置为 {size} 轮对话"
    except ValueError:
        return "请输入有效的数字"


def update_system_prompt(prompt: str) -> str:
    """Update the system prompt."""
    if not prompt or prompt.strip() == "":
        prompt = DEFAULT_SYSTEM_PROMPT
    chat_state.system_prompt = prompt.strip()
    return f"系统提示词已更新"


def chat_response(message: str, history: list[dict] | None) -> Generator[str, None, None]:
    """Generate chat response using OpenAI API with streaming."""
    # Build conversation history
    messages = []

    # Add system message
    messages.append({
        "role": "system",
        "content": chat_state.system_prompt
    })

    # Add conversation history with context limit
    # Gradio 6.0 uses dict format with 'role' and 'content' keys
    recent_history = history[-chat_state.context_size:] if history else []
    for msg in recent_history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            # Skip empty content or None
            if not content or content is None:
                continue
            # Extract content after thinking tags for assistant messages
            if role == "assistant":
                # Handle new format: >> ## 完整回复
                if ">> ## 完整回复" in content:
                    content = content.split(">> ## 完整回复")[-1].strip()
                # Handle old format: </details>
                elif "<details>" in content:
                    content = content.split("</details>")[-1].strip()
            messages.append({"role": role, "content": content})

    # Add current message
    messages.append({"role": "user", "content": message})

    try:
        # Call OpenAI API with streaming
        stream = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            stream=True,
            extra_body={
            "thinking": {
                "type": "enabled",
            },
          }
        )

        # Stream the response
        full_response = ""
        thinking_content = ""
        thinking_started = False
        thinking_ended = False

        for chunk in stream:
            delta = chunk.choices[0].delta
            # logger.info(f"Received chunk: {delta}")

            # Check for reasoning content (try multiple possible field names)
            reasoning_content = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
            if reasoning_content and chat_state.show_thinking:
                if not thinking_started:
                    # Start thinking section
                    thinking_started = True
                    yield ">> ## 思考过程\n\n"
                thinking_content += reasoning_content
                yield reasoning_content

            # Check for regular content
            if delta.content:
                content = delta.content
                if thinking_started and not thinking_ended and chat_state.show_thinking:
                    # End thinking section and start response
                    thinking_ended = True
                    yield "\n\n>> ## 完整回复\n\n"
                full_response += content
                yield content

        # Close thinking section if it was started but never ended
        if thinking_started and not thinking_ended:
            yield "\n\n--- 正式回复 ---\n\n"

        # Build the assistant message with reasoning if present
        if thinking_content and chat_state.show_thinking:
            assistant_message = f">> ## 思考过程\n\n{thinking_content}\n\n>> ## 完整回复\n\n{full_response}"
        else:
            assistant_message = full_response

        # Update state history
        chat_state.history.append((message, assistant_message))

        # Log API response
        # logger.info(f"API响应: {assistant_message[:200]}..." if len(assistant_message) > 200 else assistant_message)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"发生错误: {str(e)}")
        yield error_msg


# Create Gradio interface
with gr.Blocks(title="AI Chatbot") as demo:
    gr.Markdown("# 🤖 AI 聊天机器人")

    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
                label="对话",
                height=600
            )
            msg = gr.Textbox(
                label="输入消息",
                placeholder="请输入您的问题...",
                lines=2
            )
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                clear = gr.Button("清除对话")

        with gr.Column(scale=1):
            # Settings panel
            gr.Markdown("### ⚙️ 设置")

            # System prompt setting
            system_prompt = gr.Textbox(
                label="系统提示词",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=3,
                placeholder="自定义系统提示词...",
                info="定义机器人的角色和行为"
            )
            update_prompt_btn = gr.Button("更新提示词", size="sm")

            context_size = gr.Number(
                label="对话记忆轮数",
                value=DEFAULT_CONTEXT_SIZE,
                minimum=1,
                maximum=50,
                step=1,
                info="机器人能记住的最近对话轮数"
            )
            update_context_btn = gr.Button("更新记忆设置", size="sm")

            show_thinking = gr.Checkbox(
                label="显示思考过程",
                value=True,
                info="是否显示AI的思考内容"
            )

            # Info panel
            gr.Markdown("### ℹ️ 信息")
            gr.Textbox(
                label="模型",
                value=DEFAULT_MODEL,
                interactive=False
            )
            gr.Textbox(
                label="API Base URL",
                value=os.getenv("BASE_URL", "https://api.openai.com/v1"),
                interactive=False
            )

    # Event handlers
    def submit_message(message: str, history: list[dict] | None) -> Generator[tuple[list[dict], str], None, None]:
        if not message:
            return None, ""
        # Log user input
        logger.info(f"用户输入: {message}")
        # Gradio 6.0 expects list of messages
        if history is None:
            history = []

        # Add user message to history immediately
        history.append({"role": "user", "content": message})

        # Show user message with loading message in chatbot
        yield history, ""

        # Add loading message in chatbot
        history.append({"role": "assistant", "content": "⏳ 正在思考..."})
        yield history, ""

        # Stream the response
        response_text = ""
        for chunk in chat_response(message, history[:-2]):
            response_text += chunk
            # Update the assistant message with streaming content
            yield history[:-1] + [{"role": "assistant", "content": response_text}], ""

    submit.click(
        submit_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    msg.submit(
        submit_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    clear.click(
        reset_chat,
        outputs=[chatbot]
    )

    def update_settings(size: float | int) -> tuple[str, str]:
        status = update_context_size(size)
        return status, f"当前记忆: {chat_state.context_size} 轮"

    def update_prompt_settings(prompt: str) -> str:
        status = update_system_prompt(prompt)
        return status

    def toggle_thinking(show: bool) -> None:
        chat_state.show_thinking = show

    update_context_btn.click(
        update_settings,
        inputs=[context_size]
    )

    update_prompt_btn.click(
        update_prompt_settings,
        inputs=[system_prompt]
    )

    show_thinking.change(
        toggle_thinking,
        inputs=[show_thinking]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
