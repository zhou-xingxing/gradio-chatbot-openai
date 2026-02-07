import os
import sys
import logging
from typing import Generator, TypedDict
from dotenv import load_dotenv
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
import yaml

os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
import gradio as gr


# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Type definitions
class UserState(TypedDict):
    """User session state structure."""
    model_id: str
    context_size: int
    system_prompt: str
    enable_thinking: bool


# OpenAI client cache to avoid recreating clients
_client_cache: dict[str, OpenAI] = {}


def get_or_create_openai_client(model_id: str) -> OpenAI:
    """Create or get cached OpenAI client for the specified model."""
    if model_id not in _client_cache:
        model_config = get_model_config(model_id)
        _client_cache[model_id] = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['base_url']
        )
    return _client_cache[model_id]


def load_config() -> dict:
    """Load configuration from config.yaml or environment variables."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    if os.path.exists(config_path):
        # Use config.yaml only
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # Fallback to environment variables
        config = {
            'models': [{
                'id': os.getenv('MODEL_ID', 'gpt-4o'),
                'api_key': os.getenv('API_KEY', ''),
                'base_url': os.getenv('BASE_URL', 'https://api.openai.com/v1'),
                'supports_thinking': False
            }]
        }

    # Set default values if not present in config
    if not config.get('context_size'):
        config['context_size'] = 5
    if not config.get('system_prompt'):
        config['system_prompt'] = 'You are a helpful AI assistant.'

    # Validate configuration
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Validate configuration and exit if invalid."""
    if not config.get('models'):
        logger.error("配置错误: models 列表为空或未定义")
        sys.exit(1)

    for model in config['models']:
        if 'id' not in model:
            logger.error("配置错误: 模型缺少必需的 'id' 字段")
            sys.exit(1)
        if 'api_key' not in model:
            logger.error(f"配置错误: 模型 '{model.get('id', 'unknown')}' 缺少 'api_key'")
            sys.exit(1)
        if 'base_url' not in model:
            logger.error(f"配置错误: 模型 '{model.get('id', 'unknown')}' 缺少 'base_url'")
            sys.exit(1)

    # Set default_model_id to first model if not specified
    model_ids = [m['id'] for m in config['models']]
    config.setdefault('default_model_id', model_ids[0])

# Load environment variables
load_dotenv()

# Load configuration
CONFIG = load_config()

# Build model config mapping and choices list
MODEL_CONFIG_MAP = {model['id']: model for model in CONFIG['models']}
MODEL_CHOICES = list(MODEL_CONFIG_MAP.keys())  # Dropdown choices are model IDs


def get_model_config(model_id: str) -> dict:
    """Get configuration for a specific model."""
    return MODEL_CONFIG_MAP.get(model_id, CONFIG['models'][0])


def create_user_state(enable_thinking: bool = True) -> UserState:
    """Create a new user-specific state."""
    return {
        "model_id": CONFIG['default_model_id'],
        "context_size": CONFIG['context_size'],
        "system_prompt": CONFIG['system_prompt'],
        "enable_thinking": enable_thinking
    }


def update_model(model_id: str, state: UserState) -> UserState:
    """Update the selected model."""
    state["model_id"] = model_id

    # Update thinking capability based on model
    model_config = get_model_config(model_id)
    if not model_config.get('supports_thinking', False):
        state["enable_thinking"] = False

    return state


def update_context_size(size: float | int, state: UserState) -> UserState:
    """Update the context size for conversation memory."""
    try:
        size = int(size)
        if size < 1:
            size = 1
        state["context_size"] = size
    except ValueError:
        pass
    return state


def update_system_prompt(prompt: str, state: UserState) -> UserState:
    """Update the system prompt."""
    if not prompt or prompt.strip() == "":
        prompt = CONFIG['system_prompt']
    state["system_prompt"] = prompt.strip()
    return state


def chat_response(message: str, history: list[dict] | None, state: UserState) -> Generator[str, None, None]:
    """Generate chat response using OpenAI API with streaming."""
    model_id = state.get("model_id", CONFIG['default_model_id'])
    model_config = get_model_config(model_id)

    # Build conversation history
    messages = []

    # Add system message
    messages.append({
        "role": "system",
        "content": state["system_prompt"]
    })

    # Add conversation history with context limit
    # Multiply by 2 because each round includes user message + assistant response
    recent_history = history[-state["context_size"]*2:] if history else []
    for msg in recent_history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle Gradio 6.0 format: content is a list of dicts
            if isinstance(content, list):
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                content = text_content

            # Skip empty content
            if not content or not content.strip():
                continue

            # Extract content after thinking tags for assistant messages
            if role == "assistant":
                if ">> ## 完整回复" in content:
                    content = content.split(">> ## 完整回复")[-1].strip()
                elif "<details>" in content:
                    content = content.split("</details>")[-1].strip()

            messages.append({"role": role, "content": content})

    # Add current message
    messages.append({"role": "user", "content": message})

    try:
        # Create client for this model
        client = get_or_create_openai_client(model_id)

        # Build API request parameters
        api_params = {
            "model": model_id,
            "messages": messages,
            "stream": True,
        }

        # Only include thinking parameter if enabled and model supports it
        if state["enable_thinking"] and model_config.get('supports_thinking', False):
            api_params["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                },
                "enable_thinking": True
            }

        # Call OpenAI API with streaming
        stream = client.chat.completions.create(**api_params)

        # Stream the response
        thinking_started = False
        thinking_ended = False

        for chunk in stream:
            delta = chunk.choices[0].delta

            # Check for reasoning content
            reasoning_content = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
            if reasoning_content and state["enable_thinking"] and model_config.get('supports_thinking', False):
                if not thinking_started:
                    thinking_started = True
                    yield ">> ## 思考过程\n\n"
                yield reasoning_content

            # Check for regular content
            if delta.content:
                content = delta.content
                if thinking_started and not thinking_ended and state["enable_thinking"] and model_config.get('supports_thinking', False):
                    thinking_ended = True
                    yield "\n\n>> ## 完整回复\n\n"
                yield content

    except AuthenticationError as e:
        logger.error(f"API 认证失败: {str(e)}")
        yield "错误: API 密钥无效或已过期，请检查配置"
    except RateLimitError as e:
        logger.error(f"API 限流: {str(e)}")
        yield "错误: API 请求过于频繁，请稍后再试"
    except APIError as e:
        logger.error(f"API 错误: {str(e)}")
        yield f"错误: 模型服务异常 ({e.code if hasattr(e, 'code') else 'unknown'})"
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        yield f"错误: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="AI Chatbot") as demo:
    gr.Markdown("# 🤖 AI 聊天机器人")

    # Get default model's thinking support for initial state
    default_supports_thinking = MODEL_CONFIG_MAP[CONFIG['default_model_id']].get('supports_thinking', False)

    # User-specific state (isolated per session)
    user_state = gr.State(create_user_state(enable_thinking=default_supports_thinking))

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

            # Model selector
            model_dropdown = gr.Dropdown(
                label="选择模型",
                choices=MODEL_CHOICES,
                value=CONFIG['default_model_id'],
                info="选择要使用的AI模型"
            )

            # URL display (read-only, synced with model selection)
            model_url_display = gr.Textbox(
                label="API Base URL",
                value=MODEL_CONFIG_MAP[CONFIG['default_model_id']]['base_url'],
                interactive=False,
                info="模型对应的API地址"
            )

            # System prompt setting
            system_prompt = gr.Textbox(
                label="系统提示词",
                value=CONFIG['system_prompt'],
                lines=3,
                placeholder="自定义系统提示词...",
                info="定义机器人的角色和行为"
            )
            update_prompt_btn = gr.Button("更新提示词", size="sm")

            context_size = gr.Number(
                label="对话记忆轮数",
                value=CONFIG['context_size'],
                minimum=1,
                maximum=50,
                step=1,
                info="机器人能记住的最近对话轮数"
            )
            update_context_btn = gr.Button("更新记忆设置", size="sm")

            # Get default model's thinking support for initial state
            default_supports_thinking = MODEL_CONFIG_MAP[CONFIG['default_model_id']].get('supports_thinking', False)
            show_thinking = gr.Checkbox(
                label="启用思考能力",
                value=default_supports_thinking,
                interactive=default_supports_thinking,
                info="是否启用AI的思考功能（仅部分模型支持）"
            )

    # Event handlers
    def submit_message(message: str, history: list[dict] | None, state: UserState) -> Generator[tuple[list[dict], str], None, None]:
        if not message:
            yield history, ""
            return

        if history is None:
            history = []

        # Add user message to history immediately
        history.append({"role": "user", "content": message})
        yield history, ""

        # Add loading message in chatbot
        history.append({"role": "assistant", "content": "⏳ 正在推理..."})
        yield history, ""

        # Stream the response
        response_text = ""
        for chunk in chat_response(message, history[:-2], state):
            response_text += chunk
            yield history[:-1] + [{"role": "assistant", "content": response_text}], ""

    submit.click(
        submit_message,
        inputs=[msg, chatbot, user_state],
        outputs=[chatbot, msg]
    )

    clear.click(
        lambda: None,
        outputs=[chatbot]
    )

    # Model selection handler
    def on_model_change(model_id: str, state: UserState) -> tuple[UserState, str, dict]:
        state = update_model(model_id, state)
        model_config = get_model_config(model_id)

        # Get URL for the selected model
        url = model_config.get('base_url', '')

        # Update thinking checkbox based on model support
        supports_thinking = model_config.get('supports_thinking', False)
        if not supports_thinking:
            state["enable_thinking"] = False

        # 如果模型支持思考，保持用户当前的选择；不支持则强制设为 False
        checkbox_value = state["enable_thinking"] if supports_thinking else False
        # 返回更新后的状态、URL、以及 checkbox 的更新配置（值和是否可交互）
        return state, url, gr.update(value=checkbox_value, interactive=supports_thinking)

    model_dropdown.change(
        on_model_change,
        inputs=[model_dropdown, user_state],
        outputs=[user_state, model_url_display, show_thinking]
    )

    update_context_btn.click(
        update_context_size,
        inputs=[context_size, user_state],
        outputs=[user_state]
    )

    update_prompt_btn.click(
        update_system_prompt,
        inputs=[system_prompt, user_state],
        outputs=[user_state]
    )

    def toggle_thinking_enable(show: bool, state: UserState) -> UserState:
        state["enable_thinking"] = show
        return state

    show_thinking.change(
        toggle_thinking_enable,
        inputs=[show_thinking, user_state],
        outputs=[user_state]
    )


if __name__ == "__main__":
    logger.warning("服务器正在启动...")
    demo.queue(
        default_concurrency_limit=5,   # 支持5个并发对话
        max_size=100                   # 队列最大长度
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
    logger.warning("服务器已关闭")
