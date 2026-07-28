import json
import logging
import os
import sys
from typing import Any, Generator, NoReturn, NotRequired, TypedDict
from dotenv import load_dotenv
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
import yaml
import requests

os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
import gradio as gr


# Configure logging
logging.basicConfig(
    # level=logging.WARNING,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_SIZE = 5
DEFAULT_SYSTEM_PROMPT = 'You are a helpful AI assistant.'
INPUT_HINT_CSS = """
.message-input {
    gap: 0.35rem;
}

.input-header {
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
}

.input-label {
    color: var(--body-text-color);
    font-size: 0.95rem;
    font-weight: 600;
    line-height: 1.3;
}

.input-shortcut-hint {
    color: var(--body-text-color-subdued);
    font-size: 0.85rem;
    line-height: 1.3;
    text-align: right;
}

.input-label p,
.input-shortcut-hint p {
    margin: 0;
}

@media (max-width: 700px) {
    .input-header {
        align-items: flex-start;
        gap: 0.25rem;
    }

    .input-shortcut-hint {
        text-align: left;
    }
}
"""


# Type definitions
class UserState(TypedDict):
    """User session state structure."""
    model_key: str
    context_size: int
    system_prompt: str
    enable_thinking: bool


class ModelConfig(TypedDict):
    """模型配置结构。"""
    id: str
    name: str
    model_key: str
    api_key: str
    base_url: str
    supports_thinking: NotRequired[bool]
    max_model_len: NotRequired[int | str]


class AppConfig(TypedDict):
    """应用配置结构。"""
    models: list[ModelConfig]
    context_size: int
    system_prompt: str
    default_model_key: str


# OpenAI client cache to avoid recreating clients
_client_cache: dict[str, OpenAI] = {}

# Model context length cache (loaded at startup)
_model_context_cache: dict[str, str] = {}


def get_or_create_openai_client(model_key: str) -> OpenAI:
    """Create or get cached OpenAI client for the specified model."""
    if model_key not in _client_cache:
        model_config = get_model_config(model_key)
        _client_cache[model_key] = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['base_url']
        )
    return _client_cache[model_key]


def make_model_key(model_id: str, model_name: str) -> str:
    """将模型ID和名称编码为无拼接歧义的内部唯一标识。"""
    return json.dumps([model_id, model_name], ensure_ascii=False, separators=(',', ':'))


def load_config() -> AppConfig:
    """Load configuration from config.yaml or environment variables."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    if os.path.exists(config_path):
        # Use config.yaml only
        with open(config_path, 'r', encoding='utf-8') as f:
            config: Any = yaml.safe_load(f) or {}
    else:
        # Fallback to environment variables
        model_id = os.getenv('MODEL_ID', 'gpt-4o')
        config = {
            'models': [{
                'id': model_id,
                'name': os.getenv('MODEL_NAME', model_id),
                'api_key': os.getenv('API_KEY', ''),
                'base_url': os.getenv('BASE_URL', 'https://api.openai.com/v1'),
                'supports_thinking': False
            }]
        }

    return validate_config(config)


def exit_config_error(message: str) -> NoReturn:
    """记录配置错误并退出程序。"""
    logger.error(f"配置错误: {message}")
    sys.exit(1)


def validate_config(config: Any) -> AppConfig:
    """校验并归一化配置，返回应用可直接使用的结构。"""
    if not isinstance(config, dict):
        exit_config_error("配置文件顶层必须是字典")

    raw_models = config.get('models')
    if not isinstance(raw_models, list) or not raw_models:
        exit_config_error("models 必须是非空列表")

    models: list[ModelConfig] = []
    model_keys: set[str] = set()
    for index, raw_model in enumerate(raw_models, start=1):
        if not isinstance(raw_model, dict):
            exit_config_error(f"models[{index}] 必须是字典")

        raw_model_id = raw_model.get('id')
        if not isinstance(raw_model_id, str) or not raw_model_id.strip():
            exit_config_error(f"models[{index}] 缺少非空字符串字段 'id'")
        model_id = raw_model_id.strip()

        raw_model_name = raw_model.get('name')
        if not isinstance(raw_model_name, str) or not raw_model_name.strip():
            exit_config_error(f"模型 '{model_id}' 缺少非空字符串字段 'name'")
        model_name = raw_model_name.strip()
        model_key = make_model_key(model_id, model_name)
        if model_key in model_keys:
            exit_config_error(f"模型标识 (id='{model_id}', name='{model_name}') 重复")

        raw_api_key = raw_model.get('api_key')
        if not isinstance(raw_api_key, str) or not raw_api_key.strip():
            exit_config_error(f"模型 '{model_id}' 缺少非空字符串字段 'api_key'")

        raw_base_url = raw_model.get('base_url')
        if not isinstance(raw_base_url, str) or not raw_base_url.strip():
            exit_config_error(f"模型 '{model_id}' 缺少非空字符串字段 'base_url'")

        model_config: ModelConfig = {
            'id': model_id,
            'name': model_name,
            'model_key': model_key,
            'api_key': raw_api_key.strip(),
            'base_url': raw_base_url.strip(),
        }

        raw_supports_thinking = raw_model.get('supports_thinking')
        if raw_supports_thinking is not None:
            if not isinstance(raw_supports_thinking, bool):
                exit_config_error(f"模型 '{model_id}' 的 supports_thinking 必须是布尔值")
            model_config['supports_thinking'] = raw_supports_thinking

        raw_max_model_len = raw_model.get('max_model_len')
        if raw_max_model_len is not None:
            if isinstance(raw_max_model_len, bool) or not isinstance(raw_max_model_len, (int, str)):
                exit_config_error(f"模型 '{model_id}' 的 max_model_len 必须是整数或字符串")
            if isinstance(raw_max_model_len, str):
                raw_max_model_len = raw_max_model_len.strip()
                if not raw_max_model_len:
                    exit_config_error(f"模型 '{model_id}' 的 max_model_len 不能为空字符串")
            model_config['max_model_len'] = raw_max_model_len

        model_keys.add(model_key)
        models.append(model_config)

    raw_context_size = config.get('context_size', DEFAULT_CONTEXT_SIZE)
    if isinstance(raw_context_size, bool):
        exit_config_error("context_size 必须是正整数")
    try:
        context_size = int(raw_context_size)
    except (TypeError, ValueError):
        exit_config_error("context_size 必须是正整数")
    if context_size < 1:
        exit_config_error("context_size 必须大于等于 1")

    raw_system_prompt = config.get('system_prompt', DEFAULT_SYSTEM_PROMPT)
    if not isinstance(raw_system_prompt, str):
        exit_config_error("system_prompt 必须是字符串")
    system_prompt = raw_system_prompt.strip() or DEFAULT_SYSTEM_PROMPT

    raw_default_model_id = config.get('default_model_id')
    raw_default_model_name = config.get('default_model_name')
    if raw_default_model_id is None:
        if raw_default_model_name is not None:
            exit_config_error("配置 default_model_name 时必须同时配置 default_model_id")
        default_model = models[0]
    else:
        if not isinstance(raw_default_model_id, str) or not raw_default_model_id.strip():
            exit_config_error("default_model_id 必须是非空字符串")
        default_model_id = raw_default_model_id.strip()
        matching_models = [model for model in models if model['id'] == default_model_id]
        if not matching_models:
            exit_config_error(f"default_model_id '{default_model_id}' 不在 models 列表中")

        if raw_default_model_name is None:
            if len(matching_models) > 1:
                exit_config_error(
                    f"default_model_id '{default_model_id}' 匹配多个模型，必须配置 default_model_name"
                )
            default_model = matching_models[0]
        else:
            if not isinstance(raw_default_model_name, str) or not raw_default_model_name.strip():
                exit_config_error("default_model_name 必须是非空字符串")
            default_model_name = raw_default_model_name.strip()
            default_model_key = make_model_key(default_model_id, default_model_name)
            matching_default_models = [
                model for model in matching_models if model['model_key'] == default_model_key
            ]
            if not matching_default_models:
                exit_config_error(
                    f"默认模型 (id='{default_model_id}', name='{default_model_name}') 不在 models 列表中"
                )
            default_model = matching_default_models[0]

    return {
        'models': models,
        'context_size': context_size,
        'system_prompt': system_prompt,
        'default_model_key': default_model['model_key'],
    }

# Load environment variables
load_dotenv()

# Load configuration
CONFIG = load_config()

# Build model config mapping and dropdown choices
MODEL_CONFIG_MAP = {model['model_key']: model for model in CONFIG['models']}
MODEL_CHOICES = [(model['name'], model['model_key']) for model in CONFIG['models']]


def get_model_config(model_key: str) -> ModelConfig:
    """Get configuration for a specific model."""
    return MODEL_CONFIG_MAP.get(model_key, CONFIG['models'][0])


def fetch_max_model_len_from_api(model_id: str, model_config: ModelConfig) -> str:
    """Fetch max_model_len from API endpoint. Returns empty string if not available."""
    base_url = model_config.get('base_url', '')
    api_key = model_config.get('api_key', '')

    try:
        models_url = base_url.rstrip('/') + '/models'
        headers = {"Authorization": f"Bearer {api_key}"}
        logger.info(f"尝试从 API 获取模型信息: {models_url}")
        response = requests.get(models_url, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            logger.info(f"API 响应数据: {data}")
            # Look for the specific model in the response
            if 'data' in data and isinstance(data['data'], list):
                for model in data['data']:
                    if model.get('id') == model_id:
                        max_len = model.get('max_model_len')
                        if max_len is not None:
                            return str(max_len)
                        # Try alternative field names
                        max_len = model.get('max_tokens') or model.get('context_length')
                        if max_len is not None:
                            return str(max_len)
    except Exception:
        # Silently fail
        pass

    return ""


def fetch_max_model_len(model_key: str) -> str:
    """Get max_model_len from cache or config. Returns empty string if not available.

    Priority:
    1. Cached value from startup
    2. Config file's max_model_len field
    3. Empty string if neither available
    """
    # First check cache
    if model_key in _model_context_cache:
        return _model_context_cache[model_key]

    # Fall back to config
    model_config = get_model_config(model_key)
    max_len = model_config.get('max_model_len')
    if max_len is not None:
        return str(max_len)

    return ""


def load_all_model_contexts() -> None:
    """Load max_model_len for all models at startup and cache them."""
    global _model_context_cache
    logger.info("正在加载所有模型的上下文长度...")

    for model_key, model_config in MODEL_CONFIG_MAP.items():
        model_id = model_config['id']
        model_name = model_config['name']
        # First try API
        max_len = fetch_max_model_len_from_api(model_id, model_config)

        # If API failed, try config
        if not max_len:
            logger.info(
                f"API 无法获取模型 '{model_name}' ({model_id}) 的上下文长度，"
                "尝试从配置文件获取..."
            )
            config_max_len = model_config.get('max_model_len')
            if config_max_len is not None:
                max_len = str(config_max_len)

        # Store in cache (even if empty, to avoid repeated API calls)
        _model_context_cache[model_key] = max_len
        if max_len:
            logger.info(f"模型 '{model_name}' ({model_id}) 的最大上下文长度: {max_len}")
        else:
            logger.warning(f"模型 '{model_name}' ({model_id}) 无法获取最大上下文长度")

    logger.info("模型上下文长度加载完成")


# Load all model context lengths at startup (after all functions defined)
load_all_model_contexts()


def create_user_state(enable_thinking: bool = True) -> UserState:
    """Create a new user-specific state."""
    return {
        "model_key": CONFIG['default_model_key'],
        "context_size": CONFIG['context_size'],
        "system_prompt": CONFIG['system_prompt'],
        "enable_thinking": enable_thinking
    }


def update_model(model_key: str, state: UserState) -> UserState:
    """Update the selected model."""
    state["model_key"] = model_key

    # Update thinking capability based on model
    model_config = get_model_config(model_key)
    if model_config.get('supports_thinking', False):
        # 如果模型支持思考能力，默认启用
        state["enable_thinking"] = True
    else:
        # 如果模型不支持思考能力，禁用
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
    model_key = state.get("model_key", CONFIG['default_model_key'])
    model_config = get_model_config(model_key)
    model_id = model_config['id']

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
        client = get_or_create_openai_client(model_key)

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
                "enable_thinking": True,
                "reasoning": {
                    "enabled": True,
                }
            }

        # Call OpenAI API with streaming
        stream = client.chat.completions.create(**api_params)

        # Stream the response
        thinking_started = False
        thinking_ended = False

        for chunk in stream:
            if not chunk.choices:
                continue

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
    default_model_config = get_model_config(CONFIG['default_model_key'])
    default_supports_thinking = default_model_config.get('supports_thinking', False)

    # User-specific state (isolated per session)
    user_state = gr.State(create_user_state(enable_thinking=default_supports_thinking))

    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
                label="对话",
                height=600
            )
            with gr.Column(elem_classes="message-input"):
                with gr.Row(equal_height=False, elem_classes="input-header"):
                    gr.Markdown(
                        "输入消息",
                        elem_classes="input-label",
                        container=False
                    )
                    gr.Markdown(
                        "Enter 换行 · Shift+Enter 发送",
                        elem_classes="input-shortcut-hint",
                        container=False
                    )
                msg = gr.Textbox(
                    label="输入消息",
                    placeholder="请输入您的问题...",
                    lines=2,
                    show_label=False,
                    scale=8
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
                value=CONFIG['default_model_key'],
                info="选择要使用的AI模型"
            )

            # 模型ID展示（只读，跟随模型选择更新）
            model_id_display = gr.Textbox(
                label="Model ID",
                value=default_model_config['id'],
                interactive=False,
                info="API请求使用的模型ID"
            )

            # URL display (read-only, synced with model selection)
            model_url_display = gr.Textbox(
                label="API Base URL",
                value=default_model_config['base_url'],
                interactive=False,
                info="模型对应的API地址"
            )

            # Max context length display (read-only, synced with model selection)
            max_context_display = gr.Textbox(
                label="最大上下文长度",
                value=fetch_max_model_len(CONFIG['default_model_key']),
                interactive=False,
                info="模型支持的最大上下文token数(输入+输出)"
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
            show_thinking = gr.Checkbox(
                label="启用思考能力",
                value=default_supports_thinking,
                interactive=default_supports_thinking,
                info="是否启用AI的思考功能（仅部分模型支持）"
            )

    # Event handlers
    def submit_message(message: str, history: list[dict] | None, state: UserState) -> Generator[tuple[list[dict], str], None, None]:
        if history is None:
            history = []

        if not message:
            yield history, ""
            return

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

    msg.submit(
        submit_message,
        inputs=[msg, chatbot, user_state],
        outputs=[chatbot, msg]
    )

    clear.click(
        lambda: None,
        outputs=[chatbot]
    )

    # Model selection handler
    def on_model_change(model_key: str, state: UserState) -> tuple[UserState, str, str, str, dict]:
        state = update_model(model_key, state)
        model_config = get_model_config(model_key)

        # Get URL for the selected model
        url = model_config.get('base_url', '')

        # Fetch max context length from API or config
        max_context_len = fetch_max_model_len(model_key)

        # Update thinking checkbox based on model support
        supports_thinking = model_config.get('supports_thinking', False)

        # 返回更新后的状态、模型ID、URL、最大上下文长度和 checkbox 配置
        # update_model 已经根据模型支持情况设置了 state["enable_thinking"]
        return state, model_config['id'], url, max_context_len, gr.update(
            value=state["enable_thinking"],
            interactive=supports_thinking,
        )

    model_dropdown.change(
        on_model_change,
        inputs=[model_dropdown, user_state],
        outputs=[user_state, model_id_display, model_url_display, max_context_display, show_thinking]
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
        default_concurrency_limit=10,   # 支持10个并发对话
        max_size=100                   # 队列最大长度
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=INPUT_HINT_CSS,
        share=False
    )
    logger.warning("服务器已关闭")
