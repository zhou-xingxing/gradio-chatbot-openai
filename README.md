# AI 聊天机器人

基于 Gradio 6 的 AI 聊天机器人，支持任意兼容 OpenAI 格式的大模型API，具备多模型切换、思考过程显示、对话历史记忆、用户会话隔离等功能。

## 功能特性

- 🤖 **多模型支持**：配置文件中可定义多个模型，页面下拉框实时切换
- 🌐 **模型信息展示**：选择模型时同步显示对应的 API Base URL
- 👤 **用户会话隔离**：每个用户的模型选择、记忆轮数、系统提示词等设置完全独立
- 💭 **思考过程显示**：支持显示 AI 的推理思考过程（仅部分模型支持）
- 📝 **可配置系统提示词**：自定义机器人的角色和行为
- 🔄 **可调整对话记忆轮数**：控制机器人记住的历史对话轮数
- ⚡ **并发支持**：支持 5 个并发对话，带队列管理
- 🛡️ **友好错误提示**：区分认证失败、限流、服务异常等错误类型

## 环境要求

- Python 3.12+
- Docker（可选）

## 本地运行

### 1. 创建并激活虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

#### 方式一：配置文件（推荐）

创建 `config.yaml` 文件：

```yaml
models:
  - id: "gpt-4o"
    api_key: "${API_KEY}"
    base_url: "${BASE_URL:-https://api.openai.com/v1}"
    supports_thinking: false

  - id: "deepseek-chat"
    api_key: "${DEEPSEEK_API_KEY}"
    base_url: "https://api.deepseek.com/v1"
    supports_thinking: true
```

配置文件支持环境变量引用：
- `${VAR}` - 直接引用环境变量
- `${VAR:-default}` - 引用环境变量，不存在时使用默认值

#### 方式二：纯环境变量

如果不存在 `config.yaml`，将自动使用环境变量：

创建 `.env` 文件：

```env
API_KEY=your_api_key_here
MODEL_ID=gpt-4o
BASE_URL=https://api.openai.com/v1
DEFAULT_CONTEXT_SIZE=5
DEFAULT_SYSTEM_PROMPT=You are a helpful AI assistant.
```

### 4. 启动应用

```bash
python app.py
```

访问 http://localhost:7860

---

## Docker 部署

### 1. 构建镜像

```bash
docker build -t gradio-chatbot-openai .
```

### 2. 运行容器

使用配置文件（推荐）：
```bash
# 将本地 config.yaml 挂载到容器
docker run -d \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -p 7860:7860 \
  gradio-chatbot-openai
```

或使用环境变量：
```bash
docker run -d \
  -e API_KEY=your_api_key \
  -e MODEL_ID=gpt-4o \
  -e BASE_URL=https://api.openai.com/v1 \
  -p 7860:7860 \
  gradio-chatbot-openai
```

或使用 env 文件批量传入环境变量：
```bash
# 创建 .env 文件
cat > .env << 'EOF'
API_KEY=your_api_key
MODEL_ID=gpt-4o
BASE_URL=https://api.openai.com/v1
DEFAULT_CONTEXT_SIZE=5
EOF

# 运行容器时加载
docker run -d \
  --env-file .env \
  -p 7860:7860 \
  gradio-chatbot-openai
```

访问 http://localhost:7860

---

## 配置文件说明

### config.yaml 格式

```yaml
models:
  - id: "模型ID"
    api_key: "API密钥或${环境变量名}"
    base_url: "API基础URL"
    supports_thinking: true/false  # 是否支持思考过程

# 注：models 列表中的第一个模型将作为默认模型
```

### 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `API_KEY` | API 密钥（config.yaml不存在时必需） | - |
| `MODEL_ID` | 模型 ID（config.yaml不存在时使用） | `gpt-4o` |
| `BASE_URL` | API 基础 URL（config.yaml不存在时使用） | `https://api.openai.com/v1` |
| `DEFAULT_CONTEXT_SIZE` | 默认对话记忆轮数（config.yaml不存在时使用） | `5` |
| `DEFAULT_SYSTEM_PROMPT` | 默认系统提示词（config.yaml不存在时使用） | `You are a helpful AI assistant.` |

---

## 项目结构

```
.
├── app.py              # 主程序
├── config.yaml         # 模型配置文件（示例）
├── Dockerfile          # Docker 镜像构建文件
├── requirements.txt    # Python 依赖
├── .env                # 环境变量配置（需自行创建，可选）
├── CLAUDE.md           # 项目开发规范
└── README.md           # 本文档
```

---

## License

MIT
