# AI 聊天机器人

基于 Gradio 6 的 AI 聊天机器人，支持任意兼容 OpenAI 格式的大模型API，具备思考过程显示、对话历史记忆等功能。

## 功能特性

- 🤖 支持任意 OpenAI 格式 API
- 💭 可启用/禁用 AI 思考过程
- 📝 可配置系统提示词
- 🔄 可调整对话记忆轮数
- 🌍 多语言支持

## 环境要求

- Python 3.12+
- Docker（可选）

## 本地运行

### 1. 创建并激活虚拟环境

首先使用 Python 3.12 创建虚拟环境：

```bash
# 创建虚拟环境
python3.12 -m venv venv

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

### 3. 配置环境变量

创建 `.env` 文件：

```env
API_KEY=your_api_key_here
MODEL_ID=gpt-4o
BASE_URL=https://api.openai.com/v1
DEFAULT_CONTEXT_SIZE=10
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

### 2. 配置环境变量

创建 `.env` 文件：

```env
API_KEY=your_api_key_here
MODEL_ID=gpt-4o
BASE_URL=https://api.openai.com/v1
DEFAULT_CONTEXT_SIZE=10
DEFAULT_SYSTEM_PROMPT=You are a helpful AI assistant.
```

### 3. 运行容器

使用 `.env` 文件：
```bash
docker run -d --env-file .env -p 7860:7860 gradio-chatbot-openai
```

或直接指定环境变量：
```bash
docker run -d \
  -e API_KEY=your_api_key \
  -e MODEL_ID=gpt-4o \
  -e BASE_URL=https://api.openai.com/v1 \
  -p 7860:7860 \
  gradio-chatbot-openai
```

访问 http://localhost:7860

---

## 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `API_KEY` | API 密钥 | 必填 |
| `MODEL_ID` | 模型 ID | `gpt-4o` |
| `BASE_URL` | API 基础 URL | `https://api.openai.com/v1` |
| `DEFAULT_CONTEXT_SIZE` | 对话记忆轮数 | `10` |
| `DEFAULT_SYSTEM_PROMPT` | 系统提示词 | `You are a helpful AI assistant.` |

---

## 项目结构

```
.
├── app.py              # 主程序
├── Dockerfile           # Docker 镜像构建文件
├── requirements.txt     # Python 依赖
├── .env              # 环境变量配置（需自行创建）
└── README.md          # 本文档
```

---

## License

MIT
