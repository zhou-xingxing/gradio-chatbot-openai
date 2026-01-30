## 项目目标
- 创建一个web版AI大模型聊天机器人，不需要自己实现大模型API，只需要调用传入的API即可
- 支持通过环境变量传入自定义的API_KEY, MODEL_ID, BASE_URL
- 编写Dockerfile，因为最终要能打包成Docker镜像
- 在页面中允许用户修改聊天机器人能够记住的对话论数
- 如果调用的大模型API返回内容包含思考过程，要能够显示思考内容

## 技术栈要求
- 使用uv进行Python包管理
- 使用Python 3.12
- 使用Gradio6创建聊天机器人
- 使用openai的SDK


