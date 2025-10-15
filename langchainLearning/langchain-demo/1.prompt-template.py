import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 创建智谱LLM示例
llm = ChatOpenAI(
    temperature=0.6,
    model="glm-4.6",
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

#################### 简单对话 ####################
# # 创建消息
# message = [
#     SystemMessage(content="你是一个有用的 AI 助手"),
#     HumanMessage(content="请用一句话介绍一下agent和mcp的设计理念")
# ]
# response = llm(message)

#################### 使用提示模板 ####################
# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{domain}专家"),
    ("human", "请用一句话解释一下{topic}的概念与应用")
])

# 创建链
chain = prompt | llm

# 调用链
response = chain.invoke({
    "domain": "机器学习",
    "topic": "rag"
})

print(response.content)