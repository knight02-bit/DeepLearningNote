import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# 创建 LLM
llm = ChatOpenAI(
    temperature=0.6,
    model="glm-4.6",
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 创建提示词模板
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "你是一个与人类进行对话的友好聊天机器人"
        ),
        MessagesPlaceholder(variable_name="chat-history"),# 一个可以用来传递消息列表的占位符
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat-history",
    return_messages=True
)

# 创建对话链
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# 进行对话
response1 = conversation.invoke({"question": "给我讲一个笑话"})
print("AI: ", response1['text'])

response2 = conversation.invoke({"question": "给我再讲一个笑话"})
print("AI: ", response2['text'])