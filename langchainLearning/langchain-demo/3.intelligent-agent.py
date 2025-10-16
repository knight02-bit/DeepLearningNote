import os

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

llm = ChatOpenAI(
    model="glm-4.6",
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# Tavily是一个网络搜索工具，用于在代理执行中进行联网查询
# https://app.tavily.com/home
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# 创建工具
# 这里定义了代理可调用的工具集合，目前仅包含TavilySearchResults搜索工具
tools = [TavilySearchResults(max_results=2)] # 限制搜索结果数量为2条

# 获取提示词模板
# LangChain Hub是一个集中存放各种提示词模板（prompt templates）的仓库
# "hwchase17/react" 是一个经典的ReAct（Reason+Act）提示模板
prompt = hub.pull("hwchase17/react")

# 创建ReAct代理
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True # 启用详细日志输出，便于调试代理的推理与工具使用过程
)

# 执行任务
result = agent_executor.invoke({"input": "什么是langchain"})
print(result['output'])