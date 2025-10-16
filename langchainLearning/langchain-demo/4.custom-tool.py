import os
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city} 的天气：晴天，温度 25°C，湿度 60%"

@tool
def get_stock_price(symbol: str):
    """获取股票价格"""
    return {
        "symbol": symbol,
        "price": 150.25,
        "change": "+2.5%"
    }

llm = ChatOpenAI(
    model="glm-4.6",
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

tools = [
    get_weather,
    get_stock_price,
]

# 从LangChain Hub上拉取一个标准的ReAct模板
# 该模板帮助代理遵循“思考 → 行动 → 观察 → 再思考”的逻辑链
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# AgentExecutor -> Agent（由 create_react_agent 创建） ->  LLM + Tools
result = agent_executor.invoke({"input": "北京今天天气怎么样？然后帮我查询股票价格，股票代码是 000001"})
print(result['output'])

'''
> Entering new AgentExecutor chain...

用户问了两个问题，我需要先查询北京的天气，然后查询股票代码为000001的股票价格。
Action: get_weather
Action Input: 北京
Observation北京
Observation 的天气：晴天，温度 25°C，湿度 60%
I have the weather information for Beijing. Now I need to get the stock price for symbol 000001.
Action: get_stock_price
Action Input: 000001
Observation{'symbol': '000001\nObservation', 'price': 150.25, 'change': '+2.5%'}
Thought: I now know the final answer
Final Answer: 根据查询结果：
1. 北京今天的天气：晴天，温度 25°C，湿度 60%
2. 股票代码000001的价格为150.25元，涨幅为+2.5%

> Finished chain.
根据查询结果：
1. 北京今天的天气：晴天，温度 25°C，湿度 60%
2. 股票代码000001的价格为150.25元，涨幅为+2.5%
'''
