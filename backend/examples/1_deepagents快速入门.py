import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

llm = init_chat_model(
    model=os.getenv("LLM_MODEL"),
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    model_provider="openai",
)

tavily_key = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=tavily_key)


@tool
def search(
        query: str,
        max_results: int = 5,
        topic: Literal["news", "finance", "general"] = "general",
        include_raw_content: bool = False,
):
    """搜索网络信息"""
    print(
        f"开始调用网络搜索工具，核心参数为：{query},{max_results},{topic},{include_raw_content}"
    )
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


deep_agent = create_deep_agent(
    model=llm,
    tools=[search],
    subagents=[],
    system_prompt="""
    你是一名严谨的研究员，可以使用 search 工具检索网络信息。
    请根据检索结果进行归纳、分析和交叉验证，生成一份结构清晰、信息可靠的中文报告。
    """,
)

# resp = deep_agent.invoke({"messages": [
#     {"role": "user", "content": "请查询人工智能和机器人领域的热门新闻信息，并整理为一份简要报告。"},
# ]})
#
# print(resp)
# 
# print(resp["messages"][-1].content)

for chunk in deep_agent.stream({"messages": [
    {"role": "user", "content": "请查询人工智能和机器人领域的热门新闻信息，并整理为一份简要报告。"},
]}):
    print(chunk)
