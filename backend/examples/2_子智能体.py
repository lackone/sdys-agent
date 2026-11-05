import asyncio

from deepagents import create_deep_agent

from examples.common import llm

weather_agent = {
    "name": "weather_helper",
    "description": "用于查询天气信息。当用户询问天气时，请调用此助手。",
    "system_prompt": """
    你是一个天气查询助手。
    无论用户查询哪个城市，请统一回复：今天天气晴朗，温度25度。
    """,
    "tools": [],
}

math_agent = {
    "name": "math_helper",
    "description": "用于处理数学计算问题。当用户询问加减乘除、数字计算、算数题时，请调用此助手。",
    "system_prompt": """
    你是一个严谨的数学助手。
    请帮助用户完成数学计算，并给出清晰、准确的答案。
    """,
    "tools": [],
}

translate_agent = {
    "name": "translate_helper",
    "description": "用于处理中英互译任务。当用户需要中文和英文之间的翻译时，请调用此助手。",
    "system_prompt": """
    你是一个中英翻译助手。
    如果输入是中文，请翻译成英文；如果输入是英文，请翻译成中文。
    """,
    "tools": [],
}

main_agent = create_deep_agent(
    model=llm,
    tools=[],
    subagents=[weather_agent, math_agent, translate_agent],
    system_prompt="""
    你是一个负责统筹任务的主智能体。
    请根据用户需求选择合适的子智能体完成任务。
    你不直接执行天气查询、数学计算或翻译任务，而是通过子智能体完成。
    """,
)


def test_stream(query):
    """
    同步流式执行一次用户问题，并打印主智能体的调度过程。

    stream() 会持续产出图节点状态，常见节点包括：
    - model：模型做出下一步决策，或生成最终回复；
    - tools：工具执行完成，子智能体的 task 结果也会从这里返回。
    """
    stream = main_agent.stream({"messages": [{"role": "user", "content": query}]})

    for chunk in stream:
        # chunk 是按节点名组织的字典，例如 {"model": {"messages": [...]}}
        for node_name, state in chunk.items():
            # DeepAgents 内部可能产出空状态或非消息状态，这里只解析消息类状态
            if state is None or "messages" not in state:
                continue

            messages = state["messages"]
            if messages and isinstance(messages, list):
                # 每个节点本次产出的最后一条消息，通常就是最值得观察的信息
                last_msg = messages[-1]

                if node_name == "model":
                    # model 节点的 tool_calls 表示模型决定下一步调用工具或子智能体
                    if last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            if tool_call["name"] == "task":
                                # DeepAgents 用内置 task 工具表示“分派给某个子智能体”
                                print(
                                    f"【model】决定调用子智能体{tool_call['args']['subagent_type']}"
                                )
                            else:
                                print(
                                    f"【model】决定调用子工具{tool_call['name']},传入的参数为：{tool_call['args']}"
                                )
                    elif last_msg.content:
                        # 没有 tool_calls 且 content 非空，通常就是主智能体整理后的最终回复
                        print(f"【model】返回最终结果：{last_msg.content}")
                elif node_name == "tools":
                    # tools 节点返回普通工具结果；task 子智能体执行完后也会以工具消息形式返回
                    name = last_msg.name
                    content = last_msg.content
                    print(
                        f"【agent】调用了具体的工具{name},返回结果为：{content[:100] + '...'}"
                    )


# test_stream("北京今天的天气怎么样？")

async def test_stream2(query):
    """
    异步流式执行一次用户问题，并打印主智能体的调度过程。

    astream() 会持续产出图节点状态，常见节点包括：
    - model：模型做出下一步决策，或生成最终回复；
    - tools：工具执行完成，子智能体的 task 结果也会从这里返回。
    """
    stream = main_agent.astream({"messages": [{"role": "user", "content": query}]})

    async for chunk in stream:
        # chunk 是按节点名组织的字典，例如 {"model": {"messages": [...]}}
        for node_name, state in chunk.items():
            # DeepAgents 内部可能产出空状态或非消息状态，这里只解析消息类状态
            if state is None or "messages" not in state:
                continue

            messages = state["messages"]
            if messages and isinstance(messages, list):
                # 每个节点本次产出的最后一条消息，通常就是最值得观察的信息
                last_msg = messages[-1]

                if node_name == "model":
                    # model 节点的 tool_calls 表示模型决定下一步调用工具或子智能体
                    if last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            if tool_call["name"] == "task":
                                # DeepAgents 用内置 task 工具表示“分派给某个子智能体”
                                print(
                                    f"【model】决定调用子智能体{tool_call['args']['subagent_type']}"
                                )
                            else:
                                print(
                                    f"【model】决定调用子工具{tool_call['name']},传入的参数为：{tool_call['args']}"
                                )
                    elif last_msg.content:
                        # 没有 tool_calls 且 content 非空，通常就是主智能体整理后的最终回复
                        print(f"【model】返回最终结果：{last_msg.content}")
                elif node_name == "tools":
                    # tools 节点返回普通工具结果；task 子智能体执行完后也会以工具消息形式返回
                    name = last_msg.name
                    content = last_msg.content
                    print(
                        f"【agent】调用了具体的工具{name},返回结果为：{content[:100] + '...'}"
                    )


async def batch_run():
    task1 = test_stream2("北京今天的天气怎么样？")
    task2 = test_stream2("请将'你是最棒的'翻译成英文。")
    print("------------------")
    await asyncio.gather(task1, task2)


if __name__ == "__main__":
    asyncio.run(batch_run())
