import time
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware import SummarizationMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware, wrap_tool_call
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from examples.common import llm


# 模型调用前后	限制调用次数、切换模型、重试、记录成本	重点讲
# 工具调用前后	记录日志、权限校验、改写参数、包装结果	重点讲
# 上下文送入模型之前	摘要压缩、敏感信息处理	讲核心思路
# 最终回复生成前后	统一格式、收尾处理	只做提示
# 子智能体自己的链路中	单独设置限制、日志或权限规则	讲配置方式

# 上下文太长	SummarizationMiddleware	自动摘要旧消息，保留关键上下文
# 调用次数失控	ModelCallLimitMiddleware、ToolCallLimitMiddleware	限制模型或工具调用次数，防止循环和成本失控
# 高风险动作审批	HumanInTheLoopMiddleware	在发邮件、删库、写文件等动作前暂停，等待人工确认
# 隐私与敏感信息	PIIMiddleware	对邮箱、手机号、信用卡等信息做脱敏、遮蔽或阻断
# 失败恢复	ModelRetryMiddleware、ModelFallbackMiddleware、ToolRetryMiddleware	模型或外部工具失败时重试，必要时切换备用模型
# 工具太多难选择	LLMToolSelectorMiddleware	工具列表很长时，先筛出当前任务相关工具
# 任务规划辅助	TodoListMiddleware	给 Agent 增加待办清单能力，帮助处理多步骤任务

@tool
def delete_database(table_name: str):
    """
    删除数据库表

    这是高风险动作，所以会在 interrupt_on 中配置为执行前中断
    示例只返回模拟结果，不会真的删除数据库表
    """
    print(f"调用 delete_database 工具，准备删除 {table_name} 表")
    return f"已删除表：{table_name}"


@tool
def delete_file(file_name: str):
    """
    删除文件

    这是高风险动作，所以会在 interrupt_on 中配置为执行前中断
    示例只返回模拟结果，不会真的删除本地文件
    """
    print(f"调用 delete_file 工具，准备删除 {file_name} 文件")
    return f"已删除文件：{file_name}"


@tool
def select_database(table_name: str):
    """
    查询数据库表数据

    查询动作属于低风险读操作，本示例不会对它做人工审批
    """
    print(f"调用 select_database 工具，查询 {table_name} 表数据")
    return f"已查询表 {table_name} 的数据"


checkpointer = InMemorySaver()
thread_config = {"configurable": {"thread_id": "erdaye"}}

main_agent = create_deep_agent(
    model=llm,
    tools=[delete_database, delete_file, select_database],
    checkpointer=checkpointer,
    system_prompt="回答使用中文，调用对应的工具实现对应的功能！",
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=1,  # 同一个 thread_id 下累计最多调用 1 次模型
            run_limit=1,  # 当前这次 invoke 内最多调用 1 次模型
            exit_behavior="error",  # 超限后抛出异常，便于后端统一捕获处理
        )
    ],
    # 本章重点是 middleware 调用限制，因此这里关闭人工审批拦截
    interrupt_on={
        "delete_database": False,
        "delete_file": False,
        "select_database": False,
    },
)


# # 这个请求通常需要多次“模型规划 -> 工具调用 -> 模型整理结果”
# # run_limit=1 会让示例更容易触发模型调用上限
# result_1 = main_agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "先查询 product 表的数据，再删除 user 表，最后删除 zhaoweifeng.txt 文件",
#             }
#         ]
#     },
#     config=thread_config,
# )
#
# print(f"最终结果：{result_1['messages'][-1].content}")


# 上下文摘要：SummarizationMiddleware

# summary_middleware = SummarizationMiddleware(
#     model=llm,
#     trigger=("tokens", 4000),  # 消息累计到约 4000 token 时触发摘要
#     keep=("messages", 20),  # 摘要后保留最近 20 条原始消息
# )
#
# main_agent = create_deep_agent(
#     model=llm,
#     tools=[...],
#     middleware=[summary_middleware],
# )


# 工具调用限制：ToolCallLimitMiddleware

# global_tool_limit = ToolCallLimitMiddleware(
#     thread_limit=10,  # 同一条会话线程里，所有工具累计最多调用 10 次
#     run_limit=5,  # 当前这次 invoke 内，所有工具最多调用 5 次
# )
#
# search_tool_limit = ToolCallLimitMiddleware(
#     tool_name="search_web",  # 只限制 search_web 这个工具
#     run_limit=3,  # 当前这次 invoke 内最多搜索 3 次
#     exit_behavior="error",  # 超限后直接抛异常，交给业务层处理
# )
#
# sql_tool_limit = ToolCallLimitMiddleware(
#     tool_name="execute_sql",  # 只限制 execute_sql 这个工具
#     run_limit=2,  # 当前这次 invoke 内最多执行 2 次 SQL
#     exit_behavior="error",
# )
#
# main_agent = create_deep_agent(
#     model=llm,
#     tools=[search_web, execute_sql],
#     middleware=[
#         global_tool_limit,
#         search_tool_limit,
#         sql_tool_limit,
#     ],
# )


# 自定义工具中间件：@wrap_tool_call
@wrap_tool_call
def log_tool_call(request, handler):
    """
    工具调用中间件：在目标工具执行前后打印调用信息
    """
    print("--------进入了工具中间件----------")
    print(f"request : {request}")  # 本次工具调用请求，包含工具名、工具参数等信息
    print(f"handler : {handler}")  # 真正执行目标工具的调用器

    # 前置增强：这里可以记录日志、做权限校验，或者改写 request 中的工具参数
    # handler(request) 是真正执行目标工具的关键步骤；不调用它，工具就会被中间件拦住
    result = handler(request)

    # 后置增强：这里可以包装返回结果、做敏感信息过滤，或者记录工具耗时
    print("--------退出工具中间件----------")
    print(f"result:{result}")

    return result


@wrap_tool_call
def monitor_tool(request, handler):
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    print(f"准备调用工具: {tool_name}")
    print(f"工具参数: {tool_args}")

    result = handler(request)

    print(f"工具调用完成: {tool_name}")
    return result


@tool
def add_numbers(a: int, b: int):
    """
    计算两个数字的和

    :param a: 第一个数字
    :param b: 第二个数字
    :return: 两个数字相加后的结果
    """
    time.sleep(0.5)  # 模拟工具本身的耗时，便于观察中间件前后日志
    result = a + b
    print(f"[工具执行] {a} + {b} = {result}")
    return result


deep_agent = create_deep_agent(
    model=llm,
    tools=[add_numbers],
    checkpointer=InMemorySaver(),
    middleware=[log_tool_call],
    system_prompt="你是一个计算器助手，使用add_numbers工具完成加法计算，回答仅返回计算结果。",
)

thread_config = {"configurable": {"thread_id": "middleware_test_1"}}

# result = deep_agent.invoke(
#     {"messages": [{"role": "user", "content": "帮我计算 100 + 200 的结果"}]},
#     config=thread_config,
# )
#
# print("\n=== 最终回复 ===")
# print(result["messages"][-1].content)

# DeepAgents 中的 Skill 配置

current_dir = Path(__file__).parent.resolve()

file_backend = FilesystemBackend(
    # Agent 能访问的文件系统根目录
    root_dir=current_dir,
    # 开启虚拟沙箱，限制 Agent 只能在这个根目录下访问文件
    virtual_mode=True,
)

main_agent = create_deep_agent(
    model=llm,
    backend=file_backend,
    skills=[
        "skills",
    ],
    system_prompt="你是一个智能助手，可以使用 SKILL 技能",
)

query = "我早上起床晚了，赶公交车差点摔倒，还好最后到了公司。请你只用表情翻译技能。"

# DeepAgents 沿用 messages 输入结构，最终回复在 messages 最后一条。
result = main_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
    }
)

print(f"最终输出结果：{result['messages'][-1].content}")
