from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich import print as rprint

from examples.common import llm


# interrupt_on	哪些工具需要在执行前停下来，交给人工审批
# checkpointer	保存“Agent 暂停在哪里”的短期状态
# thread_id	找回同一次执行线程的 ID，恢复时必须保持一致
# __interrupt__	第一次执行后返回的中断信息
# Command(resume=...)	第二次恢复执行时传入的恢复指令

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


# 配置人机协作的三个关键点
# 用 checkpointer 保存暂停状态；
# 用稳定的 thread_id 找回同一次执行；
# 用 interrupt_on 标记哪些工具需要审批。

checkpointer = InMemorySaver()

thread_config = {
    "configurable": {
        "thread_id": "hitl-approval-demo",
    }
}

main_agent = create_deep_agent(
    model=llm,
    tools=[delete_database, delete_file, select_database],
    checkpointer=checkpointer,
    system_prompt="""
    你是一个负责执行数据库和文件操作的智能助手
    请根据用户需求调用合适的工具，并使用中文回复执行结果
    """,
    # interrupt_on 用工具名配置哪些动作需要人工审批
    # True 表示使用默认审批选项：approve、edit、reject
    # False 表示该工具不需要中断，可以直接执行
    interrupt_on={
        "delete_database": True,
        "delete_file": True,
        "select_database": False,
    },
)

result_1 = main_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "先查询 product 表的数据，再删除 user 表，最后删除 zhaoweifeng.txt 文件",
            }
        ]
    },
    config=thread_config,
)

# 命中人机协作中断时，结果中会出现 __interrupt__
# __interrupt__ 是一个列表，里面保存 Interrupt 对象
interrupts = result_1.get("__interrupt__", [])

print("第一次执行中断信息：")
rprint(interrupts)

if interrupts:
    action_requests = interrupts[0].value["action_requests"]
    print(
        "本次需要审核的工具数量："
        f"{len(action_requests)}，具体拦截的工具："
        f"{[action['name'] for action in action_requests]}"
    )

    decisions = []

    for action in action_requests:
        action_name = action["name"]

        # decisions 的顺序要和 action_requests 的顺序保持一致
        # reject 表示拒绝执行该工具；approve 表示允许继续执行该工具
        if action_name == "delete_database":
            # decisions.append({"type": "reject"})

            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {
                        "name": action_name,
                        "args": {
                            "table_name": "archived_user",
                        },
                    },
                }
            )
        elif action_name == "delete_file":
            decisions.append({"type": "approve"})

    # 这里有三个关键点：
    # 第二次执行不再传用户原始问题，而是传 Command(...)。
    # 第二次执行必须使用第一次相同的 config，尤其是相同的 thread_id。
    # decisions 的数量和顺序要和 action_requests 对应上。

    # 第二次 invoke 不再传用户原始问题，而是传 Command(resume=...)
    # config 必须继续使用第一次相同的 thread_id，Agent 才能找到之前暂停的位置
    result_2 = main_agent.invoke(
        Command(
            resume={
                "decisions": decisions,
            }
        ),
        config=thread_config,
    )

    print(f"最终结果：{result_2['messages'][-1].content}")