from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StoreBackend, CompositeBackend
from langgraph.store.memory import InMemoryStore

from examples.common import llm

# 短期记忆	一次调用的执行过程、节点状态、中断位置	人机协作中断恢复、流程续跑	checkpointer
# 长期记忆	一次或多次会话产生的结果、文件、用户偏好	跨线程、跨会话共享信息	Backend

# StateBackend	当前运行状态	同一 thread 内可随检查点保留	默认临时文件、中间结果
# FilesystemBackend	本地磁盘	文件长期存在于本机	本地调试、生成报告、查看文件
# StoreBackend	KV Store	取决于 Store 实现	跨线程共享、长期记忆、生产存储
# CompositeBackend	混合路由	按路由决定	不同路径走不同存储、生产项目


# FilesystemBackend 把 DeepAgents 的文件系统操作映射到本地磁盘
# root_dir 是真实存储目录
# virtual_mode=True 表示开启虚拟沙箱，限制 Agent 只能在 root_dir 范围内读写
workspace_dir = Path("./agent_workspace").resolve()
workspace_dir.mkdir(parents=True, exist_ok=True)
file_backend = FilesystemBackend(
    root_dir=workspace_dir,
    virtual_mode=True,
)

main_agent = create_deep_agent(
    model=llm,
    tools=[],
    backend=file_backend,
    system_prompt="""
    你是一个智能助手，可以使用文件工具进行文件读写
    只有在用户明确要求创建或写入文件时，才可以创建文件
    """,
)

# result_1 = main_agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "帮我查询下 Python 语言的介绍",
#             }
#         ]
#     }
# )
#
# result_2 = main_agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "帮我查询下 Java 语言的介绍，并写到 java.txt 文件中",
#             }
#         ]
#     }
# )

# StoreBackend：KV Store 存储
store = InMemoryStore()

# 这里的 backend=StoreBackend 表示文件系统操作走 Store。
# store=store 表示具体存到哪个 Store 实例中。
main_agent = create_deep_agent(
    model=llm,
    backend=StoreBackend(namespace=lambda _rt: ("test",), ),
    store=store,
    system_prompt="""
    你是一个智能助手
    当用户提供重要个人信息时，请保存到 user_profile.txt
    当用户询问个人信息时，请从 user_profile.txt 中读取
    """,
)

config_a = {"configurable": {"thread_id": "thread-a"}}
config_b = {"configurable": {"thread_id": "thread-b"}}

# result_a = main_agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "我是乌萨奇，我今年 16 岁",
#             }
#         ]
#     },
#     config=config_a,
# )
# print('==================================')
# print(result_a["messages"][-1].content)

# result_b = main_agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "我叫什么，我的年龄是多少",
#             }
#         ]
#     },
#     config=config_b,
# )
# print('==================================')
# print(result_b["messages"][-1].content)

print('==================================')
items = store.search(("test",))
for item in items:
    print(f"key = {item.key}")
    print(f"value = {item.value}")

# CompositeBackend：混合存储策略
store = InMemoryStore()

def create_composite_backend():
    """
    创建混合 Backend

    create_deep_agent 会把 runtime 传入这个工厂函数
    StoreBackend 需要 runtime 才能连接到 Agent 配置中的 store
    """
    workspace_dir = Path("./agent_workspace").resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # 默认后端：普通文件写入本地工作目录
    fs_backend = FilesystemBackend(
        root_dir=workspace_dir,
        virtual_mode=True,
    )

    # Store 后端：重要记忆写入 Key-Value Store
    store_backend = StoreBackend(namespace=lambda _rt: ("aaa",), )

    # 路由规则：
    # 普通路径，例如 local.txt，走 default 的 FilesystemBackend
    # /store/ 开头的路径，例如 /store/memory.txt，走 StoreBackend
    return CompositeBackend(
        default=fs_backend,
        routes={
            "/store/": store_backend,
        },
    )


agent = create_deep_agent(
    model=llm,
    store=store,
    backend=create_composite_backend(),
    tools=[],
    system_prompt="""
    你是一个智能助手
    普通文件请直接写入文件名，例如 local.txt
    重要记忆请写入 /store/ 目录，例如 /store/memory.txt
    """,
)

config = {"configurable": {"thread_id": "thread-composite"}}
user_input = (
    "1 创建本地文件 local.txt，内容为 本地文件\n"
    "2 创建记忆文件 /store/memory.txt，内容为 重要记忆"
)
print(f"用户指令：{user_input}")

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_input,
            }
        ]
    },
    config=config,
)

print(f"Agent 回复：{result['messages'][-1].content}")

items = store.search(("aaa",))
for item in items:
    print(item)