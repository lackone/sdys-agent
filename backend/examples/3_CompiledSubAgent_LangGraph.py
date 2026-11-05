
from typing import TypedDict, Annotated, Literal

from deepagents import create_deep_agent, CompiledSubAgent
from langchain_core.messages import AIMessage
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph

from examples.common import llm


# LangGraph 子图要接入 DeepAgents，最关键的要求是：State 里必须包含 messages 字段。
class ResearchPlanState(TypedDict):
    # DeepAgents 调用 LangGraph 子图时，会通过 messages 传入任务，也会从 messages 读取结果。
    # add_messages 表示节点返回的新消息会追加到消息链，而不是覆盖历史消息。
    messages: Annotated[list, add_messages]
    topic: str
    depth: Literal["quick", "deep"]
    plan: list[str]


def extract_topic(state: ResearchPlanState):
    """从用户任务中提取研究主题，并判断是否需要深度研究。"""
    task = state["messages"][-1].content
    depth = (
        "deep"
        if any(word in task for word in ["深度", "报告", "系统", "趋势"])
        else "quick"
    )
    topic = (
        task.replace("请", "")
        .replace("帮我", "")
        .replace("生成", "")
        .replace("一份", "")
        .strip("。 ")
    )
    print(f"【LangGraph】提取研究主题：{topic}")
    print(f"【LangGraph】判断研究深度：{depth}")
    return {
        "topic": topic,
        "depth": depth,
        "messages": [AIMessage(content=f"已识别研究主题：{topic}；研究深度：{depth}")],
    }


def route_by_depth(state: ResearchPlanState):
    """条件边：根据研究深度选择普通规划或深度规划。"""
    return state["depth"]


def build_quick_plan(state: ResearchPlanState):
    """普通查询规划：适合简短问答或轻量资料整理。"""
    topic = state["topic"]
    plan = [
        f"围绕「{topic}」搜索 3-5 条公开资料",
        "提取资料中的关键事实和来源",
        "整理成简短结论，并标注仍需补充的信息",
    ]
    print("【LangGraph】进入 quick_plan 节点")
    return {
        "plan": plan,
        "messages": [AIMessage(content="已生成普通研究计划。")],
    }


def build_deep_plan(state: ResearchPlanState):
    """深度研究规划：适合报告、趋势分析、行业调研等长链路任务。"""
    topic = state["topic"]
    plan = [
        f"明确「{topic}」的研究范围、时间窗口和核心问题",
        "搜索公开资料，优先收集权威媒体、机构报告和官方信息",
        "拆分技术、产业、公司案例、风险四个方向分别整理证据",
        "对不同来源的信息做交叉验证，记录冲突和缺口",
        "输出结构化研究报告：背景、现状、趋势、案例、风险、结论",
    ]
    print("【LangGraph】进入 deep_plan 节点")
    return {
        "plan": plan,
        "messages": [AIMessage(content="已生成深度研究计划。")],
    }


def finalize_plan(state: ResearchPlanState):
    """把规划步骤整理成一条 AIMessage，作为 LangGraph 子智能体的返回结果。"""
    plan_text = "\n".join(
        f"{index}. {item}" for index, item in enumerate(state["plan"], 1)
    )
    final = f"研究主题：{state['topic']}\n研究深度：{state['depth']}\n\n执行计划：\n{plan_text}"
    print("【LangGraph】汇总研究计划")
    return {"messages": [AIMessage(content=final)]}


workflow = StateGraph(ResearchPlanState)
workflow.add_node("extract_topic", extract_topic)
workflow.add_node("quick_plan", build_quick_plan)
workflow.add_node("deep_plan", build_deep_plan)
workflow.add_node("finalize_plan", finalize_plan)

workflow.set_entry_point("extract_topic")
workflow.add_conditional_edges(
    "extract_topic",
    route_by_depth,
    {
        "quick": "quick_plan",
        "deep": "deep_plan",
    },
)
workflow.add_edge("quick_plan", "finalize_plan")
workflow.add_edge("deep_plan", "finalize_plan")
workflow.add_edge("finalize_plan", END)

compiled_graph = workflow.compile()

research_planner_graph = CompiledSubAgent(
    name="research_planner_graph",
    description="用于把开放研究问题拆解成可执行的研究计划，适合行业趋势、技术调研、报告规划等任务。",
    runnable=compiled_graph,
)

main_agent = create_deep_agent(
    model=llm,
    tools=[],
    subagents=[research_planner_graph],
    system_prompt="""
    你是深度研搜系统的主智能体。
    当用户需要研究规划、报告大纲、趋势调研步骤时，必须调用 research_planner_graph。
    你不自己编造研究计划，而是根据子智能体返回的计划整理最终回复。
    """,
)

for chunk in main_agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "请帮我生成一份人工智能与机器人行业趋势的深度研究计划。",
            }
        ]
    }
):
    print(chunk)