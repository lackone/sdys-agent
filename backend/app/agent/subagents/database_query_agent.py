from app.agent.prompts import sub_agents_content
from app.tools.db_tools import execute_sql_query, get_table_data, list_sql_tables

# 数据库助手必须按“列出表 -> 预览表数据 -> 执行 SQL”的顺序获取真实上下文
# tools 列表中的三个工具共同约束了这个查询链路
database_query_agent = {
    "name": sub_agents_content["db"]["name"],
    "description": sub_agents_content["db"]["description"],
    "system_prompt": sub_agents_content["db"]["system_prompt"],
    "tools": [list_sql_tables, get_table_data, execute_sql_query],
}