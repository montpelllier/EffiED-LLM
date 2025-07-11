import re

from pydantic import BaseModel


class ErrorLabel(BaseModel):
    index: int
    is_error: bool


class LabelList(BaseModel):
    labels: list[ErrorLabel]


class ColumnMetadata(BaseModel):
    name: str
    description: str
    schema_org: str
    data_type: str
    related_cols: list[str]


class ColumnMetadataList(BaseModel):
    columns: list[ColumnMetadata]



def extract_label_list_json(response_content):
    """
    从模型响应中提取符合 LabelList 格式的 JSON 数据

    参数:
        response_content: 模型返回的响应文本

    返回:
        解析后的 JSON 对象或 None（如果提取失败）
    """
    # 先尝试匹配代码块中的 JSON

    json_pattern = r'(\{\s*"labels"\s*:\s*\[[\s\S]*?\]\s*\})'
    match = re.search(json_pattern, response_content)
    if match:
        json_str = match.group(1).strip()
    else:
        print("⚠️ No valid JSON found in response, using raw content")
        json_str = response_content.strip()

    return json_str
