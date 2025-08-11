from pydantic import BaseModel


class ErrorLabel(BaseModel):
    row_id: int
    is_error: bool


class LabelList(BaseModel):
    labels: list[ErrorLabel]


class ColumnMetadata(BaseModel):
    name: str
    meaning: str
    data_type: str
    format_rule: str
    null_value_rule: str


class ColumnMetadataList(BaseModel):
    columns: list[ColumnMetadata]
