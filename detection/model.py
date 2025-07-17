from pydantic import BaseModel


class ErrorLabel(BaseModel):
    row_id: int
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
