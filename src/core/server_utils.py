from pydantic import BaseModel


class HTTPBodyDf(BaseModel):
    dataframe_json: str
    method: str = 'Powell'
    target_p: float = 0.9
