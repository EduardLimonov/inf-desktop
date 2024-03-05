from pydantic import BaseModel


class NetworkSettings(BaseModel):
    STATUS_OK: int = 200
    PORT: int = 8000
    LOCAL_URL: str = f"http://127.0.0.1:{PORT}/"
    LOCAL_SERVER_START: str = f"uvicorn src.server:app --port {PORT}"
    KILL_LOCAL_SERVER: str = "kill -9 {pid}"
    CONN_CHECK_TIMEOUT: int = 4
    MAX_SERVER_AWAIT_SEC: int = 5
    DEFAULT_LOCAL_NAME: str = "локальное"
    NON_SERVER_CORE_URL: str = "<...>"
    NON_SERVER_CORE_NAME: str = "встроенное"


network_settings = NetworkSettings()
