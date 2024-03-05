from pydantic import BaseModel


class NetworkSettings(BaseModel):
    STATUS_OK = 200
    PORT = 8000
    LOCAL_URL = f"http://127.0.0.1:{PORT}/"
    LOCAL_SERVER_START = f"uvicorn src.server:app --port {PORT}"
    KILL_LOCAL_SERVER = "kill -9 {pid}"
    CONN_CHECK_TIMEOUT = 4
    MAX_SERVER_AWAIT_SEC = 5
    DEFAULT_LOCAL_NAME = "локальное"
    NON_SERVER_CORE_URL = "<...>"
    NON_SERVER_CORE_NAME = "встроенное"


network_settings = NetworkSettings()
