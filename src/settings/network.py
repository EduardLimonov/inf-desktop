from pydantic import BaseModel


class NetworkSettings(BaseModel):
    STATUS_OK = 200
    PORT = 8000
    LOCAL_URL = f"http://127.0.0.1:{PORT}/"
    LOCAL_SERVER_START = f"uvicorn src.server:app --port {PORT}"
    CONN_CHECK_TIMEOUT = 4
    MAX_SERVER_AWAIT_SEC = 5


network_settings = NetworkSettings()
