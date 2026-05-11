from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from urllib import request


@dataclass(slots=True)
class FeishuSendResult:
    status_code: int
    body: str


class FeishuWebhookNotifier:
    def __init__(self, webhook_url: str, secret: str | None = None, timeout_seconds: float = 10.0) -> None:
        self.webhook_url = webhook_url
        self.secret = secret
        self.timeout_seconds = timeout_seconds

    def send_text(self, text: str) -> FeishuSendResult:
        payload: dict[str, object] = {
            "msg_type": "text",
            "content": {"text": text},
        }
        if self.secret:
            timestamp = str(int(time.time()))
            payload["timestamp"] = timestamp
            payload["sign"] = self._sign(timestamp, self.secret)

        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.webhook_url,
            data=encoded,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            return FeishuSendResult(status_code=response.status, body=body)

    @staticmethod
    def _sign(timestamp: str, secret: str) -> str:
        string_to_sign = f"{timestamp}\n{secret}"
        digest = hmac.new(
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("utf-8")
