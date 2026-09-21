"""Independent, conservative evidence gate for automatically proposed memories.

The local model is a fallible semantic checker, not a proof engine.
"""
from __future__ import annotations

import json

import httpx

EVIDENCE_SYSTEM = """Check whether ONE proposed long-term memory is justified by ONE
quoted span of the user's latest message. Be skeptical. Return ONLY JSON:
{"supported":true|false,"reason":"short explanation"}.

Approve only when the ENTIRE new claim follows from the quote alone.
Do not use older memories or assistant messages to supply missing personal
facts. Reject bundled facts when the quote supports just one clause. Reject
invented employers, positions, countries, dates, decisions, or emotions.
Never turn a plan into an already completed action. A tentative inference
must remain tentative, and an uncertain outcome must remain unresolved.
For a revision, the LATEST quote must support the ENTIRE new replacement
claim; earlier versions remain in the history, not as free evidence for
unsourced additional claims in the replacement.

Examples:
Quote: "My current job doesn't have as much clinical exposure."
Claim: "The user works at Emory and has a new Morehouse job, but their
current job has less clinical exposure."
Decision: unsupported: employer and new job are not in quote.
Quote: "I'm going to live in Japan."
Claim: "The user has decided to live in Japan."
Decision: supported: decided, not already living there.
Quote: "I might switch jobs."
Claim: "The user has switched jobs."
Decision: unsupported: possibility is not completion.
"""


class OllamaEvidenceVerifier:
    def __init__(
        self, model: str, base_url: str, *,
        keep_alive: str | int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive

    async def check(self, action: dict) -> tuple[bool, str]:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 180, "num_ctx": 4096},
            "messages": [
                {"role": "system", "content": EVIDENCE_SYSTEM},
                {"role": "user", "content": json.dumps({
                    "quote": action["source_quote"],
                    "claim": action["text"],
                    "status": action["epistemic_status"],
                    "action": action["action"],
                }, ensure_ascii=False)},
            ],
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        async with httpx.AsyncClient(timeout=75.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        raw = data.get("message", {}).get("content")
        if not isinstance(raw, str):
            return False, "checker_missing_response"
        raw = raw.strip()
        # The MLX runner sometimes wraps otherwise valid JSON in a code fence.
        fence = chr(96) * 3
        if raw.startswith(fence) and raw.endswith(fence):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[0].lower() in {fence, fence + "json"}:
                raw = "\n".join(lines[1:-1]).strip()
        try:
            result = json.loads(raw)
        except (TypeError, ValueError):
            return False, "checker_invalid_json"
        if not isinstance(result, dict) or type(result.get("supported")) is not bool:
            return False, "checker_invalid_structure"
        if not result["supported"]:
            return False, "evidence_does_not_support_whole_claim"
        return True, "supported"
