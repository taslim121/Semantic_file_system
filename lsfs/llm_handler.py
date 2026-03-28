import json
from typing import Any, Dict, List, Optional

import requests


def _extract_first_json_block(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_first_json_array(text: str) -> Optional[str]:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class OllamaHandler:
    """Handles interaction with local Ollama LLM."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.request_timeout = 45

        if not self._test_connection():
            raise ConnectionError(f"Cannot connect to Ollama at {base_url}")

    def _test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def parse_command(self, user_input: str) -> Dict[str, Any]:
        """Parse natural language command using local LLM."""
        system_prompt = """
You are a strict JSON command parser for a local semantic file system.
Return ONLY valid JSON (no markdown, no prose).

Allowed operations:
- create_file {file_name, content?}
- create_dir {dir_name}
- write {file_name, content, append?}
- read {file_name}
- search {query, k?, file_types?}
- list {subdir?}
- delete {file_name}
- move {source, destination}
- copy {source, destination}
- reindex {}
- stats {}

Rules:
- Fix user spelling mistakes when obvious.
- If user asks for file types, include "file_types": [".json", ".ts"] etc.
- Keep k small (default 5) if user doesn't provide k.
- Output format:
{
  "operation": "...",
  "parameters": {...},
  "confidence": 0.0-1.0
}
""".strip()

        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 220},
                },
                timeout=self.request_timeout,
            )
            if response.status_code != 200:
                return {
                    "operation": "error",
                    "parameters": {"message": "LLM request failed"},
                    "confidence": 0.0,
                }

            content = response.json().get("message", {}).get("content", "").strip()
            json_str = _extract_first_json_block(content)
            if not json_str:
                return {
                    "operation": "error",
                    "parameters": {"message": "No JSON found in LLM response"},
                    "confidence": 0.0,
                }
            try:
                parsed = json.loads(json_str)
                if "operation" not in parsed:
                    raise ValueError("Missing operation key")
                if "parameters" not in parsed or not isinstance(parsed["parameters"], dict):
                    parsed["parameters"] = {}
                if "confidence" not in parsed:
                    parsed["confidence"] = 0.5
                return parsed
            except Exception as e:
                return {
                    "operation": "error",
                    "parameters": {"message": f"Parse error: {e}"},
                    "confidence": 0.0,
                }
        except Exception as e:
            return {
                "operation": "error",
                "parameters": {"message": str(e)},
                "confidence": 0.0,
            }

    def rerank_results(
        self, query: str, candidates: List[Dict[str, Any]], final_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank candidate files with LLM, returning up to final_k items."""
        if not candidates:
            return []

        short_candidates = []
        for i, item in enumerate(candidates, start=1):
            short_candidates.append(
                {
                    "id": i,
                    "file_name": item.get("file_name", ""),
                    "relative_path": item.get("relative_path", ""),
                    "preview": (item.get("preview", "") or "")[:240],
                    "semantic": item.get("semantic_score", item.get("similarity", 0.0)),
                    "lexical": item.get("lexical_score", 0.0),
                }
            )

        prompt = {
            "task": "rerank_files_for_query",
            "query": query,
            "instructions": [
                "Choose the most relevant files for the user query.",
                "Prefer exact intent match in filename/path/preview.",
                "Use semantic and lexical hints, but prioritize user intent.",
                "Return only JSON array of ids in best-first order.",
            ],
            "candidates": short_candidates,
        }

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": json.dumps(prompt),
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 120},
                },
                timeout=self.request_timeout,
            )
            if response.status_code != 200:
                return candidates[:final_k]

            text = response.json().get("response", "").strip()
            array_str = _extract_first_json_array(text)
            if not array_str:
                return candidates[:final_k]

            order = json.loads(array_str)
            if not isinstance(order, list):
                return candidates[:final_k]

            by_id = {i + 1: c for i, c in enumerate(candidates)}
            reranked: List[Dict[str, Any]] = []
            used = set()
            for raw_id in order:
                try:
                    idx = int(raw_id)
                except Exception:
                    continue
                if idx in by_id and idx not in used:
                    reranked.append(by_id[idx])
                    used.add(idx)
                if len(reranked) >= final_k:
                    break

            if len(reranked) < final_k:
                for i, item in enumerate(candidates, start=1):
                    if i in used:
                        continue
                    reranked.append(item)
                    if len(reranked) >= final_k:
                        break
            return reranked[:final_k]
        except Exception:
            return candidates[:final_k]
