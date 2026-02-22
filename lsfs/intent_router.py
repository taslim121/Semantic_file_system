import math
from typing import List, Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


def _to_list(vec) -> List[float]:
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)


class IntentRouter:
    """Route a natural language query to a target index based on intent labels."""

    def __init__(
        self,
        model_name: str,
        labels: List[Dict[str, str]],
        min_confidence: float,
        fallback_target: str,
    ):
        self.model = SentenceTransformer(model_name)
        self.labels = labels
        self.min_confidence = min_confidence
        self.fallback_target = fallback_target

        self._label_embeddings: List[List[float]] = []
        for label in labels:
            name = label.get("name", "")
            description = label.get("description", "")
            text = f"{name}: {description}".strip(": ")
            embedding = self.model.encode(text)
            self._label_embeddings.append(_normalize(_to_list(embedding)))

    def route(self, query: str) -> Tuple[str, float, Optional[str]]:
        if not self.labels or not query.strip():
            return self.fallback_target, 0.0, None

        q_embedding = self.model.encode(query)
        q_vec = _normalize(_to_list(q_embedding))

        best_idx = -1
        best_score = -1.0
        for i, label_vec in enumerate(self._label_embeddings):
            score = sum(a * b for a, b in zip(q_vec, label_vec))
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == -1:
            return self.fallback_target, 0.0, None

        if best_score < self.min_confidence:
            return self.fallback_target, best_score, self.labels[best_idx].get("name")

        label = self.labels[best_idx]
        target = label.get("target", self.fallback_target) or self.fallback_target
        return target, best_score, label.get("name")
