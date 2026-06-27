import re
import numpy as np
from datetime import datetime, timezone, timedelta
from pymongo import ASCENDING

UTC_PLUS_3 = timezone(timedelta(hours=3))


class SemanticCache:
    """Global semantic cache — stores query→response pairs with embeddings."""

    def __init__(self, mongo_db, dense_model, threshold=0.92):
        self.coll = mongo_db.semantic_cache
        self.coll.create_index([("query", ASCENDING)])
        self.dense_model = dense_model
        self.threshold = threshold
        self._cache = None

    def _load_all(self):
        """Load all cache entries into memory."""
        docs = list(self.coll.find({}, {"query": 1, "response": 1, "embedding": 1, "hit_count": 1}))
        self._cache = docs
        return docs

    def _embed(self, text: str) -> np.ndarray:
        return np.array(self.dense_model.encode(text), dtype=np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def find(self, query: str):
        """Find a semantically similar cached response. Returns dict or None."""
        if self._cache is None:
            self._load_all()
        if not self._cache:
            return None

        query_emb = self._embed(query)
        best_score = -1.0
        best_doc = None

        for doc in self._cache:
            cached_emb = np.array(doc["embedding"], dtype=np.float32)
            score = self._cosine_similarity(query_emb, cached_emb)
            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc and best_score >= self.threshold:
            self.coll.update_one(
                {"_id": best_doc["_id"]},
                {"$inc": {"hit_count": 1}},
            )
            return {
                "response": best_doc["response"],
                "query": best_doc["query"],
                "score": best_score,
            }
        return None

    def store(self, query: str, response: str, model_name: str = ""):
        """Store a new query→response pair in the cache."""
        embedding = self._embed(query).tolist()
        doc = {
            "query": query,
            "response": response,
            "embedding": embedding,
            "model_name": model_name,
            "created_at": datetime.now(UTC_PLUS_3),
            "hit_count": 0,
        }
        self.coll.insert_one(doc)
        if self._cache is not None:
            self._cache.append(doc)

    def invalidate(self):
        """Force reload on next find()."""
        self._cache = None


_MODEL_FAST = "openai/gpt-oss-20b"
_MODEL_STRONG = "openai/gpt-oss-120b"

_SIMPLE_PATTERNS_EN = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|good (morning|afternoon|evening)|yes|no|ok|okay|sure|help)\s*[!?.]*$",
    re.IGNORECASE,
)
_SIMPLE_PATTERNS_AR = re.compile(
    r"^(مرحبا|السلام عليكم|أهلاً|اهلا|صباح الخير|مساء الخير|شكرا|مع السلامة|نعم|لا|تمام|حسنًا|ممتاز|ألف شكر)\s*[!؟]*$",
)
_SIMPLE_LENGTH = 25
_STRONG_SIGNALS_EN = re.compile(
    r"\b(compare|difference|versus|vs|pros and cons|which (is )?better|should i|recommend|advice|explain in detail|step by step|deep|complex)\b",
    re.IGNORECASE,
)
_STRONG_SIGNALS_AR = re.compile(
    r"(أفضل|فرق|مقارنة|ليش|وش الفرق|ايش الفرق|افضل|شرح تفصيلي|خطوة بخطوة|عميق|معقد|مقارنة بين)",
)


def route_query(prompt: str, history: list = None) -> str:
    """Route a user query to the appropriate model.

    Returns model name: "openai/gpt-oss-20b" for simple FAQ,
    "openai/gpt-oss-120b" for complex questions.
    """
    text = prompt.strip()

    if len(text) <= _SIMPLE_LENGTH and (_SIMPLE_PATTERNS_EN.match(text) or _SIMPLE_PATTERNS_AR.match(text)):
        return _MODEL_FAST

    if _STRONG_SIGNALS_EN.search(text) or _STRONG_SIGNALS_AR.search(text):
        return _MODEL_STRONG

    if len(text.split()) > 40:
        return _MODEL_STRONG

    if text.count("?") > 1 or text.count("؟") > 1:
        return _MODEL_STRONG

    if history and len(history) > 4:
        return _MODEL_STRONG

    return _MODEL_FAST
