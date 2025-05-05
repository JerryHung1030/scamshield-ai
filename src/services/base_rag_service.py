# src/services/base_rag_service.py
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from managers.embedding_manager import EmbeddingManager
from managers.vector_store_manager import VectorStoreManager
from managers.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class BaseRAGService(ABC):
    """
    通用 RAG 骨架：
    1. 先用 Embedding Manager 產生 Query Embedding
    2. 在 Vector DB (via VectorStoreManager) 檢索 context，每段以 dict 形式: {"uid","text","score"}
    3. build_prompt() 組合 Prompt -> 呼叫 LLM 產生 JSON
    4. post_process() 做後處理 (例如插入 similarity_score, start_idx, end_idx)
    5. 回傳最終 JSON string
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store_manager: VectorStoreManager,
        llm_manager: LLMManager,
        domain_key: str,
        selected_llm_name: str | None = None,
        k: int = 15,
        extra_cfg: Dict[str, Any] | None = None,
        **kwargs
    ):
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        self.llm_manager = llm_manager
        self.domain_key = domain_key
        self.selected_llm_name = selected_llm_name
        self.top_k = k

        # 其他設定
        self.cfg = extra_cfg or {}
        self.cfg.setdefault("evidence_key", "evidence")
        self.cfg.setdefault("start_idx_key", "start_idx")
        self.cfg.setdefault("end_idx_key", "end_idx")
        self.cfg.setdefault("min_similarity", 0.5)

    # ---------- 子類別一定要實作 ---------- #
    @abstractmethod
    def build_prompt(self, user_query: str, context_docs: List[dict]) -> str:
        """
        子類必須實作：根據 user_query 與檢索到的 context_docs (List[dict]) 來組合 Prompt。
        context_docs 的每個元素結構為:
        {
          "uid": <str>,
          "text": <str>,
          "score": <float>
        }
        """

    # ---------- post_process ---------- #
    def post_process(
        self,
        user_query: str,
        raw_json: list[dict],
        hits: List[dict]
    ) -> list[dict]:
        """
        預設行為：
          - 幫 evidence -> start_idx/end_idx
          - 用 uid 對應 similarity_score
        """
        raise NotImplementedError("Subclasses must implement post_process()")


    # ---------- helper ---------- #
    def _strip_code_fence(self, text: str) -> str:
        """
        移除可能的 Markdown code fence
        """
        txt = text.strip()
        if txt.startswith("```"):
            txt = txt.split("\n", 1)[-1].strip()
        if txt.endswith("```"):
            txt = txt.rsplit("```", 1)[0].strip()
        return txt

    # ---------- RAG 流程 ---------- #
    def retrieve_context(self, user_query: str, filters: Dict[str, Any] | None) -> List[dict]:
        """
        1) Embedding user_query
        2) Search in VectorStore: return List[dict], each dict has {"uid","text","score"}
        3) 過濾 similarity < min_similarity
        4) 排序後回傳
        """
        query_vec = self.embedding_manager.generate_embedding(user_query)
        hits = self.vector_store_manager.search_similar_with_score(
            domain=self.domain_key,
            query_vector=query_vec,
            k=self.top_k,
            filters=filters
        )
        # hits: List[dict] = [{"uid":"xx","text":"...","score":0.88}, ...]

        # sort by score desc
        hits.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"[RAG] context docs = {len(hits)}")

        # min_similarity
        min_sim = self.cfg["min_similarity"]
        filtered = [h for h in hits if h["score"] >= min_sim]
        logger.debug(f"[RAG] similarity >= {min_sim}: {len(filtered)}/{len(hits)}")

        return filtered

    async def generate_answer(
        self,
        user_query: str,
        filters: Dict[str, Any] | None = None
    ) -> str:
        """
        1) retrieve_context
        2) build_prompt
        3) LLM
        4) JSON parse
        5) post_process
        6) return JSON str
        """
        try:
            hits = self.retrieve_context(user_query, filters)
            if not hits:
                return "[]"

            # build prompt
            prompt = self.build_prompt(user_query, hits)

            adapter_name = self.selected_llm_name or self.llm_manager.default_adapter_name
            adapter = self.llm_manager.get_adapter(adapter_name)
            if not adapter:
                logger.error(f"No adapter {adapter_name}, returning []")
                return "[]"

            # LLM
            raw_llm = await adapter.async_generate_response(prompt)
            cleaned = self._strip_code_fence(raw_llm)

            # parse JSON
            try:
                parsed_data = json.loads(cleaned)
            except Exception as e:
                logger.error(f"[RAG] JSON parse error: {e} raw={cleaned[:200]}")
                return "[]"

            if not isinstance(parsed_data, list):
                logger.warning("[RAG] LLM output not a JSON list, returning []")
                return "[]"

            # post_process
            final_records = self.post_process(user_query, parsed_data, hits)

            return json.dumps(final_records, ensure_ascii=False)

        except Exception as e:
            logger.error(f"[RAG] generate_answer exception: {e}")
            return "[]"
