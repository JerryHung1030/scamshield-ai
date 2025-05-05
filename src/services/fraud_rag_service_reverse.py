
import logging
from typing import List, Dict, Any
from .base_rag_service import BaseRAGService
from managers.blacklist_manager import BlacklistManager
import json
logger = logging.getLogger(__name__)


class FraudRAGService_reverse(BaseRAGService):
    """
    詐騙偵測：
    - prompt 要求 output 欄位：code、label、evidence、confidence、start_idx、end_idx
    - label = "<code><category>:<desc>"（必須跟檢索 Doc 完全一致）
    - 另外: 若發現黑名單(網址/line id...)，直接回傳 [{"label":"blacklist"...}]
    """

    def __init__(
        self,
        embedding_manager,
        vector_store_manager,
        llm_manager,
        blacklist_manager: BlacklistManager,
        domain_key="FRAUD",
        selected_llm_name=None,
    ):
        super().__init__(
            embedding_manager,
            vector_store_manager,
            llm_manager,
            domain_key,
            selected_llm_name
        )
        self.blacklist_manager = blacklist_manager
        self.prompt_header = (
            "你是詐騙分類偵測器，根據下方user貼文chunk，判斷貼文chunk內容是否有命中以下詐騙分類。如沒有命中請直接輸出[]。並請回傳貼文內容chunk對應的uid。\n"
            "請直接輸出JSON array, 格式：\n"
            "[\n"
            "  {\n"
            "    \"uid\": \"...\",\n"
            "    \"label\": \"...\",\n"
            "    \"confidence\": 0.95,\n"
            "  }\n"
            "]\n"
            "如未命中, 請輸出 []。\n"
        )

    def _hit_blacklist(self, text: str) -> bool:
        return bool(
            self.blacklist_manager.check_urls(text) + self.blacklist_manager.check_line_ids(text)
        )

    def build_prompt(self, user_query: str, context_docs: List[dict]) -> str:
        # context_docs = [{"uid":..., "text":"...", "score":...}, ...]
        lines = []
        for i, doc in enumerate(context_docs, start=1):
            lines.append(f"[Doc#{i}] uid={doc['uid']} \n{doc['text']}\n")
        docs_txt = "\n".join(lines)

        guide = """
        {
            "uid": "<請回傳貼文內容chunk對應的uid>",
            "label": "....",
            "confidence": <請你自行判斷該label的可信程度0~1>,

        }
        """

        return (
            f"{self.prompt_header}\n"
            f"貼文內容chunk與其對應uid: {user_query}\n"
            f"---候選詐騙分類---\n{docs_txt}\n"
            f"範例: {guide}\n"
            "請務必回傳JSON array，每個物件必須包含 uid, label, confidence"
        )
    def post_process(
        self,
        user_query: str,
        raw_json: list[dict],
        hits: List[dict]
    ) -> list[dict]:
        # 用 code 補 uid
        # for rec in raw_json:
        #     if "uid" not in rec or not rec["uid"]:
        #         rec["uid"] = rec.get("code", "")

        ev_key = self.cfg["evidence_key"]
        s_key, e_key = self.cfg["start_idx_key"], self.cfg["end_idx_key"]

        for rec in raw_json:
            uid_from_llm = rec.get("uid", "")
            match_doc = next((h for h in hits if h["uid"] == uid_from_llm), None)

            if match_doc:
                rec["similarity_score"] = match_doc["score"]
            else:
                rec["similarity_score"] = 0.0
                doc_text = ""

            evidence_txt = rec.get(ev_key, "")
            if evidence_txt:
                start_idx = user_query.find(evidence_txt)
                end_idx = start_idx + len(evidence_txt) if start_idx != -1 else -1
            else:
                start_idx = -1
                end_idx = -1

            rec[s_key], rec[e_key] = start_idx, end_idx

        return raw_json
    async def generate_answer(self, user_query: str, filters: Dict[str, Any] | None = None) -> str:
        if self._hit_blacklist(user_query):
            return '[{"label":"blacklist","evidence":"","confidence":1,"start_idx":-1,"end_idx":-1}]'
        raw_output_str = await super().generate_answer(user_query, filters)
        return json.loads(raw_output_str)



