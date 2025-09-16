import logging
import json
import re
from typing import List, Dict, Optional, Any, Tuple

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from openai import OpenAI
import chromadb
from chromadb.config import Settings


# =========================
# LLM
# =========================

class QwenChat:

    def __init__(
            self,
            api_key: str,
            base_url: str,
            model_name: str = "qwen-plus",
            temperature: float = 0.3,
            max_tokens: int = 1024,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _generate(self, messages: List[Any]):
        # 转换为 OpenAI chat.completions 格式
        oai_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                oai_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                oai_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                oai_messages.append({"role": "assistant", "content": m.content})
            else:
                role = getattr(m, "role", "user")
                content = getattr(m, "content", "")
                oai_messages.append({"role": role, "content": content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=oai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False
        )

        from langchain.schema import ChatGeneration, ChatResult
        generation = ChatGeneration(message=AIMessage(content=response.choices[0].message.content))
        return ChatResult(generations=[generation])

    def _llm_type(self) -> str:
        return "qwen_chat"


# =========================
# database
# =========================

class VectorDatabase:
    """
    - Image RobotAgent - Extracting Image Patterns and Objective Phenomena
    """

    def __init__(self, config: dict):
        self.config = config
        persist_dir = config["chroma"].get("persist_directory", "./chroma_db")
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            persist_directory=persist_dir
        ))
        api_cfg = config["api"]
        self.embedding_client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg["base_url"])
        self.embedding_model = api_cfg["models"].get("embedding", "text-embedding-v4")
        self.collection_names = {
            "imaging": config["chroma"]["collection_names"].get("imaging", "imaging_kb"),
            "clinical": config["chroma"]["collection_names"].get("clinical", "clinical_kb"),
        }
        self._ensure_collections()

    def _ensure_collections(self):
        for name in [self.collection_names["imaging"], self.collection_names["clinical"]]:
            try:
                self.client.get_collection(name=name)
            except Exception:
                self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def _embed(self, text: str) -> List[float]:
        resp = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return resp.data[0].embedding

    def _get_collection(self, category: str):
        if category == "imaging":
            name = self.collection_names["imaging"]
        elif category == "clinical":
            name = self.collection_names["clinical"]
        else:
            name = category
        return self.client.get_collection(name=name)

    # ---- clean  str/int/float/bool ----
    def _sanitize_metadata(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not meta:
            return {}
        safe: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif v is None:
                safe[k] = ""
            else:
                try:
                    safe[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    safe[k] = str(v)
        return safe

    def add_experience(self, category: str, content: str, meta: Dict[str, Any] = None, doc_id: Optional[str] = None):
        """
        category: 'imaging' | 'clinical'
        """
        collection = self._get_collection(category)
        embedding = self._embed(content)
        if doc_id is None:
            try:
                suffix = collection.count() + 1
            except Exception:
                suffix = 1
            doc_id = f"{category}_{suffix}"

        # replace cosin

        sim_threshold = self.config["retrieval"].get("similarity_threshold", 0.8)
        distance_threshold = 1 - sim_threshold

        similar_results = collection.query(
            query_embeddings=[embedding],
            include=['distances'],
            n_results=1
        )

        try:
            distances = similar_results.get('distances', [[]])[0]
            ids_list = similar_results.get('ids', [[]])[0]
            if distances and ids_list and distances[0] < distance_threshold:
                old_id = ids_list[0]
                collection.delete(ids=[old_id])
        except Exception:
            pass

        meta = self._sanitize_metadata(meta)

        collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta or {}],
            ids=[doc_id]
        )

    def search(self, category: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        collection = self._get_collection(category)
        q_emb = self._embed(query)
        res = collection.query(
            query_embeddings=[q_emb],
            include=['documents', 'metadatas', 'distances'],
            n_results=top_k
        )
        results = []
        ids_matrix = res.get('ids', [[]])
        docs_matrix = res.get('documents', [[]])
        metas_matrix = res.get('metadatas', [[]])
        dists_matrix = res.get('distances', [[]])

        if ids_matrix and ids_matrix[0]:
            for i in range(len(ids_matrix[0])):
                results.append({
                    "id": ids_matrix[0][i],
                    "content": (docs_matrix[0][i] if docs_matrix and docs_matrix[0] else None),
                    "metadata": (metas_matrix[0][i] if metas_matrix and metas_matrix[0] else None),
                    "distance": (dists_matrix[0][i] if dists_matrix and dists_matrix[0] else None)
                })
        return results


# =========================
# base agent
# =========================

class BaseAgent:
    """Agent"""

    def __init__(self, config: dict, name: str):
        self.config = config
        self.name = name
        api_cfg = config["api"]
        self.llm = QwenChat(
            api_key=api_cfg["api_key"],
            base_url=api_cfg["base_url"],
            model_name=api_cfg["models"].get("text", "qwen-plus"),
            temperature=api_cfg.get("temperature", 0.3),
            max_tokens=api_cfg.get("max_tokens", 1024)
        )
        self.multimodal_llm = QwenChat(
            api_key=api_cfg["api_key"],
            base_url=api_cfg["base_url"],
            model_name=api_cfg["models"].get("multimodal", "qwen-vl-plus"),
            temperature=api_cfg.get("temperature", 0.3),
            max_tokens=api_cfg.get("max_tokens", 1024)
        )
        self.logger = logging.getLogger(self.name)

    def get_system_prompt(self) -> str:
        return "You are a helpful medical AI agent."

    def chat(self, messages: List[Any]) -> str:
        result = self.llm._generate(messages)
        return result.generations[0].message.content


# =========================
#  Agent
# =========================

class StudentAgent(BaseAgent):
    """学生智能体 - 负责诊断推理和工具调用决策"""

    def __init__(self, config: dict, vector_db: VectorDatabase, tools: List = None):
        super().__init__(config, 'student')
        self.vector_db = vector_db
        self.tools = tools or []

    def get_system_prompt(self) -> str:
        return """You are a medical student AI assistant specializing in clinical diagnosis. Your responsibilities include:

1. **Initial Response**: Provide comprehensive initial diagnostic assessment
2. **Stepwise Verification**: Address verification questions systematically
3. **Tool Decision**: Decide when to use external tools (like PubMed) based on uncertainty
4. **Final Summary**: Synthesize all information into final diagnosis

Guidelines:
- Be concise and clinically relevant
- Focus strictly on the patient's question; if key info is missing, ask for up to 2 brief clarifications first
- Avoid hallucinations; clearly indicate uncertainty
- Use structured bullet points and headings
- If the patient's question is closed-ended (yes/no or multiple choice), add the final single-line format at the end: `Answer: Yes|No|A|B|C|D|E|1|2|...`"""

    def initial_response(self, query: str, imaging_evidence: str = "",
                         imaging_analysis: str = "", retrieved_experiences: List[dict] = None,
                         lang_hint: str = "") -> str:
        prefix_lang = (lang_hint + "\n\n") if lang_hint else ""
        experience_context = self._format_experiences(retrieved_experiences)

        content = f"""
{prefix_lang}Query: {query}

{f"Imaging Evidence: {imaging_evidence}" if imaging_evidence else ""}
{f"Imaging Analysis: {imaging_analysis}" if imaging_analysis else ""}
{f"Relevant Past Experiences: {experience_context}" if experience_context else ""}

Please provide your initial diagnostic assessment. Include:
1. Key findings and clinical interpretation
2. Differential diagnosis (ranked by likelihood)
3. Recommended next steps (tests, management)
4. Uncertainties and specific clarifying questions (if any)
"""
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=content)
        ]
        result = self.llm._generate(messages)
        return result.generations[0].message.content

    def stepwise_verification(self, original_query: str, verification_questions: str,
                              context: dict, lang_hint: str = "") -> str:
        prefix_lang = (lang_hint + "\n\n") if lang_hint else ""
        content = f"""
{prefix_lang}Original Query: {original_query}

Teacher's Verification Questions: {verification_questions}

Previous Context:
- Initial Response: {context.get('initial_response', '')}
- Imaging Evidence: {context.get('imaging_evidence', '')}
- Imaging Analysis: {context.get('imaging_analysis', '')}

Please address each verification question explicitly in numbered points. If any information is insufficient, state what is needed. Keep the focus strictly on the patient's question.
"""
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=content)
        ]
        result = self.llm._generate(messages)
        return result.generations[0].message.content

    def final_summary(self, query: str, context: dict, tool_results: str = "", lang_hint: str = "") -> str:
        prefix_lang = (lang_hint + "\n\n") if lang_hint else ""
        content = f"""
{prefix_lang}Original Query: {query}

Complete Diagnostic Process:
- Initial Response: {context.get('initial_response', '')}
- Verification: {context.get('verification_response', '')}
- Teacher's CoT Questions: {context.get('cot_questions', '')}
{f"- Tool Results: {tool_results}" if tool_results else ""}

Now produce the final clinical answer **strictly focused on the patient's question**:

**Final Diagnosis (if applicable)**:
- ...

**Differential Diagnosis**:
1. ...
2. ...

**Key Evidence**:
- ...

**Recommendations**:
- ...

**Uncertainties / Follow-up Questions (if any)**:
- ...

If the patient's question is closed-ended (yes/no or multiple choice), output the final single-line decision at the very end:
Answer: <Yes|No|A|B|C|D|E|1|2|...>
"""
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=content)
        ]
        result = self.llm._generate(messages)
        return result.generations[0].message.content

    def should_use_tools(self, teacher_cot: str, verification_response: str) -> bool:
        text = (teacher_cot or "") + "\n" + (verification_response or "")
        lower = text.lower()
        uncertainty_markers = [
            "uncertain", "not sure", "unclear", "might", "could", "possibly", "maybe",
            "need more information", "require further", "ambiguous", "inconclusive",
            "不确定", "不清楚", "尚不清楚", "可能", "大概", "也许", "需要进一步", "信息不足", "无法确定", "有待明确"
        ]
        hits = sum(1 for w in uncertainty_markers if w in lower)
        q_count = text.count("?") + text.count("？")
        return (hits >= 2) or (q_count >= 3)

    def _format_experiences(self, experiences: List[dict]) -> str:
        if not experiences:
            return ""
        formatted = ""
        for i, exp in enumerate(experiences[:3], 1):
            formatted += f"{i}. {exp.get('content', '')}\n"
        return formatted


class ImagingRobotAgent(BaseAgent):
    """Image Robot  Agent - Extracting Image Modes and Objective Phenomena"""

    def get_system_prompt(self) -> str:
        return """You are an Imaging Robot Agent that **only** extracts objective imaging evidence.
Rules:
- Do NOT provide interpretation or diagnosis
- List modalities and per-modality objective findings (location, size, density/signal, shape, enhancement, etc.)
- If images are poor quality, state the technical limitation
- Be concise and standardized"""

    def extract_evidence(self, images: List[str], query: str, lang_hint: str = "") -> str:
        prefix_lang = (lang_hint + "\n\n") if lang_hint else ""
        if not images:
            return "No images provided for analysis."

        try:
            messages = [
                {
                    "role": "system",
                    "content": self.get_system_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prefix_lang}Query context: {query}\n\nPlease analyze the provided medical images and extract objective findings."
                        }
                    ]
                }
            ]

            # 只添加第一张，避免过长
            for i, img_base64 in enumerate(images[:1]):
                if len(img_base64) > 1_200_000:
                    return "Image too large for processing. Please upload a smaller image (recommended < 1MB)."
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": img_base64}
                })

            response = self.multimodal_llm.client.chat.completions.create(
                model=self.multimodal_llm.model_name,
                messages=messages,
                temperature=self.multimodal_llm.temperature,
                max_tokens=self.multimodal_llm.max_tokens
            )
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Fail: {str(e)}")
            return f"Error analyzing images: {str(e)}. Please try with a smaller image or check image format."


class ImagingDoctorAgent(BaseAgent):
    """Imaging Doctor Agent - Interpreting and Analyzing Images"""

    def get_system_prompt(self) -> str:
        return """You are an Imaging Doctor Agent (radiologist). Your job is to interpret imaging **based on** the robot's objective findings and the clinical question.
Rules:
- Do NOT restate raw pixels; build clinical meaning
- Provide differentials with reasoning
- State limitations and recommend next steps
- Remain strictly relevant to the patient's question"""

    def analyze_images(self, robot_evidence: str, query: str, images: List[str] = None, lang_hint: str = "") -> str:
        prefix_lang = (lang_hint + "\n\n") if lang_hint else ""
        content = f"""
{prefix_lang}Clinical Query: {query}

Imaging Robot's Objective Findings:
{robot_evidence}

Based on these objective findings and the clinical context, please provide your radiological interpretation. Include:
1. Clinical correlation of findings
2. Differential diagnostic considerations
3. Assessment of clinical significance
4. Recommendations for management or follow-up
"""
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=content)
        ]
        result = self.llm._generate(messages)
        return result.generations[0].message.content


class TeacherAgent(BaseAgent):
    """Teacher Agent - Planning and Doub"""

    def get_system_prompt(self) -> str:
        return """You are a senior clinician (Teacher). You plan verification and CoT questions to stress-test the student's reasoning.
- Keep questions concise and clinically relevant
- Focus strictly on the patient's original query
- Avoid introducing irrelevant topics"""

    def plan_verification(self, student_response: str, original_query: str, lang_hint: str = "") -> str:
        prefix_lang = (lang_hint + "\n\n") if lang_hint else ""
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=f"""
{prefix_lang}Original Query: {original_query}

Student's Initial Response: {student_response}

Based on the student's response, generate 3-5 verification questions that would surface possible mistakes, oversights, or areas needing clarification. Focus on:
- Differential diagnosis considerations
- Evidence evaluation
- Clinical reasoning gaps
- Potential contradictions

Format as: **Planning & Inquiry**
1. [Question 1]
2. [Question 2]
3. [Question 3]
""")
        ]
        result = self.llm._generate(messages)
        return result.generations[0].message.content

    def cot_questioning(self, student_verification: str, original_query: str, lang_hint: str = "") -> str:
        prefix_lang = (lang_hint + "\n\n") if lang_hint else ""
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=f"""
{prefix_lang}Original Query: {original_query}

Student's Verification Response: {student_verification}

Now conduct final Chain of Thought questioning. Ask 2-3 penetrating questions that:
- Test the logical consistency of the diagnosis
- Verify key clinical reasoning steps
- Challenge the most critical diagnostic assumptions
- Ensure no important considerations were missed

Format as: **CoT Questioning**
1. [Critical question 1]
2. [Critical question 2]
...
""")
        ]
        result = self.llm._generate(messages)
        return result.generations[0].message.content


class JudicialAgent(BaseAgent):
    """Judg agent - more flexible and humanized evaluation
        -Automatic recognition of question types: If the reference answer contains vague words such as "possible/requires combination/uncertain", it will be scored according to the open-ended question
        -Closed ended question (clearly yes/no or A/B/C/D/1/2/...): Maintain strict equivalence comparison
        -Open ended question: Four dimensional 1-10 rigorous scoring+providing "semantic consistency alignment" with the reference answer
        alignment ∈ {"agree","lean_agree","neutral","lean_disagree","disagree"}
    """

    def get_system_prompt(self) -> str:
        return """You are a Judicial Agent that evaluates medical answers.

    For OPEN-ENDED:
    - Score strictly on 4 dimensions (1-10, up to two decimals): helpfulness, relevance, accuracy, level_of_details
    - Penalize hedging/uncertainty without justification
    - Penalize off-topic content
    - Penalize contradictions with provided evidence or reference
    - Return strict JSON only with keys: type, helpfulness, relevance, accuracy, level_of_details, overall
    """

    # ---------------- Heuristics----------------

    def _has_hedging(self, text: str) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        zh = text
        hedging_en = [
            "likely", "possible", "probable", "suggest", "suggestive", "suspicious",
            "may", "might", "could", "appears", "cannot rule out", "need further",
            "depend on", "requires correlation", "cannot confirm", "uncertain", "not sure", "inconclusive"
        ]
        hedging_zh = [
            "可能", "考虑", "提示", "倾向", "疑似", "不一定", "需结合", "需要结合", "建议进一步", "不能排除",
            "不能确诊", "待排", "尚不明确", "不明确"
        ]
        return any(w in t for w in hedging_en) or any(w in zh for w in hedging_zh)

    def _is_strict_closed_ref(self, ref: str) -> bool:
        """Is the reference answer a clear closed label (A/B/C/D/E/1-9/yes/no/definite diagnosis/definite exclusion)）"""
        if not ref:
            return False
        # If there are obvious vague words, simply do not
        if self._has_hedging(ref):
            return False
        # Clearly label
        if re.search(r'\b([A-Ea-e]|[1-9])\b', ref):
            return True
        # Clearly yes/no
        yn = self._normalize_ref_label(ref)
        return yn in {"yes", "no"}

    # -------- 既有的 yes/no 与选项解析（保留/复用） --------

    def _normalize_ref_label(self, ref: str) -> Optional[str]:
        if not ref:
            return None
        text = ref.strip()
        m = re.search(r'\b([A-Ea-e])\b', text)
        if m:
            return m.group(1).upper()
        m = re.search(r'([A-Ea-e])\s*[项项别选择题项]?', text)
        if m:
            return m.group(1).upper()
        m = re.search(r'\b([1-9])\b', text)
        if m:
            return m.group(1)
        yn = self._extract_yes_no_label(text)
        if yn:
            return yn
        if any(k in text for k in ["是", "确诊", "肯定", "阳性"]):
            return "yes"
        if any(k in text for k in ["否", "不是", "阴性", "排除"]):
            return "no"
        return None

    def _extract_choice_label(self, text: str) -> Optional[str]:
        if not text:
            return None
        t = text.strip()
        m = re.search(r'(?i)answer\s*[:：]\s*([A-E])\b', t)
        if m: return m.group(1).upper()
        m = re.search(r'(?i)answer\s*[:：]\s*([1-9])\b', t)
        if m: return m.group(1)
        m = re.search(r'[选择项]\s*([A-Ea-e])\b', t)
        if m: return m.group(1).upper()
        m = re.search(r'([A-Ea-e])\s*[项项别]', t)
        if m: return m.group(1).upper()
        m = re.search(r'(?i)\b(option|choice)\b\s*([A-Ea-e])\b', t)
        if m: return m.group(2).upper()
        m = re.search(r'(?<![A-Za-z])([A-Ea-e])(?![A-Za-z])', t)
        if m: return m.group(1).upper()
        m = re.search(r'[选择项]\s*([1-9])\b', t)
        if m: return m.group(1)
        m = re.search(r'(?<!\d)([1-9])(?!\d)', t)
        if m: return m.group(1)
        return None

    def _extract_yes_no_label(self, text: str) -> Optional[str]:
        if not text:
            return None
        t = text.strip().lower()
        neg_en = [" no ", " not ", "ruled out", "rule out", "unlikely", "negative for",
                  "does not", "isn't", "is not", "cannot be", "excluded", "absence of"]
        neg_zh = ["否", "不是", "不支持", "不符合", "未见", "排除", "无证据支持", "不考虑", "阴性", "不存在"]
        if any(k in t for k in neg_en) or any(k in text for k in neg_zh):
            return "no"
        unc_en = ["cannot confirm", "cannot be confirmed", "cannot rule out", "likely", "possible", "probable",
                  "suggestive", "suspicious", "recommend", "need further", "consider", "may", "might", "appears"]
        unc_zh = ["不能确诊", "可能", "考虑", "提示", "需结合", "建议进一步", "高度怀疑", "倾向于", "不能排除", "待排",
                  "疑似"]
        if any(k in t for k in unc_en) or any(k in text for k in unc_zh):
            return None
        pos_en = [" yes", "confirmed", "definite", "diagnosed", " is ", "positive for", "consistent with",
                  "evidence of"]
        pos_zh = ["是", "确诊", "明确", "诊断为", "阳性", "符合", "证据支持"]
        if any(k in t for k in pos_en) or any(k in text for k in pos_zh):
            return "yes"
        return None

    def _normalize_closed_label(self, text: str) -> Optional[str]:
        opt = self._extract_choice_label(text)
        if opt:
            return opt
        yn = self._extract_yes_no_label(text)
        if yn:
            return yn
        return None

    # ---------------- judge ----------------

    def _infer_polarity(self, text: str) -> str:
        """
        Return one of {'yes', 'no', 'soft_yes',' soft_no ',' unclain ',' none '}
        -Strong negation first, followed by strong affirmation; The yes/no with fuzzy words are classified as soft_yes/soft_no, respectively
        """
        if not text:
            return "none"
        t = text.lower()
        zh = text

        neg_strong_en = ["clearly not", "definitely not", "ruled out", "excluded", "no evidence of", "negative for"]
        neg_strong_zh = ["明确排除", "明确不是", "肯定不是", "已排除", "无证据支持", "阴性为主"]
        pos_strong_en = ["confirmed", "definite", "diagnosed", "positive for"]
        pos_strong_zh = ["确诊", "明确诊断", "阳性", "肯定是"]

        hedging = self._has_hedging(text)

        if any(w in t for w in neg_strong_en) or any(w in zh for w in neg_strong_zh):
            return "no"
        if any(w in t for w in pos_strong_en) or any(w in zh for w in pos_strong_zh):
            return "yes"

        # weak
        pos_weak_en = ["consistent with", "suggestive of", "evidence of", "compatible with"]
        pos_weak_zh = ["符合", "提示", "倾向", "考虑为", "可考虑", "疑似"]
        neg_weak_en = ["unlikely", "less likely", "not typical of"]
        neg_weak_zh = ["不太像", "不典型", "可能不是"]

        if hedging and (any(w in t for w in pos_weak_en) or any(w in zh for w in pos_weak_zh)):
            return "soft_yes"
        if hedging and (any(w in t for w in neg_weak_en) or any(w in zh for w in neg_weak_zh)):
            return "soft_no"

        hard = self._extract_yes_no_label(text)
        if hard == "yes":
            return "yes"
        if hard == "no":
            return "no"

        return "uncertain" if hedging else "none"

    def _alignment(self, ref_pol: str, pred_pol: str) -> str:
        """Provide consistency labels based on polarity at both ends"""
        if ref_pol == "none" and pred_pol == "none":
            return "neutral"
        same = {("yes", "yes"), ("no", "no"), ("soft_yes", "soft_yes"), ("soft_no", "soft_no"),
                ("yes", "soft_yes"), ("soft_yes", "yes"), ("no", "soft_no"), ("soft_no", "no"),
                ("uncertain", "uncertain")}
        if (ref_pol, pred_pol) in same:
            return "agree"
        lean_agree = {("soft_yes", "uncertain"), ("uncertain", "soft_yes"),
                      ("yes", "uncertain"), ("uncertain", "yes"),
                      ("soft_no", "uncertain"), ("uncertain", "soft_no"),
                      ("no", "uncertain"), ("uncertain", "no")}
        if (ref_pol, pred_pol) in lean_agree:
            return "lean_agree"
        disagree = {("yes", "no"), ("no", "yes"), ("soft_yes", "soft_no"), ("soft_no", "soft_yes")}
        if (ref_pol, pred_pol) in disagree:
            return "disagree"
        return "lean_disagree"

    # ---------------- main ----------------

    def evaluate_response(
            self,
            final_summary: str,
            reference_answer: str = "",
            q_type: str = "open",
            original_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 1) question type select
        if q_type in (None, "", "auto"):
            effective_closed = self._is_strict_closed_ref(reference_answer)
        elif q_type == "closed":
            effective_closed = self._is_strict_closed_ref(reference_answer)
        else:
            effective_closed = False

        # 2) closed-end
        if effective_closed:
            ref = self._normalize_ref_label(reference_answer)
            pred = self._normalize_closed_label(final_summary)
            acc = 1.0 if (pred is not None and ref is not None and pred == ref) else 0.0
            return {
                "type": "closed",
                "accuracy": round(acc, 2),  # 1.00 or 0.00
                "label_ref": ref,
                "label_pred": pred
            }

        # 3) open-end
        ref_pol = self._infer_polarity(reference_answer or "")
        pred_pol = self._infer_polarity(final_summary or "")
        align = self._alignment(ref_pol, pred_pol)

        ref_block = f"Reference (may be hedged; use to calibrate scoring):\n{reference_answer}\n\n" if reference_answer else ""
        prompt = f"""
    You will evaluate an open-ended medical answer with STRICT criteria.

    {ref_block}Model Final Summary:
    {final_summary}

    Scoring rules (be strict, numeric values with up to two decimals):
    - helpfulness (1-10)
    - relevance (1-10)
    - accuracy (1-10)
    - level_of_details (1-10)

    Return strict JSON ONLY with numeric values (not strings). Example:
    {{
      "type": "open",
      "helpfulness": 7.25,
      "relevance": 6.75,
      "accuracy": 6.50,
      "level_of_details": 7.00,
      "overall": 6.88
    }}
    """
        messages = [SystemMessage(content=self.get_system_prompt()), HumanMessage(content=prompt)]
        result = self.llm._generate(messages)
        raw = (result.generations[0].message.content or "").strip()
        try:
            data = json.loads(raw)

            helpfulness = round(float(data.get("helpfulness", 6)), 2)
            relevance = round(float(data.get("relevance", 6)), 2)
            accuracy = round(float(data.get("accuracy", 6)), 2)
            lod = round(float(data.get("level_of_details", 6)), 2)

            if "overall" in data:
                overall = round(float(data.get("overall", 6)), 2)
            else:

                overall = round((helpfulness + relevance + accuracy + lod) / 4.0, 2)

            return {
                "type": "open",
                "helpfulness": helpfulness,
                "relevance": relevance,
                "accuracy": accuracy,
                "level_of_details": lod,
                "overall": overall,
                "alignment": align,
                "ref_polarity": ref_pol,
                "pred_polarity": pred_pol
            }
        except Exception:

            return {
                "type": "open",
                "helpfulness": 6.00, "relevance": 6.00, "accuracy": 6.00, "level_of_details": 6.00, "overall": 6.00,
                "alignment": align, "ref_polarity": ref_pol, "pred_polarity": pred_pol
            }


class ImgExpAgent(BaseAgent):
    """img rag exp Agent"""

    def get_system_prompt(self) -> str:
        return """You extract **transferable imaging experience** from (evidence, analysis, outcome).
- Summarize reusable imaging patterns and pitfalls
- Be concise (5-8 lines)
- Avoid dataset/memo-specific identifiers"""

    def extract_experience(self, context: dict, evaluation: dict) -> Dict[str, Any]:
        robot = context.get('imaging_evidence', '')
        analysis = context.get('imaging_analysis', '')
        final_summary = context.get('final_summary', '')
        prompt = f"""
Imaging Evidence (robot): 
{robot}

Imaging Analysis (doctor):
{analysis}

Final Summary:
{final_summary}

Extract transferable imaging experience in 5-8 bullet points (objective features, archetypal patterns, pitfalls, quality issues).
"""
        messages = [SystemMessage(content=self.get_system_prompt()), HumanMessage(content=prompt)]
        result = self.llm._generate(messages).generations[0].message.content
        return {"content": result, "type": "imaging", "score": evaluation}


class ClinExpAgent(BaseAgent):
    """rag cline exp Agent"""

    def get_system_prompt(self) -> str:
        return """You extract **transferable clinical experience** from the reasoning trace and final outcome.
- Summarize diagnostic heuristics, red flags, lab/threshold rules
- Keep it concise and reusable"""

    def extract_experience(self, context: dict, evaluation: dict) -> Dict[str, Any]:
        initial = context.get('initial_response', '')
        verification = context.get('verification_response', '')
        final_summary = context.get('final_summary', '')
        prompt = f"""
Initial Response:
{initial}

Verification:
{verification}

Final Summary:
{final_summary}

Write 5-8 bullet points of clinical reasoning heuristics (e.g., when to suspect, quick rule-outs, guideline thresholds).
"""
        messages = [SystemMessage(content=self.get_system_prompt()), HumanMessage(content=prompt)]
        result = self.llm._generate(messages).generations[0].message.content
        return {"content": result, "type": "clinical", "score": evaluation}


# =========================
# Agent manage
# =========================

class AgentManager:
    """Unified management and orchestration of various agents"""

    def __init__(self, config: dict):
        self.config = config
        self.vector_db = VectorDatabase(config)
        self.student = StudentAgent(config, self.vector_db)
        self.teacher = TeacherAgent(config, 'teacher')
        self.imaging_robot = ImagingRobotAgent(config, 'imaging_robot')
        self.imaging_doctor = ImagingDoctorAgent(config, 'imaging_doctor')
        self.judicial = JudicialAgent(config, 'judicial')
        self.img_exp_agent = ImgExpAgent(config, 'img_exp')
        self.clin_exp_agent = ClinExpAgent(config, 'clin_exp')

    def set_tools(self, tools: List[Any]):
        self.student.tools = tools or []

    def retrieve_experiences(self, query: str, images: List[str] = None, top_k: int = 5) -> Tuple[
        List[dict], List[dict]]:
        clin = self.vector_db.search('clinical', query, top_k=top_k)
        img = self.vector_db.search('imaging', query, top_k=top_k)
        clin_slim = [{"id": x["id"], "content": x.get("content"), "distance": x.get("distance")} for x in clin]
        img_slim = [{"id": x["id"], "content": x.get("content"), "distance": x.get("distance")} for x in img]
        return clin_slim, img_slim

    def store_experience(self, context: dict, evaluation: dict):
        clin = self.clin_exp_agent.extract_experience(context, evaluation)
        self.vector_db.add_experience(
            category='clinical',
            content=clin['content'],
            meta={"eval": evaluation, "kind": "clinical"}
        )
        if context.get('imaging_evidence'):
            img = self.img_exp_agent.extract_experience(context, evaluation)
            self.vector_db.add_experience(
                category='imaging',
                content=img['content'],
                meta={"eval": evaluation, "kind": "imaging"}
            )
