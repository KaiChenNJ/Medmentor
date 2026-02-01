"""
MedMentor Main
"""

import os
import json
import yaml
import logging
import uuid
import time
import base64
import re
from datetime import datetime
from typing import List, Dict, Optional, Generator

from flask import Flask, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename

from agents import AgentManager
from tools import ToolManager

DEFAULT_CONFIG_PATH = "config.yaml"



def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format=fmt)
    if log_cfg.get("file"):
        os.makedirs(os.path.dirname(log_cfg["file"]), exist_ok=True)
        fh = logging.FileHandler(log_cfg["file"])
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)

# ----------------------------
# language detect
# ----------------------------

def detect_language(text: str) -> str:
    return 'zh' if re.search(r'[\u4e00-\u9fff]', text or "") else 'en'

def build_lang_hint(lang: str, role: str) -> str:
    if lang == 'zh':
        common = "请用中文回答，并严格围绕患者的原始提问作答；若信息缺失，请先用1-2句提出澄清问题。"
        if role == 'teacher':
            return "请用中文提出针对性的核验问题，仅围绕患者原始提问与当前推理，不引入无关话题。"
        if role == 'imaging':
            return "请用中文，仅报告与患者提问相关的客观影像征象，避免诊断性推断。"
        return common
    else:
        common = "Please answer in English and stay strictly focused on the patient's original question; if key info is missing, ask 1–2 brief clarifying questions first."
        if role == 'teacher':
            return "Ask concise, targeted verification questions in English strictly about the original query and current reasoning only."
        if role == 'imaging':
            return "In English, report only objective imaging findings relevant to the patient's question; avoid interpretive diagnoses."
        return common



class MedMentorSystem:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("MedMentor")
        self.agent_manager = AgentManager(config)
        self.tool_manager = ToolManager(config)

    def _json_line(self, data: dict) -> bytes:
        return ("data: " + json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")

    def _stream_text(self, text: str, chunk: int = 60) -> Generator[str, None, None]:
        if not text:
            return
        i = 0
        n = max(1, chunk)
        while i < len(text):
            yield text[i:i+n]
            i += n

    def process_consultation(
        self, query: str, images: List[str], mode: str = "consultation", reference_answer: Optional[str] = None
    ) -> Generator[bytes, None, None]:
        session_id = str(uuid.uuid4())
        lang = detect_language(query)
        hint_general  = build_lang_hint(lang, 'general')
        hint_teacher  = build_lang_hint(lang, 'teacher')
        hint_imaging  = build_lang_hint(lang, 'imaging')

        context: Dict[str, str] = {
            "session_id": session_id,
            "query": query,
            "lang": lang,
            "timestamp": datetime.now().isoformat()
        }

        try:
            yield self._json_line({"type": "system", "content": "Initializing diagnostic process..."})

            # exp rag
            top_k = self.config["retrieval"].get("top_k", 5)
            clin_exp, img_exp = self.agent_manager.retrieve_experiences(query=query, images=images, top_k=top_k)

            # Image
            imaging_evidence, imaging_analysis = "", ""
            if images:
                yield self._json_line({"type": "system", "content": "Analyzing medical images..."})
                yield self._json_line({"type": "agent", "agent": "Imaging Robot Agent", "content": ""})

                imaging_evidence = self.agent_manager.imaging_robot.extract_evidence(
                    images=images, query=query, lang_hint=hint_imaging
                )
                for chunk in self._stream_text(imaging_evidence):
                    yield self._json_line({"type": "agent", "agent": "Imaging Robot Agent", "content": chunk})

                yield self._json_line({"type": "agent", "agent": "Imaging Doctor Agent", "content": ""})
                imaging_analysis = self.agent_manager.imaging_doctor.analyze_images(
                    robot_evidence=imaging_evidence, query=query, images=images, lang_hint=hint_general
                )
                for chunk in self._stream_text(imaging_analysis):
                    yield self._json_line({"type": "agent", "agent": "Imaging Doctor Agent", "content": chunk})

            context["imaging_evidence"] = imaging_evidence
            context["imaging_analysis"] = imaging_analysis

            # student initial
            yield self._json_line({"type": "system", "content": "Generating initial response..."})
            yield self._json_line({"type": "agent", "agent": "Student", "content": ""})
            initial_response = self.agent_manager.student.initial_response(
                query=query,
                imaging_evidence=imaging_evidence,
                imaging_analysis=imaging_analysis,
                retrieved_experiences=(clin_exp + img_exp),
                lang_hint=hint_general
            )
            for chunk in self._stream_text(initial_response):
                yield self._json_line({"type": "agent", "agent": "Student", "content": chunk})
            context["initial_response"] = initial_response

            # teacher planning
            yield self._json_line({"type": "agent", "agent": "Teacher", "content": ""})
            verification_questions = self.agent_manager.teacher.plan_verification(
                student_response=initial_response, original_query=query, lang_hint=hint_teacher
            )
            for chunk in self._stream_text(verification_questions):
                yield self._json_line({"type": "agent", "agent": "Teacher", "content": chunk})
            context["verification_questions"] = verification_questions

            # student step verif
            yield self._json_line({"type": "agent", "agent": "Student", "content": ""})
            verification_response = self.agent_manager.student.stepwise_verification(
                original_query=query,
                verification_questions=verification_questions,
                context=context,
                lang_hint=hint_general
            )
            for chunk in self._stream_text(verification_response):
                yield self._json_line({"type": "agent", "agent": "Student", "content": chunk})
            context["verification_response"] = verification_response

            # teacher CoT
            yield self._json_line({"type": "agent", "agent": "Teacher", "content": ""})
            cot_questions = self.agent_manager.teacher.cot_questioning(
                student_verification=verification_response, original_query=query, lang_hint=hint_teacher
            )
            for chunk in self._stream_text(cot_questions):
                yield self._json_line({"type": "agent", "agent": "Teacher", "content": chunk})
            context["cot_questions"] = cot_questions

            #tools
            tool_results_text = ""
            if self.agent_manager.student.should_use_tools(cot_questions, verification_response):
                yield self._json_line({"type": "system", "content": "Consulting external tools..."})
                pubmed = self.tool_manager.get_tool("pubmed")
                if pubmed:
                    res = pubmed.search(query, max_results=self.tool_manager.max_results)
                    tool_results_text = res.get("summary", "")
                    yield self._json_line({"type": "tool", "tool": "PubMed", "content": json.dumps(res, ensure_ascii=False)})
                else:
                    yield self._json_line({"type": "tool", "tool": "PubMed", "content": "Tool not available."})

            # student summary
            yield self._json_line({"type": "agent", "agent": "Student", "content": ""})
            final_summary = self.agent_manager.student.final_summary(
                query=query, context=context, tool_results=tool_results_text, lang_hint=hint_general
            )
            for chunk in self._stream_text(final_summary):
                yield self._json_line({"type": "agent", "agent": "Student", "content": chunk})
            context["final_summary"] = final_summary

            # Eval
            yield self._json_line({"type": "system", "content": "Evaluating and extracting experience..."})
            # q_type = "closed" if (mode == "training" and reference_answer) else "open"
            q_type = "auto"

            evaluation = self.agent_manager.judicial.evaluate_response(
                final_summary=final_summary, reference_answer=(reference_answer or ""), q_type=q_type
            )

            min_score = self.config["retrieval"].get("min_score_for_experience", 6)
            allow_store = True
            if evaluation.get("type") == "open":
                allow_store = evaluation.get("overall", 6) >= min_score

            # cline exp
            clin_exp_obj = self.agent_manager.clin_exp_agent.extract_experience(context, evaluation)
            yield self._json_line({"type": "experience", "agent": "Clinical Experience Agent", "content": clin_exp_obj["content"]})
            if allow_store:
                self.agent_manager.vector_db.add_experience(
                    category="clinical",
                    content=clin_exp_obj["content"],
                    meta={"eval": evaluation, "kind": "clinical", "session_id": session_id}
                )

            # Imaging experience (written only with imaging evidence)
            if context.get("imaging_evidence"):
                img_exp_obj = self.agent_manager.img_exp_agent.extract_experience(context, evaluation)
                yield self._json_line({"type": "experience", "agent": "Imaging Experience Agent", "content": img_exp_obj["content"]})
                if allow_store:
                    self.agent_manager.vector_db.add_experience(
                        category="imaging",
                        content=img_exp_obj["content"],
                        meta={"eval": evaluation, "kind": "imaging", "session_id": session_id}
                    )

            # judge
            yield self._json_line({"type": "evaluation", "content": evaluation})
            yield self._json_line({"type": "system", "content": "Diagnostic process completed."})

        except Exception as e:
            self.logger.exception("An error occurred while handling the consultation")
            yield self._json_line({"type": "error", "content": f"An error occurred: {str(e)}"})

# ----------------------------
# Flask
# ----------------------------

def create_app() -> Flask:
    config_path = os.environ.get("MEDMENTOR_CONFIG", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)
    setup_logging(config)

    server_cfg = config.get("server", {})
    upload_folder = server_cfg.get("upload_folder", "./uploads")
    os.makedirs(upload_folder, exist_ok=True)

    static_root = os.path.abspath(server_cfg.get("static_root", "static"))
    os.makedirs(static_root, exist_ok=True)

    app = Flask(__name__, static_folder=static_root, static_url_path="")
    app.config["JSON_AS_ASCII"] = False
    app.config["MAX_CONTENT_LENGTH"] = int(server_cfg.get("max_content_length", 16 * 1024 * 1024))

    system = MedMentorSystem(config)

    @app.route("/")
    def index():
        index_path = os.path.join(app.static_folder, "index.html")
        if os.path.exists(index_path):
            return send_from_directory(app.static_folder, "index.html")

        return """
        <h2>MedMentor backend is running ✅</h2>
        <p>Please place your front-end file at <code>./static/index.html</code> and refresh.</p>
        <ul>
          <li>POST <code>/consult</code></li>
          <li>POST <code>/upload</code></li>
          <li>GET  <code>/health</code></li>
        </ul>
        """, 200


    # ---- API: con ----
    @app.route("/consult", methods=["POST"])
    def consult():
        payload = request.get_json(force=True, silent=True) or {}
        query = (payload.get("query") or "").strip()
        images = payload.get("images", []) or []
        mode = payload.get("mode", config.get("system", {}).get("default_mode", "consultation"))
        reference_answer = payload.get("reference_answer", "")

        if not query:
            return jsonify({"error": "Empty query"}), 400

        def generate():
            for chunk in system.process_consultation(query=query, images=images, mode=mode, reference_answer=reference_answer):
                yield chunk

        return Response(generate(), mimetype="text/event-stream")

    # ---- API: upload----
    @app.route("/upload", methods=["POST"])
    def upload():
        files = request.files.getlist("files")
        results = []
        for f in files:
            filename = secure_filename(f.filename)
            path = os.path.join(upload_folder, filename)
            f.save(path)
            with open(path, "rb") as imgf:
                b64 = base64.b64encode(imgf.read()).decode("utf-8")
            mime = "image/jpeg"
            low = filename.lower()
            if low.endswith(".png"):
                mime = "image/png"
            elif low.endswith(".gif"):
                mime = "image/gif"
            elif low.endswith(".bmp"):
                mime = "image/bmp"
            elif low.endswith(".tif") or low.endswith(".tiff"):
                mime = "image/tiff"
            results.append({"filename": filename, "base64": f"data:{mime};base64,{b64}"})
        return jsonify({"files": results})

    # --
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    return app

app = create_app()

#  `python main.py` 
if __name__ == "__main__":
    cfg = load_config(os.environ.get("MEDMENTOR_CONFIG", DEFAULT_CONFIG_PATH))
    setup_logging(cfg)
    server = cfg.get("server", {})
    host = server.get("host", "0.0.0.0")
    port = int(server.get("port", 5000))
    debug = bool(server.get("debug", True))
    print("=" * 60)
    print(f"🚀 MedMentor server at http://{host}:{port}")
    print("🗂  Static root:", os.path.abspath(server.get("static_root", "static")))
    print("📁 Upload dir :", os.path.abspath(server.get("upload_folder", "./uploads")))
    print("🔧 Debug mode :", debug)
    print("=" * 60)
    app.run(host=host, port=port, debug=debug, threaded=True)
