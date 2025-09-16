from typing import List, Dict, Optional, Any
import logging
import requests
import xml.etree.ElementTree as ET
import yaml
import re
import os

DEFAULT_CONFIG_PATH = "config.yaml"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PubMedTool:

    def __init__(self, config: dict):
        self.logger = logging.getLogger("PubMedTool")
        tools_cfg = (config or {}).get("tools", {})
        pm_cfg = tools_cfg.get("pubmed", {})
        self.enabled: bool = True
        self.timeout: int = int(pm_cfg.get("timeout", 15))
        self.base_url = pm_cfg.get("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
        self.max_results_default = int(pm_cfg.get("max_results", 3))

    def is_available(self) -> bool:
        return self.enabled

    def _search_pubmed(self, query: str, max_results: int) -> List[str]:
        url = f"{self.base_url}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
            "sort": "relevance"
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        pmids = [e.text for e in root.findall(".//IdList/Id")]
        return pmids

    def _fetch_details(self, pmids: List[str]) -> List[Dict[str, str]]:
        if not pmids:
            return []
        url = f"{self.base_url}efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        results: List[Dict[str, str]] = []
        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            abst_el = article.find(".//Abstract/AbstractText")
            title = title_el.text if title_el is not None else "(No Title)"
            abstract = abst_el.text if abst_el is not None else "(No Abstract)"
            results.append({"title": title, "abstract": abstract})
        return results

    def search(self, query: str, max_results: int = None) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "summary": "PubMed tool is disabled."}
        if max_results is None:
            max_results = self.max_results_default

        try:
            pmids = self._search_pubmed(query, max_results=max_results)
            details = self._fetch_details(pmids)
            bullets = []
            for i, item in enumerate(details, 1):
                bullets.append(f"{i}. {item['title']}\n{item['abstract'][:400]}...")
            summary = "Top PubMed hits:\n" + "\n\n".join(bullets) if bullets else "No relevant articles found."
            return {"enabled": True, "pmids": pmids, "results": details, "summary": summary}
        except Exception as e:
            self.logger.exception("PubMed 搜索失败")
            return {"enabled": True, "error": str(e), "summary": "PubMed search error."}


class ToolManager:

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.logger = logging.getLogger("ToolManager")
        self._tools: Dict[str, Any] = {}

        pubmed = PubMedTool(self.config)
        if pubmed.is_available():
            self._tools["pubmed"] = pubmed

        self.max_results = self.config.get("tools", {}).get("pubmed", {}).get("max_results", 3)

    def get_tool(self, name: str) -> Optional[Any]:
        return self._tools.get(name)




def should_call_tool(teacher_doubts: str, verification_text: str, threshold: float = 0.5) -> bool:

    text = f"{teacher_doubts or ''}\n{verification_text or ''}"
    lower = text.lower()

    uncertainty_keywords = [
        "not sure","uncertain","unclear","might be","could be","possibly","maybe",
        "need more information","require further","ambiguous","inconclusive",
        "不确定","不清楚","尚不清楚","可能","大概","也许","需要进一步","信息不足","无法确定","有待明确"
    ]
    hits = sum(1 for w in uncertainty_keywords if w in lower)

    q_count = text.count("?") + text.count("？")

    uncertainty_score = min(hits / 4.0, 1.0)
    complexity_score = min(q_count / 3.0, 1.0)
    score = 0.5 * (uncertainty_score + complexity_score)
    return score >= threshold
