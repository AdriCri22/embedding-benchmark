"""
Generation implementation for the embedding benchmark suite.
"""
import httpx
import json
from typing import List, Tuple, Dict, Any

from ..interface import GenerationInterface
from embedding_benchmark.metrics import get_metrics


class GenerationImpl(GenerationInterface):
    """Implementation of LLM generation and judging."""
    
    def __init__(self):
        self._metrics = get_metrics()
    
    def generate_answer(
        self, 
        model_cfg: Dict[str, Any], 
        query: str, 
        context_chunks: List[str]
    ) -> str:
        """Generate answer using LLM local via llama.cpp /v1/chat/completions."""
        context = "\n\n---\n\n".join(context_chunks[:5])
        prompt = f"""Context:
{context}

Question: {query}

Responde DIRECTAMENTE basándote SOLO en el contexto. Si el contexto no tiene la información, di exactamente: "El contexto no contiene esta información". Sé conciso. Responde en el mismo idioma de la pregunta."""

        resp = httpx.post(
            f"http://localhost:{model_cfg['server_port']}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Eres un asistente experto. Responde directo y conciso basándote solo en el contexto proporcionado. No des rodeos."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 512
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    
    def judge_answer(
        self, 
        judge_cfg: Dict[str, Any], 
        query: str, 
        answer: str, 
        expected_answer: str, 
        lang: str
    ) -> Tuple[int, str]:
        """Evaluate answer with LLM judge. Returns (score, reasoning)."""
        
        prompt = f"""{self._get_judge_rubric()}

IDIOMA: {lang}
PREGUNTA: {query}
ESPERADA: {expected_answer}
RESPUESTA: {answer}

Evalúa y devuelve SOLO JSON."""

        resp = httpx.post(
            f"http://localhost:{judge_cfg['server_port']}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Eres un evaluador estricto. Devuelve SOLO JSON válido: {\"score\": 1-5, \"reasoning\": \"...\"}"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 256
            },
            timeout=60
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                score = int(data.get("score", 1))
                reasoning = str(data.get("reasoning", ""))
                return max(1, min(5, score)), reasoning
        except Exception as e:
            print(f"      Judge parse error: {e}")
        return 1, "Error parsing judge response"
    
    def _get_judge_rubric(self) -> str:
        return """
Eres un evaluador experto. Clasifica la respuesta del 1 al 5 según esta rúbrica:

**5 - EXCELENTE**: Respuesta directa, completa y precisa. Resuelve la intención sin rodeos. Cita fuentes.
**4 - BUENA**: Correcta y mayormente directa. Pequeños rodeos u omisión menor. Cita fuentes.
**3 - ACEPTABLE**: Parcialmente correcta. Rodeos, evasiva en parte, mezcla info relevante/irrelevante.
**2 - DEFICIENTE**: Evasiva, genérica, no resuelve la intención. No cita fuentes o cita incorrectas.
**1 - INÚTIL**: No responde. Alucina, inventa, o dice "no sé" sin intentar ayudar.

CRITERIOS DE PENALIZACIÓN:
- Respuesta evasiva ("depende", "consulte documentación", "es complejo") → -1 a -2
- No cita la fuente esperada → -1
- Información incorrecta/alucinada → -2
- Rodeos innecesarios antes de responder → -1
- No responde en el idioma de la pregunta → -1

Devuelve SOLO JSON: {"score": 1-5, "reasoning": "breve justificación"}
"""