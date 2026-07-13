#app/services/recommender.py
from typing import Dict, List

from app.services.rules import recomendacion_para


class Recommender:
    """Traduce las alertas detectadas a texto accionable (Cuadro 10 del documento de dominio)."""

    def generar(self, alertas: List[Dict]) -> List[Dict[str, str]]:
        if not alertas:
            return [{"tipo": "normal", "texto": recomendacion_para("normal")}]
        vistos = set()
        recomendaciones = []
        for alerta in alertas:
            tipo = alerta["tipo"]
            if tipo in vistos:
                continue
            vistos.add(tipo)
            recomendaciones.append({"tipo": tipo, "texto": recomendacion_para(tipo)})
        return recomendaciones

    def texto_consolidado(self, alertas: List[Dict]) -> str:
        recos = self.generar(alertas)
        return " | ".join(r["texto"] for r in recos)
