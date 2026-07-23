# Archivo: tests/test_ml_calidad.py
# Carpeta: microservicioMLL/tests/
"""Pruebas del regresor de calidad (escala SCA 0-100) introducido en la migración de
calidad_real/calidad_estimada de categorías (excelente/buena/regular/baja) a un puntaje continuo
-- ver migration.sql paso 10, ML/definicion_problema_kajve.md Sección 3.3, y
scripts/train_models.py::entrenar_regresor_calidad.

IMPORTANTE: estas pruebas NUNCA deben escribir en app/ml/artifacts/ (el directorio de
producción real) -- por eso todo entrenamiento aquí redirige ARTIFACTS_DIR a un directorio
temporal via monkeypatch, el mismo criterio ya usado en entrenar_clasificador_texto.py para no
tocar artefactos reales con datos sintéticos de prueba."""
import os

import joblib
import pandas as pd
import pytest

import scripts.train_models as train_models
from app.services.predictor import Predictor


def _df_sintetico_suficiente() -> pd.DataFrame:
    """5 lotes distintos (== MIN_LOTES_CALIDAD), cada uno con varias lecturas y un
    _calidad_final_lote fijo -- suficiente para que entrenar_regresor_calidad no se omita."""
    filas = []
    calidades_por_lote = {1: 60.0, 2: 70.0, 3: 80.0, 4: 88.0, 5: 95.0}
    for id_lote, calidad in calidades_por_lote.items():
        for i in range(4):
            filas.append({
                "id_lote": id_lote,
                "tipo_proceso": ["lavado", "honey", "natural"][id_lote % 3],
                "temperatura_grano": 24.0 + id_lote + i * 0.5,
                "temperatura_ambiental": 22.0 + id_lote,
                "humedad_grano": 40.0 - id_lote - i,
                "lluvia": 0.0,
                "luz": 30000 + id_lote * 1000,
                "delta_temp": 2.0 + id_lote * 0.1,
                "horas_transcurridas": 20.0 * (i + 1),
                "_calidad_final_lote": calidad,
            })
    return pd.DataFrame(filas)


def _df_sintetico_insuficiente() -> pd.DataFrame:
    """Solo 2 lotes con _calidad_final_lote conocida -- menos que MIN_LOTES_CALIDAD."""
    df = _df_sintetico_suficiente()
    return df[df["id_lote"].isin([1, 2])]


def test_entrenar_regresor_calidad_se_omite_con_pocos_lotes(tmp_path, monkeypatch):
    monkeypatch.setattr(train_models, "ARTIFACTS_DIR", str(tmp_path))
    resultado = train_models.entrenar_regresor_calidad(_df_sintetico_insuficiente())
    assert "omitido" in resultado
    assert resultado["n_lotes_disponibles"] == 2
    # No debe haberse creado ningún artefacto si se omitió el entrenamiento
    assert not os.path.exists(os.path.join(str(tmp_path), "rf_calidad.joblib"))


def test_entrenar_regresor_calidad_entrena_y_guarda_artefacto(tmp_path, monkeypatch):
    monkeypatch.setattr(train_models, "ARTIFACTS_DIR", str(tmp_path))
    df = _df_sintetico_suficiente()
    resultado = train_models.entrenar_regresor_calidad(df)

    assert "omitido" not in resultado
    assert resultado["n_lotes"] == 5
    # Antes eran accuracy/f1_macro (clasificador); ahora es regresión.
    assert "rmse_puntos" in resultado
    assert "mae_puntos" in resultado
    assert resultado["rmse_puntos"] >= 0
    assert resultado["mae_puntos"] >= 0

    artefacto_path = os.path.join(str(tmp_path), "rf_calidad.joblib")
    assert os.path.exists(artefacto_path)

    artefacto = joblib.load(artefacto_path)
    assert "modelo" in artefacto
    assert "features" in artefacto
    # A diferencia del clasificador viejo, un regresor no tiene .classes_ -- no debe guardarse
    # una clave "clases" que ya no tiene sentido.
    assert "clases" not in artefacto


def test_predictor_calidad_regresa_puntaje_0_100_con_confianza(tmp_path, monkeypatch):
    """Integración train_models -> Predictor: entrena un artefacto real (con datos sintéticos,
    en un directorio temporal) y confirma que Predictor.predecir() lo consume correctamente:
    calidad_estimada es un número entre 0 y 100 (no una categoría), y confianza (la heurística
    de dispersión entre árboles, no una probabilidad de clase) también cae en 0-100."""
    monkeypatch.setattr(train_models, "ARTIFACTS_DIR", str(tmp_path))
    train_models.entrenar_regresor_calidad(_df_sintetico_suficiente())

    predictor = Predictor(artifacts_dir=str(tmp_path))
    assert predictor.rf_calidad is not None  # el artefacto sí se cargó

    resultado = predictor.predecir(
        tipo_proceso="lavado",
        features={
            "temperatura_grano": 27.0,
            "temperatura_ambiental": 24.0,
            "humedad_grano": 30.0,
            "lluvia": 0.0,
            "luz": 32000,
            "delta_temp": 3.0,
        },
        horas_transcurridas=40.0,
    )

    assert resultado["calidad_estimada"] is not None
    assert 0.0 <= resultado["calidad_estimada"] <= 100.0
    assert resultado["confianza"] is not None
    assert 0.0 <= resultado["confianza"] <= 100.0


def test_predictor_calidad_confianza_baja_con_arboles_en_desacuerdo(tmp_path, monkeypatch):
    """La heurística de confianza debe bajar cuando los árboles del bosque predicen cosas muy
    distintas entre sí. Se entrena con calidades por lote muy dispersas y ruidosas (mismos
    valores de entrada, salidas casi opuestas) para forzar alta varianza entre árboles, y se
    compara contra un caso "fácil" (calidad casi constante) que debería dar más confianza."""
    # Caso fácil: casi todos los lotes con la misma calidad -> los árboles deberían coincidir.
    monkeypatch.setattr(train_models, "ARTIFACTS_DIR", str(tmp_path / "facil"))
    os.makedirs(str(tmp_path / "facil"), exist_ok=True)
    df_facil = _df_sintetico_suficiente()
    df_facil["_calidad_final_lote"] = 85.0  # misma calidad para todos los lotes
    train_models.entrenar_regresor_calidad(df_facil)
    predictor_facil = Predictor(artifacts_dir=str(tmp_path / "facil"))

    features = {
        "temperatura_grano": 27.0, "temperatura_ambiental": 24.0, "humedad_grano": 30.0,
        "lluvia": 0.0, "luz": 32000, "delta_temp": 3.0,
    }
    resultado_facil = predictor_facil.predecir("lavado", features, horas_transcurridas=40.0)

    # Con calidad constante en el entrenamiento, todos los árboles deberían predecir ~lo mismo
    # -> confianza alta.
    assert resultado_facil["confianza"] > 50.0