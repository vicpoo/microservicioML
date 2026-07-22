#scripts/monitorear_modelos.py
"""
Paso 12 del pipeline de ML — Monitoreo y reentrenamiento (versión cron/terminal).

Corre ML/monitoreo.py contra la BD real (usa DATABASE_URL de .env, igual que el resto del
servicio) e imprime un reporte legible. Pensado para correr periódicamente en el SERVIDOR
donde vive el microservicio desplegado (ej. cron semanal) -- Cowork no tiene acceso a ese
servidor, así que este script no se programa desde aquí, se agenda en el cron del servidor:

    # cada lunes 6am, revisa salud de los modelos
    0 6 * * 1 cd /ruta/al/proyecto && .venv/bin/python scripts/monitorear_modelos.py

Sale con código 1 si `necesita_reentrenamiento` es True, para que el cron pueda encadenar una
alerta (ej. "|| mail -s 'kajve: revisar modelos' tu@correo") sin tener que parsear el texto.

Este script NO reentrena nada por sí solo -- solo diagnostica. Ver ML/monitoreo.py para la
razón (el paso 10 ya mostró que un modelo "ganador" en validación puede fallar en datos
reales; reentrenar sin revisión humana sería peligroso).
"""
import sys

from app.models.database import SessionLocal
from ML import monitoreo


def main() -> int:
    db = SessionLocal()
    try:
        reporte = monitoreo.resumen_salud(db)
    finally:
        db.close()

    reentrenamiento = reporte["reentrenamiento"]
    alertas = reporte["monitoreo_alertas"]
    datos = reentrenamiento["disponibilidad_datos"]
    desempeno = reentrenamiento["desempeno_produccion"]

    print(f"=== Salud de modelos kajve — {reporte['fecha_reporte']} ===\n")

    print("-- Datos --")
    print(f"Último entrenamiento: {datos['fecha_ultimo_entrenamiento'] or 'desconocido'} "
          f"({datos['n_filas_train_ultimo_entrenamiento']} filas)")
    print(f"Lecturas hoy: {datos['n_filas_lecturas_ambientales_hoy']} "
          f"({datos['filas_nuevas_desde_ultimo_entrenamiento']} nuevas desde entonces, "
          f"umbral {datos['umbral_filas_nuevas']})")
    print(f"Lotes con retroalimentación real: {datos['n_lotes_con_retroalimentacion_real']} "
          f"(faltan {datos['lotes_faltantes_para_tiempo_restante']} para tiempo_restante, "
          f"{datos['lotes_faltantes_para_calidad']} para calidad)")

    print("\n-- Desempeño en producción (predicciones vs retroalimentación real) --")
    if "omitido" in desempeno:
        print(f"  omitido: {desempeno['omitido']}")
    else:
        print(f"  {desempeno['n_lotes_comparados']} lote(s) comparados de "
              f"{desempeno['n_lotes_con_retroalimentacion']} con retroalimentación real")
        print(f"  tiempo_restante: {desempeno['tiempo_restante']}")
        print(f"  calidad: {desempeno['calidad']}")

    print(f"\n-- Alertas últimos {alertas['ventana_dias']} días --")
    print(f"  actual={alertas['alertas_ventana_actual']} vs "
          f"anterior={alertas['alertas_ventana_anterior']} "
          f"(razón={alertas['razon_actual_vs_anterior']}) "
          f"posible_drift={alertas['posible_drift']}")

    print(f"\n-- ¿Reentrenar? {'SÍ' if reentrenamiento['necesita_reentrenamiento'] else 'no por ahora'} --")
    for razon in reentrenamiento["razones"]:
        print(f"  - {razon}")

    return 1 if reentrenamiento["necesita_reentrenamiento"] else 0


if __name__ == "__main__":
    sys.exit(main())
