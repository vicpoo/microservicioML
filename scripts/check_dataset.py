#scripts/check_dataset.py
"""Inspección rápida del dataset REAL recolectado con scripts/recolectar_datos_reales.py
(ya no del CSV sintético de scripts/generar_dataset.py, deprecado)."""
import pandas as pd

df = pd.read_csv("data/raw/lecturas_reales_entrenamiento.csv")
print(f"Lecturas totales: {len(df):,}")
print(f"Lotes: {df['id_lote'].nunique()}")
if len(df):
    print(f"Normales: {(~df['_es_anomalia']).sum():,} ({(~df['_es_anomalia']).mean()*100:.1f}%)")
    print(f"Anomalias: {df['_es_anomalia'].sum():,} ({df['_es_anomalia'].mean()*100:.1f}%)")
for col in ["temperatura_grano", "temperatura_ambiental", "humedad_grano"]:
    print(f"Nulos {col}: {df[col].isna().sum()}")
print(f"Filas con horas_restantes conocida: {df['horas_restantes'].notna().sum()}")
print(f"Filas con calidad_final conocida: {df['_calidad_final_lote'].notna().sum()}")
if len(df):
    print("\nSeveridad:")
    print(df["_severidad"].value_counts().to_string())
    print("\nTipos de anomalia:")
    print(df.loc[df["_es_anomalia"], "_tipo_anomalia"].value_counts().to_string())
    print("\nCalidad final por lote (solo lotes con dato real, escala SCA 0-100):")
    # value_counts() ya no tiene sentido con un puntaje continuo (migration.sql paso 10); describe()
    # da min/max/media/percentiles, más útil para ver si el rango reportado es razonable.
    print(df.dropna(subset=["_calidad_final_lote"]).drop_duplicates("id_lote")["_calidad_final_lote"].describe().to_string())
