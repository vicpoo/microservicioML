#scripts/check_dataset.py
import pandas as pd

df = pd.read_csv("data/raw/lecturas_ml_training.csv")
print(f"Lecturas totales: {len(df):,}")
print(f"Lotes: {df['id_lote'].nunique()}")
print(f"Normales: {(~df['_es_anomalia']).sum():,} ({(~df['_es_anomalia']).mean()*100:.1f}%)")
print(f"Anomalias: {df['_es_anomalia'].sum():,} ({df['_es_anomalia'].mean()*100:.1f}%)")
for col in ["temperatura_grano", "temperatura_ambiental", "humedad_ambiental", "humedad_grano"]:
    print(f"Nulos {col}: {df[col].isna().sum()}")
print("\nSeveridad:")
print(df["_severidad"].value_counts().to_string())
print("\nTipos de anomalia:")
print(df.loc[df["_es_anomalia"], "_tipo_anomalia"].value_counts().to_string())
print("\nCalidad final por lote:")
print(df.drop_duplicates("id_lote")["_calidad_final_lote"].value_counts().to_string())
