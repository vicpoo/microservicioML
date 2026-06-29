#scripts/check_dataset.py
import pandas as pd

df = pd.read_csv("data/raw/lecturas_ml_training.csv")
print(f"Lecturas totales: {len(df):,}")
print(f"Normales: {(~df['_es_anomalia']).sum():,} ({(~df['_es_anomalia']).mean()*100:.1f}%)")
print(f"Anomalias: {df['_es_anomalia'].sum():,} ({df['_es_anomalia'].mean()*100:.1f}%)")
print(f"Nulos temperatura: {df['temperatura'].isna().sum()}")
print(f"Nulos humedad: {df['humedad'].isna().sum()}")
print("\nTipos de anomalia:")
print(df.loc[df["_es_anomalia"], "_tipo_anomalia"].value_counts().to_string())
