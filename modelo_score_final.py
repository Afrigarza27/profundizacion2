"""
============================================================
PROYECTO: TURISMO CALDAS
ARCHIVO:  modelo_score_final.py
AUTOR:    Sebastián Garzón Henao (Data Scientist)
DESC:     Modelo de Score de Visitas (Random Forest)
          Lee datos directamente desde SQL Server (SSMS 21)
          y exporta todos los artefactos para el Backend.
============================================================

INSTRUCCIONES:
  1. Cambia SERVIDOR por el nombre que ves en SSMS 21
  2. Corre en terminal:  python modelo_score_final.py
  3. Listo — genera 4 archivos para el Backend

REQUISITOS (instalar si no los tienes):
  pip install sqlalchemy pyodbc pandas scikit-learn
"""

import pandas as pd
import numpy as np
import pickle
import json
import urllib
import sys
import pyodbc
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ════════════════════════════════════════════════════════════
#  ⚙️  CONFIGURACIÓN — CAMBIA SOLO ESTAS LÍNEAS
# ════════════════════════════════════════════════════════════

# El nombre que ves en SSMS 21 al conectarte
# Ejemplos: "LAPTOP-ABC123\\SQLEXPRESS"  |  "localhost\\SQLEXPRESS"  |  ".\\SQLEXPRESS"
SERVIDOR      = "REICERX\\SQLEXPRESS"   # ← CAMBIA ESTO

BASE_DE_DATOS = "TurismoCaldas"           # ← deja igual si no cambiaste el nombre

# Carpeta donde se guardarán los archivos del modelo
# Ejemplo Windows: "C:/Users/Sebastián/Desktop/turismo/"
# Deja "./" para guardar en la misma carpeta donde está este script
RUTA_SALIDA   = "./"

# ════════════════════════════════════════════════════════════
#  1. CONEXIÓN A SQL SERVER (detecta el driver automáticamente)
# ════════════════════════════════════════════════════════════
print("=" * 62)
print("  MODELO SCORE DE VISITAS — TURISMO CALDAS")
print("=" * 62)

print("\n[1/7] Conectando a SQL Server...")

# Detectar automáticamente el driver ODBC disponible
drivers_disponibles = [d for d in pyodbc.drivers() if "SQL Server" in d]

if not drivers_disponibles:
    print("\n❌ ERROR: No se encontró ningún driver ODBC para SQL Server.")
    print("   Solución: descarga e instala el driver desde:")
    print("   https://aka.ms/downloadmsodbcsql")
    sys.exit(1)

# Preferir driver 17, luego 18, luego el que sea
driver = next((d for d in drivers_disponibles if "17" in d), None) or \
         next((d for d in drivers_disponibles if "18" in d), None) or \
         drivers_disponibles[0]

print(f"  Driver detectado: {driver}")
print(f"  Servidor:         {SERVIDOR}")
print(f"  Base de datos:    {BASE_DE_DATOS}")

params = urllib.parse.quote_plus(
    f"DRIVER={{{driver}}};"
    f"SERVER={SERVIDOR};"
    f"DATABASE={BASE_DE_DATOS};"
    f"Trusted_Connection=yes;"
    f"TrustServerCertificate=yes;"   # necesario en versiones recientes de SSMS
)

try:
    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        fast_executemany=True
    )
    # Probar la conexión antes de continuar
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("  ✅ Conexión exitosa a SQL Server")
except Exception as e:
    print(f"\n❌ ERROR de conexión: {e}")
    print("\n  Posibles soluciones:")
    print("  1. Verifica el nombre del servidor en SSMS 21")
    print("     (el que aparece al abrir SSMS en 'Nombre del servidor')")
    print("  2. Asegúrate de que SQL Server esté corriendo")
    print("     (busca 'SQL Server Configuration Manager' y verifica)")
    print("  3. Prueba cambiar SERVIDOR a '.\\SQLEXPRESS' o 'localhost\\SQLEXPRESS'")
    sys.exit(1)

# ════════════════════════════════════════════════════════════
#  2. CARGAR DATOS DESDE SQL SERVER
# ════════════════════════════════════════════════════════════
print("\n[2/7] Cargando datos desde SQL Server...")

# Consulta principal: visitas + lugares + usuarios en un solo JOIN
# Esto es más eficiente que cargar 3 tablas separadas
QUERY_PRINCIPAL = """
SELECT
    v.visita_id,
    v.lugar_id,
    v.usuario_id,
    v.mes,
    v.dia_semana,
    v.es_festivo,
    v.duracion_horas,
    v.calificacion_dada,
    v.comentario_sentimiento,
    v.clima_dia_visita,
    v.gasto_total_cop,
    v.fecha_visita,

    l.categoria           AS categoria_lugar,
    l.altitud_msnm,
    l.precio_entrada_cop,
    l.capacidad_max_dia,
    l.dificultad_acceso,
    l.clima_predominante,
    l.calificacion_promedio,

    u.edad,
    u.presupuesto_preferido,
    u.categoria_favorita,
    u.nivel_actividad_fisica,
    u.viaja_con,
    u.num_visitas_total

FROM dbo.visitas v
INNER JOIN dbo.lugares  l ON v.lugar_id   = l.lugar_id
INNER JOIN dbo.usuarios u ON v.usuario_id = u.usuario_id
"""

# Clima promedio diario (para enriquecer el modelo)
QUERY_CLIMA = """
SELECT
    fecha,
    AVG(temperatura_max_c) AS temp_max_prom,
    AVG(precipitacion_mm)  AS precip_prom,
    MAX(temporada)         AS temporada
FROM dbo.clima_historico
GROUP BY fecha
"""

df          = pd.read_sql(QUERY_PRINCIPAL, engine, parse_dates=["fecha_visita"])
df_clima    = pd.read_sql(QUERY_CLIMA,     engine, parse_dates=["fecha"])
df_lugares  = pd.read_sql("SELECT * FROM dbo.lugares",  engine)
df_usuarios = pd.read_sql("SELECT * FROM dbo.usuarios", engine)

# Unir con clima
df = df.merge(df_clima, left_on="fecha_visita", right_on="fecha", how="left")
df["temp_max_prom"] = df["temp_max_prom"].fillna(df["temp_max_prom"].median())
df["precip_prom"]   = df["precip_prom"].fillna(df["precip_prom"].median())
df["temporada"]     = df["temporada"].fillna("media")

print(f"  ✅ {len(df):,} registros cargados desde SQL Server")
print(f"     ({len(df_lugares)} lugares | {len(df_usuarios)} usuarios | {len(df)} visitas con joins)")

# ════════════════════════════════════════════════════════════
#  3. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
print("\n[3/7] Construyendo features...")

# Mapas de conversión numérica
presup_map = {"bajo": 50000, "medio": 150000, "alto": 999999}
act_map    = {"bajo": 1, "medio": 2, "alto": 3}
dif_map    = {"fácil": 1, "moderado": 2, "difícil": 3}

# Features de compatibilidad usuario-lugar
df["match_categoria"]       = (df["categoria_favorita"] == df["categoria_lugar"]).astype(int)
df["presupuesto_max"]       = df["presupuesto_preferido"].map(presup_map)
df["precio_compatible"]     = (df["precio_entrada_cop"] <= df["presupuesto_max"]).astype(int)
df["nivel_act_num"]         = df["nivel_actividad_fisica"].map(act_map).fillna(2)
df["dificultad_num"]        = df["dificultad_acceso"].map(dif_map).fillna(2)
df["actividad_compatible"]  = (df["nivel_act_num"] >= df["dificultad_num"]).astype(int)

# Features temporales
df["es_fin_semana"]    = df["dia_semana"].isin(["sábado","domingo"]).astype(int)
df["clima_favorable"]  = df["clima_dia_visita"].isin(["soleado","nublado"]).astype(int)

# Ocupación estimada del lugar
visitas_prom             = df.groupby("lugar_id")["visita_id"].count() / max(df["fecha_visita"].nunique(), 1)
df["ocupacion_estimada"] = df["lugar_id"].map(visitas_prom).fillna(0.3)

# Codificación de categóricas
cols_categoricas = [
    "categoria_lugar", "dificultad_acceso", "clima_predominante",
    "presupuesto_preferido", "categoria_favorita",
    "nivel_actividad_fisica", "viaja_con", "temporada",
    "clima_dia_visita", "comentario_sentimiento"
]
encoders = {}
for col in cols_categoricas:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

print(f"  ✅ {len(cols_categoricas)} variables categóricas codificadas")

# ════════════════════════════════════════════════════════════
#  4. ENTRENAMIENTO DEL MODELO
# ════════════════════════════════════════════════════════════
print("\n[4/7] Entrenando modelo Random Forest...")

FEATURES = [
    # Del lugar
    "altitud_msnm", "precio_entrada_cop", "capacidad_max_dia",
    "calificacion_promedio", "dificultad_num",
    "categoria_lugar_enc", "clima_predominante_enc",
    # Del usuario
    "edad", "num_visitas_total", "nivel_act_num",
    "presupuesto_preferido_enc", "categoria_favorita_enc", "viaja_con_enc",
    # De la visita
    "mes", "es_festivo", "es_fin_semana", "duracion_horas",
    "gasto_total_cop", "clima_dia_visita_enc", "clima_favorable",
    # Features construidas
    "match_categoria", "precio_compatible", "actividad_compatible",
    "ocupacion_estimada",
    # Del clima
    "temp_max_prom", "precip_prom", "temporada_enc",
]
TARGET = "calificacion_dada"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
modelo.fit(X_train, y_train)
print(f"  ✅ Modelo entrenado: {len(X_train)} muestras train / {len(X_test)} test")

# ════════════════════════════════════════════════════════════
#  5. EVALUACIÓN
# ════════════════════════════════════════════════════════════
print("\n[5/7] Evaluando modelo...")

y_pred = modelo.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
cv     = cross_val_score(modelo, X, y, cv=5, scoring="r2", n_jobs=-1)

print(f"\n  {'Métrica':<30} {'Valor':>10}")
print(f"  {'-'*42}")
print(f"  {'MAE  (error prom. en escala 1-5)':<30} {mae:>10.4f}")
print(f"  {'RMSE':<30} {rmse:>10.4f}")
print(f"  {'R² Test':<30} {r2:>10.4f}")
print(f"  {'CV R² promedio (5 folds)':<30} {cv.mean():>10.4f}")
print(f"  {'CV R² desviación':<30} {cv.std():>10.4f}")

# ════════════════════════════════════════════════════════════
#  6. DEMO DEL SISTEMA DE RECOMENDACIÓN
# ════════════════════════════════════════════════════════════
print("\n[6/7] Demo sistema de recomendación...")

def recomendar(usuario_id, top_n=5):
    """
    Función que el Backend usará para servir recomendaciones.
    Recibe un usuario_id y devuelve los top N lugares recomendados.
    """
    u = df_usuarios[df_usuarios["usuario_id"] == usuario_id].iloc[0]
    pres_max  = presup_map.get(u["presupuesto_preferido"], 100000)
    nivel_act = act_map.get(u["nivel_actividad_fisica"], 2)

    filas = []
    for _, lugar in df_lugares.iterrows():
        dif = dif_map.get(lugar["dificultad_acceso"], 2)
        fila = {
            "altitud_msnm":             lugar["altitud_msnm"],
            "precio_entrada_cop":        lugar["precio_entrada_cop"],
            "capacidad_max_dia":         lugar["capacidad_max_dia"],
            "calificacion_promedio":     lugar["calificacion_promedio"],
            "dificultad_num":            dif,
            "categoria_lugar_enc":       encoders["categoria_lugar"].transform([lugar["categoria"]])[0],
            "clima_predominante_enc":    encoders["clima_predominante"].transform([lugar["clima_predominante"]])[0],
            "edad":                      u["edad"],
            "num_visitas_total":         u["num_visitas_total"],
            "nivel_act_num":             nivel_act,
            "presupuesto_preferido_enc": encoders["presupuesto_preferido"].transform([u["presupuesto_preferido"]])[0],
            "categoria_favorita_enc":    encoders["categoria_favorita"].transform([u["categoria_favorita"]])[0],
            "viaja_con_enc":             encoders["viaja_con"].transform([u["viaja_con"]])[0],
            "mes":                       pd.Timestamp.now().month,
            "es_festivo":                0,
            "es_fin_semana":             1 if pd.Timestamp.now().weekday() >= 5 else 0,
            "duracion_horas":            3.0,
            "gasto_total_cop":           min(lugar["precio_entrada_cop"] + 30000, pres_max),
            "clima_dia_visita_enc":      encoders["clima_dia_visita"].transform(["soleado"])[0],
            "clima_favorable":           1,
            "match_categoria":           int(u["categoria_favorita"] == lugar["categoria"]),
            "precio_compatible":         int(lugar["precio_entrada_cop"] <= pres_max),
            "actividad_compatible":      int(nivel_act >= dif),
            "ocupacion_estimada":        0.3,
            "temp_max_prom":             18.0,
            "precip_prom":               5.0,
            "temporada_enc":             encoders["temporada"].transform(["media"])[0],
        }
        filas.append(fila)

    scores = modelo.predict(pd.DataFrame(filas))
    df_rec = df_lugares.copy()
    df_rec["score_pred"] = np.round(scores, 2)
    return df_rec.sort_values("score_pred", ascending=False)[
        ["lugar_id","nombre","municipio","categoria","precio_entrada_cop","score_pred"]
    ].head(top_n)

for uid in [1, 15, 42]:
    u = df_usuarios[df_usuarios["usuario_id"] == uid].iloc[0]
    print(f"\n  👤 Usuario {uid} | {u['edad']} años | {u['ciudad_origen']} | "
          f"le gusta: {u['categoria_favorita']} | presupuesto: {u['presupuesto_preferido']}")
    for _, r in recomendar(uid).iterrows():
        precio = f"${r['precio_entrada_cop']:,}" if r["precio_entrada_cop"] > 0 else "Gratis"
        print(f"     {r['score_pred']:.2f}⭐  {r['nombre']:<35} ({r['categoria']}) — {precio}")

# ════════════════════════════════════════════════════════════
#  7. EXPORTAR ARTEFACTOS PARA EL BACKEND
# ════════════════════════════════════════════════════════════
print("\n[7/7] Exportando artefactos...")

# Modelo entrenado
with open(f"{RUTA_SALIDA}modelo_score_visitas.pkl", "wb") as f:
    pickle.dump(modelo, f)
print(f"  ✅ modelo_score_visitas.pkl  → el Backend lo carga con pickle.load()")

# Encoders (siempre van junto al modelo)
with open(f"{RUTA_SALIDA}encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print(f"  ✅ encoders.pkl             → necesario para procesar datos nuevos")

# Importancia de features (para el dashboard del Frontend)
pd.DataFrame({
    "feature":    FEATURES,
    "importance": modelo.feature_importances_
}).sort_values("importance", ascending=False).to_csv(
    f"{RUTA_SALIDA}feature_importance.csv", index=False
)
print(f"  ✅ feature_importance.csv   → el Frontend lo grafica en el dashboard")

# Métricas del modelo (para el dashboard del Frontend)
metricas = {
    "fuente_datos":  f"SQL Server — {SERVIDOR} / {BASE_DE_DATOS}",
    "modelo":        "RandomForestRegressor",
    "n_estimators":  200,
    "max_depth":     12,
    "n_features":    len(FEATURES),
    "n_train":       len(X_train),
    "n_test":        len(X_test),
    "MAE_test":      round(mae,      4),
    "RMSE_test":     round(rmse,     4),
    "R2_test":       round(r2,       4),
    "CV_R2_mean":    round(cv.mean(),4),
    "CV_R2_std":     round(cv.std(), 4),
    "features":      FEATURES
}
with open(f"{RUTA_SALIDA}metricas_modelo.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)
print(f"  ✅ metricas_modelo.json     → métricas para el dashboard y el profesor")

# Predicciones del set de prueba (para análisis y gráficas)
df_pred = X_test.copy()
df_pred["y_real"] = y_test.values
df_pred["y_pred"] = np.round(y_pred, 2)
df_pred["error"]  = np.abs(df_pred["y_real"] - df_pred["y_pred"]).round(2)
df_pred.to_csv(f"{RUTA_SALIDA}predicciones_test.csv", index=False)
print(f"  ✅ predicciones_test.csv    → para gráficas de evaluación")

# ════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  ✅ PIPELINE COMPLETO — TODO LISTO")
print("=" * 62)
print(f"  Fuente de datos : SQL Server ({SERVIDOR})")
print(f"  Registros usados: {len(df):,} visitas enriquecidas con joins")
print(f"  MAE test        : {mae:.4f}  (error ±{mae:.2f} en escala 1-5)")
print(f"  R² test         : {r2:.4f}")
print(f"  CV R²           : {cv.mean():.4f} ± {cv.std():.4f}")
print(f"\n  Archivos generados en '{RUTA_SALIDA}':")
print(f"    📦 modelo_score_visitas.pkl  → Backend")
print(f"    📦 encoders.pkl              → Backend")
print(f"    📊 feature_importance.csv   → Frontend/Dashboard")
print(f"    📊 metricas_modelo.json     → Frontend/Dashboard")
print(f"    📊 predicciones_test.csv    → Análisis")
print("=" * 62)
