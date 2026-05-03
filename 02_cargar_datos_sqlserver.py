"""
============================================================
PROYECTO: TURISMO CALDAS
ARCHIVO:  02_cargar_datos_sqlserver.py
AUTOR:    Sebastián Garzón Henao (Data Scientist)
DESC:     Carga los 5 CSVs generados hacia SQL Server.
          Se ejecuta UNA sola vez después de crear las tablas.
============================================================

"""

import pandas as pd
from sqlalchemy import create_engine, text
import urllib

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE CONEXIÓN 
# ─────────────────────────────────────────────────────────────────

SERVIDOR = "ReicerX\SQLEXPRESS"
BASE_DE_DATOS = "TurismoCaldas"

# Ruta donde están los CSVs 
RUTA_CSV = r"C:\Users\Sebastian\Desktop\OCTAVO_SEMESTRE\PROFUNDIZACION II\PROYECTO_TURISMO\TABLAS_BD_TURISMO\\"   # ← cambia esto si los CSVs están en otra carpeta

# ─────────────────────────────────────────────────────────────────
# CREAR CONEXIÓN 
# ─────────────────────────────────────────────────────────────────
params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVIDOR};"
    f"DATABASE={BASE_DE_DATOS};"
    f"Trusted_Connection=yes;"  
)

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)

print("="*60)
print("CARGANDO DATOS EN SQL SERVER")
print(f"  Servidor: {SERVIDOR}")
print(f"  Base de datos: {BASE_DE_DATOS}")
print("="*60)

# ─────────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR PARA LA VERIFICACIÓN DE LOS DATOS
# ─────────────────────────────────────────────────────────────────
def cargar_tabla(nombre_csv, nombre_tabla, engine, ruta):
    """Carga un CSV en una tabla SQL Server con reporte de progreso."""
    df = pd.read_csv(f"{ruta}{nombre_csv}")

    # Se limpian las columnas con nombres raros (espacios, caracteres especiales)
    df.columns = [c.strip().replace(" ", "_").replace("ó", "o")
                   .replace("é", "e").replace("á", "a")
                   .replace("í", "i").replace("ú", "u") for c in df.columns]
    
    df = df.replace({'True': 1, 'False': 0, True: 1, False: 0})

    # Renombrar columna del CSV de visitas
    if "visit_con" in df.columns:
        df = df.rename(columns={"visit_con": "visito_con"})
    if "visitó_con" in df.columns:
        df = df.rename(columns={"visitó_con": "visito_con"})

    # Cargar a SQL Server con if_exists='append' lo cual respeta los datos existentes
    df.to_sql(
        name=nombre_tabla,
        con=engine,
        schema="dbo",
        if_exists="append",   
        index=False,
        chunksize=500          # inserta de 500 en 500 filas 
    )

    print(f"   {nombre_tabla:<30} → {len(df):>5} filas cargadas")
    return len(df)

# ─────────────────────────────────────────────────────────────────
# CARGAR LAS 5 TABLAS EN ORDEN:
# (primero las tablas sin llave foránea, luego las que dependen de ellas)
# ─────────────────────────────────────────────────────────────────
print("\nCargando tablas (orden importante por llaves foráneas):")

total = 0

# 1. lugares (sin FK — va primero)
total += cargar_tabla("tb_lugares.csv",              "lugares",               engine, RUTA_CSV)

# 2. usuarios (sin FK — va segundo)
total += cargar_tabla("tb_usuarios.csv",             "usuarios",              engine, RUTA_CSV)

# 3. visitas (depende de lugares y usuarios)
total += cargar_tabla("tb_visitas.csv",              "visitas",               engine, RUTA_CSV)

# 4. clima_historico (sin FK)
total += cargar_tabla("tb_clima_historico.csv",      "clima_historico",       engine, RUTA_CSV)

# 5. analitica_lugar_diaria (depende de lugares — va al final)
total += cargar_tabla("tb_analitica_lugar_diaria.csv","analitica_lugar_diaria",engine, RUTA_CSV)

print(f"\n{'='*60}")
print(f"  TOTAL registros cargados: {total:,}")
print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────────
# VERIFICACIÓN FINAL DESDE SQL Server
# ─────────────────────────────────────────────────────────────────
print("\nVerificando conteo de filas en SQL Server:")

tablas = ["lugares", "usuarios", "visitas", "clima_historico", "analitica_lugar_diaria"]

with engine.connect() as conn:
    for tabla in tablas:
        resultado = conn.execute(text(f"SELECT COUNT(*) FROM dbo.{tabla}"))
        conteo = resultado.scalar()
        print(f"  {tabla:<30} → {conteo:>5} filas en SQL Server")

print("\n ¡Carga completada exitosamente!")
print("   ver los datos en TurismoCaldas.")