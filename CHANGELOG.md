# CHANGELOG

## [0.1.1] - 2025-12-26
- Notebook de limpieza: análisis detallado de partidos con `FTR` no informado y grupos con resultados no informados.
- Eliminación de grupos con >10% de partidos no informados (FTHG/FTAG/FTR) y reporte por `Division`, `Competition` y `Season`.
- Construcción y consolidación de columnas `ODDS_H`, `ODDS_D`, `ODDS_A` priorizando Bet365 y eliminación de columnas de casas de apuestas originales.
- Diagnóstico de `ODDS_A` nulo: resumen agrupado por `Division`, `Competition` y `Season`.
- Decisión actual: no imputar `ODDS_A`; se mantiene sin imputación para evaluar el comportamiento del modelo.
- Exportación del dataframe final limpio a `data/intermediate/unified_cleaned_football_data.csv`.

## [0.1.0] - 2025-12-25
- Configuración inicial del entorno Python 3.11 y requisitos.
- Instalación de paquetes necesarios (incluyendo xgboost y lightgbm).
- Creación del script `src/data/process_football_data.py` para unificar todos los CSV de football_data en un único archivo `data/intermediate/unified_football_data.csv`.
- Limpieza y estandarización de la variable `Season` al formato XX-YY.
- Análisis y eliminación de columnas y registros con alta presencia de NaN.
- Posible Eliminación de registros de las temporadas 93 a 99 por alta cantidad de datos faltantes.
- Creación de notebooks para análisis y limpieza del dataset.
- Preparación del proyecto para su subida a un nuevo repositorio Git.
