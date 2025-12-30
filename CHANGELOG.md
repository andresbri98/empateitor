# CHANGELOG

## [0.1.3] - 2025-12-30
- Finalización del notebook de limpieza de engineered: [src/notebooks/clean_engineered_football_data.ipynb](src/notebooks/clean_engineered_football_data.ipynb).
- Imputaciones y normalizaciones aplicadas:
	- `HTR <- FTR` para nulos.
	- H2H y estado de forma (`*_last5`, `*_ma_5`, `*_median_5`, `*_std_5`, `draw_tendency_*`, `form_diff_*`, `*_last_*`) rellenados con 0.
	- Descanso: `home_rest_days`, `away_rest_days`, `rest_days_diff` a 0; `rest_days_diff` recalculado si aplica.
	- PPG season-to-date y diferenciales recalculados con divisiones seguras.
	- Acumulados y `matches_played*` rellenados con 0.
	- Cuotas: imputación por mediana `Competition+Season` con fallback global; probabilidades normalizadas y métricas derivadas.
	- NaN numéricos residuales a 0.
- Exportaciones con timestamp en `data/intermediate`:
	- `engineered_football_data_cleaned_all_<timestamp>.csv`
	- `engineered_football_data_cleaned_no_odds_nulls_<timestamp>.csv`
- Sincronización a `data/processed` copiando estos CSV para garantizar igualdad exacta:
	- `engineered_football_data_cleaned_all.csv`
	- `engineered_football_data_cleaned_no_odds_nulls.csv`
- Verificación: igualdad por hash SHA256 y conteo de filas entre `intermediate` y `processed`.

## [0.1.2] - 2025-12-26
- Script `feature_engineering.py` actualizado para generar solo features útiles:
	- Añadidas features de estado de forma contextual: `home_*_last5_homeonly` y `away_*_last5_awayonly`.
	- Eliminadas columnas cruzadas no informativas del partido actual (p. ej., `home_gf_away`, `away_gf_home`).
	- Mantenidas las acumuladas por temporada útiles (`*_home_season_todate`, `*_away_season_todate`).
- Sincronización de salidas: `engineered_football_data.csv` (full) y `engineered_football_data_lite.csv` (lite) ahora tienen el mismo nº de filas (una por partido).
- Notebook de preparación: creación de datasets para EDA/modelado en `data/processed`:
	- Full imputed (odds por mediana Competition+Season, H2H=0) y full filtered (sin nulos en odds).
	- Lite imputed y lite filtered con el mismo criterio.
- Decisión operativa actual: dejar la imputación avanzada y EDA para el siguiente paso, y revisar fiabilidad de datos y columnas duplicadas antes de modelizar.

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
