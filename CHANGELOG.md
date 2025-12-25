# CHANGELOG

## [0.1.0] - 2025-12-25
- Configuración inicial del entorno Python 3.11 y requisitos.
- Instalación de paquetes necesarios (incluyendo xgboost y lightgbm).
- Creación del script `src/data/process_football_data.py` para unificar todos los CSV de football_data en un único archivo `data/intermediate/unified_football_data.csv`.
- Limpieza y estandarización de la variable `Season` al formato XX-YY.
- Análisis y eliminación de columnas y registros con alta presencia de NaN.
- Posible Eliminación de registros de las temporadas 93 a 99 por alta cantidad de datos faltantes.
- Creación de notebooks para análisis y limpieza del dataset.
- Preparación del proyecto para su subida a un nuevo repositorio Git.
