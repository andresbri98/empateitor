# Predicción de Empates en Fútbol (v0.1.4)

Este proyecto tiene como objetivo predecir empates en partidos de fútbol europeo utilizando datos históricos, features ingenieriles y modelos de machine learning. El enfoque es maximizar aciertos en predicciones semanales de empates, con un recall aceptable (al menos 10 empates por cada 100 partidos) y con una precisión superior al 0.36.

## Estructura del Proyecto

```
empates/
├── CHANGELOG.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── football_data/
│   └── intermediate/
│   │   ├── unified_football_data.csv          # Datos unificados
│   │   ├── football_data_with_features.csv    # Con nuevas features
│   │   └── football_data_cleaned.csv          # Datos limpios y con un tratamiento de NaN
│   ├── processed/
│   │   └── football_data_model.csv            # Datos para modelizar con sólo variables prepartido
│   │   └── football_data_model_smote.csv      # Datos para modelizar con balanceo de clases
├── models/                                    # Modelos entrenados 
├── notebooks/
│   └── test_football_data.ipynb               # Notebook para revisar datos y las features creadas
│   └── eda_football_data.ipynb                # EDA con visualizaciones e interpretaciones 
└── src/
    ├── data/
    │   ├── download_external.py               # Descarga de datos externos, posible implemento futuro
    │   ├── process_football_data.py           # Unificación de CSVs
   │   ├── feature_engineering.py             # Generación de features (forma general y contextual)
    │   └── data_cleaning.py                   # Limpieza de datos
    │   └── data_model.py                      # Elimina las variables prepartido
    └── models/                                # Scripts de modeladado
    │   ├── train_logisticRegression.py        # Entrenemaiento modelo de regresion
    │   ├── train_xgb.py                       # Entrenamiento modelo XGBoost
    └── notebooks/       
        ├── clean_unified_football_data.pynb   # Primera limpieza de data raw              
```

## Flujo de Trabajo

### Nota importante para notebooks
Si usas los notebooks de la carpeta `notebooks/`, asegúrate de que el directorio de trabajo sea el raíz del proyecto antes de cargar datos. Añade al inicio del notebook:

```python
import os
os.chdir("C:/Users/andre/Desktop/proyectos/empates")  # Ajusta si tu ruta es diferente
```
Esto evita errores de FileNotFoundError al cargar archivos como `data/processed/football_data_cleaned.csv`.

1. **Datos**:
   - Los datos utilizados son los de Football-Data.co.uk ya descargados localmente en `data/raw/football_data/`.
   - Se deja para futuro la posibilidad de combinar estos datos junto con otros datasets y así generar nuevas variables

2. **Unificación de Datos**:
   - Ejecuta `src/data/process_football_data.py` para unificar CSVs de múltiples ligas en un solo dataset.
   - Maneja encodings y columnas estándar.
   - Resultado: `data/intermediate/unified_football_data.csv` (~250k filas, 27 columnas).
   - Ahora analizamos estos datos y revisamos posibles incoherencias o falta de información muy grande en `data/notebooks/clean_unified_football_data.pynb`. Y dejamos el csv con sólo datos útiles y preparado para añadir nueva features a partir de las que ya existen. 
  - Resultado: `data/intermediate/unified_cleaned_football_data.csv` 

3. **Feature Engineering (v0.1.2)**:
   - Ejecuta `src/data/feature_engineering.py` para generar features pre-partido.
   - Incluye dos tipos de estado de forma:
     - General: `*_last5` (últimos 5 partidos del equipo).
     - Contextual: `home_*_last5_homeonly` y `away_*_last5_awayonly` (solo partidos previos en la misma condición de local/visitante).
   - Se eliminan columnas cruzadas no informativas del partido actual (p. ej., `home_gf_away`, `away_gf_home`).
   - Se mantienen acumuladas útiles por temporada: `*_home_season_todate`, `*_away_season_todate`.
   - Reset por temporada; las features se calculan pre-partido (sin fuga de información).
   - Resultados sincronizados en `data/intermediate/engineered_football_data.csv` (full) y `data/intermediate/engineered_football_data_lite.csv` (lite) con el mismo nº de filas.

5. **Limpieza de Engineered (v0.1.3)**:
    - Ejecuta el notebook [src/notebooks/clean_engineered_football_data.ipynb](src/notebooks/clean_engineered_football_data.ipynb).
    - Imputación y normalización:
       - `HTR <- FTR` cuando falte.
       - H2H y estado de forma (`*_last5`, `*_ma_5`, `*_median_5`, `*_std_5`, `draw_tendency_*`, `form_diff_*`, `*_last_*`) a 0.
       - Descanso: `home_rest_days`, `away_rest_days`, `rest_days_diff` a 0; recalcular `rest_days_diff` si aplica.
       - PPG season-to-date con divisiones seguras y diferenciales (`ppg_diff_season_td`, `ppg_homeaway_ctx_diff_season_td`).
       - Acumulados y `matches_played*` a 0.
       - Cuotas: imputación por mediana `Competition+Season` con fallback a mediana global; probabilidades normalizadas y métricas derivadas.
       - NaN numéricos residuales a 0.
    - Salidas (con timestamp) en `data/intermediate`:
       - `engineered_football_data_cleaned_all_<timestamp>.csv`
       - `engineered_football_data_cleaned_no_odds_nulls_<timestamp>.csv`
      - Opcional: sincroniza a `data/processed` copiando los últimos CSV con timestamp para que sean idénticos.

6. **Pipeline Final Consolidado (v0.1.4)**:
    - Ejecuta el script final para generar directamente los datasets procesados en `data/processed`:
       - Preferencia: si existe `data/intermediate/engineered_football_data.csv`, se usará para la limpieza final.
       - Alternativa: si no existe engineered, se leerá `data/intermediate/unified_cleaned_football_data.csv`, se calcularán las features en memoria y se aplicará la misma limpieza final.
    - Resultados (idénticos a los intermedios más recientes):
       - `data/processed/engineered_footbal_data_cleaned_all.csv`
       - `data/processed/engineered_football_data_cleaned_no_odds_nulls.csv`
    - Comando sugerido:
     
       ```bash
       python -m src.data.process_football_data_final
       ```

7. **Validación de Datos**:
   - Verificar fiabilidad: revisar columnas duplicadas y coherencia de features (estado de forma y acumulados).
   - Próximo paso: imputación de nulos (H2H=0, odds por mediana Competition+Season) y EDA antes de modelizar.

8. **Datasets procesados para EDA/Modelado (v0.1.3)**:
    - Copias de los últimos CSV con timestamp de `intermediate` hacia `processed` para garantizar igualdad exacta:
       - `data/processed/engineered_football_data_cleaned_all.csv`
       - `data/processed/engineered_football_data_cleaned_no_odds_nulls.csv`
    - Recomendado: ejecutar las últimas celdas del notebook de limpieza para sincronizar y verificar igualdad (hash SHA256 y conteo de filas).

9. **EDA (Análisis Exploratorio)**:
   - Usa `notebooks/eda_football_data.ipynb` para explorar distribuciones, correlaciones y outliers del dataset `data/processed/football_data_cleaned.csv`.
   - Visualizaciones: Distribuciones de FTR, odds, correlaciones con empates, análisis por liga y temporada.
   - Insights: Features positivas para empates (e.g., Prob_D alta, H2H_Draws); ligas con más empates (e.g., Serie A); patrones temporales.
   - Identifica features predictoras de empates (e.g., Prob_D, H2H_Draws).

10. **Modelado**:
   - Entrena modelos (XGBoost, Random Forest) para clasificar H/D/A o binario D vs no-D.
   - Métricas: Precisión y recall para "D" (empates).
   - Objetivo: Predecir top 10-15 empates por semana con ~40% precisión.
   - Scripts en `src/models/`.

seguiremos con más pasos... mejora de modelos, evaluación de resultados etc..

## Requisitos

- Python 3.11 (para compatibilidad con librerías).
- Librerías: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn.
- Instala con `pip install -r requirements.txt`.

## Uso

1. Configura entorno: `python -m venv .venv; .venv\Scripts\activate`.
2. Instala dependencias: `pip install -r requirements.txt`.
3. Genera features (opcional si usas el pipeline final v0.1.4):
   - `python src/data/feature_engineering.py -i data/intermediate/unified_cleaned_football_data.csv -o data/intermediate/engineered_football_data.csv -ol data/intermediate/engineered_football_data_lite.csv -w 5`
4. Limpia engineered y sincroniza procesados (vía notebook) o usa el pipeline final:
   - Opción A (notebook):
   - Ejecuta [src/notebooks/clean_engineered_football_data.ipynb](src/notebooks/clean_engineered_football_data.ipynb) y genera CSVs con timestamp en `data/intermediate`.
   - Sincroniza a `data/processed` copiando los últimos CSV de `intermediate` para asegurar igualdad exacta.
   - Opción B (script final v0.1.4):
     
     ```bash
     python -m src.data.process_football_data_final
     ```
     
     Genera directamente los dos CSV en `data/processed` y verifica igualdad con los intermedios.

## Notas Técnicas

- **Reset por Temporada**: forma, puntos y acumulados se calculan dentro de cada temporada.
- **Pre-partido**: todas las features se computan con `shift(1)` o `cumcount()` para evitar fuga de información.
- **Odds**: al imputar por mediana por `Competition+Season`, recalcular `prob_*` y señales `odds_*` para consistencia.

## Diccionario de Datos (resumen)

- `home_*_last5_homeonly` / `away_*_last5_awayonly`: suma de GF/GA/PTS de los últimos 5 partidos previos en esa misma condición.
- `*_home_season_todate` / `*_away_season_todate`: acumulados hasta el encuentro (excluye el partido actual).
- `h2h_*_5y`: acumulados H2H en los 5 años previos; `h2h_draw_rate_5y` como ratio.
- `prob_*`, `bookmaker_margin`, `odds_*`: derivados de `ODDS_H/D/A`.