# Predicción de Empates en Fútbol

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
    │   ├── download_external.py               # Descarga de datos externos
    │   ├── process_football_data.py           # Unificación de CSVs
    │   ├── feature_engineering.py             # Generación de features
    │   └── data_cleaning.py                   # Limpieza de datos
    │   └── data_model.py                      # Elimina las variables prepartido
    └── models/                                # Scripts de modeladado
    │   ├── train_logisticRegression.py        # Entrenemaiento modelo de regresion
    │   ├── train_xgb.py                       # Entrenamiento modelo XGBoost
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
   - Ahora analizamos estos datos y revisamos posibles incoherencias o falta de información muy grande en `notebooks/clean_unified_football_data.pynb`. Y dejamos el csv con sólo datos útiles y preparado para añadir nueva features a partir de las que ya existen. 
  - Resultado: `data/intermediate/cleaned_football_data.csv` 

3. **Feature Engineering**:
   - Ejecuta `src/data/feature_engineering.py` para generar features pre-partido.
   - Features incluyen: forma del equipo (últimos 5 partidos), xG, puntos acumulados, puntos por partido, diferencia de goles, H2H y features de odds.
   - Variables se resetean por temporada (e.g., forma y puntos no acumulan entre años).
   - Al inicio de temporada, features como forma y puntos por partido son NaN (menos de 5 partidos).
   - Resultado: `data/intermediate/football_data_with_features.csv` (~250k filas, +15 columnas).

5. **Limpieza de Datos**:
   - Ejecuta `src/data/data_cleaning.py` para filtrar y limpiar.
   - Elimina partidos de principios de temporada (NaN en features clave, ~21k filas eliminadas).
   - Imputa NaN: mediana por liga para forma, xG, posiciones, goal diff; 0 para H2H; mediana para odds.
   - Resultado: `data/intermediate/football_data_cleaned.csv` (229,311 filas, 47 columnas, distribución: 45% H, 28% A, 27% D).

6. **Validación de Datos**:
   - Ejecuta `notebooks/test_football_data.ipynb` para verificar fiabilidad del dataset, muestra partidos aleatorios y verifica esas variables

7. **Generación del dataset para modelado**:
   - Ejecuta `src/data/data_model.py` para eliminar las variables prepartido y quedarnos solo con las variables que se van a utilizar en el modelado
   - - Resultado: `data/processed/football_data_cleaned.csv` 

8. **EDA (Análisis Exploratorio)**:
   - Usa `notebooks/eda_football_data.ipynb` para explorar distribuciones, correlaciones y outliers del dataset `data/processed/football_data_cleaned.csv`.
   - Visualizaciones: Distribuciones de FTR, odds, correlaciones con empates, análisis por liga y temporada.
   - Insights: Features positivas para empates (e.g., Prob_D alta, H2H_Draws); ligas con más empates (e.g., Serie A); patrones temporales.
   - Identifica features predictoras de empates (e.g., Prob_D, H2H_Draws).

9. **Modelado**:
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

1. Configura entorno: `python -m venv venv311; venv311\Scripts\activate`.
.....

## Notas Técnicas

- **Features por Temporada**: Variables como forma, xG y posición se calculan solo dentro de la temporada actual para evitar ruido de cambios en equipos.
- **Principios de Temporada**: Features son NaN hasta que equipos jueguen suficientes partidos; se filtran en limpieza para fiabilidad.
- **Odds**: Se eligen las mejores (Pinnacle si > NaN que Bet365; sino Bet365).
- **Rendimiento Esperado**: Modelo baseline ~26% precisión para empates; con features, ~35-45%.

## Diccionario de Datos

aqui hacer un diccionario de datos