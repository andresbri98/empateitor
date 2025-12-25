
import os
import glob
import pandas as pd

RAW_DATA_PATH = os.path.join('data', 'raw', 'football_data')
OUTPUT_PATH = os.path.join('data', 'intermediate', 'unified_football_data.csv')

def extract_season(filename):
    # Ejemplo: SP1_00_01.csv -> 00-01, SP1_95_96.csv -> 95-96
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 2:
        # Formato esperado: XXX_YY_ZZ.csv
        ini = parts[1]
        fin = parts[2].replace('.csv', '')
        if len(ini) == 2 and len(fin) == 2 and ini.isdigit() and fin.isdigit():
            return f"{ini}-{fin}"
    elif len(parts) > 1:
        # Formato alternativo: XXX_YYYY.csv o XXX_YYZZ.csv
        season_part = parts[1].replace('.csv', '')
        if len(season_part) == 4 and season_part.isdigit():
            return f"{season_part[:2]}-{season_part[2:]}"
        elif len(season_part) == 5 and '_' in season_part:
            ini, fin = season_part.split('_')
            return f"{ini}-{fin}"
    return ''

def collect_csv_files():
    # Recursivamente encuentra todos los CSVs en football_data
    return glob.glob(os.path.join(RAW_DATA_PATH, '*', '*', '*.csv'))


def try_read_csv(csv_path):
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252', 'windows-1252']
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding, on_bad_lines='skip')
        except Exception as e:
            continue
    print(f"No se pudo leer {csv_path} con ningún encoding.")
    return None

def main():
    all_dfs = []
    for csv_path in collect_csv_files():
        rel_path = os.path.relpath(csv_path, RAW_DATA_PATH)
        parts = rel_path.split(os.sep)
        if len(parts) < 3:
            continue
        competition = parts[0]
        division = parts[1]
        season = extract_season(parts[2])
        df = try_read_csv(csv_path)
        if df is None:
            print(f"Error leyendo {csv_path}: no se pudo cargar con ningún encoding.")
            continue
        df['Season'] = season
        df['Competition'] = competition
        df['Division'] = division
        all_dfs.append(df)
    if all_dfs:
        unified_df = pd.concat(all_dfs, ignore_index=True, sort=True)
        # Ordenar columnas según lo solicitado
        ordered_cols = [
            'Season',
            'Division',
            'Competition',
            'Div',
            'Date',
            'Time',
            'HomeTeam',
            'AwayTeam'
        ]
        # Añadir el resto de columnas conservando el orden original
        rest_cols = [col for col in unified_df.columns if col not in ordered_cols]
        final_cols = ordered_cols + rest_cols
        unified_df = unified_df.reindex(columns=final_cols)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        unified_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Unified CSV saved to {OUTPUT_PATH}")
    else:
        print("No CSV files found or loaded.")

if __name__ == '__main__':
    main()
