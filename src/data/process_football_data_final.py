import os
import glob
import argparse
import importlib.util
import pandas as pd
import numpy as np


# Rutas base
RAW_FD_PATH = os.path.join('data', 'raw', 'football_data')
INTERMEDIATE_DIR = os.path.join('data', 'intermediate')
PROCESSED_DIR = os.path.join('data', 'processed')

# Productos intermedios y finales
UNIFIED_PATH = os.path.join(INTERMEDIATE_DIR, 'unified_football_data.csv')
UNIFIED_CLEANED_PATH = os.path.join(INTERMEDIATE_DIR, 'unified_cleaned_football_data.csv')
ENGINEERED_PATH = os.path.join(INTERMEDIATE_DIR, 'engineered_football_data.csv')
ENGINEERED_LITE_PATH = os.path.join(INTERMEDIATE_DIR, 'engineered_football_data_lite.csv')
OUTPUT_ALL_PATH = os.path.join(PROCESSED_DIR, 'engineered_footbal_data_cleaned_all.csv')
OUTPUT_NO_ODDS_NULLS_PATH = os.path.join(PROCESSED_DIR, 'engineered_football_data_cleaned_no_odds_nulls.csv')


def _ensure_dirs():
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


# === 1) process_football_data.py ===
def extract_season(filename: str) -> str:
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 2:
        ini = parts[1]
        fin = parts[2].replace('.csv', '')
        if len(ini) == 2 and len(fin) == 2 and ini.isdigit() and fin.isdigit():
            return f"{ini}-{fin}"
    elif len(parts) > 1:
        season_part = parts[1].replace('.csv', '')
        if len(season_part) == 4 and season_part.isdigit():
            return f"{season_part[:2]}-{season_part[2:]}"
        elif len(season_part) == 5 and '_' in season_part:
            ini, fin = season_part.split('_')
            return f"{ini}-{fin}"
    return ''


def collect_fd_csv_files() -> list[str]:
    return glob.glob(os.path.join(RAW_FD_PATH, '*', '*', '*.csv'))


def try_read_csv(csv_path: str) -> pd.DataFrame | None:
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252', 'windows-1252']
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding, on_bad_lines='skip')
        except Exception:
            continue
    print(f"No se pudo leer {csv_path} con ningún encoding.")
    return None


def build_unified_from_football_data(out_path: str = UNIFIED_PATH) -> str:
    dfs = []
    for csv_path in collect_fd_csv_files():
        rel_path = os.path.relpath(csv_path, RAW_FD_PATH)
        parts = rel_path.split(os.sep)
        if len(parts) < 3:
            continue
        competition = parts[0]
        division = parts[1]
        season = extract_season(parts[2])
        df = try_read_csv(csv_path)
        if df is None:
            continue
        df['Season'] = season
        df['Competition'] = competition
        df['Division'] = division
        dfs.append(df)
    if not dfs:
        raise RuntimeError('No se encontraron CSVs en data/raw/football_data.')
    unified_df = pd.concat(dfs, ignore_index=True, sort=True)
    ordered_cols = ['Season', 'Division', 'Competition', 'Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam']
    rest_cols = [c for c in unified_df.columns if c not in ordered_cols]
    final_cols = ordered_cols + rest_cols
    unified_df = unified_df.reindex(columns=final_cols)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    unified_df.to_csv(out_path, index=False)
    return out_path


# === 2) clean_unified_football_data.ipynb ===
def _to_numeric_if_exists(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    return pd.Series(np.nan, index=df.index)


def _coalesce_with_primary(df: pd.DataFrame, primary: str, others: list[str]) -> pd.Series:
    base = _to_numeric_if_exists(df, primary)
    out = base.copy()
    for c in others:
        s = _to_numeric_if_exists(df, c)
        out = out.fillna(s)
    return out


def clean_unified_data(in_path: str = UNIFIED_PATH, out_path: str = UNIFIED_CLEANED_PATH) -> str:
    df = pd.read_csv(in_path, low_memory=False)

    # Eliminar columnas con >50% NaN
    nan_percent = df.isna().mean() * 100
    cols_muchos_nan = nan_percent[nan_percent > 50].index.tolist()
    if cols_muchos_nan:
        df = df.drop(columns=cols_muchos_nan)

    # Construir ODDS_H/D/A priorizando B365 y luego otras casas
    houses = {
        'H': {'primary': 'B365H', 'others': ['BWH', 'WHH', 'VCH', 'IWH']},
        'D': {'primary': 'B365D', 'others': ['BWD', 'WHD', 'VCD', 'IWD']},
        'A': {'primary': 'B365A', 'others': ['BWA', 'WHA', 'VCA', 'IWA']},
    }
    for oc, cfg in houses.items():
        df[f'ODDS_{oc}'] = _coalesce_with_primary(df, cfg['primary'], cfg['others'])

    # Eliminar columnas de casas de apuestas (nos quedamos con ODDS_*)
    cols_houses = ['B365H','B365D','B365A','BWH','BWD','BWA','WHH','WHD','WHA','VCH','VCD','VCA','IWH','IWD','IWA']
    df = df.drop(columns=[c for c in cols_houses if c in df.columns])

    # Marcar partidos no informados (FTHG/FTAG/FTR)
    FTHG_num = pd.to_numeric(df['FTHG'], errors='coerce') if 'FTHG' in df.columns else pd.Series(np.nan, index=df.index)
    FTAG_num = pd.to_numeric(df['FTAG'], errors='coerce') if 'FTAG' in df.columns else pd.Series(np.nan, index=df.index)
    if 'FTR' in df.columns:
        FTR_clean = df['FTR'].astype(str).str.strip().str.upper().replace({'': np.nan})
        FTR_valid = FTR_clean.where(FTR_clean.isin(['H','D','A']))
    else:
        FTR_valid = pd.Series(np.nan, index=df.index)
    uninformed = FTHG_num.isna() | FTAG_num.isna() | FTR_valid.isna()

    # Eliminar grupos con >10% no informados por (Division/Div, Competition, Season)
    group_cols = []
    if 'Division' in df.columns:
        group_cols.append('Division')
    elif 'Div' in df.columns:
        group_cols.append('Div')
    if 'Competition' in df.columns:
        group_cols.append('Competition')
    if 'Season' in df.columns:
        group_cols.append('Season')
    if group_cols:
        df['__uninformed'] = uninformed
        summary = df.groupby(group_cols, dropna=False)['__uninformed'].agg(total='size', uninformed_count='sum')
        summary['uninformed_ratio'] = (summary['uninformed_count'] / summary['total']).round(4)
        high_missing_groups = summary[summary['uninformed_ratio'] > 0.10].reset_index()

        def build_composite_key(df_like: pd.DataFrame, cols: list[str]) -> pd.Series:
            if df_like.empty:
                return pd.Series([], dtype=object)
            tmp = df_like[cols].copy()
            for c in cols:
                tmp[c] = tmp[c].astype(str).where(tmp[c].notna(), '<<NA>>')
            return tmp.apply(lambda r: tuple(r.values.tolist()), axis=1)

        keys_df = build_composite_key(df, group_cols)
        keys_remove = set(build_composite_key(high_missing_groups, group_cols))
        remove_mask = keys_df.isin(keys_remove)
        if remove_mask.any():
            df = df.loc[~remove_mask].copy()
        df = df.drop(columns=['__uninformed'])

    # Eliminar filas con FTR no informado
    if 'FTR' in df.columns:
        FTR_clean = df['FTR'].astype(str).str.strip().str.upper().replace({'': np.nan})
        df = df.loc[FTR_clean.isin(['H','D','A'])].copy()

    # Eliminar superliga_griega en seasons específicas
    if {'Competition','Season'}.issubset(df.columns):
        comp_vals = df['Competition'].astype(str).str.strip().str.lower()
        season_vals = df['Season'].astype(str).str.strip()
        target_comp = 'superliga_griega'
        target_seasons = {'00-01', '02-03', '03-04', '04-05'}
        mask = comp_vals.eq(target_comp) & season_vals.isin(target_seasons)
        if mask.any():
            df = df.loc[~mask].copy()

    # Rellenar tarjetas con moda y completar HTAG/HTHG/HTR desde FTAG/FTHG/FTR
    for c in ['AR','AY','HR','HY']:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            mode_vals = pd.Series(s).mode(dropna=True)
            if len(mode_vals) > 0:
                df[c] = s.fillna(mode_vals.iloc[0])
    half_final_map = {'HTAG': 'FTAG', 'HTHG': 'FTHG', 'HTR': 'FTR'}
    for half_col, final_col in half_final_map.items():
        if half_col in df.columns and final_col in df.columns:
            if half_col in ['HTAG','HTHG']:
                df[half_col] = pd.to_numeric(df[half_col], errors='coerce').fillna(pd.to_numeric(df[final_col], errors='coerce'))
            else:
                half = df[half_col].astype(str).str.strip().replace({'': np.nan})
                final = df[final_col].astype(str).str.strip().str.upper().replace({'': np.nan})
                final_valid = final.where(final.isin(['H','D','A']))
                df[half_col] = half.where(half.notna(), final_valid)

    # Garantizar columnas HR/HY/AR/AY existen para la etapa de features
    for c in ['AR','AY','HR','HY']:
        if c not in df.columns:
            df[c] = np.nan

    df.to_csv(out_path, index=False)
    return out_path


# === 3) feature_engineering.py ===
def _import_feature_engineering(module_path: str) -> object:
    spec = importlib.util.spec_from_file_location('feature_engineering', module_path)
    if spec is None or spec.loader is None:
        raise ImportError('No se pudo cargar feature_engineering.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_feature_engineering(input_path: str = UNIFIED_CLEANED_PATH,
                            output_path: str = ENGINEERED_PATH,
                            output_lite_path: str = ENGINEERED_LITE_PATH) -> tuple[str, str | None]:
    fe = _import_feature_engineering(os.path.join('src', 'data', 'feature_engineering.py'))
    return fe.engineer_features(input_path, output_path, windows=(5,), output_lite_path=output_lite_path)


# === 4) clean_engineered_football_data.ipynb ===
def engineer_in_memory(unified_cleaned_path: str) -> pd.DataFrame:
    # Cargar unified limpio y garantizar columnas de tarjetas si faltan
    df = pd.read_csv(unified_cleaned_path, low_memory=False, dtype={'Division': 'string', 'Div': 'string'})
    for c in ['AR','AY','HR','HY']:
        if c not in df.columns:
            df[c] = np.nan

    # Importar funciones de feature_engineering y construir dataset con features sin escribir a disco
    fe = _import_feature_engineering(os.path.join('src', 'data', 'feature_engineering.py'))
    df = fe.parse_dates(df)
    long_df = fe.prepare_long_format(df)
    long_df = fe.add_team_rolling_features(long_df, windows=(5,))
    long_df = fe.add_season_to_date_home_away(long_df)
    long_df = fe.add_rest_days(long_df)
    long_df = fe.add_low_scoring_features(long_df)
    long_df = fe.add_competition_season_zscores(long_df)
    home_feat, away_feat = fe.split_home_away_features(long_df)
    h2h = fe.add_h2h_last5years(df)

    base_cols = [c for c in df.columns]
    out = df.reset_index().rename(columns={'index': 'match_id'})[base_cols + ['match_id']]
    out = out.merge(home_feat, on='match_id', how='left').merge(away_feat, on='match_id', how='left')
    out = out.merge(h2h, on='match_id', how='left')
    out = fe.add_odds_features(out)
    out = fe.add_draw_focused_match_features(out)
    out = fe.drop_empty_xg_columns(out)

    # Diferenciales útiles
    if {'home_pts_last5','away_pts_last5'}.issubset(out.columns):
        out['pts_last5_diff'] = out['home_pts_last5'] - out['away_pts_last5']
    if {'home_gf_last5','away_gf_last5'}.issubset(out.columns):
        out['gf_last5_diff'] = out['home_gf_last5'] - out['away_gf_last5']
    if {'home_ga_last5','away_ga_last5'}.issubset(out.columns):
        out['ga_last5_diff'] = out['home_ga_last5'] - out['away_ga_last5']
    if {'home_rest_days','away_rest_days'}.issubset(out.columns):
        out['rest_days_diff'] = out['home_rest_days'] - out['away_rest_days']
    if {'home_ppg_season_td','away_ppg_season_td'}.issubset(out.columns):
        out['ppg_diff_season_td'] = out['home_ppg_season_td'] - out['away_ppg_season_td']
    if {'home_ppg_home_season_td','away_ppg_away_season_td'}.issubset(out.columns):
        out['ppg_homeaway_ctx_diff_season_td'] = out['home_ppg_home_season_td'] - out['away_ppg_away_season_td']

    if 'Date' in out.columns:
        sort_cols = [c for c in ['Date','Competition','Season','Div'] if c in out.columns]
        out = out.sort_values(sort_cols, kind='mergesort')
    return out


def clean_engineered_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_odds_cols = [c for c in ['ODDS_H','ODDS_D','ODDS_A'] if c in df.columns]
    pre_odds_na_mask = df[base_odds_cols].isna().any(axis=1) if base_odds_cols else pd.Series(False, index=df.index)

    if {'HTR','FTR'}.issubset(df.columns):
        df['HTR'] = df['HTR'].fillna(df['FTR'])

    h2h_cols = [c for c in df.columns if c.startswith('h2h_')]
    form_patterns = ['_last5', '_ma_5', '_median_5', '_std_5']
    form_cols = [c for c in df.columns if any(p in c for p in form_patterns)]
    extra_form = [c for c in df.columns if c.startswith('form_diff_') or c.startswith('draw_tendency_') or '_last_' in c]
    for c in set(h2h_cols + form_cols + extra_form):
        df[c] = df[c].fillna(0)

    for c in ['home_rest_days','away_rest_days','rest_days_diff']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    if {'home_rest_days','away_rest_days','rest_days_diff'}.issubset(df.columns):
        mask_need = df['rest_days_diff'].eq(0) & (df['home_rest_days'].notna() & df['away_rest_days'].notna())
        df.loc[mask_need, 'rest_days_diff'] = df.loc[mask_need, 'home_rest_days'] - df.loc[mask_need, 'away_rest_days']

    def safe_div(a: pd.Series, b: pd.Series) -> np.ndarray:
        return np.where(b.fillna(0) > 0, a.fillna(0) / b.fillna(0), 0.0)

    def exist(*cols: str) -> bool:
        return all(c in df.columns for c in cols)

    if exist('home_pts_home_season_todate','home_matches_played_home_season_td'):
        df['home_ppg_home_season_td'] = pd.Series(safe_div(df['home_pts_home_season_todate'], df['home_matches_played_home_season_td']))
    if exist('home_pts_away_season_todate','home_matches_played_away_season_td'):
        df['home_ppg_away_season_td'] = pd.Series(safe_div(df['home_pts_away_season_todate'], df['home_matches_played_away_season_td']))
    if exist('away_pts_home_season_todate','away_matches_played_home_season_td'):
        df['away_ppg_home_season_td'] = pd.Series(safe_div(df['away_pts_home_season_todate'], df['away_matches_played_home_season_td']))
    if exist('away_pts_away_season_todate','away_matches_played_away_season_td'):
        df['away_ppg_away_season_td'] = pd.Series(safe_div(df['away_pts_away_season_todate'], df['away_matches_played_away_season_td']))

    if exist('home_pts_home_season_todate','home_pts_away_season_todate','home_matches_played_home_season_td','home_matches_played_away_season_td'):
        thp = df['home_pts_home_season_todate'].fillna(0) + df['home_pts_away_season_todate'].fillna(0)
        thm = df['home_matches_played_home_season_td'].fillna(0) + df['home_matches_played_away_season_td'].fillna(0)
        df['home_ppg_season_td'] = np.where(thm > 0, thp / thm, 0.0)
    else:
        if 'home_ppg_season_td' in df.columns:
            df['home_ppg_season_td'] = df['home_ppg_season_td'].fillna(0)

    if exist('away_pts_home_season_todate','away_pts_away_season_todate','away_matches_played_home_season_td','away_matches_played_away_season_td'):
        tap = df['away_pts_home_season_todate'].fillna(0) + df['away_pts_away_season_todate'].fillna(0)
        tam = df['away_matches_played_home_season_td'].fillna(0) + df['away_matches_played_away_season_td'].fillna(0)
        df['away_ppg_season_td'] = np.where(tam > 0, tap / tam, 0.0)
    else:
        if 'away_ppg_season_td' in df.columns:
            df['away_ppg_season_td'] = df['away_ppg_season_td'].fillna(0)

    if exist('home_ppg_season_td','away_ppg_season_td'):
        df['ppg_diff_season_td'] = df['home_ppg_season_td'].fillna(0) - df['away_ppg_season_td'].fillna(0)
    if exist('home_ppg_home_season_td','away_ppg_away_season_td'):
        df['ppg_homeaway_ctx_diff_season_td'] = df['home_ppg_home_season_td'].fillna(0) - df['away_ppg_away_season_td'].fillna(0)

    season_acc_cols = [c for c in df.columns if ('_season_todate' in c) or ('_season_td' in c) or ('matches_played' in c)]
    for c in season_acc_cols:
        df[c] = df[c].fillna(0)

    if base_odds_cols:
        keys = [k for k in ['Competition','Season'] if k in df.columns]
        if keys:
            med_group = df.groupby(keys)[base_odds_cols].transform('median')
            df[base_odds_cols] = df[base_odds_cols].fillna(med_group)
        df[base_odds_cols] = df[base_odds_cols].fillna(df[base_odds_cols].median())

        inv_h = 1.0 / df['ODDS_H']
        inv_d = 1.0 / df['ODDS_D']
        inv_a = 1.0 / df['ODDS_A']
        inv_sum = inv_h + inv_d + inv_a
        df['prob_h'] = inv_h / inv_sum
        df['prob_d'] = inv_d / inv_sum
        df['prob_a'] = inv_a / inv_sum
        df['bookmaker_margin'] = inv_sum - 1.0
        if 'odds_ha_prob_spread' in df.columns:
            df['odds_ha_prob_spread'] = (df['prob_h'] - df['prob_a']).abs()
        if 'odds_draw_over_mean_ha' in df.columns:
            df['odds_draw_over_mean_ha'] = df['prob_d'] - ((df['prob_h'] + df['prob_a']) / 2.0)
        if 'odds_draw_ratio_ha' in df.columns:
            df['odds_draw_ratio_ha'] = df['prob_d'] / ((df['prob_h'] + df['prob_a']) / 2.0)
        if 'odds_draw_vs_max_ha' in df.columns:
            df['odds_draw_vs_max_ha'] = df['prob_d'] - np.maximum(df['prob_h'], df['prob_a'])

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    df_no_odds = df.loc[~pre_odds_na_mask].copy()
    return df, df_no_odds


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Pipeline final: unificación, limpieza, features y limpieza final.')
    p.add_argument('--run', action='store_true', help='Ejecuta todo el pipeline end-to-end.')
    p.add_argument('--skip-unify', action='store_true', help='Saltar la unificación (usar unified existente).')
    p.add_argument('--unified-in', default=UNIFIED_PATH, help='Ruta del unified existente si se usa --skip-unify.')
    return p


def run_from_unified_cleaned(unified_cleaned_path: str) -> tuple[str, str]:
    _ensure_dirs()
    engineered_df = engineer_in_memory(unified_cleaned_path)
    cleaned_all_df, cleaned_no_odds_df = clean_engineered_df(engineered_df)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cleaned_all_df.to_csv(OUTPUT_ALL_PATH, index=False)
    cleaned_no_odds_df.to_csv(OUTPUT_NO_ODDS_NULLS_PATH, index=False)
    return OUTPUT_ALL_PATH, OUTPUT_NO_ODDS_NULLS_PATH


def run_from_engineered_file(engineered_path: str) -> tuple[str, str]:
    _ensure_dirs()
    df = pd.read_csv(engineered_path, low_memory=False)
    cleaned_all_df, cleaned_no_odds_df = clean_engineered_df(df)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cleaned_all_df.to_csv(OUTPUT_ALL_PATH, index=False)
    cleaned_no_odds_df.to_csv(OUTPUT_NO_ODDS_NULLS_PATH, index=False)
    return OUTPUT_ALL_PATH, OUTPUT_NO_ODDS_NULLS_PATH


def main():
    _ensure_dirs()
    args = build_argparser().parse_args()
    # Preferir engineered existente para igualar los intermedios; si no, usar unified_cleaned
    engineered_in_default = ENGINEERED_PATH
    if os.path.exists(engineered_in_default):
        out_all, out_no_odds = run_from_engineered_file(engineered_in_default)
    else:
        unified_in = args.unified_in if os.path.exists(args.unified_in) else UNIFIED_CLEANED_PATH
        if not os.path.exists(unified_in):
            raise FileNotFoundError(f'No existe el unified limpio: {unified_in}')
        out_all, out_no_odds = run_from_unified_cleaned(unified_in)
    print('Generados:')
    print(f' - {out_all}')
    print(f' - {out_no_odds}')


if __name__ == '__main__':
    main()
