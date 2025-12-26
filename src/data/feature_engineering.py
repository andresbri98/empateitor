import os
import argparse
import pandas as pd
import numpy as np


DEFAULT_INPUT = os.path.join('data', 'intermediate', 'unified_cleaned_football_data.csv')
DEFAULT_OUTPUT = os.path.join('data', 'intermediate', 'engineered_football_data.csv')
DEFAULT_OUTPUT_LITE = os.path.join('data', 'intermediate', 'engineered_football_data_lite.csv')


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if 'Date' in df.columns:
        # Try strict format first (dd/mm/yy), then fallback to dayfirst parsing
        s = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
        mask = s.isna()
        if mask.any():
            s2 = pd.to_datetime(df.loc[mask, 'Date'], dayfirst=True, errors='coerce')
            s.loc[mask] = s2
        df['Date'] = s
    return df


def prepare_long_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index().rename(columns={'index': 'match_id'})

    # Detect optional xG columns (si existen)
    xg_home_candidates = ['xG_home', 'xg_home', 'home_xg', 'HXG', 'HXg', 'HXG_total']
    xg_away_candidates = ['xG_away', 'xg_away', 'away_xg', 'AXG', 'AXg', 'AXG_total']
    xg_home_col = next((c for c in xg_home_candidates if c in df.columns), None)
    xg_away_col = next((c for c in xg_away_candidates if c in df.columns), None)

    home_cols = {
        'HomeTeam': 'Team',
        'AwayTeam': 'Opponent',
        'FTHG': 'goals_for',
        'FTAG': 'goals_against',
        'HR': 'red_cards',
        'HY': 'yellow_cards',
    }
    away_cols = {
        'AwayTeam': 'Team',
        'HomeTeam': 'Opponent',
        'FTAG': 'goals_for',
        'FTHG': 'goals_against',
        'AR': 'red_cards',
        'AY': 'yellow_cards',
    }

    common_cols = ['match_id', 'Date', 'Season', 'Division', 'Competition', 'Div', 'FTR']
    home_cols_list = common_cols + list(home_cols.keys()) + ([xg_home_col] if xg_home_col else [])
    away_cols_list = common_cols + list(away_cols.keys()) + ([xg_away_col] if xg_away_col else [])

    home = df[home_cols_list].copy()
    home = home.rename(columns=home_cols)
    home['is_home'] = True
    if xg_home_col:
        home = home.rename(columns={xg_home_col: 'xg'})
    else:
        home['xg'] = np.nan

    away = df[away_cols_list].copy()
    away = away.rename(columns=away_cols)
    away['is_home'] = False
    if xg_away_col:
        away = away.rename(columns={xg_away_col: 'xg'})
    else:
        away['xg'] = np.nan

    long_df = pd.concat([home, away], ignore_index=True)

    long_df['goals_for'] = pd.to_numeric(long_df['goals_for'], errors='coerce')
    long_df['goals_against'] = pd.to_numeric(long_df['goals_against'], errors='coerce')
    long_df['red_cards'] = pd.to_numeric(long_df['red_cards'], errors='coerce')
    long_df['yellow_cards'] = pd.to_numeric(long_df['yellow_cards'], errors='coerce')
    long_df['xg'] = pd.to_numeric(long_df['xg'], errors='coerce')
    long_df['goal_diff'] = long_df['goals_for'] - long_df['goals_against']

    # Points desde la perspectiva del equipo
    home_points = (long_df['FTR'] == 'H') & (long_df['is_home'])
    away_points = (long_df['FTR'] == 'A') & (~long_df['is_home'])
    draw_points = (long_df['FTR'] == 'D')
    long_df['points'] = 0
    long_df.loc[home_points | away_points, 'points'] = 3
    long_df.loc[draw_points, 'points'] = 1

    long_df['is_draw'] = (long_df['FTR'] == 'D').astype(int)

    # Orden para asegurar rolling correcto (por competición y equipo)
    long_df = (
        long_df.sort_values(['Competition', 'Season', 'Team', 'Date', 'match_id'], kind='mergesort')
        .reset_index(drop=True)
    )
    return long_df


def add_team_rolling_features(long_df: pd.DataFrame, windows=(5,)) -> pd.DataFrame:
    # Ventanas para medias móviles (si se desean adicionales), pero por defecto usamos 5
    metrics_ma = ['goals_for', 'goals_against', 'goal_diff', 'points', 'is_draw', 'red_cards', 'yellow_cards']
    for w in windows:
        for m in metrics_ma:
            col = f'{m}_ma_{w}'
            long_df[col] = (
                long_df.groupby(['Competition', 'Season', 'Team'], sort=False)[m]
                .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
            )

        long_df[f'matches_played_{w}'] = (
            long_df.groupby(['Competition', 'Season', 'Team'], sort=False)['match_id']
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).count())
        )

    # Formas requeridas: últimos 5 partidos (sumas)
    long_df['gf_last5'] = (
        long_df.groupby(['Competition', 'Season', 'Team'], sort=False)['goals_for']
        .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).sum())
    )
    long_df['ga_last5'] = (
        long_df.groupby(['Competition', 'Season', 'Team'], sort=False)['goals_against']
        .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).sum())
    )
    long_df['pts_last5'] = (
        long_df.groupby(['Competition', 'Season', 'Team'], sort=False)['points']
        .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).sum())
    )

    # xG en últimos 5 si existe
    if 'xg' in long_df.columns:
        long_df['xg_last5'] = (
            long_df.groupby(['Competition', 'Season', 'Team'], sort=False)['xg']
            .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).sum())
        )
    else:
        long_df['xg_last5'] = np.nan

    # Features de último partido (sin fuga de datos)
    long_df['last_points'] = long_df.groupby(['Competition', 'Season', 'Team'], sort=False)['points'].shift(1)
    long_df['last_goal_diff'] = long_df.groupby(['Competition', 'Season', 'Team'], sort=False)['goal_diff'].shift(1)
    long_df['last_is_draw'] = long_df.groupby(['Competition', 'Season', 'Team'], sort=False)['is_draw'].shift(1)

    # Contexto específico: últimos 5 solo como local o solo como visitante
    grp = long_df.groupby(['Competition', 'Season', 'Team'], sort=False)

    def _rolling_sum_context(g: pd.DataFrame, col: str, mask: pd.Series, w: int = 5):
        prev = g[col].shift(1)
        prev = prev.where(mask, 0.0)
        return prev.rolling(window=w, min_periods=1).sum()

    for _, g in grp:
        idx = g.index
        # Solo partidos previos del equipo actuando de local
        long_df.loc[idx, 'gf_last5_homeonly'] = _rolling_sum_context(g, 'goals_for', g['is_home'])
        long_df.loc[idx, 'ga_last5_homeonly'] = _rolling_sum_context(g, 'goals_against', g['is_home'])
        long_df.loc[idx, 'pts_last5_homeonly'] = _rolling_sum_context(g, 'points', g['is_home'])
        # Solo partidos previos del equipo actuando de visitante
        away_mask = ~g['is_home']
        long_df.loc[idx, 'gf_last5_awayonly'] = _rolling_sum_context(g, 'goals_for', away_mask)
        long_df.loc[idx, 'ga_last5_awayonly'] = _rolling_sum_context(g, 'goals_against', away_mask)
        long_df.loc[idx, 'pts_last5_awayonly'] = _rolling_sum_context(g, 'points', away_mask)
    return long_df


def add_season_to_date_home_away(long_df: pd.DataFrame) -> pd.DataFrame:
    # Variables específicas por condición de local/visitante acumuladas en la temporada hasta el partido
    long_df['gf_home'] = np.where(long_df['is_home'], long_df['goals_for'], 0.0)
    long_df['ga_home'] = np.where(long_df['is_home'], long_df['goals_against'], 0.0)
    long_df['pts_home'] = np.where(long_df['is_home'], long_df['points'], 0.0)

    long_df['gf_away'] = np.where(~long_df['is_home'], long_df['goals_for'], 0.0)
    long_df['ga_away'] = np.where(~long_df['is_home'], long_df['goals_against'], 0.0)
    long_df['pts_away'] = np.where(~long_df['is_home'], long_df['points'], 0.0)

    grp = long_df.groupby(['Competition', 'Team', 'Season'], sort=False)
    long_df['gf_home_season_td'] = grp['gf_home'].cumsum().shift(0)  # cumsum incluye el partido actual
    long_df['ga_home_season_td'] = grp['ga_home'].cumsum().shift(0)
    long_df['pts_home_season_td'] = grp['pts_home'].cumsum().shift(0)

    long_df['gf_away_season_td'] = grp['gf_away'].cumsum().shift(0)
    long_df['ga_away_season_td'] = grp['ga_away'].cumsum().shift(0)
    long_df['pts_away_season_td'] = grp['pts_away'].cumsum().shift(0)

    # Excluir el partido actual (hasta ese encuentro)
    long_df['gf_home_season_td'] = grp['gf_home_season_td'].shift(1)
    long_df['ga_home_season_td'] = grp['ga_home_season_td'].shift(1)
    long_df['pts_home_season_td'] = grp['pts_home_season_td'].shift(1)

    long_df['gf_away_season_td'] = grp['gf_away_season_td'].shift(1)
    long_df['ga_away_season_td'] = grp['ga_away_season_td'].shift(1)
    long_df['pts_away_season_td'] = grp['pts_away_season_td'].shift(1)

    # Puntos totales de temporada hasta el partido (todas las condiciones)
    long_df['pts_season_td'] = grp['points'].cumsum().shift(1)

    # Partidos jugados hasta el encuentro
    long_df['matches_played_overall_season_td'] = grp.cumcount()
    long_df['matches_played_home_season_td'] = grp['is_home'].transform(lambda s: s.astype(int).shift(1).cumsum())
    long_df['matches_played_away_season_td'] = grp['is_home'].transform(lambda s: (~s).astype(int).shift(1).cumsum())

    # Puntos por partido (PPG) hasta el encuentro
    def _safe_div(a, b):
        return np.where((b.notna()) & (b > 0), a / b, np.nan)

    long_df['ppg_season_td'] = _safe_div(long_df['pts_season_td'], long_df['matches_played_overall_season_td'])
    long_df['ppg_home_season_td'] = _safe_div(long_df['pts_home_season_td'], long_df['matches_played_home_season_td'])
    long_df['ppg_away_season_td'] = _safe_div(long_df['pts_away_season_td'], long_df['matches_played_away_season_td'])

    return long_df


def add_low_scoring_features(long_df: pd.DataFrame) -> pd.DataFrame:
    # Señales de baja anotación (correlacionadas con empates)
    long_df['goals_total_match'] = long_df['goals_for'] + long_df['goals_against']
    ggrp = long_df.groupby(['Competition', 'Season', 'Team'], sort=False)['goals_total_match']
    long_df['goals_total_median_5'] = ggrp.transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).median())
    long_df['goals_total_std_5'] = ggrp.transform(lambda s: s.shift(1).rolling(window=5, min_periods=2).std())
    return long_df


def add_competition_season_zscores(long_df: pd.DataFrame) -> pd.DataFrame:
    # Z-scores por competición y season (comparabilidad inter-ligas)
    z_cols = ['pts_last5', 'gf_last5', 'ga_last5', 'goals_total_median_5']
    for c in z_cols:
        if c in long_df.columns:
            grp = long_df.groupby(['Competition', 'Season'])[c]
            mean = grp.transform('mean')
            std = grp.transform('std')
            long_df[f'{c}_z'] = np.where(std > 0, (long_df[c] - mean) / std, 0.0)
    return long_df


def add_rest_days(long_df: pd.DataFrame) -> pd.DataFrame:
    # Días de descanso desde el último partido por equipo y competición
    long_df['rest_days'] = (
        long_df.groupby(['Competition', 'Season', 'Team'], sort=False)['Date']
        .transform(lambda s: (s - s.shift(1)).dt.days)
    )
    return long_df


def add_h2h_last5years(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features H2H de los últimos 5 años para cada partido.
    - No se resetea por season, pero sí se ignoran encuentros con más de 5 años.
    - Calculado por competición y par de equipos (independiente del orden).
    """
    if 'Date' not in df.columns:
        return pd.DataFrame({'match_id': [],
                     'h2h_home_gf_5y': [], 'h2h_home_ga_5y': [], 'h2h_home_pts_5y': [],
                     'h2h_away_gf_5y': [], 'h2h_away_ga_5y': [], 'h2h_away_pts_5y': [],
                     'h2h_meetings_5y': [], 'h2h_draws_5y': []})

    m = df.reset_index().rename(columns={'index': 'match_id'}).copy()
    # Asegurar tipos
    m['Date'] = pd.to_datetime(m['Date'], dayfirst=True, errors='coerce')
    for col in ['FTHG', 'FTAG']:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors='coerce')

    # Puntos por partido, perspectiva de local y de visitante
    m['points_home'] = np.where(m['FTR'] == 'H', 3, np.where(m['FTR'] == 'D', 1, 0))
    m['points_away'] = np.where(m['FTR'] == 'A', 3, np.where(m['FTR'] == 'D', 1, 0))

    # Clave del par de equipos (independiente del orden)
    pair_keys = np.where(m['HomeTeam'] < m['AwayTeam'],
                         m['HomeTeam'] + '|' + m['AwayTeam'],
                         m['AwayTeam'] + '|' + m['HomeTeam'])
    m['pair_key'] = pair_keys

    curr = m[['match_id', 'Date', 'Competition', 'HomeTeam', 'AwayTeam', 'pair_key']].copy()
    past = m[['match_id', 'Date', 'Competition', 'HomeTeam', 'AwayTeam', 'pair_key', 'FTHG', 'FTAG', 'points_home', 'points_away']].copy()
    past = past.rename(columns={
        'match_id': 'match_id_past', 'Date': 'Date_past', 'HomeTeam': 'HomeTeam_past',
        'AwayTeam': 'AwayTeam_past', 'FTHG': 'FTHG_past', 'FTAG': 'FTAG_past',
        'points_home': 'points_home_past', 'points_away': 'points_away_past'
    })

    merged = curr.merge(
        past,
        on=['Competition', 'pair_key'],
        how='left',
        suffixes=('', '_p')
    )

    # Solo partidos anteriores y en los últimos 5 años
    merged = merged[merged['Date_past'] < merged['Date']]
    # 5 años ~ 1826 días (considerando un día extra por bisiestos)
    merged['delta_days'] = (merged['Date'] - merged['Date_past']).dt.days
    merged = merged[merged['delta_days'] <= 1826]

    # GF/GA/PTS desde la perspectiva del equipo local actual
    same_side_home = merged['HomeTeam'] == merged['HomeTeam_past']
    merged['h2h_home_gf'] = np.where(same_side_home, merged['FTHG_past'], merged['FTAG_past'])
    merged['h2h_home_ga'] = np.where(same_side_home, merged['FTAG_past'], merged['FTHG_past'])
    merged['h2h_home_pts'] = np.where(same_side_home, merged['points_home_past'], merged['points_away_past'])

    # GF/GA/PTS desde la perspectiva del equipo visitante actual
    same_side_away = merged['AwayTeam'] == merged['HomeTeam_past']
    merged['h2h_away_gf'] = np.where(same_side_away, merged['FTHG_past'], merged['FTAG_past'])
    merged['h2h_away_ga'] = np.where(same_side_away, merged['FTAG_past'], merged['FTHG_past'])
    merged['h2h_away_pts'] = np.where(same_side_away, merged['points_home_past'], merged['points_away_past'])

    merged['h2h_draw_flag'] = ((merged['points_home_past'] == 1) & (merged['points_away_past'] == 1)).astype(int)

    agg = merged.groupby('match_id', as_index=False).agg(
        h2h_home_gf_5y=('h2h_home_gf', 'sum'),
        h2h_home_ga_5y=('h2h_home_ga', 'sum'),
        h2h_home_pts_5y=('h2h_home_pts', 'sum'),
        h2h_away_gf_5y=('h2h_away_gf', 'sum'),
        h2h_away_ga_5y=('h2h_away_ga', 'sum'),
        h2h_away_pts_5y=('h2h_away_pts', 'sum'),
        h2h_meetings_5y=('match_id_past', 'count'),
        h2h_draws_5y=('h2h_draw_flag', 'sum')
    )

    return agg


def add_draw_focused_match_features(out: pd.DataFrame) -> pd.DataFrame:
    # Odds-based signals
    if {'prob_h', 'prob_a', 'prob_d'}.issubset(out.columns):
        out['odds_ha_prob_spread'] = (out['prob_h'] - out['prob_a']).abs()
        mean_ha = (out['prob_h'] + out['prob_a']) / 2.0
        out['odds_draw_over_mean_ha'] = out['prob_d'] - mean_ha
        out['odds_draw_ratio_ha'] = np.where(mean_ha > 0, out['prob_d'] / mean_ha, np.nan)
        out['odds_draw_vs_max_ha'] = out['prob_d'] - np.maximum(out['prob_h'], out['prob_a'])

    # Symmetry metrics (last 5)
    if {'home_pts_last5', 'away_pts_last5'}.issubset(out.columns):
        out['form_diff_pts5_abs'] = (out['home_pts_last5'] - out['away_pts_last5']).abs()
    if {'home_gf_last5', 'home_ga_last5', 'away_gf_last5', 'away_ga_last5'}.issubset(out.columns):
        home_gd5 = out['home_gf_last5'] - out['home_ga_last5']
        away_gd5 = out['away_gf_last5'] - out['away_ga_last5']
        out['form_diff_gd5_abs'] = (home_gd5 - away_gd5).abs()
        out['goals_total_last5'] = (out['home_gf_last5'] + out['home_ga_last5'] + out['away_gf_last5'] + out['away_ga_last5'])

    # Draw tendency (rolling mean of draws)
    if {'home_is_draw_ma_5', 'away_is_draw_ma_5'}.issubset(out.columns):
        out['draw_tendency_mean_5'] = (out['home_is_draw_ma_5'] + out['away_is_draw_ma_5']) / 2.0
        out['draw_tendency_diff_5'] = (out['home_is_draw_ma_5'] - out['away_is_draw_ma_5']).abs()

    # H2H draw rate
    if {'h2h_draws_5y', 'h2h_meetings_5y'}.issubset(out.columns):
        out['h2h_draw_rate_5y'] = np.where(out['h2h_meetings_5y'] > 0, out['h2h_draws_5y'] / out['h2h_meetings_5y'], np.nan)

    return out


def split_home_away_features(long_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # Seleccionar columnas de features relevantes
    exclude = ['Team', 'Opponent', 'FTR', 'Season', 'Division', 'Competition', 'Div', 'Date', 'is_home']
    # Excluir métricas del partido actual y columnas cruzadas no informativas
    exclude_raw = [
        'goals_for', 'goals_against', 'goal_diff', 'points', 'is_draw', 'red_cards', 'yellow_cards', 'xg',
        'gf_home', 'ga_home', 'pts_home', 'gf_away', 'ga_away', 'pts_away'
    ]
    feature_cols = [c for c in long_df.columns if c not in exclude + exclude_raw + ['match_id']]

    home_feat = long_df[long_df['is_home'] == True][['match_id'] + feature_cols].copy()
    away_feat = long_df[long_df['is_home'] == False][['match_id'] + feature_cols].copy()

    home_feat = home_feat.add_prefix('home_')
    away_feat = away_feat.add_prefix('away_')

    # Restaurar el nombre de la clave
    home_feat = home_feat.rename(columns={'home_match_id': 'match_id'})
    away_feat = away_feat.rename(columns={'away_match_id': 'match_id'})

    # Renombrar algunas columnas clave para mayor claridad solicitada
    rename_map_home = {
        'home_gf_last5': 'home_gf_last5',
        'home_ga_last5': 'home_ga_last5',
        'home_pts_last5': 'home_pts_last5',
        # Contexto específico (local como local)
        'home_gf_last5_homeonly': 'home_gf_last5_homeonly',
        'home_ga_last5_homeonly': 'home_ga_last5_homeonly',
        'home_pts_last5_homeonly': 'home_pts_last5_homeonly',
        'home_gf_home_season_td': 'home_gf_home_season_todate',
        'home_ga_home_season_td': 'home_ga_home_season_todate',
        'home_pts_home_season_td': 'home_pts_home_season_todate',
        # Mantener acumulados cruzados (informativos)
        'home_gf_away_season_td': 'home_gf_away_season_todate',
        'home_ga_away_season_td': 'home_ga_away_season_todate',
        'home_pts_away_season_td': 'home_pts_away_season_todate',
    }
    rename_map_away = {
        'away_gf_last5': 'away_gf_last5',
        'away_ga_last5': 'away_ga_last5',
        'away_pts_last5': 'away_pts_last5',
        # Contexto específico (visitante como visitante)
        'away_gf_last5_awayonly': 'away_gf_last5_awayonly',
        'away_ga_last5_awayonly': 'away_ga_last5_awayonly',
        'away_pts_last5_awayonly': 'away_pts_last5_awayonly',
        'away_gf_home_season_td': 'away_gf_home_season_todate',
        'away_ga_home_season_td': 'away_ga_home_season_todate',
        'away_pts_home_season_td': 'away_pts_home_season_todate',
        'away_gf_away_season_td': 'away_gf_away_season_todate',
        'away_ga_away_season_td': 'away_ga_away_season_todate',
        'away_pts_away_season_td': 'away_pts_away_season_todate',
    }
    home_feat = home_feat.rename(columns=rename_map_home)
    away_feat = away_feat.rename(columns=rename_map_away)
    return home_feat, away_feat


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['ODDS_H', 'ODDS_D', 'ODDS_A']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    inv_h = 1.0 / df['ODDS_H'] if 'ODDS_H' in df.columns else np.nan
    inv_d = 1.0 / df['ODDS_D'] if 'ODDS_D' in df.columns else np.nan
    inv_a = 1.0 / df['ODDS_A'] if 'ODDS_A' in df.columns else np.nan

    implied_sum = inv_h + inv_d + inv_a
    df['prob_h'] = inv_h / implied_sum
    df['prob_d'] = inv_d / implied_sum
    df['prob_a'] = inv_a / implied_sum
    df['bookmaker_margin'] = implied_sum - 1.0
    return df


def engineer_features(input_path: str, output_path: str, windows=(5,), output_lite_path: str | None = None) -> tuple[str, str | None]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'No se encontró el archivo de entrada: {input_path}')

    df = pd.read_csv(input_path, low_memory=False, dtype={'Division': 'string', 'Div': 'string'})
    df = parse_dates(df)

    # Keep essential columns even if some optional stats are missing
    expected_cols = [
        'Season', 'Division', 'Competition', 'Div', 'Date',
        'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
        'HR', 'HY', 'AR', 'AY', 'ODDS_H', 'ODDS_D', 'ODDS_A'
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f'Aviso: Faltan columnas esperadas: {missing}. Continuando con las disponibles.')

    # Build long format and compute rolling stats + season-to-date + rest days
    long_df = prepare_long_format(df)
    long_df = add_team_rolling_features(long_df, windows=windows)
    long_df = add_season_to_date_home_away(long_df)
    long_df = add_rest_days(long_df)
    long_df = add_low_scoring_features(long_df)
    long_df = add_competition_season_zscores(long_df)
    home_feat, away_feat = split_home_away_features(long_df)

    # H2H últimos 5 años (por competición y par de equipos)
    h2h = add_h2h_last5years(df)

    # Merge features back to the original match-level dataframe
    base_cols = [c for c in df.columns]  # preserve all original columns
    out = df.reset_index().rename(columns={'index': 'match_id'})[base_cols + ['match_id']]
    out = out.merge(home_feat, on='match_id', how='left').merge(away_feat, on='match_id', how='left')
    out = out.merge(h2h, on='match_id', how='left')
    out = add_odds_features(out)
    out = add_draw_focused_match_features(out)

    # Si las columnas xG están completamente vacías, elimínalas para evitar ruido
    out = drop_empty_xg_columns(out)

    # Diferenciales útiles (tras imputación)
    if {'home_pts_last5', 'away_pts_last5'}.issubset(out.columns):
        out['pts_last5_diff'] = out['home_pts_last5'] - out['away_pts_last5']
    if {'home_gf_last5', 'away_gf_last5'}.issubset(out.columns):
        out['gf_last5_diff'] = out['home_gf_last5'] - out['away_gf_last5']
    if {'home_ga_last5', 'away_ga_last5'}.issubset(out.columns):
        out['ga_last5_diff'] = out['home_ga_last5'] - out['away_ga_last5']
    if {'home_rest_days', 'away_rest_days'}.issubset(out.columns):
        out['rest_days_diff'] = out['home_rest_days'] - out['away_rest_days']
    if {'home_ppg_season_td', 'away_ppg_season_td'}.issubset(out.columns):
        out['ppg_diff_season_td'] = out['home_ppg_season_td'] - out['away_ppg_season_td']
    if {'home_ppg_home_season_td', 'away_ppg_away_season_td'}.issubset(out.columns):
        out['ppg_homeaway_ctx_diff_season_td'] = out['home_ppg_home_season_td'] - out['away_ppg_away_season_td']

    # Sort by date for readability
    if 'Date' in out.columns:
        sort_cols = [c for c in ['Date', 'Competition', 'Season', 'Div'] if c in out.columns]
        out = out.sort_values(sort_cols, kind='mergesort')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

    lite_written = None
    # Generate lite dataset if requested (path provided) else default to write alongside full
    if output_lite_path is None:
        output_lite_path = DEFAULT_OUTPUT_LITE
    if output_lite_path:
        out_lite = build_lite_dataset(out)
        os.makedirs(os.path.dirname(output_lite_path), exist_ok=True)
        out_lite.to_csv(output_lite_path, index=False)
        lite_written = output_lite_path

    return output_path, lite_written


def build_lite_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Base identifiers and target
    base_cols = [
        'Season', 'Division', 'Competition', 'Div', 'Date',
        'HomeTeam', 'AwayTeam', 'FTR', 'ODDS_H', 'ODDS_D', 'ODDS_A'
    ]

    # Core predictive features for draws
    core_cols = [
        'prob_h', 'prob_d', 'prob_a', 'bookmaker_margin', 'odds_ha_prob_spread',
        'home_pts_last5', 'away_pts_last5', 'pts_last5_diff', 'form_diff_pts5_abs', 'form_diff_gd5_abs',
        'home_ppg_home_season_td', 'away_ppg_away_season_td', 'ppg_homeaway_ctx_diff_season_td',
        'rest_days_diff',
        'h2h_draw_rate_5y', 'h2h_meetings_5y',
        'home_goals_total_median_5', 'away_goals_total_median_5'
    ]

    # Keep only columns that exist
    cols = [c for c in base_cols + core_cols if c in df.columns]
    return df[cols].copy()


def drop_empty_xg_columns(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        'home_xg', 'away_xg', 'home_xg_last5', 'away_xg_last5'
    ]
    to_drop = [c for c in candidates if c in df.columns and not df[c].notna().any()]
    if to_drop:
        return df.drop(columns=to_drop)
    return df


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Feature engineering para datos de fútbol unificados.')
    p.add_argument('--input', '-i', default=DEFAULT_INPUT, help='Ruta del CSV de entrada (limpio).')
    p.add_argument('--output', '-o', default=DEFAULT_OUTPUT, help='Ruta del CSV de salida con features (completo).')
    p.add_argument('--output-lite', '-ol', default=DEFAULT_OUTPUT_LITE, help='Ruta del CSV de salida "lite" con columnas esenciales.')
    p.add_argument('--windows', '-w', type=int, nargs='+', default=[5], help='Ventanas de medias móviles por equipo (para *_ma_*).')
    return p


def main():
    args = build_argparser().parse_args()
    output_path, lite_path = engineer_features(args.input, args.output, windows=tuple(args.windows), output_lite_path=args.output_lite)
    print(f'Features (completo) generado en: {output_path}')
    if lite_path:
        print(f'Features (lite) generado en: {lite_path}')


if __name__ == '__main__':
    main()
