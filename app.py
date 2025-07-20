import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Inicializar session_state si no existe - DEBE ESTAR AL PRINCIPIO
if 'df_con_rendimiento' not in st.session_state:
    st.session_state.df_con_rendimiento = None

if 'ticker_seleccionado' not in st.session_state:
    st.session_state.ticker_seleccionado = "ALAB"

if 'current_page' not in st.session_state:
    st.session_state.current_page = "principal"

st.title("Dashboard Financiero ALAB")
st.write("Datos extraídos de Yahoo Finance")

# === SISTEMA DE NAVEGACIÓN ===
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("📊 Dashboard Principal", type="primary" if st.session_state.current_page == "principal" else "secondary", use_container_width=True):
        st.session_state.current_page = "principal"

with col2:
    if st.button("🔍 Patrones Identificados", type="primary" if st.session_state.current_page == "patrones" else "secondary", use_container_width=True):
        st.session_state.current_page = "patrones"

with col3:
    if st.session_state.df_con_rendimiento is not None:
        st.success(f"✅ Datos cargados para {st.session_state.ticker_seleccionado}")
    else:
        st.warning("⚠️ Cargue datos primero")

st.markdown("---")

# === FUNCIÓN PARA OBTENER DATOS DE YAHOO FINANCE ===
def get_stock_data(ticker, period, custom_start=None, custom_end=None):
    try:
        # Mapeo de períodos personalizados a Yahoo Finance
        yahoo_period_map = {
            # Períodos básicos directos
            "1d": "1d",
            "5d": "5d", 
            "1w": "5d",
            "1mo": "1mo",
            "3mo": "3mo",
            "6mo": "6mo",
            "ytd": "ytd",
            "1y": "1y",
            "2y": "2y",
            "3y": "3y",
            "5y": "5y",
            "max": "max",
            
            # Períodos customizados
            "yesterday": "2d",  # Últimos 2 días para incluir ayer
            "3w": "21d",
            "3q": "9mo",
            "mtd": "1mo",  # Mes hasta la fecha
            "qtd": "3mo", # Trimestre hasta la fecha
            
            # Períodos relativos pasados
            "this_week": "5d",
            "this_month": "1mo",
            "this_quarter": "3mo", 
            "last_week": "2w",
            "last_month": "2mo",
            "last_quarter": "6mo",
            "last_year": "2y",
            
            # Períodos futuros (limitados por datos disponibles)
            "next_5d": "5d",
            "next_week": "5d",
            "next_2w": "10d", 
            "next_30d": "1mo",
            "next_month": "1mo",
            "next_3m": "3mo",
            "next_6m": "6mo",
            "next_quarter": "3mo",
            "next_year": "1y",
            "next_3y": "3y",
            
            # Personalizable
            "custom": "1y"  # Default para personalizable
        }
        
        # Obtener el período de Yahoo Finance
        yahoo_period = yahoo_period_map.get(period, "1y")
        
        # Construir URL según el tipo de período
        if yahoo_period in ["1d", "5d", "1mo", "3mo", "6mo", "ytd", "1y", "2y", "3y", "5y", "max"]:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={yahoo_period}&interval=1d"
        else:
            # Para períodos en días
            days = yahoo_period.replace('d', '')
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={days}d&interval=1d"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            timestamps = data['chart']['result'][0]['timestamp']
            quotes = data['chart']['result'][0]['indicators']['quote'][0]

            df = pd.DataFrame({
                'Date': pd.to_datetime(timestamps, unit='s'),
                'Open': quotes['open'],
                'High': quotes['high'],
                'Low': quotes['low'],
                'Close': quotes['close'],
                'Volume': quotes['volume']
            })

            # Filtros adicionales para períodos específicos
            now = pd.Timestamp.now()
            
            # Intervalo personalizable con fechas específicas
            if period == "custom" and custom_start and custom_end:
                df = df[(df['Date'].dt.date >= custom_start) & (df['Date'].dt.date <= custom_end)]
            
            # Períodos específicos de día
            elif period == "yesterday":
                yesterday = now - pd.Timedelta(days=1)
                df = df[df['Date'].dt.date == yesterday.date()]
            
            # Períodos "hasta la fecha"
            elif period == "mtd":  # Mes hasta la fecha
                start_month = now.replace(day=1)
                df = df[df['Date'] >= start_month.normalize()]
            elif period == "qtd":  # Trimestre hasta la fecha
                quarter_start = pd.Timestamp(now.year, ((now.quarter - 1) * 3) + 1, 1)
                df = df[df['Date'] >= quarter_start]
            
            # Períodos "este/esta"
            elif period == "this_week":
                start_week = now - pd.Timedelta(days=now.weekday())
                df = df[df['Date'] >= start_week.normalize()]
            elif period == "this_month":
                start_month = now.replace(day=1)
                df = df[df['Date'] >= start_month.normalize()]
            elif period == "this_quarter":
                quarter_start = pd.Timestamp(now.year, ((now.quarter - 1) * 3) + 1, 1)
                df = df[df['Date'] >= quarter_start]
            
            # Períodos "anterior/pasado"
            elif period == "last_week":
                end_week = now - pd.Timedelta(days=now.weekday())
                start_week = end_week - pd.Timedelta(days=7)
                df = df[(df['Date'] >= start_week.normalize()) & (df['Date'] < end_week.normalize())]
            elif period == "last_month":
                if now.month == 1:
                    last_month = now.replace(year=now.year-1, month=12, day=1)
                    this_month = now.replace(day=1)
                else:
                    last_month = now.replace(month=now.month-1, day=1)
                    this_month = now.replace(day=1)
                df = df[(df['Date'] >= last_month.normalize()) & (df['Date'] < this_month.normalize())]
            elif period == "last_quarter":
                if now.quarter == 1:
                    last_q_start = pd.Timestamp(now.year-1, 10, 1)  # Q4 del año anterior
                    this_q_start = pd.Timestamp(now.year, 1, 1)
                else:
                    last_q_start = pd.Timestamp(now.year, ((now.quarter - 2) * 3) + 1, 1)
                    this_q_start = pd.Timestamp(now.year, ((now.quarter - 1) * 3) + 1, 1)
                df = df[(df['Date'] >= last_q_start) & (df['Date'] < this_q_start)]
            elif period == "last_year":
                last_year_start = pd.Timestamp(now.year-1, 1, 1)
                this_year_start = pd.Timestamp(now.year, 1, 1)
                df = df[(df['Date'] >= last_year_start) & (df['Date'] < this_year_start)]
            
            # Períodos futuros (nota: limitados por datos disponibles)
            elif period in ["next_5d", "next_week", "next_2w", "next_30d", "next_month", "next_3m", "next_6m", "next_quarter", "next_year", "next_3y"]:
                # Para períodos futuros, mostramos mensaje informativo
                st.info(f"📅 Período futuro seleccionado: '{period_label}'. Mostrando datos históricos disponibles.")
            
            # Intervalo personalizable
            elif period == "custom":
                st.sidebar.info("🛠️ Función de intervalo personalizable próximamente disponible")

            return df
        else:
            st.warning(f"No se pudieron obtener datos para {ticker}. Código de estado: {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"Error al procesar datos para {ticker}: {e}")
        return None

# === FUNCIONES PARA PATRONES DE TRADING ===

def analyze_volatility_patterns(df):
    """Análisis de patrones de volatilidad"""
    if df is None or len(df) < 30:
        return None
    
    # Calcular volatilidad realizada en ventanas móviles
    df['Vol_5d'] = df['Rendimiento'].rolling(5).std() * np.sqrt(5)
    df['Vol_10d'] = df['Rendimiento'].rolling(10).std() * np.sqrt(10) 
    df['Vol_20d'] = df['Rendimiento'].rolling(20).std() * np.sqrt(20)
    
    # Percentiles de volatilidad
    vol_20d_clean = df['Vol_20d'].dropna()
    vol_percentiles = {
        'p25': vol_20d_clean.quantile(0.25),
        'p50': vol_20d_clean.quantile(0.50),
        'p75': vol_20d_clean.quantile(0.75),
        'p90': vol_20d_clean.quantile(0.90)
    }
    
    # Clasificar regímenes de volatilidad
    df['Vol_Regime'] = 'Normal'
    df.loc[df['Vol_20d'] < vol_percentiles['p25'], 'Vol_Regime'] = 'Baja'
    df.loc[df['Vol_20d'] > vol_percentiles['p75'], 'Vol_Regime'] = 'Alta'
    df.loc[df['Vol_20d'] > vol_percentiles['p90'], 'Vol_Regime'] = 'Extrema'
    
    # Duración de regímenes
    df['Regime_Change'] = (df['Vol_Regime'] != df['Vol_Regime'].shift()).cumsum()
    regime_durations = df.groupby(['Regime_Change', 'Vol_Regime']).size()
    avg_durations = regime_durations.groupby(level=1).mean()
    
    # Retornos por régimen de volatilidad
    regime_returns = df.groupby('Vol_Regime')['Rendimiento'].agg([
        'mean', 'std', 'count', 'min', 'max'
    ]) * 100
    
    return {
        'data': df,
        'percentiles': vol_percentiles,
        'avg_durations': avg_durations,
        'regime_returns': regime_returns,
        'current_regime': df['Vol_Regime'].iloc[-1] if not df.empty else None
    }

def analyze_temporal_patterns(df):
    """Análisis de patrones temporales y estacionales"""
    if df is None or len(df) < 60:
        return None
    
    df_temp = df.copy()
    df_temp['Month'] = df_temp['Date'].dt.month
    df_temp['DayOfWeek'] = df_temp['Date'].dt.dayofweek
    df_temp['Hour'] = df_temp['Date'].dt.hour
    df_temp['Quarter'] = df_temp['Date'].dt.quarter
    
    # Mapeo de días
    day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    # Análisis por día de la semana
    daily_stats = df_temp.groupby('DayOfWeek')['Rendimiento'].agg([
        'mean', 'std', 'count', 
        lambda x: (x > 0).sum() / len(x)  # Win rate
    ]).round(4)
    daily_stats.columns = ['Promedio', 'Volatilidad', 'Observaciones', 'Tasa_Exito']
    daily_stats.index = [day_names[i] for i in daily_stats.index if i < len(day_names)]
    
    # Análisis por mes
    monthly_stats = df_temp.groupby('Month')['Rendimiento'].agg([
        'mean', 'std', 'count',
        lambda x: (x > 0).sum() / len(x)
    ]).round(4)
    monthly_stats.columns = ['Promedio', 'Volatilidad', 'Observaciones', 'Tasa_Exito']
    monthly_stats.index = [month_names[i-1] for i in monthly_stats.index]
    
    # Análisis por trimestre
    quarterly_stats = df_temp.groupby('Quarter')['Rendimiento'].agg([
        'mean', 'std', 'count',
        lambda x: (x > 0).sum() / len(x)
    ]).round(4)
    quarterly_stats.columns = ['Promedio', 'Volatilidad', 'Observaciones', 'Tasa_Exito']
    
    # Efecto "Sell in May" - Comparar mayo-septiembre vs resto
    df_temp['SellInMay'] = df_temp['Month'].isin([5, 6, 7, 8, 9])
    sell_in_may_stats = df_temp.groupby('SellInMay')['Rendimiento'].agg([
        'mean', 'std', 'count'
    ]).round(4)
    sell_in_may_stats.index = ['Oct-Abr', 'May-Sep']
    
    return {
        'daily_stats': daily_stats,
        'monthly_stats': monthly_stats,
        'quarterly_stats': quarterly_stats,
        'sell_in_may': sell_in_may_stats
    }

def analyze_momentum_patterns(df):
    """Análisis de patrones de momentum"""
    if df is None or len(df) < 50:
        return None
    
    df_mom = df.copy()
    
    # Calcular momentum en diferentes períodos
    periods = [5, 10, 20, 50]
    for period in periods:
        if len(df_mom) > period:
            df_mom[f'Momentum_{period}d'] = df_mom['Close'].pct_change(period) * 100
            df_mom[f'Future_Return_{period}d'] = df_mom['Rendimiento'].shift(-period).rolling(period).sum() * 100
    
    # Análisis de persistencia del momentum
    momentum_analysis = {}
    for period in periods:
        mom_col = f'Momentum_{period}d'
        fut_col = f'Future_Return_{period}d'
        
        if mom_col in df_mom.columns and fut_col in df_mom.columns:
            valid_data = df_mom[[mom_col, fut_col]].dropna()
            
            if len(valid_data) > 20:
                # Dividir en quintiles
                valid_data['Momentum_Quintile'] = pd.qcut(
                    valid_data[mom_col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
                )
                
                quintile_analysis = valid_data.groupby('Momentum_Quintile')[fut_col].agg([
                    'mean', 'std', 'count'
                ]).round(2)
                
                # Calcular spread (Q5 - Q1)
                momentum_spread = quintile_analysis.loc['Q5', 'mean'] - quintile_analysis.loc['Q1', 'mean']
                
                momentum_analysis[period] = {
                    'quintiles': quintile_analysis,
                    'spread': momentum_spread,
                    'correlation': valid_data[mom_col].corr(valid_data[fut_col])
                }
    
    return momentum_analysis

def analyze_gap_patterns(df):
    """Análisis de patrones de gaps"""
    if df is None or len(df) < 30:
        return None
    
    df_gap = df.copy()
    
    # Calcular gaps (diferencia entre apertura y cierre anterior)
    df_gap['Gap'] = ((df_gap['Open'] - df_gap['Close'].shift(1)) / df_gap['Close'].shift(1)) * 100
    df_gap['Gap_Size'] = abs(df_gap['Gap'])
    
    # Clasificar tipos de gaps
    df_gap['Gap_Type'] = 'Normal'
    df_gap.loc[df_gap['Gap'] > 2, 'Gap_Type'] = 'Gap Up Significativo'
    df_gap.loc[df_gap['Gap'] < -2, 'Gap_Type'] = 'Gap Down Significativo'
    df_gap.loc[(df_gap['Gap'] > 0.5) & (df_gap['Gap'] <= 2), 'Gap_Type'] = 'Gap Up Menor'
    df_gap.loc[(df_gap['Gap'] < -0.5) & (df_gap['Gap'] >= -2), 'Gap_Type'] = 'Gap Down Menor'
    
    # Análisis de reversión de gaps (gap fill)
    df_gap['Gap_Fill'] = False
    df_gap['Days_To_Fill'] = np.nan
    
    for i in range(1, len(df_gap)):
        if abs(df_gap['Gap'].iloc[i]) > 0.5:  # Solo gaps significativos
            gap_price = df_gap['Close'].iloc[i-1]
            
            # Buscar si el gap se llena en los próximos 10 días
            for j in range(i+1, min(i+11, len(df_gap))):
                if df_gap['Gap'].iloc[i] > 0:  # Gap up
                    if df_gap['Low'].iloc[j] <= gap_price:
                        df_gap.loc[df_gap.index[i], 'Gap_Fill'] = True
                        df_gap.loc[df_gap.index[i], 'Days_To_Fill'] = j - i
                        break
                else:  # Gap down
                    if df_gap['High'].iloc[j] >= gap_price:
                        df_gap.loc[df_gap.index[i], 'Gap_Fill'] = True
                        df_gap.loc[df_gap.index[i], 'Days_To_Fill'] = j - i
                        break
    
    # Estadísticas de gaps
    gap_stats = df_gap[df_gap['Gap_Size'] > 0.5].groupby('Gap_Type').agg({
        'Gap': ['count', 'mean'],
        'Rendimiento': 'mean',
        'Gap_Fill': lambda x: x.sum() / len(x) * 100,  # % que se llenan
        'Days_To_Fill': 'mean'
    }).round(2)
    
    return {
        'data': df_gap,
        'gap_stats': gap_stats,
        'recent_gaps': df_gap[df_gap['Gap_Size'] > 0.5].tail(10)
    }

def analyze_support_resistance(df):
    """Análisis de niveles de soporte y resistencia"""
    if df is None or len(df) < 50:
        return None
    
    df_sr = df.copy()
    
    # Encontrar máximos y mínimos locales
    from scipy.signal import find_peaks
    
    # Máximos locales (resistencias potenciales)
    high_peaks, _ = find_peaks(df_sr['High'], distance=5, prominence=df_sr['High'].std()*0.5)
    low_peaks, _ = find_peaks(-df_sr['Low'], distance=5, prominence=df_sr['Low'].std()*0.5)
    
    # Obtener niveles de soporte y resistencia
    resistance_levels = df_sr['High'].iloc[high_peaks].values
    support_levels = df_sr['Low'].iloc[low_peaks].values
    
    # Agrupar niveles cercanos (dentro del 2%)
    def cluster_levels(levels, tolerance=0.02):
        if len(levels) == 0:
            return []
        
        levels_sorted = sorted(levels)
        clusters = []
        current_cluster = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        clusters.append(np.mean(current_cluster))
        
        return clusters
    
    resistance_clusters = cluster_levels(resistance_levels)
    support_clusters = cluster_levels(support_levels)
    
    # Calcular fuerza de los niveles (cuántas veces fueron tocados)
    current_price = df_sr['Close'].iloc[-1]
    
    def calculate_level_strength(levels, prices, tolerance=0.02):
        level_strength = []
        for level in levels:
            touches = sum(abs(price - level) / level <= tolerance for price in prices)
            level_strength.append({
                'level': level,
                'touches': touches,
                'distance_from_current': abs(level - current_price) / current_price * 100
            })
        return sorted(level_strength, key=lambda x: x['touches'], reverse=True)
    
    resistance_strength = calculate_level_strength(resistance_clusters, df_sr['High'])
    support_strength = calculate_level_strength(support_clusters, df_sr['Low'])
    
    return {
        'resistance_levels': resistance_strength[:5],  # Top 5
        'support_levels': support_strength[:5],       # Top 5
        'current_price': current_price,
        'peak_dates_resistance': df_sr['Date'].iloc[high_peaks].tolist(),
        'peak_dates_support': df_sr['Date'].iloc[low_peaks].tolist()
    }

# Interfaz de usuario para carga de datos (siempre visible en ambas páginas)
st.sidebar.header("⚙️ Configuración de Datos")

tickers_disponibles = [
    # Portfolio ACCIONES Y ETF
    "ALAB", "SAP", "AMZN", "TSLA", "BABA", "PANW", "SPY", "AMD", "GTLB", "CRM", "EDIT","META",
    
    # Portfolio PROTECT 2
    "SLV", "GLD", "SVM", "BTC-USD",
    
    # Portfolio de Crypto (formato Yahoo Finance)
    "ZIL-USD", "STX4847-USD", "BEAM28298-USD", "LTC-USD", "VRA-USD", "MANA-USD", 
    "SOL-USD", "AVAX-USD", "RVN-USD", "TON11419-USD", "DOT-USD", "ETH-USD", 
    "ENA-USD", "LINK-USD", "FLUX-USD", "ZANO-USD", "ADA-USD",
    ]
ticker_seleccionado = st.sidebar.selectbox("Seleccione una acción", tickers_disponibles,
                                   index=tickers_disponibles.index(st.session_state.get('ticker_seleccionado', 'ALAB')),
                                   key="ticker_principal")
st.session_state.ticker_seleccionado = ticker_seleccionado

period_options = {
    # Períodos básicos de Imagen 1
    "Hoy": "1d",
    "Ayer": "yesterday",
    "Esta semana": "this_week",
    "Este mes": "this_month", 
    "Este trimestre": "this_quarter",
    "Este año": "ytd",
    "Últimos 5 días": "5d",
    "Últimos 7 días": "1w",
    "Últimos 30 días": "1mo",
    "Últimos 3 meses": "3mo",
    "Año hasta la fecha": "ytd",
    "Mes hasta la fecha": "mtd",
    "Trimestre hasta la fecha": "qtd",
    
    # Períodos de Imagen 2
    "Año anterior": "last_year",
    "Próximo año": "next_year",
    "Últimos 3 años": "3y",
    "Próximos 3 años": "next_3y",
    "Trimestre anterior": "last_quarter",
    "Próximo trimestre": "next_quarter", 
    "Últimos 3 trimestres": "3q",
    "Mes anterior": "last_month",
    "Mes siguiente": "next_month",
    "Últimos 6 meses": "6mo",
    "Próximos 3 meses": "next_3m",
    
    # Períodos de Imagen 3
    "Próximos 6 meses": "next_6m",
    "Próximas 2 semanas": "next_2w",
    "Semana anterior": "last_week",
    "Próxima semana": "next_week",
    "Últimas 3 semanas": "3w", 
    "30 días siguientes": "next_30d",
    "5 días siguientes": "next_5d",
    
    # Períodos adicionales estándar
    "Últimos 12 meses": "1y",
    "Últimos 2 años": "2y",
    "Últimos 5 años": "5y",
    "Todo el histórico": "max",
    
    # Intervalo personalizable (placeholder)
    "Intervalo personalizar...": "custom"
}

period_label = st.sidebar.selectbox(
    "Seleccione período",
    options=list(period_options.keys()),
    index=31,  # "Últimos 12 meses" como default
    key="period_principal"
)

period = period_options[period_label]

# Funcionalidad especial para intervalo personalizable
if period == "custom":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🛠️ Intervalo Personalizado**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio", value=pd.Timestamp.now() - pd.Timedelta(days=365))
    with col2:
        end_date = st.date_input("Fecha fin", value=pd.Timestamp.now())
    
    if start_date >= end_date:
        st.sidebar.error("❌ La fecha de inicio debe ser anterior a la fecha de fin")
        period = "1y"  # Fallback
    else:
        # Calcular diferencia y convertir a período aproximado
        diff_days = (end_date - start_date).days
        if diff_days <= 7:
            period = "5d"
        elif diff_days <= 30:
            period = "1mo"
        elif diff_days <= 90:
            period = "3mo"
        elif diff_days <= 180:
            period = "6mo"
        elif diff_days <= 365:
            period = "1y"
        elif diff_days <= 1095:
            period = "3y"
        else:
            period = "5y"
        
        st.sidebar.success(f"📅 Rango: {diff_days} días")
        st.sidebar.markdown("---")

if st.sidebar.button("🔄 Actualizar Datos", type="primary"):
    with st.spinner(f"Cargando datos para {ticker_seleccionado.upper()} ({period_label})..."):
        # Obtener fechas personalizadas si es necesario
        custom_start_date = None
        custom_end_date = None
        if period == "custom":
            custom_start_date = start_date if 'start_date' in locals() else None
            custom_end_date = end_date if 'end_date' in locals() else None
        
        df = get_stock_data(ticker_seleccionado.upper(), period, custom_start_date, custom_end_date)
        if df is not None and not df.empty:
            df = df.sort_values('Date').reset_index(drop=True)
            df['Rendimiento'] = df['Close'].pct_change() * 100
            df['Subida'] = df['Rendimiento'] > 0

            # Calcular Rango solo si High y Low existen
            if 'High' in df.columns and 'Low' in df.columns:
                df['Rango'] = df['High'] - df['Low']
            else:
                df['Rango'] = None

            st.session_state.df_con_rendimiento = df
            st.sidebar.success(f"✅ Datos actualizados para {ticker_seleccionado.upper()}")
            st.sidebar.info(f"📅 Período: {period_label}")
            st.sidebar.info("💡 Navegue entre las páginas para ver diferentes análisis")
        else:
            st.sidebar.error(f"❌ No se encontraron datos válidos para {ticker_seleccionado.upper()}")

# === FUNCIÓN PARA PREPARAR DATOS DE DESCARGA ===
def prepare_download_data():
    """Prepara los datos para descarga en CSV"""
    if st.session_state.df_con_rendimiento is None:
        return None
    
    df_download = st.session_state.df_con_rendimiento.copy()
    
    # Formatear fecha para CSV
    df_download['Date'] = df_download['Date'].dt.strftime('%Y-%m-%d')
    
    # Redondear valores numéricos
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Rendimiento', 'Rango']
    for col in numeric_columns:
        if col in df_download.columns:
            df_download[col] = df_download[col].round(4)
    
    # Agregar información adicional si existe
    if 'ATR14' in df_download.columns:
        df_download['ATR14'] = df_download['ATR14'].round(4)
    
    if 'Volatilidad' in df_download.columns:
        df_download['Volatilidad_Categoria'] = df_download['Volatilidad']
    
    # Reordenar columnas
    column_order = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Rendimiento', 'Subida', 'Rango']
    available_columns = [col for col in column_order if col in df_download.columns]
    other_columns = [col for col in df_download.columns if col not in available_columns]
    final_columns = available_columns + other_columns
    
    return df_download[final_columns]

# Mostrar estado actual en sidebar
if st.session_state.df_con_rendimiento is not None:
    st.sidebar.success(f"📊 Datos cargados: {st.session_state.ticker_seleccionado.upper()}")
    last_date = st.session_state.df_con_rendimiento['Date'].iloc[-1].strftime('%Y-%m-%d')
    first_date = st.session_state.df_con_rendimiento['Date'].iloc[0].strftime('%Y-%m-%d')
    total_days = len(st.session_state.df_con_rendimiento)
    st.sidebar.info(f"📅 Período: {first_date} a {last_date}")
    st.sidebar.info(f"📊 Total de días: {total_days}")
    
    # === BOTÓN DE DESCARGA ===
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Descargar Datos")
    
    # Preparar datos para descarga
    download_data = prepare_download_data()
    
    if download_data is not None:
        # Convertir a CSV
        csv_data = download_data.to_csv(index=False, encoding='utf-8')
        
        # Nombre del archivo
        filename = f"{st.session_state.ticker_seleccionado}_{first_date}_to_{last_date}.csv"
        
        # Botón de descarga principal
        st.sidebar.download_button(
            label="📊 Descargar Datos Completos",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help="Descarga todos los datos históricos con indicadores calculados"
        )
        
        # Información adicional sobre la descarga
        with st.sidebar.expander("ℹ️ Información de Descarga"):
            st.write("**El archivo CSV incluye:**")
            st.write("• Datos OHLCV básicos")
            st.write("• Rendimientos diarios calculados")
            st.write("• Indicadores técnicos (ATR, rangos)")
            st.write("• Clasificaciones de volatilidad")
            st.write("• Fechas en formato YYYY-MM-DD")
            st.write(f"• **Total de registros:** {len(download_data)}")
            
            # Mostrar preview de las columnas
            st.write("**Columnas incluidas:**")
            columns_preview = ", ".join(download_data.columns[:8])
            if len(download_data.columns) > 8:
                columns_preview += f"... (+{len(download_data.columns)-8} más)"
            st.write(columns_preview)
    
else:
    st.sidebar.warning("⚠️ Sin datos - Haga clic en 'Actualizar Datos'")

st.sidebar.markdown("---")

# === PÁGINA PRINCIPAL ===
if st.session_state.current_page == "principal":
    st.header("📊 Dashboard Principal")
    
    # Mostrar resultados solo si hay datos
    if st.session_state.df_con_rendimiento is not None:
        df = st.session_state.df_con_rendimiento

        # === ANÁLISIS BÁSICO ===
        
        # Gráfico de rango diario + ATR(14)
        if 'Rango' in df.columns and df['Rango'].notna().any():
            st.subheader("Rango diario (High - Low) + ATR(14)")

            df['TR'] = pd.DataFrame({
                'HL': df['High'] - df['Low'],
                'HC': abs(df['High'] - df['Close'].shift(1)),
                'LC': abs(df['Low'] - df['Close'].shift(1))
            }).max(axis=1)

            df['ATR14'] = df['TR'].rolling(window=14).mean()

            fig_rango = go.Figure()
            fig_rango.add_trace(go.Scatter(x=df['Date'], y=df['Rango'], mode='lines+markers', name='Rango diario'))
            fig_rango.add_trace(go.Scatter(x=df['Date'], y=df['ATR14'], mode='lines', name='ATR(14)', line=dict(color='purple')))
            fig_rango.update_layout(title="Rango diario y ATR(14)", xaxis_title="Fecha", yaxis_title="Rango (USD)")
            st.plotly_chart(fig_rango, use_container_width=True)
        else:
            st.warning("No se puede mostrar el rango. Datos no disponibles.")

        # Clasificación de volatilidad
        if 'Rendimiento' in df.columns and df['Rendimiento'].notna().any():
            st.subheader("Clasificación de Volatilidad Diaria")
            df['Volatilidad'] = pd.cut(abs(df['Rendimiento']), bins=[-1, 0.5, 1.5, float('inf')], labels=['Baja', 'Normal', 'Alta'])
            conteo_volatilidad = df['Volatilidad'].value_counts()

            fig_dona = px.pie(conteo_volatilidad, values=conteo_volatilidad.values, names=conteo_volatilidad.index,
                              hole=0.4, color_discrete_sequence=px.colors.sequential.Viridis)
            fig_dona.update_traces(textposition='inside', textinfo='percent+label')
            fig_dona.update_layout(showlegend=False, title="Distribución de Días por Nivel de Volatilidad")
            st.plotly_chart(fig_dona, use_container_width=True)
        else:
            st.warning("No se pudo calcular la distribución de días por nivel de volatilidad.")

        # Gráfico de cierre con estadísticas
        st.subheader(f"Precio de cierre de {st.session_state.ticker_seleccionado.upper()}")
        
        # Calcular estadísticas del precio de cierre
        precio_media = df['Close'].mean()
        precio_mediana = df['Close'].median()
        precio_std = df['Close'].std()
        precio_varianza = df['Close'].var()
        precio_cv = (precio_std / precio_media) * 100 if precio_media != 0 else 0
        
        fig_cierre = go.Figure()
        fig_cierre.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Precio de cierre', line=dict(color='blue', width=2)))
        
        # Agregar líneas estadísticas
        fig_cierre.add_hline(y=precio_media, line_dash="solid", line_color="red", line_width=2, 
                           annotation_text=f"Media: ${precio_media:.2f}")
        fig_cierre.add_hline(y=precio_mediana, line_dash="dash", line_color="green", line_width=2,
                           annotation_text=f"Mediana: ${precio_mediana:.2f}")
        fig_cierre.add_hline(y=precio_media + precio_std, line_dash="dot", line_color="orange", line_width=1,
                           annotation_text=f"+1σ: ${precio_media + precio_std:.2f}")
        fig_cierre.add_hline(y=precio_media - precio_std, line_dash="dot", line_color="orange", line_width=1,
                           annotation_text=f"-1σ: ${precio_media - precio_std:.2f}")
        fig_cierre.add_hline(y=precio_media + 2*precio_std, line_dash="dashdot", line_color="purple", line_width=1,
                           annotation_text=f"+2σ: ${precio_media + 2*precio_std:.2f}")
        fig_cierre.add_hline(y=precio_media - 2*precio_std, line_dash="dashdot", line_color="purple", line_width=1,
                           annotation_text=f"-2σ: ${precio_media - 2*precio_std:.2f}")
        
        fig_cierre.update_layout(
            title="Precio de Cierre con Estadísticas Descriptivas",
            xaxis_title="Fecha", 
            yaxis_title="Precio (USD)",
            height=500
        )
        st.plotly_chart(fig_cierre, use_container_width=True)
        
        # Mostrar estadísticas en métricas
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Media", f"${precio_media:.2f}")
        col2.metric("Mediana", f"${precio_mediana:.2f}")
        col3.metric("Desv. Estándar", f"${precio_std:.2f}")
        col4.metric("Varianza", f"${precio_varianza:.2f}")
        col5.metric("Coef. Variación", f"{precio_cv:.2f}%")

        # Cálculos estadísticos
        st.subheader("Estadísticas de Rendimiento")
        promedio = df['Rendimiento'].mean()
        mediana = df['Rendimiento'].median()
        desviacion = df['Rendimiento'].std()

        rendimientos_al_alza = df[df['Rendimiento'] > 0]['Rendimiento']
        rendimientos_a_la_baja = df[df['Rendimiento'] < 0]['Rendimiento']

        promedio_correccion_al_alza = rendimientos_al_alza.mean() if not rendimientos_al_alza.empty else 0
        promedio_correccion_a_la_baja = abs(rendimientos_a_la_baja.mean()) if not rendimientos_a_la_baja.empty else 0
        promedio_correccion_total = abs(df['Rendimiento']).mean()

        col1, col2, col3 = st.columns(3)
        col1.metric(label="Promedio", value=f"{promedio:.2f}%")
        col2.metric(label="Mediana", value=f"{mediana:.2f}%")
        col3.metric(label="Desv. Estándar", value=f"{desviacion:.2f}%")

        col4, col5, col6 = st.columns(3)
        col4.metric(label="Corrección al alza", value=f"{promedio_correccion_al_alza:.2f}%")
        col5.metric(label="Corrección a la baja", value=f"{promedio_correccion_a_la_baja:.2f}%")
        col6.metric(label="Corrección total promedio", value=f"{promedio_correccion_total:.2f}%")

        # Gráfico de rendimiento diario con líneas de referencia
        st.subheader(f"Rendimiento diario de {st.session_state.ticker_seleccionado.upper()}")
        fig_rendimiento = go.Figure()
        fig_rendimiento.add_trace(go.Scatter(x=df['Date'], y=df['Rendimiento'], mode='markers+lines', name='Rendimiento (%)'))

        fig_rendimiento.add_hline(y=promedio, line_dash="solid", line_color="blue", annotation_text="Promedio")
        fig_rendimiento.add_hline(y=mediana, line_dash="dashdot", line_color="green", annotation_text="Mediana")
        fig_rendimiento.add_hline(y=promedio + desviacion, line_dash="dot", line_color="red", annotation_text="+1σ")
        fig_rendimiento.add_hline(y=promedio - desviacion, line_dash="dot", line_color="red", annotation_text="-1σ")

        fig_rendimiento.update_layout(title="Rendimiento Diario con Referencias Estadísticas", xaxis_title="Fecha", yaxis_title="Rendimiento (%)")
        st.plotly_chart(fig_rendimiento, use_container_width=True)

        # Leyenda de líneas horizontales
        with st.expander("📋 Ver Leyenda de Líneas Horizontales"):
            leyenda_datos = {
                "Descripción": [
                    "Promedio del rendimiento",
                    "Mediana del rendimiento",
                    "+1 Desviación estándar",
                    "-1 Desviación estándar"
                ],
                "Valor (%)": [
                    f"{promedio:.2f}",
                    f"{mediana:.2f}",
                    f"{promedio + desviacion:.2f}",
                    f"{promedio - desviacion:.2f}"
                ],
                "Color": ["🔵 Azul", "🟢 Verde", "🔴 Rojo", "🔴 Rojo"],
                "Estilo de línea": [
                    "——— Continua",
                    "- · - Puntos y guiones",
                    "····· Punteada",
                    "····· Punteada"
                ]
            }
            st.table(pd.DataFrame(leyenda_datos))

        # Top días de rendimiento
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🔝 Top 5 Mejores Días")
            top_5_mayor = df.nlargest(5, 'Rendimiento')
            for _, row in top_5_mayor.iterrows():
                st.write(f"📅 {row['Date'].strftime('%Y-%m-%d')}: **{row['Rendimiento']:.2f}%**")
        
        with col2:
            st.subheader(f"📉 Top 5 Peores Días")
            top_5_menor = df.nsmallest(5, 'Rendimiento')
            for _, row in top_5_menor.iterrows():
                st.write(f"📅 {row['Date'].strftime('%Y-%m-%d')}: **{row['Rendimiento']:.2f}%**")

        # Rachas consecutivas
        if 'Subida' in df.columns and df['Subida'].notna().any():
            st.subheader("Análisis de Rachas Consecutivas")
            df['RachaID'] = (df['Subida'] != df['Subida'].shift()).cumsum()
            rachas = df.groupby(['RachaID', 'Subida']).size().reset_index(name='Días')
            rachas['Tipo'] = rachas['Subida'].map({True: 'Al alza', False: 'A la baja'})

            # Separar por tipo
            rachas_al_alza = rachas[rachas['Subida']]['Días'].value_counts().sort_index()
            rachas_a_la_baja = rachas[~rachas['Subida']]['Días'].value_counts().sort_index()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 Rachas al alza")
                if not rachas_al_alza.empty:
                    st.bar_chart(rachas_al_alza)
                else:
                    st.write("No hubo rachas al alza en este periodo.")
            with col2:
                st.markdown("#### 📉 Rachas a la baja")
                if not rachas_a_la_baja.empty:
                    st.bar_chart(rachas_a_la_baja)
                else:
                    st.write("No hubo rachas a la baja en este periodo.")

            # Gráfico de evolución con rachas coloreadas
            st.markdown("#### Evolución del precio con rachas consecutivas")
            fig = go.Figure()

            current_start = None
            current_type = None

            for i, row in df.iterrows():
                subida = row['Subida']
                if pd.isna(subida):
                    continue

                if current_type is None:
                    current_type = subida
                    current_start = row['Date']
                elif subida != current_type:
                    segment = df[(df['Date'] >= current_start) & (df['Date'] <= row['Date'])]
                    fig.add_trace(go.Scatter(
                        x=segment['Date'], y=segment['Close'],
                        mode='lines',
                        line=dict(color="green" if current_type else "red"),
                        showlegend=False,
                        hoverinfo="skip"
                    ))
                    current_type = subida
                    current_start = row['Date']

            if current_type is not None:
                segment = df[df['Date'] >= current_start]
                fig.add_trace(go.Scatter(
                    x=segment['Date'], y=segment['Close'],
                    mode='lines',
                    line=dict(color="green" if current_type else "red"),
                    name="Última racha"
                ))

            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='markers', name='Precio de cierre', marker=dict(size=4)))
            fig.update_layout(title="Evolución del Precio con Rachas Coloreadas", xaxis_title="Fecha", yaxis_title="Precio (USD)")
            st.plotly_chart(fig, use_container_width=True)

            # Detalles de rachas en expander
            with st.expander("📊 Ver Detalles de Rachas"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(rachas[['Días', 'Tipo']].rename(columns={'Días': 'Días consecutivos'}), use_container_width=True)
                with col2:
                    rachas_csv = rachas[['Días', 'Tipo']].to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label="📥 CSV Rachas",
                        data=rachas_csv,
                        file_name=f"rachas_consecutivas_{st.session_state.ticker_seleccionado}.csv",
                        mime="text/csv",
                        help="Descargar análisis de rachas consecutivas"
                    )
        
        # === SECCIÓN DE DESCARGA COMPLETA ===
        st.markdown("---")
        st.markdown("### 📥 Descargar Análisis Completo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Datos básicos con indicadores
            download_data = prepare_download_data()
            if download_data is not None:
                csv_data = download_data.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="📊 Datos + Indicadores",
                    data=csv_data,
                    file_name=f"completo_{st.session_state.ticker_seleccionado}_{first_date}_to_{last_date}.csv",
                    mime="text/csv",
                    help="Todos los datos históricos con indicadores técnicos calculados",
                    use_container_width=True
                )
        
        with col2:
            # Solo estadísticas resumen
            stats_data = {
                'Métrica': ['Promedio', 'Mediana', 'Desv_Estándar', 'Corrección_Alza', 'Corrección_Baja', 'Corrección_Total'],
                'Valor_%': [
                    round(promedio, 4),
                    round(mediana, 4),
                    round(desviacion, 4),
                    round(promedio_correccion_al_alza, 4),
                    round(promedio_correccion_a_la_baja, 4),
                    round(promedio_correccion_total, 4)
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_csv = stats_df.to_csv(index=False, encoding='utf-8')
            
            st.download_button(
                label="📈 Solo Estadísticas",
                data=stats_csv,
                file_name=f"estadisticas_{st.session_state.ticker_seleccionado}.csv",
                mime="text/csv",
                help="Resumen estadístico del análisis",
                use_container_width=True
            )
        
        with col3:
            # Top días (mejores y peores)
            top_data = []
            
            # Mejores días
            for _, row in top_5_mayor.iterrows():
                top_data.append({
                    'Fecha': row['Date'].strftime('%Y-%m-%d'),
                    'Tipo': 'Mejor',
                    'Rendimiento_%': round(row['Rendimiento'], 4),
                    'Precio_Close': round(row['Close'], 2)
                })
            
            # Peores días
            for _, row in top_5_menor.iterrows():
                top_data.append({
                    'Fecha': row['Date'].strftime('%Y-%m-%d'),
                    'Tipo': 'Peor',
                    'Rendimiento_%': round(row['Rendimiento'], 4),
                    'Precio_Close': round(row['Close'], 2)
                })
            
            top_df = pd.DataFrame(top_data)
            top_csv = top_df.to_csv(index=False, encoding='utf-8')
            
            st.download_button(
                label="🏆 Top Días",
                data=top_csv,
                file_name=f"top_dias_{st.session_state.ticker_seleccionado}.csv",
                mime="text/csv",
                help="Top 5 mejores y peores días de rendimiento",
                use_container_width=True
            )

    else:
        st.info("📊 Dashboard de Análisis Financiero")
        st.markdown("""
        **Este dashboard incluye:**
        - 📈 Gráficos de precio y volatilidad
        - 📊 Estadísticas de rendimiento  
        - 🔄 Análisis de rachas consecutivas
        - 📊 Clasificación de volatilidad diaria
        - 📈 Análisis técnico básico (ATR, rangos)
        
        **👈 Use el panel lateral para cargar datos y comenzar el análisis**
        """)

# === PÁGINA DE PATRONES IDENTIFICADOS ===
elif st.session_state.current_page == "patrones":
    st.header("🔍 Patrones Identificados")
    
    if st.session_state.df_con_rendimiento is not None:
        df = st.session_state.df_con_rendimiento
        
        # Crear tabs para los análisis de patrones
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Patrones de Volatilidad", 
            "📅 Patrones Temporales", 
            "🚀 Momentum", 
            "📈 Gaps", 
            "🎯 Soporte/Resistencia"
        ])
        
        # TAB 1: PATRONES DE VOLATILIDAD
        with tab1:
            st.subheader("Análisis de Patrones de Volatilidad")
            
            vol_analysis = analyze_volatility_patterns(df)
            if vol_analysis:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_regime = vol_analysis['current_regime']
                    st.metric("Régimen Actual", current_regime)
                
                with col2:
                    current_vol = vol_analysis['data']['Vol_20d'].iloc[-1] if not vol_analysis['data']['Vol_20d'].isna().all() else 0
                    st.metric("Volatilidad 20d", f"{current_vol:.1%}")
                
                with col3:
                    avg_duration = vol_analysis['avg_durations'].get('Alta', 0)
                    st.metric("Duración Prom. Alta Vol.", f"{avg_duration:.0f} días")
                
                # Gráfico de evolución de volatilidad
                fig_vol = go.Figure()
                vol_data = vol_analysis['data']
                
                # Colorear por régimen
                regime_colors = {'Baja': 'green', 'Normal': 'blue', 'Alta': 'orange', 'Extrema': 'red'}
                
                for regime, color in regime_colors.items():
                    regime_data = vol_data[vol_data['Vol_Regime'] == regime]
                    if not regime_data.empty:
                        fig_vol.add_trace(go.Scatter(
                            x=regime_data['Date'],
                            y=regime_data['Vol_20d'] * 100,
                            mode='markers',
                            name=f'Vol. {regime}',
                            marker=dict(color=color, size=4)
                        ))
                
                # Líneas de percentiles
                for pct_name, pct_val in vol_analysis['percentiles'].items():
                    fig_vol.add_hline(y=pct_val * 100, line_dash="dash", 
                                    annotation_text=f"{pct_name}: {pct_val:.1%}")
                
                fig_vol.update_layout(
                    title="Evolución de Volatilidad y Regímenes",
                    xaxis_title="Fecha",
                    yaxis_title="Volatilidad 20d (%)",
                    height=500
                )
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Tabla de retornos por régimen
                st.subheader("Retornos por Régimen de Volatilidad")
                regime_returns = vol_analysis['regime_returns']
                regime_returns_display = regime_returns.round(2)
                st.dataframe(regime_returns_display)
                
                # Botón de descarga para análisis de volatilidad
                col1, col2 = st.columns([3, 1])
                with col2:
                    vol_csv = regime_returns_display.to_csv(encoding='utf-8')
                    st.download_button(
                        label="📥 CSV",
                        data=vol_csv,
                        file_name=f"volatilidad_analysis_{st.session_state.ticker_seleccionado}.csv",
                        mime="text/csv",
                        help="Descargar análisis de volatilidad"
                    )
                
                # Interpretación
                st.markdown("#### 💡 Interpretación:")
                if current_regime == 'Baja':
                    st.success("🟢 **Volatilidad Baja**: Entorno favorable para estrategias de momentum")
                elif current_regime == 'Alta':
                    st.warning("🟡 **Volatilidad Alta**: Considerar estrategias defensivas")
                elif current_regime == 'Extrema':
                    st.error("🔴 **Volatilidad Extrema**: Alta probabilidad de reversión")
                else:
                    st.info("🔵 **Volatilidad Normal**: Condiciones estándar de mercado")
        
        # TAB 2: PATRONES TEMPORALES
        with tab2:
            st.subheader("Análisis de Patrones Temporales y Estacionales")
            
            temporal_analysis = analyze_temporal_patterns(df)
            if temporal_analysis:
                
                # Análisis por día de la semana
                st.markdown("#### 📊 Rendimientos por Día de la Semana")
                daily_stats = temporal_analysis['daily_stats']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_daily = px.bar(
                        x=daily_stats.index,
                        y=daily_stats['Promedio'] * 100,
                        color=daily_stats['Tasa_Exito'],
                        title="Rendimiento Promedio por Día",
                        labels={'y': 'Rendimiento (%)', 'color': 'Tasa de Éxito'},
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                with col2:
                    # Mejores y peores días
                    best_day = daily_stats['Promedio'].idxmax()
                    worst_day = daily_stats['Promedio'].idxmin()
                    
                    st.metric("Mejor Día", best_day, f"{daily_stats.loc[best_day, 'Promedio']*100:.2f}%")
                    st.metric("Peor Día", worst_day, f"{daily_stats.loc[worst_day, 'Promedio']*100:.2f}%")
                    st.metric("Efecto Lunes", f"{daily_stats.loc['Lunes', 'Tasa_Exito']*100:.1f}%")
                
                # Tabla detallada
                daily_display = daily_stats.copy()
                daily_display['Promedio'] = (daily_display['Promedio'] * 100).round(2)
                daily_display['Volatilidad'] = (daily_display['Volatilidad'] * 100).round(2)
                daily_display['Tasa_Exito'] = (daily_display['Tasa_Exito'] * 100).round(1)
                st.dataframe(daily_display, use_container_width=True)
                
                # Botón de descarga para patrones temporales
                col1, col2 = st.columns([3, 1])
                with col2:
                    temporal_csv = daily_display.to_csv(encoding='utf-8')
                    st.download_button(
                        label="📥 CSV Diario",
                        data=temporal_csv,
                        file_name=f"patrones_temporales_diario_{st.session_state.ticker_seleccionado}.csv",
                        mime="text/csv",
                        help="Descargar análisis por día de la semana"
                    )
                
                # Análisis mensual
                st.markdown("#### 📅 Rendimientos por Mes")
                monthly_stats = temporal_analysis['monthly_stats']
                
                fig_monthly = px.bar(
                    x=monthly_stats.index,
                    y=monthly_stats['Promedio'] * 100,
                    color=monthly_stats['Tasa_Exito'],
                    title="Rendimiento Promedio por Mes",
                    labels={'y': 'Rendimiento (%)', 'color': 'Tasa de Éxito'},
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Mostrar tabla mensual y botón de descarga
                col1, col2 = st.columns([3, 1])
                with col1:
                    monthly_display = monthly_stats.copy()
                    monthly_display['Promedio'] = (monthly_display['Promedio'] * 100).round(2)
                    monthly_display['Volatilidad'] = (monthly_display['Volatilidad'] * 100).round(2)
                    monthly_display['Tasa_Exito'] = (monthly_display['Tasa_Exito'] * 100).round(1)
                    st.dataframe(monthly_display, use_container_width=True)
                
                with col2:
                    monthly_csv = monthly_display.to_csv(encoding='utf-8')
                    st.download_button(
                        label="📥 CSV Mensual",
                        data=monthly_csv,
                        file_name=f"patrones_temporales_mensual_{st.session_state.ticker_seleccionado}.csv",
                        mime="text/csv",
                        help="Descargar análisis por mes"
                    )
                
                # Efecto "Sell in May"
                st.markdown("#### 🌸 Efecto 'Sell in May'")
                sell_may = temporal_analysis['sell_in_may']
                
                col1, col2 = st.columns(2)
                with col1:
                    oct_apr_return = sell_may.loc['Oct-Abr', 'mean'] * 100
                    st.metric("Oct-Abr (Favorable)", f"{oct_apr_return:.2f}%")
                
                with col2:
                    may_sep_return = sell_may.loc['May-Sep', 'mean'] * 100
                    st.metric("May-Sep (Desfavorable)", f"{may_sep_return:.2f}%")
                
                difference = oct_apr_return - may_sep_return
                if difference > 1:
                    st.success(f"✅ **Efecto Confirmado**: Diferencia de {difference:.2f}% a favor Oct-Abr")
                else:
                    st.info(f"ℹ️ **Efecto Débil**: Diferencia de solo {difference:.2f}%")
                
                # === ANÁLISIS DE RANGO DIARIO ===
                if 'Rango' in df.columns and df['Rango'].notna().any():
                    st.markdown("---")
                    st.markdown("#### 📏 Análisis de Rango Diario")
                    
                    # Preparar datos de rango con información temporal
                    df_rango = df.copy()
                    df_rango['Month'] = df_rango['Date'].dt.month
                    df_rango['DayOfWeek'] = df_rango['Date'].dt.dayofweek
                    
                    # Mapeo de días y meses
                    day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
                    
                    # Análisis de rango por mes
                    monthly_range = df_rango.groupby('Month')['Rango'].agg([
                        'mean', 'median', 'std', 'count'
                    ]).round(4)
                    monthly_range.index = [month_names[i-1] for i in monthly_range.index]
                    
                    # Análisis de rango por día de la semana
                    daily_range = df_rango.groupby('DayOfWeek')['Rango'].agg([
                        'mean', 'median', 'std', 'count'
                    ]).round(4)
                    daily_range.index = [day_names[i] for i in daily_range.index if i < len(day_names)]
                    
                    # Gráficos de rango
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_range_month = px.bar(
                            x=monthly_range.index,
                            y=monthly_range['mean'],
                            title="Rango Diario Promedio por Mes",
                            labels={'y': 'Rango Promedio ($)', 'x': 'Mes'},
                            color=monthly_range['mean'],
                            color_continuous_scale='Viridis'
                        )
                        fig_range_month.update_layout(showlegend=False)
                        st.plotly_chart(fig_range_month, use_container_width=True)
                        
                        # Tabla mensual
                        st.markdown("**Estadísticas Mensuales:**")
                        monthly_display = monthly_range.copy()
                        monthly_display.columns = ['Promedio', 'Mediana', 'Desv_Est', 'Días']
                        st.dataframe(monthly_display, use_container_width=True)
                    
                    with col2:
                        fig_range_day = px.bar(
                            x=daily_range.index,
                            y=daily_range['mean'],
                            title="Rango Diario Promedio por Día de la Semana",
                            labels={'y': 'Rango Promedio ($)', 'x': 'Día de la Semana'},
                            color=daily_range['mean'],
                            color_continuous_scale='Plasma'
                        )
                        fig_range_day.update_layout(showlegend=False)
                        st.plotly_chart(fig_range_day, use_container_width=True)
                        
                        # Tabla diaria
                        st.markdown("**Estadísticas por Día:**")
                        daily_display = daily_range.copy()
                        daily_display.columns = ['Promedio', 'Mediana', 'Desv_Est', 'Días']
                        st.dataframe(daily_display, use_container_width=True)
                    
                    # Insights del rango
                    st.markdown("#### 💡 Insights del Rango Diario:")
                    
                    # Mejor y peor mes para volatilidad
                    best_month_vol = monthly_range['mean'].idxmax()
                    worst_month_vol = monthly_range['mean'].idxmin()
                    
                    # Mejor y peor día para volatilidad
                    best_day_vol = daily_range['mean'].idxmax()
                    worst_day_vol = daily_range['mean'].idxmin()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"📈 **Mayor Volatilidad**: {best_month_vol} (${monthly_range.loc[best_month_vol, 'mean']:.2f})")
                        st.info(f"📉 **Menor Volatilidad**: {worst_month_vol} (${monthly_range.loc[worst_month_vol, 'mean']:.2f})")
                    
                    with col2:
                        st.success(f"📈 **Día más Volátil**: {best_day_vol} (${daily_range.loc[best_day_vol, 'mean']:.2f})")
                        st.info(f"📉 **Día menos Volátil**: {worst_day_vol} (${daily_range.loc[worst_day_vol, 'mean']:.2f})")
                    
                    # Botones de descarga para análisis de rango
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col2:
                        monthly_range_csv = monthly_display.to_csv(encoding='utf-8')
                        st.download_button(
                            label="📥 CSV Mensual",
                            data=monthly_range_csv,
                            file_name=f"rango_mensual_{st.session_state.ticker_seleccionado}.csv",
                            mime="text/csv",
                            help="Descargar análisis de rango por mes"
                        )
                    
                    with col3:
                        daily_range_csv = daily_display.to_csv(encoding='utf-8')
                        st.download_button(
                            label="📥 CSV Semanal",
                            data=daily_range_csv,
                            file_name=f"rango_semanal_{st.session_state.ticker_seleccionado}.csv",
                            mime="text/csv",
                            help="Descargar análisis de rango por día"
                        )
        
        # TAB 3: MOMENTUM
        with tab3:
            st.subheader("Análisis de Patrones de Momentum")
            
            momentum_analysis = analyze_momentum_patterns(df)
            if momentum_analysis:
                
                # Gráfico de correlaciones de momentum
                correlations = {period: data['correlation'] for period, data in momentum_analysis.items() if 'correlation' in data}
                spreads = {period: data['spread'] for period, data in momentum_analysis.items() if 'spread' in data}
                
                if correlations:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_corr = px.bar(
                            x=list(correlations.keys()),
                            y=list(correlations.values()),
                            title="Correlación Momentum vs Retornos Futuros",
                            labels={'x': 'Período (días)', 'y': 'Correlación'}
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                    
                    with col2:
                        fig_spread = px.bar(
                            x=list(spreads.keys()),
                            y=list(spreads.values()),
                            title="Spread Quintiles (Q5 - Q1)",
                            labels={'x': 'Período (días)', 'y': 'Spread (%)'}
                        )
                        st.plotly_chart(fig_spread, use_container_width=True)
                
                # Análisis detallado por período
                period_selected = st.selectbox("Seleccione período para análisis detallado:", list(momentum_analysis.keys()))
                
                if period_selected in momentum_analysis:
                    quintile_data = momentum_analysis[period_selected]['quintiles']
                    
                    st.markdown(f"#### Análisis de Quintiles - Momentum {period_selected} días")
                    
                    # Gráfico de quintiles
                    fig_quintiles = px.bar(
                        x=quintile_data.index,
                        y=quintile_data['mean'],
                        title=f"Retornos Futuros por Quintil de Momentum ({period_selected}d)",
                        labels={'x': 'Quintil de Momentum', 'y': 'Retorno Futuro Promedio (%)'}
                    )
                    st.plotly_chart(fig_quintiles, use_container_width=True)
                    
                    # Tabla de quintiles
                    st.dataframe(quintile_data.round(2), use_container_width=True)
                    
                    # Botón de descarga para análisis de momentum
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        momentum_csv = quintile_data.round(2).to_csv(encoding='utf-8')
                        st.download_button(
                            label="📥 CSV",
                            data=momentum_csv,
                            file_name=f"momentum_{period_selected}d_{st.session_state.ticker_seleccionado}.csv",
                            mime="text/csv",
                            help=f"Descargar análisis de momentum {period_selected} días"
                        )
                    
                    # Interpretación
                    momentum_strength = momentum_analysis[period_selected]['spread']
                    correlation_strength = momentum_analysis[period_selected]['correlation']
                    
                    st.markdown("#### 💡 Interpretación del Momentum:")
                    if momentum_strength > 2:
                        st.success(f"🚀 **Momentum Fuerte**: Spread de {momentum_strength:.1f}% - Estrategia de momentum viable")
                    elif momentum_strength > 0.5:
                        st.warning(f"📈 **Momentum Moderado**: Spread de {momentum_strength:.1f}% - Momentum presente pero débil")
                    else:
                        st.info(f"📊 **Sin Momentum**: Spread de {momentum_strength:.1f}% - No hay persistencia clara")
                    
                    if correlation_strength > 0.1:
                        st.success(f"✅ **Correlación Positiva**: {correlation_strength:.3f} - Momentum predice retornos futuros")
                    elif correlation_strength < -0.1:
                        st.error(f"🔄 **Reversión**: {correlation_strength:.3f} - Tendencia a revertir")
                    else:
                        st.info(f"➡️ **Neutral**: {correlation_strength:.3f} - Sin patrón claro")
        
        # TAB 4: GAPS
        with tab4:
            st.subheader("Análisis de Patrones de Gaps")
            
            gap_analysis = analyze_gap_patterns(df)
            if gap_analysis:
                gap_stats = gap_analysis['gap_stats']
                
                # Estadísticas de gaps
                st.markdown("#### 📊 Estadísticas de Gaps por Tipo")
                
                # Preparar datos para visualización
                if not gap_stats.empty:
                    gap_summary = gap_stats.copy()
                    gap_summary.columns = ['_'.join(col).strip() for col in gap_summary.columns.values]
                    
                    # Mostrar tabla de gaps
                    st.dataframe(gap_summary.round(2), use_container_width=True)
                    
                    # Gráfico de fill rate
                    if 'Gap_Fill_<lambda>' in gap_summary.columns:
                        fig_fill = px.bar(
                            x=gap_summary.index,
                            y=gap_summary['Gap_Fill_<lambda>'],
                            title="Tasa de Llenado de Gaps por Tipo",
                            labels={'y': 'Tasa de Llenado (%)', 'x': 'Tipo de Gap'}
                        )
                        st.plotly_chart(fig_fill, use_container_width=True)
                
                # Gaps recientes
                st.markdown("#### 🔍 Gaps Recientes")
                recent_gaps = gap_analysis['recent_gaps']
                
                if not recent_gaps.empty:
                    gap_display = recent_gaps[['Date', 'Gap', 'Gap_Type', 'Rendimiento', 'Gap_Fill', 'Days_To_Fill']].copy()
                    gap_display['Date'] = gap_display['Date'].dt.strftime('%Y-%m-%d')
                    gap_display['Gap'] = gap_display['Gap'].round(2)
                    gap_display['Rendimiento'] = gap_display['Rendimiento'].round(2)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.dataframe(gap_display, use_container_width=True)
                    
                    with col2:
                        gaps_csv = gap_display.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            label="📥 CSV Gaps",
                            data=gaps_csv,
                            file_name=f"gaps_recientes_{st.session_state.ticker_seleccionado}.csv",
                            mime="text/csv",
                            help="Descargar análisis de gaps recientes"
                        )
                    
                    # Análisis del último gap significativo
                    last_significant_gap = recent_gaps[recent_gaps['Gap'].abs() > 1].tail(1)
                    if not last_significant_gap.empty:
                        gap_row = last_significant_gap.iloc[0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Último Gap Significativo", f"{gap_row['Gap']:.2f}%")
                        with col2:
                            fill_status = "✅ Llenado" if gap_row['Gap_Fill'] else "❌ Sin llenar"
                            st.metric("Estado", fill_status)
                        with col3:
                            if gap_row['Gap_Fill']:
                                st.metric("Días para llenar", f"{gap_row['Days_To_Fill']:.0f}")
                            else:
                                days_since = (df['Date'].iloc[-1] - gap_row['Date']).days
                                st.metric("Días transcurridos", f"{days_since}")
                    
                    # Estrategia recomendada
                    st.markdown("#### 💡 Estrategia de Gaps:")
                    avg_fill_rate = recent_gaps['Gap_Fill'].mean() * 100
                    
                    if avg_fill_rate > 70:
                        st.success(f"🎯 **Estrategia Fade**: {avg_fill_rate:.0f}% de gaps se llenan - Trade contra el gap")
                    elif avg_fill_rate > 40:
                        st.warning(f"⚖️ **Estrategia Mixta**: {avg_fill_rate:.0f}% de llenado - Analizar contexto")
                    else:
                        st.info(f"🚀 **Estrategia Momentum**: {avg_fill_rate:.0f}% de llenado - Follow the gap")
        
        # TAB 5: SOPORTE Y RESISTENCIA
        with tab5:
            st.subheader("Análisis de Niveles de Soporte y Resistencia")
            
            sr_analysis = analyze_support_resistance(df)
            if sr_analysis:
                current_price = sr_analysis['current_price']
                
                # Mostrar niveles principales
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔴 Niveles de Resistencia")
                    resistance_levels = sr_analysis['resistance_levels']
                    
                    for i, level_info in enumerate(resistance_levels[:3]):
                        level = level_info['level']
                        touches = level_info['touches']
                        distance = level_info['distance_from_current']
                        
                        st.metric(
                            f"R{i+1}", 
                            f"${level:.2f}", 
                            f"{distance:.1f}% | {touches} toques"
                        )
                
                with col2:
                    st.markdown("#### 🟢 Niveles de Soporte")
                    support_levels = sr_analysis['support_levels']
                    
                    for i, level_info in enumerate(support_levels[:3]):
                        level = level_info['level']
                        touches = level_info['touches']
                        distance = level_info['distance_from_current']
                        
                        st.metric(
                            f"S{i+1}", 
                            f"${level:.2f}", 
                            f"{distance:.1f}% | {touches} toques"
                        )
                
                # Gráfico de precio con S/R
                fig_sr = go.Figure()
                
                # Precio
                fig_sr.add_trace(go.Scatter(
                    x=df['Date'], 
                    y=df['Close'],
                    mode='lines',
                    name='Precio',
                    line=dict(color='blue', width=2)
                ))
                
                # Niveles de resistencia
                for i, level_info in enumerate(resistance_levels[:3]):
                    level = level_info['level']
                    touches = level_info['touches']
                    fig_sr.add_hline(
                        y=level,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"R{i+1}: ${level:.2f} ({touches}x)"
                    )
                
                # Niveles de soporte
                for i, level_info in enumerate(support_levels[:3]):
                    level = level_info['level']
                    touches = level_info['touches']
                    fig_sr.add_hline(
                        y=level,
                        line_dash="dash", 
                        line_color="green",
                        annotation_text=f"S{i+1}: ${level:.2f} ({touches}x)"
                    )
                
                # Precio actual
                fig_sr.add_hline(
                    y=current_price,
                    line_dash="solid",
                    line_color="orange",
                    line_width=3,
                    annotation_text=f"Actual: ${current_price:.2f}"
                )
                
                fig_sr.update_layout(
                    title=f"Niveles de Soporte y Resistencia - {st.session_state.ticker_seleccionado}",
                    xaxis_title="Fecha",
                    yaxis_title="Precio ($)",
                    height=600
                )
                st.plotly_chart(fig_sr, use_container_width=True)
                
                # Análisis de posicionamiento
                st.markdown("#### 🎯 Análisis de Posicionamiento Actual")
                
                # Encontrar el soporte y resistencia más cercanos
                nearest_resistance = min(resistance_levels, key=lambda x: x['distance_from_current'] if x['level'] > current_price else float('inf'))
                nearest_support = min(support_levels, key=lambda x: x['distance_from_current'] if x['level'] < current_price else float('inf'))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Precio Actual", f"${current_price:.2f}")
                
                with col2:
                    if nearest_resistance['level'] > current_price:
                        dist_resistance = (nearest_resistance['level'] / current_price - 1) * 100
                        st.metric("Resistencia Cercana", f"${nearest_resistance['level']:.2f}", f"+{dist_resistance:.1f}%")
                    else:
                        st.metric("Resistencia Cercana", "Por encima de R1")
                
                with col3:
                    if nearest_support['level'] < current_price:
                        dist_support = (1 - nearest_support['level'] / current_price) * 100
                        st.metric("Soporte Cercano", f"${nearest_support['level']:.2f}", f"-{dist_support:.1f}%")
                    else:
                        st.metric("Soporte Cercano", "Por debajo de S1")
                
                # Recomendación estratégica
                st.markdown("#### 💡 Estrategia Recomendada:")
                
                resistance_distance = (nearest_resistance['level'] / current_price - 1) * 100 if nearest_resistance['level'] > current_price else 100
                support_distance = (1 - nearest_support['level'] / current_price) * 100 if nearest_support['level'] < current_price else 100
                
                if resistance_distance < 2:
                    st.warning(f"⚠️ **Cerca de Resistencia**: Precio a {resistance_distance:.1f}% de resistencia - Considerar toma de ganancias")
                elif support_distance < 2:
                    st.info(f"🛡️ **Cerca de Soporte**: Precio a {support_distance:.1f}% de soporte - Zona de compra potencial")
                elif resistance_distance < support_distance:
                    st.success(f"🚀 **Zona de Compra**: Más cerca del soporte ({support_distance:.1f}%) que resistencia ({resistance_distance:.1f}%)")
                else:
                    st.warning(f"📈 **Zona Neutral**: Equidistante entre S/R - Esperar confirmación direccional")
                
                # Preparar datos de S/R para descarga
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("#### 📊 Niveles de Soporte y Resistencia")
                    
                    # Crear DataFrame combinado
                    sr_data = []
                    
                    # Agregar resistencias
                    for i, level_info in enumerate(resistance_levels[:5]):
                        sr_data.append({
                            'Tipo': 'Resistencia',
                            'Nivel': f'R{i+1}',
                            'Precio': round(level_info['level'], 2),
                            'Toques': level_info['touches'],
                            'Distancia_Actual_%': round(level_info['distance_from_current'], 2)
                        })
                    
                    # Agregar soportes
                    for i, level_info in enumerate(support_levels[:5]):
                        sr_data.append({
                            'Tipo': 'Soporte',
                            'Nivel': f'S{i+1}',
                            'Precio': round(level_info['level'], 2),
                            'Toques': level_info['touches'],
                            'Distancia_Actual_%': round(level_info['distance_from_current'], 2)
                        })
                    
                    # Agregar precio actual
                    sr_data.append({
                        'Tipo': 'Precio_Actual',
                        'Nivel': 'Actual',
                        'Precio': round(current_price, 2),
                        'Toques': '-',
                        'Distancia_Actual_%': 0
                    })
                    
                    sr_df = pd.DataFrame(sr_data)
                    st.dataframe(sr_df, use_container_width=True)
                
                with col2:
                    sr_csv = sr_df.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label="📥 CSV S/R",
                        data=sr_csv,
                        file_name=f"soporte_resistencia_{st.session_state.ticker_seleccionado}.csv",
                        mime="text/csv",
                        help="Descargar niveles de soporte y resistencia"
                    )
    
    else:
        st.info("📈 Análisis Avanzado de Patrones")
        st.markdown("""
        **Esta sección incluye análisis avanzados de:**
        - 📊 **Patrones de Volatilidad**: Regímenes y clusters de volatilidad
        - 📅 **Patrones Temporales**: Efectos estacionales y de día de la semana
        - 🚀 **Momentum**: Persistencia y reversión de tendencias
        - 📈 **Gaps**: Análisis de brechas de precio y su llenado
        - 🎯 **Soporte/Resistencia**: Niveles técnicos clave
        
        **👈 Cargue datos primero desde el panel lateral**
        """)
