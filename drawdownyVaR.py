import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from dash import dcc
from dash import html
import plotly.graph_objs as go
import requests
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

def obtener_datos(tickers, period, custom_start=None, custom_end=None):
    datos = {}
    for ticker in tickers:
        datos[ticker] = get_stock_data(ticker, period, custom_start, custom_end)
    return datos

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
                    last_quarter = now.replace(year=now.year-1, month=12, day=1)
                    this_quarter = now.replace(month=((now.quarter - 1) * 3) + 1, day=1)
                else:
                    last_quarter = now.replace(month=((now.quarter - 2) * 3) + 1, day=1)
                    this_quarter = now.replace(month=((now.quarter - 1) * 3) + 1, day=1)
                df = df[(df['Date'] >= last_quarter) & (df['Date'] < this_quarter)]
            
            return df
        else:
            return None
    except Exception as e:
        print(f"Error al obtener datos de Yahoo Finance: {e}")
        return None

def main():
    tickers = ['ALAB', 'SAP', 'AMZN', 'TSLA', 'BABA', 'PANW', 'SPY', 'AMD', 'GTLB', 'CRM']
    period = "1y"
    datos = obtener_datos(tickers, period)

    # Calcula el drawdown histórico
    def calcular_drawdown(df):
        drawdown = pd.DataFrame({
            'Fecha': df.index,
            'Drawdown': df['Close'].rolling(window=30).max() - df['Close']
        })
        return drawdown

    # Calcula el VaR
    def calcular_var(df):
        var = pd.DataFrame({
            'Fecha': df.index,
            'VaR': df['Close'].rolling(window=30).mean() - df['Close']
        })
        return var

    # Crea el dashboard
    app = dash.Dash(__name__)

    # Define la estructura de la interfaz de usuario
    app.layout = html.Div([
        html.H1('Dashboard de Drawdown y VaR'),
        dcc.Graph(id='drawdown-graph'),
        dcc.Graph(id='var-graph'),
        dcc.Dropdown(
            id='period-dropdown',
            options=[
                {'label': '1 día', 'value': '1d'},
                {'label': '5 días', 'value': '5d'},
                {'label': '1 mes', 'value': '1mo'},
                {'label': '3 meses', 'value': '3mo'},
                {'label': '6 meses', 'value': '6mo'},
                {'label': '1 año', 'value': '1y'},
                {'label': '2 años', 'value': '2y'},
                {'label': '3 años', 'value': '3y'},
                {'label': '5 años', 'value': '5y'},
                {'label': 'Max', 'value': 'ax'},
            ],
            value='1y'
        ),
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[
                {'label': 'ALAB', 'value': 'ALAB'},
                {'label': 'SAP', 'value': 'SAP'},
                {'label': 'AMZN', 'value': 'AMZN'},
                {'label': 'TSLA', 'value': 'TSLA'},
                {'label': 'BABA', 'value': 'BABA'},
                {'label': 'PANW', 'value': 'PANW'},
                {'label': 'SPY', 'value': 'SPY'},
                {'label': 'AMD', 'value': 'AMD'},
                {'label': 'GTLB', 'value': 'GTLB'},
                {'label': 'CRM', 'value': 'CRM'},
            ],
            value='ALAB'
        ),
        dcc.Dropdown(
            id='custom-start-dropdown',
            options=[
                {'label': 'Custom', 'value': 'custom'},
            ],
            value='custom'
        ),
        dcc.Dropdown(
            id='custom-end-dropdown',
            options=[
                {'label': 'Custom', 'value': 'custom'},
            ],
            value='custom'
        ),
    ])

    # Define las funciones para actualizar los gráficos y tablas
    @app.callback(
        Output('drawdown-graph', 'figure'),
        [Input('period-dropdown', 'value'),
         Input('ticker-dropdown', 'value'),
         Input('custom-start-dropdown', 'value'),
         Input('custom-end-dropdown', 'value')]
    )
    def update_drawdown_graph(period, ticker, custom_start, custom_end):
        # Crea el gráfico de línea para el drawdown
        df = get_stock_data(ticker, period, custom_start, custom_end)
        drawdown = calcular_drawdown(df)
        fig = go.Figure(data=[go.Scatter(x=drawdown['Fecha'], y=drawdown['Drawdown'])])
        return fig

    @app.callback(
        Output('var-graph', 'figure'),
        [Input('period-dropdown', 'value'),
         Input('ticker-dropdown', 'value'),
         Input('custom-start-dropdown', 'value'),
         Input('custom-end-dropdown', 'value')]
    )
    def update_var_graph(period, ticker, custom_start, custom_end):
        # Crea el gráfico de barras para el VaR
        df = get_stock_data(ticker, period, custom_start, custom_end)
        var = calcular_var(df)
        fig = go.Figure(data=[go.Bar(x=var['Fecha'], y=var['VaR'])])
        return fig

    # Inicia el servidor del dashboard
    app.run_server(debug=True)

