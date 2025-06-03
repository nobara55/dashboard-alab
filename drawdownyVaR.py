import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Carga los datos históricos del fondo
df = pd.read_csv('datos_historicos.csv')

# Calcula el drawdown histórico
def calcular_drawdown(df):
    drawdown = pd.DataFrame({
        'Fecha': df['Fecha'],
        'Drawdown': df['Valor'].rolling(window=30).max() - df['Valor']
    })
    return drawdown

# Calcula el VaR
def calcular_var(df):
    var = pd.DataFrame({
        'Fecha': df['Fecha'],
        'VaR': df['Valor'].rolling(window=30).mean() - df['Valor']
    })
    return var

# Crea el dashboard
app = dash.Dash(__name__)

# Define los componentes del dashboard
app.layout = html.Div([
    html.H1('Análisis de Drawdown y VaR'),
    dcc.Graph(id='drawdown-graph'),
    dcc.Graph(id='var-graph'),
    html.Table(id='drawdown-table'),
    html.Table(id='var-table')
])

# Define las funciones para actualizar los gráficos y tablas
@app.callback(
    dash.dependencies.Output('drawdown-graph', 'figure'),
    [dash.dependencies.Input('drawdown-data', 'data')]
)
def update_drawdown_graph(data):
    # Crea el gráfico de línea para el drawdown
    fig = go.Figure(data=[go.Scatter(x=data['Fecha'], y=data['Drawdown'])])
    return fig

@app.callback(
    dash.dependencies.Output('var-graph', 'figure'),
    [dash.dependencies.Input('var-data', 'data')]
)
def update_var_graph(data):
    # Crea el gráfico de barras para el VaR
    fig = go.Figure(data=[go.Bar(x=data['Fecha'], y=data['VaR'])])
    return fig

@app.callback(
    dash.dependencies.Output('drawdown-table', 'children'),
    [dash.dependencies.Input('drawdown-data', 'data')]
)
def update_drawdown_table(data):
    # Crea la tabla para el drawdown
    table = html.Table([
        html.Tr([html.Th('Fecha'), html.Th('Drawdown')]),
        html.Tr([html.Td(data['Fecha'][0]), html.Td(data['Drawdown'][0])]),
        html.Tr([html.Td(data['Fecha'][1]), html.Td(data['Drawdown'][1])]),
        #...
    ])
    return table

@app.callback(
    dash.dependencies.Output('var-table', 'children'),
    [dash.dependencies.Input('var-data', 'data')]
)
def update_var_table(data):
    # Crea la tabla para el VaR
    table = html.Table([
        html.Tr([html.Th('Fecha'), html.Th('VaR')]),
        html.Tr([html.Td(data['Fecha'][0]), html.Td(data['VaR'][0])]),
        html.Tr([html.Td(data['Fecha'][1]), html.Td(data['VaR'][1])]),
        #...
    ])
    return table

# Calcula el drawdown y VaR
drawdown = calcular_drawdown(df)
var = calcular_var(df)

# Actualiza los gráficos y tablas
app.callback(
    dash.dependencies.Output('drawdown-graph', 'figure'),
    [dash.dependencies.Input('drawdown-data', 'data')]
)(update_drawdown_graph)(drawdown)

app.callback(
    dash.dependencies.Output('var-graph', 'figure'),
    [dash.dependencies.Input('var-data', 'data')]
)(update_var_graph)(var)

app.callback(
    dash.dependencies.Output('drawdown-table', 'children'),
    [dash.dependencies.Input('drawdown-data', 'data')]
)(update_drawdown_table)(drawdown)

app.callback(
    dash.dependencies.Output('var-table', 'children'),
    [dash.dependencies.Input('var-data', 'data')]
)(update_var_table)(var)

# Inicia el servidor del dashboard
if __name__ == '__main__':
    app.run_server(debug=True)

