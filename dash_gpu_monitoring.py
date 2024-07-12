import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# timeout 86400 nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > results-file.csv &

# Load the data
df = pd.read_csv('data/results-file.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove units from column names
df.columns = df.columns.str.replace(r'\s*\[.*\]', '', regex=True)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("GPU Monitoring Dashboard"),

    dcc.Dropdown(
        id='device-dropdown',
        options=[{'label': i, 'value': i} for i in df['name'].unique()],
        value=df['name'].unique()[0],
        multi=False
    ),

    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'Temperature', 'value': 'temperature.gpu'},
            {'label': 'GPU Utilization', 'value': 'utilization.gpu'},
            {'label': 'Memory Utilization', 'value': 'utilization.memory'}
        ],
        value='temperature.gpu',
        multi=False
    ),

    dcc.Graph(id='time-series-chart'),

    dcc.Graph(id='utilization-scatter'),

    dcc.Graph(id='memory-pie-chart')
])


@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('device-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_time_series(selected_device, selected_metric):
    filtered_df = df[df['name'] == selected_device]
    fig = px.line(filtered_df, x='timestamp', y=selected_metric,
                  title=f'{selected_metric} Over Time for {selected_device}')
    return fig


@app.callback(
    Output('utilization-scatter', 'figure'),
    [Input('device-dropdown', 'value')]
)
def update_scatter(selected_device):
    filtered_df = df[df['name'] == selected_device]
    fig = px.scatter(filtered_df, x='utilization.gpu', y='utilization.memory',
                     title=f'GPU vs Memory Utilization for {selected_device}')
    return fig


@app.callback(
    Output('memory-pie-chart', 'figure'),
    [Input('device-dropdown', 'value')]
)
def update_pie_chart(selected_device):
    filtered_df = df[df['name'] == selected_device].iloc[-1]  # Get the latest record
    values = [filtered_df['memory.used'], filtered_df['memory.free']]
    labels = ['Used', 'Free']
    fig = px.pie(values=values, names=labels,
                 title=f'Memory Usage for {selected_device}')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
