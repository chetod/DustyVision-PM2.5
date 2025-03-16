import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pycaret.regression import load_model, predict_model
from datetime import datetime, timedelta
import numpy as np
import random

# Load models and data
pm25_model = load_model(r'C:\Users\ASUS\Desktop\projectforecastpm2_5\models\best_model')
rainfall_model = load_model(r'C:\Users\ASUS\Desktop\projectforecastpm2_5\models\rainny_model')

# Load PM2.5 data
pm25_file_path = r"C:\Users\ASUS\Desktop\projectforecastpm2_5\dataforecast\last_test.csv"
pm25_df = pd.read_csv(pm25_file_path)
pm25_df['timestamp'] = pd.to_datetime(pm25_df['timestamp'])

# Load external data for PM2.5
external_file_path = r"C:\Users\ASUS\Desktop\projectforecastpm2_5\dataforecast\cleandata_hours.csv"
external_data = pd.read_csv(external_file_path)
external_data['timestamp'] = pd.to_datetime(external_data['timestamp'])

# Load rainfall data
rainfall_file_path = r"C:\Users\ASUS\Desktop\projectforecastpm2_5\dataforecast\raw_data\raindata.csv"
rainfall_df = pd.read_csv(rainfall_file_path)
rainfall_df['DATE'] = pd.to_datetime(rainfall_df['DATE'])

# Update PM2.5 color scheme to softer tones
PM25_LEVELS = {
    "Good": {"range": (0, 12.0), "color": "#7BC8A4"},  # Soft green
    "Moderate": {"range": (12.1, 35.4), "color": "#FED766"},  # Soft yellow
    "Unhealthy for Sensitive Groups": {"range": (35.5, 55.4), "color": "#FE9F5B"},  # Soft orange
    "Unhealthy": {"range": (55.5, 150.4), "color": "#FA7C7C"},  # Soft red
    "Very Unhealthy": {"range": (150.5, 250.4), "color": "#B39DDB"},  # Soft purple
    "Hazardous": {"range": (250.5, float('inf')), "color": "#9E9E9E"}  # Soft gray
}

def get_pm25_quality(value):
    for quality, info in PM25_LEVELS.items():
        if info["range"][0] <= value <= info["range"][1]:
            return quality
    return "Hazardous"

def forecast_pm25(model, last_data, external_data):
    last_date = last_data['timestamp'].max()
    future_hours = [last_date + timedelta(hours=i+1) for i in range(7 * 24)]  # 7 days * 24 hours
    future_data = []
    current_data = last_data.copy()
    
    for future_hour in future_hours:
        new_row = {'timestamp': future_hour}
        
        # ใช้ค่า temperature และ humidity จาก external_data
        if future_hour in external_data['timestamp'].values:
            matched_row = external_data[external_data['timestamp'] == future_hour].iloc[0]
            new_row['temperature'] = matched_row['temperature']
            new_row['humidity'] = matched_row['humidity']
        else:
            new_row['temperature'] = current_data['temperature'].iloc[-1]  # ใช้ค่าล่าสุดที่มี
            new_row['humidity'] = current_data['humidity'].iloc[-1]
        
        # Create Lag Features for PM2.5
        for lag in range(1, 4):
            if len(current_data) >= lag:
                new_row[f'pm_2_5_Lag{lag}'] = current_data['pm_2_5'].iloc[-lag]
            else:
                new_row[f'pm_2_5_Lag{lag}'] = current_data['pm_2_5'].mean()
        
        # Predict PM2.5 for this hour
        new_df = pd.DataFrame([new_row])
        prediction = predict_model(model, data=new_df)
        new_row['pm_2_5'] = prediction['prediction_label'].iloc[0]
        future_data.append(new_row)
        
        # Add predicted data to current_data for next hour prediction
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
    
    forecast_df = pd.DataFrame(future_data)
    
    # Add additional columns for easier filtering and display
    forecast_df['date'] = forecast_df['timestamp'].dt.date
    forecast_df['hour'] = forecast_df['timestamp'].dt.hour
    forecast_df['day'] = forecast_df['timestamp'].dt.day_name()
    forecast_df['date_str'] = forecast_df['timestamp'].dt.strftime('%Y-%m-%d')
    forecast_df['quality'] = forecast_df['pm_2_5'].apply(get_pm25_quality)
    
    return forecast_df

def forecast_rainfall(model, user_input_features):
    """
    ทำนายปริมาณฝนสำหรับ 7 วันข้างหน้าโดยใช้ค่าที่ผู้ใช้กรอก
    """
    # สร้างรายการวันที่สำหรับ 7 วันข้างหน้า (ทุกชั่วโมง)
    last_date = rainfall_df['DATE'].max()
    future_hours = [last_date + timedelta(hours=i+1) for i in range(7 * 24)]  # 7 วัน * 24 ชั่วโมง
    future_data = []
    
    # ใช้ค่าที่ผู้ใช้กรอก
    T2M = user_input_features.get('T2M', 25.0)
    RH2M = user_input_features.get('RH2M', 70.0)
    WS10M = user_input_features.get('WS10M', 5.0)
    WSC = user_input_features.get('WSC', 2.0)
    
    # ทำนายปริมาณฝนสำหรับแต่ละชั่วโมง
    for i, future_hour in enumerate(future_hours):
        # สุ่มค่าของ feature แต่ละตัวให้แตกต่างกันในแต่ละวัน
        day_offset = (i // 24) + 1  # คำนวณวันที่ (1-7)
        
        new_row = {
            'DATE': future_hour,
            'T2M': max(0, T2M + random.uniform(-1, 1) * day_offset),  # สุ่มค่า T2M และป้องกันค่าติดลบ
            'RH2M': max(0, RH2M + random.uniform(-2, 2) * day_offset),  # สุ่มค่า RH2M และป้องกันค่าติดลบ
            'WS10M': max(0, WS10M + random.uniform(-0.5, 0.5) * day_offset),  # สุ่มค่า WS10M และป้องกันค่าติดลบ
            'WSC': max(0, WSC + random.uniform(-0.2, 0.2) * day_offset),  # สุ่มค่า WSC และป้องกันค่าติดลบ
        }
        
        # ทำนายปริมาณฝน
        prediction_df = pd.DataFrame([new_row])
        rainfall_prediction = predict_model(model, data=prediction_df)
        new_row['PRECTOTCORR'] = rainfall_prediction['prediction_label'].iloc[0]
        
        # ทำให้ค่าเป็นบวกเสมอ
        new_row['PRECTOTCORR'] = max(0, new_row['PRECTOTCORR'])
        
        future_data.append(new_row)
    
    return pd.DataFrame(future_data)

# Initialize Dash app with custom styling
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
        "https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600&display=swap"
    ]
)

# Custom CSS with dark theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Environmental Forecast Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-color: #2c3e50;
                --secondary-color: #3498db;
                --accent-color: #e74c3c;
                --background-color: #1a1a1a;
                --card-background: #2d2d2d;
                --text-color: #ffffff;
                --border-color: #404040;
            }
            
            body {
                font-family: 'Prompt', sans-serif;
                margin: 0;
                background-color: var(--background-color);
                color: var(--text-color);
            }
            
            .main-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                padding: 25px;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            
            .card {
                background-color: var(--card-background);
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                padding: 20px;
                margin-bottom: 25px;
                border: 1px solid var(--border-color);
            }
            
            .tab-container {
                display: flex;
                background-color: var(--card-background);
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 20px;
            }
            
            .tab {
                flex: 1;
                padding: 15px;
                text-align: center;
                cursor: pointer;
                border-radius: 8px;
                transition: all 0.3s ease;
                color: var(--text-color);
            }
            
            .tab:hover {
                background-color: var(--secondary-color);
            }
            
            .tab-selected {
                background-color: var(--secondary-color);
                color: white;
            }
            
            .input-group {
                margin-bottom: 15px;
            }
            
            .input-group label {
                display: block;
                margin-bottom: 5px;
                color: var(--text-color);
            }
            
            .input-group input {
                width: 100%;
                padding: 8px;
                border-radius: 5px;
                border: 1px solid var(--border-color);
                background-color: var(--card-background);
                color: var(--text-color);
            }
            
            button {
                background-color: var(--secondary-color);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            button:hover {
                background-color: #2980b9;
            }
            
            .chart-container {
                background-color: var(--card-background);
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                background-color: var(--card-background);
            }
            
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid var(--border-color);
                color: var(--text-color);
            }
            
            th {
                background-color: var(--primary-color);
            }
            
            tr:hover {
                background-color: rgba(52, 152, 219, 0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App Layout
app.layout = html.Div(className='main-container', children=[
    html.Div(className='header', children=[
        html.H1("Environmental Forecast Dashboard", style={'margin-bottom': '5px'}),
        html.P("7-Day Forecast for PM2.5 and Rainfall")
    ]),
    
    # Main Tabs
    html.Div(className='tab-container', children=[
        html.Div(id='pm25-tab', className='tab tab-selected', children=[
            html.I(className="fas fa-wind", style={'margin-right': '8px'}),
            "PM2.5 Forecast"
        ]),
        html.Div(id='rainfall-tab', className='tab', children=[
            html.I(className="fas fa-cloud-rain", style={'margin-right': '8px'}),
            "Rainfall Forecast"
        ])
    ]),
    
    # PM2.5 Section
    html.Div(id='pm25-content', className='content', children=[
        html.Div(className='card', children=[
            html.Div(style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '20px'}, children=[
                html.H3("PM2.5 7-Day Forecast", style={'margin': '0', 'color': '#fff'}),
                html.Div(className='quality-legend', style={'display': 'flex', 'gap': '10px'}, children=[
                    html.Div(style={'display': 'flex', 'align-items': 'center'}, children=[
                        html.Div(style={'width': '12px', 'height': '12px', 'borderRadius': '50%', 'backgroundColor': color['color'], 'marginRight': '5px'}),
                        html.Span(level, style={'fontSize': '12px', 'color': '#fff'})
                    ]) for level, color in PM25_LEVELS.items()
                ])
            ]),
            dcc.Graph(id='pm25-forecast-graph')
        ]),
        html.Div(className='card', children=[
            html.Div(style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '20px'}, children=[
                html.H3("Hourly Breakdown", style={'margin': '0', 'color': '#fff'}),
                dcc.DatePickerSingle(
                    id='date-picker',
                    min_date_allowed=datetime.now().date(),
                    max_date_allowed=datetime.now().date() + timedelta(days=7),
                    initial_visible_month=datetime.now().date(),
                    date=datetime.now().date(),
                    style={'backgroundColor': 'var(--card-background)'}
                )
            ]),
            dcc.Graph(id='pm25-hourly-graph')
        ])
    ]),
    
    # Rainfall Section
    html.Div(id='rainfall-content', className='content', style={'display': 'none'}, children=[
        html.Div(className='card', children=[
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
                html.Div(className='input-group', children=[
                    html.Label("Temperature (°C):"),
                    dcc.Input(id='temp-input', type='number', value=25, step=0.1)
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Relative Humidity (%):"),
                    dcc.Input(id='humidity-input', type='number', value=70, step=0.1)
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Wind Speed at 10m (m/s):"),
                    dcc.Input(id='wind-speed-input', type='number', value=5, step=0.1)
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Wind Speed Component (m/s):"),
                    dcc.Input(id='wind-component-input', type='number', value=2, step=0.1)
                ])
            ]),
            html.Button('Generate Forecast', id='forecast-button', n_clicks=0)
        ]),
        html.Div(className='card', children=[
            dcc.Graph(id='rainfall-forecast-graph')
        ]),
        html.Div(className='card', children=[
            dcc.Graph(id='daily-rainfall-summary')
        ])
    ]),
    
    # Store components for state management
    dcc.Store(id='active-tab', data='pm25'),
    dcc.Interval(id='pm25-update', interval=3600000)  # Update every hour
])

# Callbacks
@app.callback(
    [Output('pm25-tab', 'className'),
     Output('rainfall-tab', 'className'),
     Output('pm25-content', 'style'),
     Output('rainfall-content', 'style'),
     Output('active-tab', 'data')],
    [Input('pm25-tab', 'n_clicks'),
     Input('rainfall-tab', 'n_clicks')],
    [dash.dependencies.State('active-tab', 'data')]
)
def update_tabs(pm25_clicks, rainfall_clicks, active_tab):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'tab tab-selected', 'tab', {'display': 'block'}, {'display': 'none'}, 'pm25'
    
    clicked_tab = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if clicked_tab == 'pm25-tab':
        return 'tab tab-selected', 'tab', {'display': 'block'}, {'display': 'none'}, 'pm25'
    else:
        return 'tab', 'tab tab-selected', {'display': 'none'}, {'display': 'block'}, 'rainfall'

@app.callback(
    [Output('pm25-forecast-graph', 'figure'),
     Output('pm25-hourly-graph', 'figure')],
    [Input('pm25-update', 'n_intervals'),
     Input('date-picker', 'date')]
)
def update_pm25_graphs(n, selected_date):
    # Generate PM2.5 forecast
    forecast_data = forecast_pm25(pm25_model, pm25_df, external_data)
    
    # Daily forecast graph
    daily_forecast = forecast_data.groupby('date').agg({
        'pm_2_5': ['mean', 'max', 'min'],
        'temperature': ['mean', 'max', 'min'],
        'humidity': ['mean', 'max', 'min'],
        'quality': lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
    }).reset_index()
    
    # Flatten multi-level columns
    daily_forecast.columns = ['date', 'pm25_mean', 'pm25_max', 'pm25_min', 
                            'temp_mean', 'temp_max', 'temp_min',
                            'humidity_mean', 'humidity_max', 'humidity_min', 'quality']
    
    colors = [PM25_LEVELS[quality]['color'] for quality in daily_forecast['quality']]
    
    # Create daily forecast figure
    daily_fig = make_subplots(rows=2, cols=1,
                             shared_xaxes=True,
                             vertical_spacing=0.1,
                             subplot_titles=("PM2.5 Levels", "Environmental Conditions"))
    
    # Add PM2.5 bars with gradient fill
    daily_fig.add_trace(
        go.Bar(x=daily_forecast['date'],
               y=daily_forecast['pm25_mean'],
               name='PM2.5',
               marker=dict(
                   color=colors,
                   line=dict(color=colors, width=1)
               ),
               error_y=dict(
                   type='data',
                   symmetric=False,
                   array=daily_forecast['pm25_max'] - daily_forecast['pm25_mean'],
                   arrayminus=daily_forecast['pm25_mean'] - daily_forecast['pm25_min'],
                   color='rgba(255,255,255,0.3)'
               ),
               hovertemplate="<b>%{x}</b><br>" +
                           "PM2.5: %{y:.1f} μg/m³<br>" +
                           "Max: %{error_y.array:.1f}<br>" +
                           "Min: %{error_y.arrayminus:.1f}<br>" +
                           "<extra></extra>"),
        row=1, col=1
    )
    
    # Add temperature area
    daily_fig.add_trace(
        go.Scatter(x=daily_forecast['date'],
                  y=daily_forecast['temp_max'],
                  name='Temperature Range',
                  line=dict(color='rgba(0,0,0,0)'),
                  showlegend=False),
        row=2, col=1
    )
    
    daily_fig.add_trace(
        go.Scatter(x=daily_forecast['date'],
                  y=daily_forecast['temp_min'],
                  name='Temperature',
                  fill='tonexty',
                  fillcolor='rgba(255,152,0,0.2)',
                  line=dict(color='rgba(255,152,0,0.', width=2),
                  hovertemplate="<b>%{x}</b><br>" +
                              "Temperature: %{y:.1f}°C<br>" +
                              "<extra></extra>"),
        row=2, col=1
    )
    
    # Add humidity area
    daily_fig.add_trace(
        go.Scatter(x=daily_forecast['date'],
                  y=daily_forecast['humidity_max'],
                  name='Humidity Range',
                  line=dict(color='rgba(0,0,0,0)'),
                  showlegend=False),
        row=2, col=1
    )
    
    daily_fig.add_trace(
        go.Scatter(x=daily_forecast['date'],
                  y=daily_forecast['humidity_min'],
                  name='Humidity',
                  fill='tonexty',
                  fillcolor='rgba(0,188,212,0.2)',
                  line=dict(color='rgba(0,188,212,0.', width=2),
                  hovertemplate="<b>%{x}</b><br>" +
                              "Humidity: %{y:.1f}%<br>" +
                              "<extra></extra>"),
        row=2, col=1
    )
    
    # Add reference lines for air quality levels
    for level, info in PM25_LEVELS.items():
        if info['range'][1] != float('inf'):
            daily_fig.add_shape(
                type="line",
                x0=daily_forecast['date'].min(),
                y0=info['range'][1],
                x1=daily_forecast['date'].max(),
                y1=info['range'][1],
                line=dict(color=info['color'], width=1, dash="dot"),
                row=1, col=1
            )
            daily_fig.add_annotation(
                x=daily_forecast['date'].max(),
                y=info['range'][1],
                text=level,
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(size=10, color=info['color']),
                row=1, col=1
            )
    
    # Update layout
    daily_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    daily_fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        title_text="Date",
        row=2, col=1
    )
    
    daily_fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        title_text="PM2.5 (μg/m³)",
        row=1, col=1
    )
    
    daily_fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        title_text="Value",
        row=2, col=1
    )
    
    # Hourly forecast graph
    if selected_date:
        selected_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
        hourly_data = forecast_data[forecast_data['date'] == selected_date]
    else:
        hourly_data = forecast_data[forecast_data['date'] == forecast_data['date'].min()]
    
    hourly_fig = go.Figure()
    
    # Add bars for PM2.5 levels with quality colors
    hourly_fig.add_trace(
        go.Bar(x=hourly_data['hour'],
               y=hourly_data['pm_2_5'],
               name='PM2.5',
               marker=dict(
                   color=[PM25_LEVELS[q]['color'] for q in hourly_data['quality']],
                   line=dict(color=[PM25_LEVELS[q]['color'] for q in hourly_data['quality']], width=1)
               ),
               hovertemplate="Hour: %{x}:00<br>" +
                           "PM2.5: %{y:.1f} μg/m³<br>" +
                           "Quality: %{text}<br>" +
                           "<extra></extra>",
               text=hourly_data['quality'])
    )
    
    # Add smooth line for trend
    hourly_fig.add_trace(
        go.Scatter(x=hourly_data['hour'],
                  y=hourly_data['pm_2_5'],
                  mode='lines',
                  line=dict(shape='spline', color='rgba(255,255,255,0.5)', width=2),
                  name='Trend',
                  hoverinfo='skip')
    )
    
    # Add reference lines
    for level, info in PM25_LEVELS.items():
        if info['range'][1] != float('inf'):
            hourly_fig.add_shape(
                type="line",
                x0=-0.5,
                y0=info['range'][1],
                x1=23.5,
                y1=info['range'][1],
                line=dict(color=info['color'], width=1, dash="dot")
            )
    
    # Update layout
    hourly_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=f"Hourly PM2.5 Levels - {hourly_data['day'].iloc[0]}, {selected_date}",
        xaxis=dict(
            title="Hour of Day",
            tickmode='array',
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="PM2.5 (μg/m³)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode='x unified'
    )
    
    return daily_fig, hourly_fig

@app.callback(
    [Output('rainfall-forecast-graph', 'figure'),
     Output('daily-rainfall-summary', 'figure')],
    [Input('forecast-button', 'n_clicks')],
    [Input('temp-input', 'value'),
     Input('humidity-input', 'value'),
     Input('wind-speed-input', 'value'),
     Input('wind-component-input', 'value')]
)
def update_rainfall_graphs(n_clicks, temp, humidity, wind_speed, wind_component):
    if n_clicks == 0:
        return {}, {}
    
    # Generate rainfall forecast
    user_input = {
        'T2M': temp,
        'RH2M': humidity,
        'WS10M': wind_speed,
        'WSC': wind_component
    }
    
    forecast_df = forecast_rainfall(rainfall_model, user_input)
    
    # Hourly rainfall plot
    hourly_fig = go.Figure()
    
    hourly_fig.add_trace(
        go.Scatter(x=forecast_df['DATE'],
                  y=forecast_df['PRECTOTCORR'],
                  fill='tozeroy',
                  name='Rainfall',
                  line=dict(color='#3498db'))
    )
    
    hourly_fig.update_layout(
        template="plotly_dark",
        title="Hourly Rainfall Forecast",
        xaxis_title="Time",
        yaxis_title="Rainfall (mm)",
        height=400
    )
    
    # Daily summary
    forecast_df['DATE_DAY'] = forecast_df['DATE'].dt.date
    daily_summary = forecast_df.groupby('DATE_DAY').agg({
        'PRECTOTCORR': ['sum', 'mean', 'max']
    }).reset_index()
    
    daily_fig = go.Figure()
    
    daily_fig.add_trace(
        go.Bar(x=daily_summary['DATE_DAY'],
               y=daily_summary['PRECTOTCORR']['sum'],
               name='Total Daily Rainfall',
               marker_color='#3498db')
    )
    
    daily_fig.add_trace(
        go.Scatter(x=daily_summary['DATE_DAY'],
                  y=daily_summary['PRECTOTCORR']['max'],
                  name='Max Hourly Rainfall',
                  mode='lines+markers',
                  line=dict(color='#e74c3c'))
    )
    
    daily_fig.update_layout(
        template="plotly_dark",
        title="Daily Rainfall Summary",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return hourly_fig, daily_fig

if __name__ == '__main__':
    print("Starting the dashboard server...")
    print("Please open http://127.0.0.1:8050 in your web browser")
    app.run_server(debug=True, host='0.0.0.0', port=8050) 