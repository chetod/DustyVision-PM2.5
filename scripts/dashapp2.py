import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pycaret.regression import load_model, predict_model
from datetime import datetime, timedelta
import numpy as np

# Load data
file_path = r"C:\Users\ASUS\Desktop\projectforecastpm2_5\dataforecast\last_test.csv"
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Load the trained model
model = load_model(r'C:\Users\ASUS\Desktop\projectforecastpm2_5\models\best_model')
np.random.seed(42)

def forecast_next_7_days(model, last_data, external_data):
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
        # for lag in range(1, 4):
        #     if len(current_data) >= lag:
        #         new_row[f'pm_2_5_Lag{lag}'] = current_data['pm_2_5'].iloc[-lag]
        #     else:
        #         new_row[f'pm_2_5_Lag{lag}'] = current_data['pm_2_5'].mean()
        
        # Predict PM2.5 for this hour
        new_df = pd.DataFrame([new_row])
        prediction = predict_model(model, data=new_df)
        new_row['pm_2_5'] = prediction['prediction_label'].iloc[0]
        future_data.append(new_row)
        
        # Add predicted data to current_data for next hour prediction
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
    
    return pd.DataFrame(future_data)

# โหลด external data ที่มีค่า temperature และ humidity
external_file_path = r"C:\Users\ASUS\Desktop\projectforecastpm2_5\dataforecast\cleandata_hours.csv"
external_data = pd.read_csv(external_file_path)
external_data['timestamp'] = pd.to_datetime(external_data['timestamp'])

# Generate forecast data
forecast_data = forecast_next_7_days(model, df, external_data)

# Add hour and date columns for easier filtering
forecast_data['date'] = forecast_data['timestamp'].dt.date
forecast_data['hour'] = forecast_data['timestamp'].dt.hour
forecast_data['day'] = forecast_data['timestamp'].dt.day_name()
forecast_data['date_str'] = forecast_data['timestamp'].dt.strftime('%Y-%m-%d')

# Define PM2.5 quality levels
def get_pm25_quality(value):
    if value <= 12.0:
        return "Good"
    elif value <= 35.4:
        return "Moderate"
    elif value <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif value <= 150.4:
        return "Unhealthy"
    elif value <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"

forecast_data['quality'] = forecast_data['pm_2_5'].apply(get_pm25_quality)

# Define color mapping for PM2.5 quality
color_map = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023"
}

# Create Dash application with custom CSS
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
        "https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600&display=swap"
    ]
)

# Define custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>PM2.5 Forecast Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Prompt', sans-serif;
                margin: 0;
                background-color: #f0f4f8;
            }
            .main-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                border-radius: 10px;
            }
            .header {
                background: linear-gradient(135deg, #2c3e50, #3498db);
                color: white;
                padding: 25px;
                text-align: center;
                border-radius: 10px 10px 0 0;
                margin-bottom: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .header h1 {
                margin-bottom: 5px;
                font-weight: 600;
            }
            .header p {
                font-weight: 300;
                font-size: 1.1em;
                opacity: 0.9;
            }
            .card {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.08);
                padding: 20px;
                margin-bottom: 25px;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover {
                transform: translateY(-3px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .chart-container {
                height: 450px;
            }
            .info-box {
                padding: 20px;
                background-color: #e3f2fd;
                border-left: 5px solid #2196f3;
                border-radius: 8px;
                margin-bottom: 25px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .info-box h4 {
                margin-top: 0;
                color: #1565c0;
                font-weight: 500;
            }
            .footer {
                text-align: center;
                padding: 20px;
                color: #555;
                font-size: 0.9em;
                margin-top: 30px;
                border-top: 1px solid #eee;
            }
            .quality-indicator {
                display: inline-block;
                width: 15px;
                height: 15px;
                border-radius: 50%;
                margin-right: 5px;
            }
            .quality-legend {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                margin: 15px 0;
            }
            .quality-item {
                display: flex;
                align-items: center;
                margin: 6px 10px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.08);
                text-align: center;
                transition: transform 0.3s ease;
            }
            .stat-card:hover {
                transform: translateY(-5px);
            }
            .stat-value {
                font-size: 28px;
                font-weight: 600;
                margin: 12px 0;
                color: #2c3e50;
            }
            .stat-label {
                color: #7f8c8d;
                font-weight: 300;
            }
            .stat-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
                margin-bottom: 25px;
            }
            .date-picker-container {
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 15px;
            }
            .tab-selected {
                border-bottom: 3px solid #2196f3 !important;
                color: #2196f3 !important;
            }
            .tab-container {
                margin-bottom: 15px;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                display: inline-block;
                margin-right: 10px;
                font-weight: 500;
                color: #555;
                transition: all 0.3s;
            }
            .tab:hover {
                color: #2196f3;
            }
            .param-card {
                padding: 15px;
                border-radius: 8px;
                background: #f9f9f9;
                margin-bottom: 15px;
                border-left: 4px solid #3498db;
            }
            .param-title {
                font-weight: 500;
                margin-bottom: 5px;
                color: #2c3e50;
            }
            .param-value {
                font-size: 1.2em;
                font-weight: 600;
                color: #3498db;
            }
            .param-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .date-nav-btn {
                background: #f0f4f8;
                border: none;
                border-radius: 50%;
                width: 36px;
                height: 36px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s;
            }
            .date-nav-btn:hover {
                background: #e3e8ed;
            }
            .date-display {
                font-size: 1.2em;
                font-weight: 500;
                color: #2c3e50;
                display: flex;
                align-items: center;
            }
            .tip-box {
                background: #fffde7;
                border-left: 4px solid #fbc02d;
                padding: 15px;
                margin-top: 20px;
                border-radius: 8px;
            }
            .tip-title {
                color: #f57f17;
                font-weight: 500;
                margin-bottom: 5px;
                display: flex;
                align-items: center;
            }
            .tip-title i {
                margin-right: 8px;
            }
            .hour-band {
                background: #f1f8e9;
                border-radius: 8px;
                padding: 10px;
                margin: 15px 0;
            }
            .hour-band-title {
                font-weight: 500;
                color: #33691e;
                margin-bottom: 5px;
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

# Create the app layout
app.layout = html.Div(className='main-container', children=[
    html.Div(className='header', children=[
        html.H1("PM2.5 Forecast Dashboard", style={'margin-bottom': '5px'}),
        html.P("30-day prediction of PM2.5 levels with environmental parameters")
    ]),
    
    html.Div(className='info-box', children=[
        html.H4("PM2.5 Air Quality Index"),
        html.Div(className='quality-legend', children=[
            html.Div(className='quality-item', children=[
                html.Div(className='quality-indicator', style={'background-color': color}) for color in color_map.values()
            ]),
            html.Div(className='quality-item', children=[
                html.Span(level, style={'margin-right': '15px'}) for level in color_map.keys()
            ])
        ])
    ]),
    
    html.Div(className='stat-grid', children=[
        html.Div(className='stat-card', children=[
            html.I(className="fas fa-calendar-alt", style={'color': '#3498db', 'font-size': '28px'}),
            html.Div(className='stat-value', id='forecast-days'),
            html.Div(className='stat-label', children="Days Forecasted")
        ]),
        html.Div(className='stat-card', children=[
            html.I(className="fas fa-wind", style={'color': '#3498db', 'font-size': '28px'}),
            html.Div(className='stat-value', id='avg-pm25'),
            html.Div(className='stat-label', children="Average PM2.5")
        ]),
        html.Div(className='stat-card', children=[
            html.I(className="fas fa-temperature-high", style={'color': '#3498db', 'font-size': '28px'}),
            html.Div(className='stat-value', id='avg-temp'),
            html.Div(className='stat-label', children="Average Temperature (°C)")
        ]),
        html.Div(className='stat-card', children=[
            html.I(className="fas fa-tint", style={'color': '#3498db', 'font-size': '28px'}),
            html.Div(className='stat-value', id='avg-humidity'),
            html.Div(className='stat-label', children="Average Humidity (%)")
        ])
    ]),
    
    html.Div(className='card', children=[
        html.H3("Daily PM2.5 Forecast with Environmental Parameters", style={'margin-top': '0', 'color': '#2c3e50'}),
        html.P("Click on a day to see hourly breakdown", style={'color': '#7f8c8d'}),
        dcc.Graph(id='7-day-forecast', className='chart-container', config={'responsive': True})
    ]),
    
    html.Div(className='card', children=[
        html.H3("Hourly Breakdown", style={'margin-top': '0', 'color': '#2c3e50'}),
        
        html.Div(className='date-picker-container', children=[
            html.Button(id='prev-day', className='date-nav-btn', children=[
                html.I(className="fas fa-chevron-left")
            ]),
            html.Div(id='selected-date-display', className='date-display'),
            html.Button(id='next-day', className='date-nav-btn', children=[
                html.I(className="fas fa-chevron-right")
            ]),
            dcc.Store(id='current-date', data=forecast_data['date_str'].min()),
            dcc.Store(id='min-date', data=forecast_data['date_str'].min()),
            dcc.Store(id='max-date', data=forecast_data['date_str'].max())
        ]),
        
        html.Div(className='tab-container', children=[
            html.Div(id='tab-pm25', className='tab tab-selected', children="PM2.5 Levels"),
            html.Div(id='tab-temp', className='tab', children="Temperature"),
            html.Div(id='tab-humidity', className='tab', children="Humidity"),
            html.Div(id='tab-combined', className='tab', children="Combined View"),
            dcc.Store(id='active-tab', data='pm25')
        ]),
        
        html.Div(id='daily-parameter-summary', className='param-grid'),
        dcc.Graph(id='hourly-forecast', className='chart-container', config={'responsive': True}),
        
        html.Div(className='tip-box', children=[
            html.Div(className='tip-title', children=[
                html.I(className="fas fa-lightbulb"),
                "Air Quality Tips"
            ]),
            html.Div(id='air-quality-tips')
        ])
    ]),
    
    html.Div(className='footer', children=[
        html.P("PM2.5 Forecast Dashboard © 2025"),
        html.P("Powered by PyCaret, Dash, and Plotly")
    ])
])

# Callback for date navigation
@app.callback(
    [Output('current-date', 'data'),
     Output('selected-date-display', 'children')],
    [Input('prev-day', 'n_clicks'),
     Input('next-day', 'n_clicks'),
     Input('7-day-forecast', 'clickData')],
    [dash.dependencies.State('current-date', 'data'),
     dash.dependencies.State('min-date', 'data'),
     dash.dependencies.State('max-date', 'data')]
)
def update_date(prev_clicks, next_clicks, click_data, current_date, min_date, max_date):
    ctx = dash.callback_context
    if not ctx.triggered:
        selected_date = current_date
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == 'prev-day':
            # Get previous date
            current_dt = datetime.strptime(current_date, '%Y-%m-%d')
            new_dt = current_dt - timedelta(days=1)
            selected_date = new_dt.strftime('%Y-%m-%d')
            # Check if we're at the minimum date
            if selected_date < min_date:
                selected_date = min_date
                
        elif triggered_id == 'next-day':
            # Get next date
            current_dt = datetime.strptime(current_date, '%Y-%m-%d')
            new_dt = current_dt + timedelta(days=1)
            selected_date = new_dt.strftime('%Y-%m-%d')
            # Check if we're at the maximum date
            if selected_date > max_date:
                selected_date = max_date
                
        elif triggered_id == '7-day-forecast' and click_data:
            # Get date from click data
            selected_date = click_data['points'][0]['x']
    
    # Format the date display
    date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    day_name = date_obj.strftime('%A')
    formatted_date = date_obj.strftime('%d %B %Y')
    date_display = [
        html.I(className="fas fa-calendar-day", style={'margin-right': '8px'}),
        f"{day_name}, {formatted_date}"
    ]
    
    return selected_date, date_display

# Callback for tab selection
@app.callback(
    [Output('tab-pm25', 'className'),
     Output('tab-temp', 'className'),
     Output('tab-humidity', 'className'),
     Output('tab-combined', 'className'),
     Output('active-tab', 'data')],
    [Input('tab-pm25', 'n_clicks'),
     Input('tab-temp', 'n_clicks'),
     Input('tab-humidity', 'n_clicks'),
     Input('tab-combined', 'n_clicks')],
    [dash.dependencies.State('active-tab', 'data')]
)
def update_active_tab(pm25_clicks, temp_clicks, humidity_clicks, combined_clicks, active_tab):
    ctx = dash.callback_context
    if not ctx.triggered:
        # Default to PM2.5 tab
        return 'tab tab-selected', 'tab', 'tab', 'tab', 'pm25'
    else:
        # Get the tab that was clicked
        clicked_tab = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if clicked_tab == 'tab-pm25':
            return 'tab tab-selected', 'tab', 'tab', 'tab', 'pm25'
        elif clicked_tab == 'tab-temp':
            return 'tab', 'tab tab-selected', 'tab', 'tab', 'temp'
        elif clicked_tab == 'tab-humidity':
            return 'tab', 'tab', 'tab tab-selected', 'tab', 'humidity'
        elif clicked_tab == 'tab-combined':
            return 'tab', 'tab', 'tab', 'tab tab-selected', 'combined'
        
        # If no match, return current state
        return 'tab tab-selected', 'tab', 'tab', 'tab', active_tab

# Callback for statistics
@app.callback(
    [Output('forecast-days', 'children'),
     Output('avg-pm25', 'children'),
     Output('avg-temp', 'children'),
     Output('avg-humidity', 'children')],
    [Input('current-date', 'data')]
)
def update_stats(date):
    days = len(forecast_data['date'].unique())
    avg_pm25 = f"{forecast_data['pm_2_5'].mean():.1f}"
    avg_temp = f"{forecast_data['temperature'].mean():.1f}°C"
    avg_humidity = f"{forecast_data['humidity'].mean():.1f}%"
    return days, avg_pm25, avg_temp, avg_humidity

# Callback for daily parameter summary
@app.callback(
    Output('daily-parameter-summary', 'children'),
    [Input('current-date', 'data')]
)
def update_daily_summary(date):
    # Filter data for the selected date
    date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    daily_data = forecast_data[forecast_data['date'] == date_obj]
    
    # Calculate summary statistics
    pm25_avg = daily_data['pm_2_5'].mean()
    pm25_max = daily_data['pm_2_5'].max()
    pm25_min = daily_data['pm_2_5'].min()
    
    temp_avg = daily_data['temperature'].mean()
    temp_max = daily_data['temperature'].max()
    temp_min = daily_data['temperature'].min()
    
    humidity_avg = daily_data['humidity'].mean()
    humidity_max = daily_data['humidity'].max()
    humidity_min = daily_data['humidity'].min()
    
    # Determine air quality
    quality = get_pm25_quality(pm25_avg)
    quality_color = color_map[quality]
    
    # Create summary cards
    return [
        html.Div(className='param-card', children=[
            html.Div(className='param-title', children="Daily Air Quality"),
            html.Div(className='param-value', children=[
                html.Span(quality, style={'color': quality_color, 'font-weight': 'bold'}),
                html.Div(className='quality-indicator', style={
                    'background-color': quality_color,
                    'display': 'inline-block',
                    'margin-left': '8px',
                    'vertical-align': 'middle'
                })
            ])
        ]),
        html.Div(className='param-card', children=[
            html.Div(className='param-title', children="PM2.5 (μg/m³)"),
            html.Div(className='param-value', children=[
                f"Avg: {pm25_avg:.1f} | Max: {pm25_max:.1f} | Min: {pm25_min:.1f}"
            ])
        ]),
        html.Div(className='param-card', children=[
            html.Div(className='param-title', children="Temperature (°C)"),
            html.Div(className='param-value', children=[
                f"Avg: {temp_avg:.1f} | Max: {temp_max:.1f} | Min: {temp_min:.1f}"
            ])
        ]),
        html.Div(className='param-card', children=[
            html.Div(className='param-title', children="Humidity (%)"),
            html.Div(className='param-value', children=[
                f"Avg: {humidity_avg:.1f} | Max: {humidity_max:.1f} | Min: {humidity_min:.1f}"
            ])
        ])
    ]

# Callback for air quality tips
@app.callback(
    Output('air-quality-tips', 'children'),
    [Input('current-date', 'data')]
)
def update_air_quality_tips(date):
    # Filter data for the selected date
    date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    daily_data = forecast_data[forecast_data['date'] == date_obj]
    
    # Get average PM2.5 for the day
    pm25_avg = daily_data['pm_2_5'].mean()
    quality = get_pm25_quality(pm25_avg)
    
    # Generate tips based on air quality
    tips = []
    
    if quality == "Good":
        tips = [
            "Enjoy outdoor activities - air quality is good today!",
            "Great day for outdoor exercise."
        ]
    elif quality == "Moderate":
        tips = [
            "Consider reducing prolonged outdoor exertion if you're sensitive to air pollution.",
            "Watch for symptoms such as coughing or shortness of breath."
        ]
    elif quality == "Unhealthy for Sensitive Groups":
        tips = [
            "People with respiratory or heart conditions, elderly, and children should limit prolonged outdoor activities.",
            "Consider moving outdoor activities indoors or rescheduling."
        ]
    elif quality == "Unhealthy":
        tips = [
            "Everyone should reduce prolonged or heavy outdoor exertion.",
            "Consider wearing a mask if going outside is necessary.",
            "Keep windows and doors closed to prevent outdoor air from coming in."
        ]
    elif quality == "Very Unhealthy":
        tips = [
            "Everyone should avoid all outdoor physical activity.",
            "Stay indoors and keep windows and doors closed.",
            "Use air purifiers if available."
        ]
    else:  # Hazardous
        tips = [
            "EMERGENCY CONDITIONS: Everyone should avoid all outdoor activities.",
            "Stay indoors with windows and doors closed.",
            "Use HEPA air purifiers and consider relocating temporarily if possible."
        ]
    
    # Add time-based tips
    morning_pm25 = daily_data[(daily_data['hour'] >= 6) & (daily_data['hour'] < 12)]['pm_2_5'].mean()
    afternoon_pm25 = daily_data[(daily_data['hour'] >= 12) & (daily_data['hour'] < 18)]['pm_2_5'].mean()
    evening_pm25 = daily_data[(daily_data['hour'] >= 18) & (daily_data['hour'] < 24)]['pm_2_5'].mean()
    night_pm25 = daily_data[(daily_data['hour'] >= 0) & (daily_data['hour'] < 6)]['pm_2_5'].mean()
    
    time_tips = []
    
    best_time = min(
        ("Morning (6-12)", morning_pm25),
        ("Afternoon (12-18)", afternoon_pm25),
        ("Evening (18-24)", evening_pm25),
        ("Night (0-6)", night_pm25),
        key=lambda x: x[1]
    )
    
    worst_time = max(
        ("Morning (6-12)", morning_pm25),
        ("Afternoon (12-18)", afternoon_pm25),
        ("Evening (18-24)", evening_pm25),
        ("Night (0-6)", night_pm25),
        key=lambda x: x[1]
    )
    
    time_tips.append(f"Best time for outdoor activities: {best_time[0]} (PM2.5: {best_time[1]:.1f} μg/m³)")
    time_tips.append(f"Avoid outdoor activities: {worst_time[0]} (PM2.5: {worst_time[1]:.1f} μg/m³)")
    
    return [
        html.Div(className='tip-content', children=[
            html.Ul([html.Li(tip) for tip in tips], style={'margin-top': '5px', 'padding-left': '20px'})
        ]),
        html.Div(className='hour-band', children=[
            html.Div(className='hour-band-title', children=[
                html.I(className="fas fa-clock", style={'margin-right': '8px'}),
                "Time-based Recommendations"
            ]),
            html.Ul([html.Li(tip) for tip in time_tips], style={'margin-top': '5px', 'padding-left': '20px'})
        ])
    ]

# Callback for 7-day forecast chart
@app.callback(
    Output('7-day-forecast', 'figure'),
    [Input('current-date', 'data')]
)
def update_7_day_forecast(date):
    # Group by date and calculate daily averages
    daily_forecast = forecast_data.groupby('date').agg({
        'pm_2_5': 'mean',
        'temperature': 'mean',
        'humidity': 'mean',
        'quality': lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
    }).reset_index()
    
    # Create a color list based on the quality values
    colors = [color_map[quality] for quality in daily_forecast['quality']]
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.1,
                        subplot_titles=("PM2.5 Daily Average", "Temperature & Humidity"),
                        row_heights=[0.6, 0.4])
    
    # Add PM2.5 line and markers
    fig.add_trace(
        go.Scatter(
            x=daily_forecast['date'],
            y=daily_forecast['pm_2_5'],
            mode='lines+markers',
            name='PM2.5',
            line=dict(width=3, color='#3498db'),
            marker=dict(size=12, color=colors, line=dict(width=2, color='#3498db'))
        ),
        row=1, col=1
    )
    
    # Add quality annotations
    # Add quality annotations
    for i, row in daily_forecast.iterrows():
        fig.add_annotation(
            x=row['date'],
            y=row['pm_2_5'],
            text=f"{row['pm_2_5']:.1f}",
            showarrow=False,
            yshift=15,
            font=dict(size=12, color="#2c3e50")
        )
    
    # Add temperature line
    fig.add_trace(
        go.Scatter(
            x=daily_forecast['date'],
            y=daily_forecast['temperature'],
            mode='lines+markers',
            name='Temperature (°C)',
            line=dict(width=2, color='#e74c3c'),
            marker=dict(size=8, color='#e74c3c')
        ),
        row=2, col=1
    )
    
    # Add humidity line
    fig.add_trace(
        go.Scatter(
            x=daily_forecast['date'],
            y=daily_forecast['humidity'],
            mode='lines+markers',
            name='Humidity (%)',
            line=dict(width=2, color='#2ecc71'),
            marker=dict(size=8, color='#2ecc71')
        ),
        row=2, col=1
    )
    
    # Add reference lines for air quality levels
    reference_levels = [
        (12.0, "Good", "#00E400"),
        (35.4, "Moderate", "#FFFF00"),
        (55.4, "Unhealthy for Sensitive Groups", "#FF7E00"),
        (150.4, "Unhealthy", "#FF0000")
    ]
    
    for level, label, color in reference_levels:
        fig.add_shape(
            type="line",
            x0=daily_forecast['date'].min(),
            y0=level,
            x1=daily_forecast['date'].max(),
            y1=level,
            line=dict(color=color, width=1, dash="dash"),
            row=1, col=1
        )
        fig.add_annotation(
            x=daily_forecast['date'].max(),
            y=level,
            xref="x",
            yref="y",
            text=label,
            showarrow=False,
            xanchor="right",
            font=dict(size=10, color=color),
            bgcolor="rgba(255, 255, 255, 0.7)",
            borderpad=2,
            row=1, col=1
        )
    
    # Customize layout
    fig.update_layout(
        title=None,
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        hovermode="x unified"
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    
    # Highlight the selected date
    if date:
        date_obj = datetime.strptime(date, '%Y-%m-%d').date()
        if date_obj in daily_forecast['date'].values:
            # Add vertical line to highlight selected date
            fig.add_shape(
                type="line",
                x0=date_obj,
                y0=0,
                x1=date_obj,
                y1=daily_forecast['pm_2_5'].max() * 1.1,
                line=dict(color="#2c3e50", width=2),
                row=1, col=1
            )
            fig.add_shape(
                type="line",
                x0=date_obj,
                y0=0,
                x1=date_obj,
                y1=max(daily_forecast['temperature'].max(), daily_forecast['humidity'].max()) * 1.1,
                line=dict(color="#2c3e50", width=2),
                row=2, col=1
            )
    
    return fig

# Callback for hourly forecast chart
@app.callback(
    Output('hourly-forecast', 'figure'),
    [Input('current-date', 'data'),
     Input('active-tab', 'data')]
)
def update_hourly_forecast(date, active_tab):
    if date is None:
        date = forecast_data['date_str'].min()
    
    # Convert string date to datetime.date
    if isinstance(date, str):
        selected_date = datetime.strptime(date, '%Y-%m-%d').date()
    else:
        selected_date = date
    
    # Filter data for selected date
    hourly_data = forecast_data[forecast_data['date'] == selected_date].copy()
    
    # Create figure based on active tab
    if active_tab == 'pm25':
        # Create PM2.5 figure
        fig = go.Figure()
        
        # Add bar chart for PM2.5 levels
        fig.add_trace(go.Bar(
            x=hourly_data['hour'],
            y=hourly_data['pm_2_5'],
            name='PM2.5',
            marker_color=[color_map[q] for q in hourly_data['quality']],
            hovertemplate='Hour: %{x}:00<br>PM2.5: %{y:.1f} μg/m³<br>Quality: %{text}<extra></extra>',
            text=hourly_data['quality']
        ))
        
        # Add line for better visualization
        fig.add_trace(go.Scatter(
            x=hourly_data['hour'],
            y=hourly_data['pm_2_5'],
            mode='lines',
            line=dict(color='#2c3e50', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add horizontal reference lines for air quality levels
        reference_levels = [
            (12.0, "Good", "#00E400"),
            (35.4, "Moderate", "#FFFF00"),
            (55.4, "Unhealthy for Sensitive Groups", "#FF7E00"),
            (150.4, "Unhealthy", "#FF0000"),
            (250.4, "Very Unhealthy", "#8F3F97")
        ]
        
        for level, label, color in reference_levels:
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=level,
                x1=23.5,
                y1=level,
                line=dict(color=color, width=1, dash="dash"),
            )
            fig.add_annotation(
                x=23.5,
                y=level,
                xref="x",
                yref="y",
                text=label,
                showarrow=False,
                xanchor="right",
                font=dict(size=10, color=color),
                bgcolor="rgba(255, 255, 255, 0.7)",
                borderpad=2
            )
        
        # Customize layout
        day_name = hourly_data['day'].iloc[0] if not hourly_data.empty else ""
        fig.update_layout(
            title=f"Hourly PM2.5 Levels - {day_name}, {selected_date}",
            xaxis=dict(
                title="Hour of Day",
                tickmode='array',
                tickvals=list(range(0, 24, 2)),  # Show every 2 hours
                ticktext=[f"{h}:00" for h in range(0, 24, 2)]  # Format as HH:00
            ),
            yaxis_title="PM2.5 (μg/m³)",
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=12),
            barmode='relative',
            margin=dict(l=40, r=40, t=80, b=40),
            height=450
        )
    
    elif active_tab == 'temp':
        # Create temperature figure
        fig = go.Figure()
        
        # Add area chart for temperature
        fig.add_trace(go.Scatter(
            x=hourly_data['hour'],
            y=hourly_data['temperature'],
            fill='tozeroy',
            name='Temperature',
            line=dict(color='#e74c3c', width=3),
            hovertemplate='Hour: %{x}:00<br>Temperature: %{y:.1f}°C<extra></extra>'
        ))
        
        # Customize layout
        day_name = hourly_data['day'].iloc[0] if not hourly_data.empty else ""
        fig.update_layout(
            title=f"Hourly Temperature - {day_name}, {selected_date}",
            xaxis=dict(
                title="Hour of Day",
                tickmode='array',
                tickvals=list(range(0, 24, 2)),
                ticktext=[f"{h}:00" for h in range(0, 24, 2)]
            ),
            yaxis_title="Temperature (°C)",
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=12),
            margin=dict(l=40, r=40, t=80, b=40),
            height=450
        )
        
    elif active_tab == 'humidity':
        # Create humidity figure
        fig = go.Figure()
        
        # Add area chart for humidity
        fig.add_trace(go.Scatter(
            x=hourly_data['hour'],
            y=hourly_data['humidity'],
            fill='tozeroy',
            name='Humidity',
            line=dict(color='#2ecc71', width=3),
            hovertemplate='Hour: %{x}:00<br>Humidity: %{y:.1f}%<extra></extra>'
        ))
        
        # Customize layout
        day_name = hourly_data['day'].iloc[0] if not hourly_data.empty else ""
        fig.update_layout(
            title=f"Hourly Humidity - {day_name}, {selected_date}",
            xaxis=dict(
                title="Hour of Day",
                tickmode='array',
                tickvals=list(range(0, 24, 2)),
                ticktext=[f"{h}:00" for h in range(0, 24, 2)]
            ),
            yaxis_title="Humidity (%)",
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=12),
            margin=dict(l=40, r=40, t=80, b=40),
            height=450
        )
        
    else:  # combined view
        # Create subplot figure
        fig = make_subplots(rows=3, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.08,
                            subplot_titles=("PM2.5 Levels", "Temperature", "Humidity"),
                            row_heights=[0.4, 0.3, 0.3])
        
        # Add PM2.5 bars
        fig.add_trace(
            go.Bar(
                x=hourly_data['hour'],
                y=hourly_data['pm_2_5'],
                name='PM2.5',
                marker_color=[color_map[q] for q in hourly_data['quality']],
                hovertemplate='Hour: %{x}:00<br>PM2.5: %{y:.1f} μg/m³<br>Quality: %{text}<extra></extra>',
                text=hourly_data['quality']
            ),
            row=1, col=1
        )
        
        # Add PM2.5 line
        fig.add_trace(
            go.Scatter(
                x=hourly_data['hour'],
                y=hourly_data['pm_2_5'],
                mode='lines',
                line=dict(color='#2c3e50', width=2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Add temperature line
        fig.add_trace(
            go.Scatter(
                x=hourly_data['hour'],
                y=hourly_data['temperature'],
                mode='lines',
                name='Temperature',
                line=dict(color='#e74c3c', width=3),
                hovertemplate='Hour: %{x}:00<br>Temperature: %{y:.1f}°C<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add humidity line
        fig.add_trace(
            go.Scatter(
                x=hourly_data['hour'],
                y=hourly_data['humidity'],
                mode='lines',
                name='Humidity',
                line=dict(color='#2ecc71', width=3),
                hovertemplate='Hour: %{x}:00<br>Humidity: %{y:.1f}%<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add reference lines for PM2.5
        reference_levels = [
            (12.0, "Good", "#00E400"),
            (35.4, "Moderate", "#FFFF00"),
            (55.4, "Unhealthy for Sensitive Groups", "#FF7E00")
        ]
        
        for level, label, color in reference_levels:
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=level,
                x1=23.5,
                y1=level,
                line=dict(color=color, width=1, dash="dash"),
                row=1, col=1
            )
        
        # Customize layout
        day_name = hourly_data['day'].iloc[0] if not hourly_data.empty else ""
        fig.update_layout(
            title=f"Combined Environmental Parameters - {day_name}, {selected_date}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=12),
            height=600,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        fig.update_xaxes(title_text="Hour of Day", row=3, col=1,
                        tickmode='array',
                        tickvals=list(range(0, 24, 2)),
                        ticktext=[f"{h}:00" for h in range(0, 24, 2)])
        
        fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_yaxes(title_text="Humidity (%)", row=3, col=1)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
