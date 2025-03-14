import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from pycaret.regression import load_model, predict_model
from datetime import datetime, timedelta
import numpy as np

# โหลดข้อมูล
file_path = r"C:\Users\ASUS\Desktop\projectforecastpm2_5\dataforecast\cleandata_vtest_hours.csv"
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# โหลดโมเดลที่ฝึกไว้
model = load_model(r'C:\Users\ASUS\Desktop\projectforecastpm2_5\models\best_model')

# ฟังก์ชันทำนาย PM2.5 ล่วงหน้า 7 วัน
def forecast_next_7_days(model, last_data):
    last_date = last_data['timestamp'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(7)]
    
    future_data = []
    current_data = last_data.copy()
    
    for future_date in future_dates:
        new_row = {'timestamp': future_date}
        
        temp_mean = current_data['temperature'].tail(7).mean()
        humidity_mean = current_data['humidity'].tail(7).mean()
        
        new_row['temperature'] = np.random.normal(temp_mean, 2)
        new_row['humidity'] = np.clip(np.random.normal(humidity_mean, 5), 0, 100)
        
        for lag in range(1, 8):
            if len(current_data) >= lag:
                new_row[f'pm_2_5_Lag{lag}'] = current_data['pm_2_5'].iloc[-lag]
            else:
                new_row[f'pm_2_5_Lag{lag}'] = current_data['pm_2_5'].mean()
        
        new_df = pd.DataFrame([new_row])
        prediction = predict_model(model, data=new_df)
        
        new_row['pm_2_5'] = prediction['prediction_label'].iloc[0]
        future_data.append(new_row)
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
    
    return pd.DataFrame(future_data)

# สร้างแอป Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("PM2.5 Dashboard"),
    dcc.DatePickerRange(
        id='date-picker',
        start_date=df["timestamp"].min(),
        end_date=df["timestamp"].max(),
        display_format='YYYY-MM-DD'
    ),
    dcc.Graph(id='pm25-line-chart'),
    dcc.Graph(id='pm25-heatmap'),
    html.Hr(),
    html.H3("Predict PM2.5"),
    html.Button("Predict PM2.5", id="predict-btn", n_clicks=0),
    html.Div(id='prediction-output', style={'fontSize': 20, 'marginTop': 10})
])

# Callback สำหรับอัปเดตกาฟ
@app.callback(
    [Output('pm25-line-chart', 'figure'),
     Output('pm25-heatmap', 'figure')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_graph(start_date, end_date):
    filtered_df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
    line_fig = px.line(filtered_df, x='timestamp', y='pm_2_5', title='PM2.5 Trend')
    heatmap_fig = px.density_heatmap(filtered_df, x='timestamp', y='pm_2_5', title='PM2.5 Heatmap')
    return line_fig, heatmap_fig

latest_data = df.copy()

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks')
)
def update_prediction(n_clicks):
    if n_clicks > 0:
        future_df = forecast_next_7_days(model, latest_data)
        return f"Predicted PM2.5 for next 7 days:\n{future_df[['timestamp', 'pm_2_5']].to_string(index=False)}"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)