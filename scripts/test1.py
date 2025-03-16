file_path = r"C:\Users\ASUS\Desktop\projectforecastpm2_5\dataforecast\cleandata_vtest_hours.csv"
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# โหลดโมเดลที่ฝึกไว้
model = load_model(r'C:\Users\ASUS\Desktop\projectforecastpm2_5\models\best_model')

def forecast_next_7_days(model, last_data):
    last_date = last_data['timestamp'].max()
    future_hours = [last_date + timedelta(hours=i+1) for i in range(7 * 24)]  # 7 วัน * 24 ชั่วโมง
    
    future_data = []
    current_data = last_data.copy()
    
    for future_hour in future_hours:
        new_row = {'timestamp': future_hour}
        
        # คำนวณค่าเฉลี่ยของ temperature และ humidity จากข้อมูลล่าสุด
        temp_mean = current_data['temperature'].tail(24).mean()  # ใช้ข้อมูล 24 ชั่วโมงล่าสุด
        humidity_mean = current_data['humidity'].tail(24).mean()
        
        # สร้างค่า temperature และ humidity โดยใช้การกระจายแบบปกติ
        new_row['temperature'] = np.random.normal(temp_mean, 2)
        new_row['humidity'] = np.clip(np.random.normal(humidity_mean, 5), 0, 100)
        
        # สร้าง Lag Features สำหรับ PM2.5
        for lag in range(1, 8):
            if len(current_data) >= lag:
                new_row[f'pm_2_5_Lag{lag}'] = current_data['pm_2_5'].iloc[-lag]
            else:
                new_row[f'pm_2_5_Lag{lag}'] = current_data['pm_2_5'].mean()
        
        # ทำนาย PM2.5 สำหรับชั่วโมงนี้
        new_df = pd.DataFrame([new_row])
        prediction = predict_model(model, data=new_df)
        
        new_row['pm_2_5'] = prediction['prediction_label'].iloc[0]
        future_data.append(new_row)
        
        # เพิ่มข้อมูลที่ทำนายเข้าไปใน current_data เพื่อใช้ในการทำนายชั่วโมงถัดไป
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
    
    return pd.DataFrame(future_data)
