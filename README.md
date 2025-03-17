# วิธีการรันโปรเจกต์ Data Forecast

## ขั้นตอนการรันโปรเจกต์

### 1. สร้าง Virtual Environment และติดตั้ง Dependencies
ก่อนเริ่มต้น ให้สร้าง virtual environment และติดตั้ง dependencies จาก `requirements.txt`
```bash
python -m venv .venv
source .venv/bin/activate  # สำหรับ macOS/Linux
.venv\Scripts\activate    # สำหรับ Windows
pip install -r requirements.txt
```

---

## 2. การทำความสะอาดข้อมูล (Data Cleaning)

### 2.1 Clean Data ในส่วนของการทำนายฝุ่น pm2.5  ในไฟล์ clean_data.ipynb
โดยใน repository นี้ได้ทำการ push ไฟล์ที่คลีนไว้แล้ว โดยจะมี 2 ไฟล์ได้แก่ last_test.csv (ลบไป 7 วันเพื่อจะนำข้อมูลใน cleandata_hours มาเปรียบเทียบในส่วนของการทำนาย) และ cleandata_hours.csv
โดยอาจจะต้องการแก้ไขไฟล์ path ในโค๊ดเองในส่วนของการอ่านไฟล์ โดยจะอ่านไฟล์จากในส่วนของ raw data ที่ชื่อว่า  export-jsps001-1h และกำหนดการส่งออกข้อมูลเป็น path ไฟล์ของตัวเอง หรือ ที่มีอยู่แล้วใน repository นี้


### 2.2 Clean Data Rainfall  ในส่วนของการทำนายปริมาณน้ำฝน  ในไฟล์ cleandata_rainfall.ipynb
โดยใน repository นี้ได้ทำการ push ไฟล์ที่คลีนไว้แล้ว โดยจะมี 2 ไฟล์ได้แก่ raindata.csv
โดยอาจจะต้องการแก้ไขไฟล์ path ในโค๊ดเองในส่วนของการอ่านไฟล์โดยจะอ่านไฟล์จากในส่วนของ raw data ที่ชื่อว่า POWER_Point_Hourly_20210316_20250316_009d14N_099d31E_LST
raindata.csv และกำหนดการส่งออกข้อมูลเป็น path ไฟล์ของตัวเอง หรือ ที่มีอยู่แล้วใน repository นี้

## 3. การเทรนโมเดล
### 3.1 เทรนโมเดลสำหรับ Clean Data ในส่วนของการทำนายฝุ่น pm2.5 ในไฟล์ finalmodel.ipynb
โดยอาจจะต้องแก้ไข file path ในการนำเข้าอ่านไฟล์
### 3.2 เทรนโมเดลสำหรับ Raindata ในไฟล์ rainnymodel.ipynb
โดยอาจจะต้องแก้ไข file path ในการนำเข้าอ่านไฟล์

## 4. การแสดงผลบน Dash Dashboard Dashapp.py
```bash
python scripts/dashapp.py
```
เพื่อทำการรัน Dash

ภาพตัวอย่างโครงสร้าง folder

![image](https://github.com/user-attachments/assets/dd150517-9c00-42c3-80bd-68f86c1b707b)

