# 🔎 Phishing URL Detector (BiLSTM + Heuristic + LLM)

โปรเจ็กต์นี้เป็นระบบสำหรับตรวจจับ **Phishing URL** โดยใช้การผสมผสานของ 3 วิธีหลัก:
1. **BiLSTM Model** → วิเคราะห์ URL ด้วย Deep Learning (sequence + structured features)
2. **Heuristic Rules** → วิเคราะห์โครงสร้างของ URL/HTML ด้วยกฎที่ออกแบบขึ้น
3. **LLM Reasoning** → ใช้โมเดล GPT วิเคราะห์และอธิบายผลลัพธ์


# 📂 Project Structure

├── utils/  
│ ├── model.keras # BiLSTM model ที่ train ไว้  
│ ├── scaler.pkl # StandardScaler สำหรับ features  
│ ├── tokenizer.pkl # Tokenizer สำหรับ URL  
│ └── labelencoder.pkl # LabelEncoder สำหรับ mapping labels  
├── phishing_detector.ipynb # Notebook หลัก (Colab)  
└── README.md # ไฟล์แนะนำโปรเจ็กต์


## ⚙️ Installation

1. Clone repository และเปิดใน Google Colab:
   ```bash
   git clone https://github.com/polakrit-pipi/bilstm-phishing-detector.git
   pip install -r requirements.txt

## 🚀 Usage

### 1. Mount Google Drive (ถ้าไฟล์โมเดลเก็บใน Drive)

`from google.colab import drive
drive.mount('/content/drive')`

### 2. โหลดโมเดล + Utils

`from tensorflow.keras.models import load_model import pickle with  open("utils/scaler.pkl", "rb") as f:
    scaler = pickle.load(f) with  open("utils/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f) with  open("utils/labelencoder.pkl", "rb") as f:
    le = pickle.load(f)

model = load_model("utils/model.keras", custom_objects={"Attention": Attention})` 

**โหลดได้จาก**
`https://drive.google.com/drive/folders/1e97KCXadwmg2ugQqUXHcT8rAYBkZIpak?usp=sharing` 


### 3. ทดสอบ URL

`url_test = "http://pantip.com/topic/40410662.com" label, pred = predict_url(url_test) print("Label:", label) print("Probabilities:", pred)` 

Output:

`Label: 0  Probabilities: [0.9565845  0.04341554]` 

### 4. วิเคราะห์ URL แบบผสม (BiLSTM + Heuristic + LLM)

`result = run_analysis_on_url("https://mycourses.ict.mahidol.ac.th/login/index.php")` 

ตัวอย่างผลลัพธ์:

`------ Phishing Analysis (LLM) ------ สรุป: เว็บไซต์นี้มี subdomain เยอะและ meta keywords ไม่ตรงกับ host แต่โดยรวมแล้ว URL มีลักษณะเหมือนเว็บไซต์จริง คำตอบ: Likely Safe`

## 📊 Features Extracted

ระบบจะดึง features หลายแบบ เช่น:

-   **URL-based**: length, entropy, subdomain count, digits, symbols
    
-   **Domain-based**: IP host, brand keyword, prefix-suffix patterns
    
-   **HTML-based**: abnormal links, suspicious forms, meta mismatch

## OpenAI API Key
เพื่อให้ LLM วิเคราะห์สรุปผลลัพธ์ ต้องใส่ API key:

`from getpass import getpass
api_key = getpass("🔑 Enter your OpenAI API key: ")`


# 📝 Example Workflow


-   ใส่ URL → Extract Features
    
-   Predict ด้วย BiLSTM
    
-   คำนวณ Heuristic Score
    
-   ส่งข้อมูลเข้า LLM → อธิบาย + สรุป

## 📌 To-Do / Improvements

-   แก้ไขตัว Prompt ให้ผลลัพธ์ LLM ดีขึ้น
-   ปรับปรุง Heuristic Rules ให้แม่นยำขึ้น
    
-   เพิ่ม dataset สำหรับ train BiLSTM
    
-   ทำ Web UI สำหรับ demo
    
-   เพิ่ม Dockerfile สำหรับ deployment

    
