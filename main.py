# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import requests
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import re
import math
from collections import Counter
from urllib.parse import urlparse
import joblib
from openai import OpenAI
import json
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from dotenv import load_dotenv
import os



# -------------------------
# 1️⃣ FastAPI setup
# -------------------------
app = FastAPI(title="Phishing URL Analyzer")
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str
    call_llm: bool = True  # whether to call LLM or not

# -------------------------
# 2️⃣ Load Models / Tokenizer / Scaler / LabelEncoder
# -------------------------
scaler = joblib.load('utils/scaler-2.joblib')
tokenizer = joblib.load('utils/tokenizer-2.joblib')
le = joblib.load('utils/labelencoder-2.joblib')
maxlen = 200

# -------------------------
# 3️⃣ Custom Attention Layer
# -------------------------
@register_keras_serializable()
class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)

    def call(self, x):
        u_it = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a_it = tf.nn.softmax(tf.tensordot(u_it, self.u, axes=1), axis=1)
        return tf.reduce_sum(x * a_it, axis=1)

# -------------------------
# 4️⃣ Load BILSTM model
# -------------------------
model = load_model("utils/model.keras", custom_objects={"Attention": Attention})

# -------------------------
# 5️⃣ Utility functions
# -------------------------
BRAND_KEYWORDS = ["paypal","apple","amazon","bank","chase","facebook","meta","google","microsoft",
                  "outlook","office365","instagram","line","kbank","scb","krungsri","kplus"]
COMMON_TLDS = set([
 "com","net","org","info","biz","co","io","ai","app","edu","gov","mil","ru","de","uk","cn","fr","jp","br","in","it","es","au","nl","se","no"
])

def parse_host_and_scheme(url: str):
    p = urlparse(url if '://' in url else 'http://' + url)
    return (p.hostname or "").lower(), (p.scheme or "").lower()

def is_ip_host(host: str):
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))

def count_subdomains(host: str):
    if not host: return 0
    return max(0, len(host.split(".")) - 2)

def has_double_slash_in_path(url: str):
    return "//" in (urlparse(url if '://' in url else 'http://' + url).path or "")

def has_tld_in_path(url: str):
    path = (urlparse(url if '://' in url else 'http://' + url).path or "").lower()
    return any(("."+tld) in path for tld in COMMON_TLDS)

def has_symbols_in_domain(host: str):
    return bool(re.search(r"[^a-z0-9\.-]", host))

def domain_prefix_suffix_like_brand(host: str):
    if not host: return False
    first = host.split(".")[0]
    return any(b in first and "-" in first for b in BRAND_KEYWORDS)

def brand_in_path_or_subdomain(host: str, url: str):
    text = (host + " " + urlparse(url).path + " " + urlparse(url).query).lower()
    return any(b in text for b in BRAND_KEYWORDS)

def digit_count(url: str):
    return sum(c.isdigit() for c in url)

def url_length(url: str):
    return len(url)

def url_entropy(url: str):
    if not url: return 0.0
    counts = Counter(url)
    total = len(url)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

def fetch_html(url):
    try:
        r = requests.get(url, timeout=5)
        return r.text
    except:
        return ""

def extract_html_features(html):
    hrefs = re.findall(r'href=["\'](.*?)["\']', html or '', flags=re.IGNORECASE)
    forms = re.findall(r'<form[^>]+action=["\'](.*?)["\']', html or '', flags=re.IGNORECASE)
    imgs = re.findall(r'<img[^>]+src=["\'](.*?)["\']', html or '', flags=re.IGNORECASE)
    scripts = re.findall(r'<script[^>]+src=["\'](.*?)["\']', html or '', flags=re.IGNORECASE)
    links_tag = re.findall(r'<link[^>]+href=["\'](.*?)["\']', html or '', flags=re.IGNORECASE)
    meta_keywords = re.findall(r'<meta[^>]+name=["\']keywords["\'][^>]+content=["\'](.*?)["\']', html or '', flags=re.IGNORECASE)
    return {'hrefs': hrefs, 'forms': forms, 'imgs': imgs, 'scripts': scripts, 'links_tag': links_tag, 'meta_keywords': meta_keywords}

def abnormal_links(hrefs):
    return any(h.strip().lower().startswith(('javascript:','mailto:','data:')) for h in hrefs)

def forms_action_abnormal(forms, host):
    for a in forms:
        if a and host not in a and not a.startswith('/') and not a.startswith('#'):
            return True
    return False

def anchors_point_elsewhere(hrefs, host):
    count = sum(1 for h in hrefs if host and host not in h and h.startswith('http'))
    total = max(1,len(hrefs))
    return (count / total) > 0.5

def meta_keyword_mismatch(meta_keywords, host):
    if not meta_keywords: return False
    for kw in meta_keywords:
        if host.split('.')[0] not in kw:
            return True
    return False

# -------------------------
# 6️⃣ Rule-based phishing score
# -------------------------
def phishing_score(url, html):
    host, scheme = parse_host_and_scheme(url)
    features = extract_html_features(html)
    score = 0
    reasons = []

    if is_ip_host(host):
        score += 2; reasons.append("Host เป็น IP address")
    if count_subdomains(host) > 2:
        score += 1; reasons.append("มี subdomain เยอะ")
    if has_symbols_in_domain(host):
        score += 1; reasons.append("มี symbol แปลกใน domain")
    if domain_prefix_suffix_like_brand(host):
        score += 2; reasons.append("ชื่อ domain คล้าย brand แต่มี -")
    if brand_in_path_or_subdomain(host,url):
        score +=1; reasons.append("มี brand keyword ใน path หรือ subdomain")
    if has_double_slash_in_path(url):
        score +=1; reasons.append("path มี double slash")
    if has_tld_in_path(url):
        score +=1; reasons.append("TLD ปรากฏใน path")
    if abnormal_links(features['hrefs']):
        score +=1; reasons.append("มีลิงก์ผิดปกติ")
    if forms_action_abnormal(features['forms'], host):
        score +=2; reasons.append("form action ผิดปกติ")
    if anchors_point_elsewhere(features['hrefs'], host):
        score +=1; reasons.append("anchor ชี้ไปเว็บอื่นเยอะ")
    if meta_keyword_mismatch(features['meta_keywords'], host):
        score +=1; reasons.append("meta keywords ไม่ตรงกับ host")

    # extra features
    dcount = digit_count(url)
    ulen = url_length(url)
    uentropy = url_entropy(url)
    if dcount>5: score+=1; reasons.append(f"มีตัวเลขเยอะ (digits={dcount})")
    if ulen>75: score+=1; reasons.append(f"URL ยาวผิดปกติ (length={ulen})")
    if uentropy>4.0: score+=1; reasons.append(f"Entropy ของ URL สูง (entropy={uentropy:.2f})")

    features.update({"digit_count": dcount,"url_length": ulen,"url_entropy": uentropy})
    return score, reasons, features, host, scheme

# -------------------------
# 7️⃣ BiLSTM prediction
# -------------------------
def predict_url(url):
    struct_feat = scaler.transform([list({
        "is_ip_host": int(is_ip_host(urlparse(url).hostname or '')),
        "subdomain_count": count_subdomains(urlparse(url).hostname or ''),
        "double_slash_in_path": int(has_double_slash_in_path(url)),
        "tld_in_path": int(has_tld_in_path(url)),
        "symbols_in_domain": int(has_symbols_in_domain(urlparse(url).hostname or '')),
        "prefix_suffix_like_brand": int(domain_prefix_suffix_like_brand(urlparse(url).hostname or '')),
        "brand_in_path_or_subdomain": int(brand_in_path_or_subdomain(urlparse(url).hostname or '', url)),
        "url_length": len(url),
        "scheme_https": 1 if urlparse(url).scheme=='https' else 0,
        "digit_count_domain": digit_count(url),
        "url_entropy": url_entropy(url)
    }.values())])
    seq = pad_sequences(tokenizer.texts_to_sequences([url]), maxlen=maxlen)
    pred = model.predict([seq, struct_feat])[0]
    label = le.inverse_transform([np.argmax(pred)])[0]
    return label, pred

# -------------------------
# 8️⃣ API endpoint
# -------------------------

@app.post("/analyze")
def analyze(request: URLRequest):
    url = request.url
    html = fetch_html(url)
    score, reasons, features, host, scheme = phishing_score(url, html)

    # BiLSTM prediction
    bilstm_label, bilstm_prob_array = predict_url(url)
    label_idx = np.argmax(bilstm_prob_array)
    bilstm_label = le.inverse_transform([label_idx])[0]
    bilstm_prob = float(bilstm_prob_array[label_idx])  # ใช้ probability ของ class ที่เลือกจริง

    llm_result = None
    if request.call_llm:
        prompt = f"""
URL: {url}
Host: {host}
Scheme: {scheme}

BiLSTM Prediction: {bilstm_label} (probability={bilstm_prob:.2f})
Rule-based Heuristic Score: {score}

Reasons triggered:
- {"\n- ".join(reasons)}

Extracted Features:
- Digit count: {features.get('digit_count')}
- URL length: {features.get('url_length')}
- URL entropy: {features.get('url_entropy'):.2f}
- Hrefs: {features['hrefs']}
- Images: {features['imgs']}
- Scripts: {features['scripts']}
- Links tag: {features['links_tag']}
- Forms: {features['forms']}
- Meta keywords: {features['meta_keywords']}

โปรดให้เหตุผลสั้น ๆ (2–3 ประโยค) และสรุปว่า 'Likely Phishing' หรือ 'Likely Safe' 
โดยตอบกลับเป็น JSON แบบนี้:
{{
    "verdict": "...",
    "reason_list": ["...","..."],
    "summary": "..."
}}
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content": prompt}],
            temperature=0
        )
        try:
            # แก้ไขตรงนี้
            raw_text = response.choices[0].message.content.strip()
            if raw_text.startswith("```"):
                raw_text = "\n".join(raw_text.splitlines()[1:-1])  # ตัด ```json และ ```
            llm_result = json.loads(raw_text)
        except Exception as e:
            llm_result = {"verdict":"Unknown","reason_list":[],"summary":response.choices[0].message.content}

    return {
        "url": url,
        "score": score,
        "reasons": reasons,
        "features": features,
        "bilstm_label": bilstm_label,
        "bilstm_prob": bilstm_prob,
        "llm_result": llm_result,
        "host": host,
        "scheme": scheme
    }


# -------------------------
# 9️⃣ Frontend
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phishing URL Analyzer</title>
<style>
body { font-family: 'Segoe UI', Tahoma, Verdana, sans-serif; background: #f0f4f8; margin:0;padding:0;}
.container { max-width:700px; margin:50px auto; background:#fff; padding:25px; border-radius:12px; box-shadow:0 10px 20px rgba(0,0,0,0.1);}
h1{text-align:center;color:#007BFF;}
input[type="text"]{width:75%;padding:12px;margin-right:10px;border-radius:6px;border:1px solid #ccc;font-size:16px;}
button{padding:12px 18px;border:none;border-radius:6px;background:#007BFF;color:white;font-weight:bold;cursor:pointer;transition:0.2s;}
button:hover{background:#0056b3;}
.output-card{margin-top:20px;background:#f7f9fc;padding:15px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.1);}
.section-title{font-weight:bold;margin-top:10px;margin-bottom:5px;color:#333;}
.feature-list,.reason-list{padding-left:20px;}
.bilstm,.rule,.llm{border-left:4px solid #007BFF;padding-left:10px;margin-bottom:15px;}
</style>
</head>
<body>
<div class="container">
<h1>🔍 Phishing URL Analyzer</h1>
<div style="text-align:center;">
<input id="urlInput" type="text" placeholder="ใส่ URL ที่นี่">
<button onclick="analyze()">วิเคราะห์</button>
</div>
<div id="resultContainer"></div>
</div>

<script>
async function analyze() {
    const url = document.getElementById("urlInput").value;
    if(!url) return alert("กรุณาใส่ URL");
    try {
        const res = await fetch("/analyze",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({url:url, call_llm:true})
        });
        const data = await res.json();

        const llm_html = data.llm_result ? 
            `<ul>${data.llm_result.reason_list.map(r=>`<li>${r}</li>`).join('')}</ul>
            <b>Verdict:</b> ${data.llm_result.verdict}<br>
            <b>Summary:</b> ${data.llm_result.summary}`
            : "ไม่มีผลลัพธ์ LLM";

        const html = `
        <div class="output-card">
            <div class="section-title">✅ BiLSTM Prediction</div>
            <div class="bilstm">
                Label: ${data.bilstm_label} <br>
                Probability: ${(data.bilstm_prob*100).toFixed(2)}%
            </div>

            <div class="section-title">⚖️ Rule-based Analysis</div>
            <div class="rule">
                Score: ${data.score} <br>
                Reasons:
                <ul class="reason-list">
                    ${data.reasons.map(r=>`<li>${r}</li>`).join('')}
                </ul>
            </div>

            <div class="section-title">🧾 Extracted Features</div>
            <div class="rule">
                Digit count: ${data.features.digit_count} <br>
                URL length: ${data.features.url_length} <br>
                URL entropy: ${data.features.url_entropy.toFixed(2)} <br>
                Hrefs: ${data.features.hrefs.length} <br>
                Images: ${data.features.imgs.length} <br>
                Scripts: ${data.features.scripts.length} <br>
                Links tag: ${data.features.links_tag.length} <br>
                Forms: ${data.features.forms.length} <br>
                Meta keywords: ${data.features.meta_keywords.length}
            </div>

            <div class="section-title">🤖 LLM Summary</div>
            <div class="llm">
                ${llm_html}
            </div>
        </div>
        `;
        document.getElementById("resultContainer").innerHTML = html;
    } catch(err){
        document.getElementById("resultContainer").innerHTML = `<div class="output-card">❌ Error: ${err}</div>`;
    }
}
</script>
</body>
</html>
"""

# -------------------------
# 10️⃣ Run server
# -------------------------
if __name__=="__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
