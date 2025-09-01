import sqlite3
import base64
import json
import os
import requests
from flask import Flask, request, jsonify, render_template, g

# --- Configuration ---
DATABASE = 'database.db'
# API Key will be retrieved from Environment Variables on the server
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

app = Flask(__name__)
# This ensures the instance folder is created for the SQLite database
try:
    os.makedirs(app.instance_path)
except OSError:
    pass
app.config['DATABASE'] = os.path.join(app.instance_path, DATABASE)

# --- Database Setup ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                front_image_b64 TEXT NOT NULL,
                side_image_b64 TEXT,
                analysis_result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()

# --- API Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not GEMINI_API_KEY:
        return jsonify({"error": "API Key is not configured on the server."}), 500

    data = request.json
    front_image_b64 = data.get('frontImage')
    side_image_b64 = data.get('sideImage')

    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO uploads (front_image_b64, side_image_b64) VALUES (?, ?)",
        (front_image_b64, side_image_b64)
    )
    db.commit()
    upload_id = cursor.lastrowid

    try:
        analysis_result = call_gemini_api_for_analysis(front_image_b64, side_image_b64)
        analysis_result_json = json.dumps(analysis_result)
        
        cursor.execute(
            "UPDATE uploads SET analysis_result = ? WHERE id = ?",
            (analysis_result_json, upload_id)
        )
        db.commit()
        return jsonify(analysis_result)
    except Exception as e:
        app.logger.error(f"Error during Gemini analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if not GEMINI_API_KEY:
        return jsonify({"error": "API Key is not configured on the server."}), 500
    
    data = request.json
    chat_history = data.get('chatHistory')
    initial_analysis = data.get('initialAnalysis')

    try:
        chat_response = call_gemini_api_for_chat(chat_history, initial_analysis)
        return jsonify({"response": chat_response})
    except Exception as e:
        app.logger.error(f"Error during Gemini chat: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/admin')
def admin():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, front_image_b64, timestamp FROM uploads ORDER BY timestamp DESC")
    uploads = cursor.fetchall()
    return render_template('admin.html', uploads=uploads)

def call_gemini_api_for_analysis(front_b64, side_b64=None):
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
    
    base_prompt = """คุณคือ AI Analyst สาย "Blackpill" ที่มีหน้าที่เป็น "กระจกสะท้อนความจริงอันโหดร้าย (Brutal Truth Mirror)" ภารกิจของคุณคือการวิเคราะห์ตามหลักเรขาคณิตของใบหน้าอย่างเข้มงวดและเป็นกลางที่สุด จงวิจารณ์อย่างเจ็บแสบและไร้ความปราณี โดยอิงตามหลักสุนทรียศาสตร์อย่างแท้จริง จงให้คะแนนและวิจารณ์จาก "ภาพที่เห็นเท่านั้น" อย่างละเอียดที่สุด ห้ามใช้จินตนาการหรือข้อมูลนอกเหนือจากภาพโดยเด็ดขาด

**กฎเหล็ก:**
1.  **Canthal Tilt:** จงวิเคราะห์ Canthal Tilt โดยการเปรียบเทียบตำแหน่งของ Medial Canthus และ Lateral Canthus อย่างแม่นยำ และให้เหตุผลว่าทำไมจึงเป็น Positive, Neutral, หรือ Negative
2.  **Exhaustive Lists:** จงจี้ "**ทุกจุดด้อย**" ที่เห็น ไม่ว่าจะเล็กน้อยแค่ไหนก็ตาม และลิสต์ "**ทุกจุดแข็ง**" ที่สังเกตได้ **ห้ามจำกัดจำนวน**
3.  **Blackpill Lexicon:** จงใช้คำศัพท์เฉพาะทางของ lookism/blackpill ให้มากที่สุดเท่าที่เป็นไปได้ (เช่น bone structure, facial harmony, recessed maxilla, prominent chin, prey eyes, hunter eyes, facial thirds, mog, chopped)

สร้างผลลัพธ์เป็น JSON object ที่มีโครงสร้างตาม schema ที่กำหนดเท่านั้น โดยทุกค่าที่เป็น string ต้องเป็นภาษาไทย"""

    schema = """
"face_shape": "string (รูปทรงใบหน้าจากภาพ)",
"eye_analysis": {
    "shape": "string (เช่น Hunter Eyes, Almond Eyes, Round Eyes จากภาพ)",
    "canthal_tilt": "string (Positive, Neutral, Negative จากภาพ พร้อมเหตุผลทางเรขาคณิต)",
    "assessment": "string (วิจารณ์ดวงตาตามหลัก Blackpill โดยอิงจากภาพอย่างเจ็บแสบ)"
},
"eyebrow_shape": "string (เช่น Straight, Arched, Rounded จากภาพ)",
"mouth_shape": "string (เช่น Full Lips, Thin Lips, Heart-shaped จากภาพ)",
"facial_thirds_balance": "string (ความสมดุลของใบหน้า 3 ส่วนจากภาพ)",
"symmetry_assessment": "string (การประเมินความสมมาตรจากภาพ)",
"hairstyle_analysis": {
    "overall_recommendation": "string (สรุปภาพรวมทรงผมที่เหมาะ)",
    "recommended_styles": [ { "name": "string (ชื่อทรงผม)", "reason": "string (เหตุผลว่าทำไมถึงเหมาะกับโครงหน้า)" } ],
    "styles_to_avoid": ["string (ทรงผมที่ควรหลีกเลี่ยงพร้อมเหตุผลสั้นๆ)"]
},
"halo_features": ["string", "... (ลิสต์จุดแข็ง (Halo Features) ทั้งหมดที่สังเกตได้จากภาพ)"],
"flaws_and_chopped_features": ["string", "... (ลิสต์จุดด้อยหรือจุดที่ Chopped ทั้งหมดที่เห็นในภาพ ไม่ว่าจะเล็กน้อยแค่ไหนก็ตาม)"],
"feature_ratings": { "overall_score": "integer (0-100)", "eyes": "integer (0-100)", "nose": "integer (0-100)", "lips": "integer (0-100)", "jawline_and_chin": "integer (0-100)", "forehead_and_brows": "integer (0-100)" },
"psl_scale": { "rating": "float (1.0-10.0)", "tier": "string", "summary": "string (สรุปเหตุผลการให้คะแนนตามหลัก Blackpill อย่างตรงไปตรงมา โดยอ้างอิงจากรูป)" },
"ratings_summary": "string (สรุปภาพรวมของคะแนนอย่างโหดเหี้ยม โดยอ้างอิงจากสิ่งที่เห็นในรูปเท่านั้น)"
"""
    
    parts = []
    
    if side_b64:
        prompt = f'{base_prompt}\nSchema: {{\n  "front_profile_analysis": {{ {schema} }},\n  "side_profile_analysis": {{\n    "gonial_angle_degrees": "integer (110-130)", "gonial_angle_assessment": "string (เฉียบคม/ปกติ/ป้าน จากภาพ)", "ramus_length_assessment": "string (สั้น/ปกติ/ยาว จากภาพ)", "maxilla_projection": "string (ปกติ/ยื่น/หุบ จากภาพ)", "mandible_projection": "string (ปกติ/ยื่น/หุบ จากภาพ)", "facial_convexity": "string (ตรง/นูน/เว้า จากภาพ)", "recommendations": ["string", "string (คำแนะนำเชิงปฏิบัติที่ทำได้จริง 2 ข้อจากภาพ)"]\n  }}\n}}'
        parts = [
            {"text": prompt},
            {"inlineData": {"mimeType": "image/jpeg", "data": front_b64}},
            {"inlineData": {"mimeType": "image/jpeg", "data": side_b64}}
        ]
    else:
        prompt = f'{base_prompt}\nSchema: {{\n  "front_profile_analysis": {{ {schema} }}\n}}'
        parts = [
            {"text": prompt},
            {"inlineData": {"mimeType": "image/jpeg", "data": front_b64}}
        ]

    payload = {"contents": [{"parts": parts}], "generationConfig": {"responseMimeType": "application/json"}}
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    result_json = response.json()
    json_text = result_json['candidates'][0]['content']['parts'][0]['text']
    return json.loads(json_text)

def call_gemini_api_for_chat(chat_history, initial_analysis):
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

    system_instruction = {
        "parts": [{"text": f"""คุณคือ AI Lookmaxxing Advisor ที่มีความรู้แบบ Blackpill กำลังสนทนากับผู้ใช้
        ข้อมูลการวิเคราะห์ใบหน้าของผู้ใช้อยู่ด้านล่างในรูปแบบ JSON:
        --- ANALYSIS DATA ---
        {initial_analysis}
        --- END ANALYSIS DATA ---
        หน้าที่ของคุณคือตอบคำถามของผู้ใช้และให้คำแนะนำเพิ่มเติมโดยอิงจาก "ข้อมูลการวิเคราะห์" ที่ให้มาเท่านั้น ห้ามสร้างข้อมูลใหม่ จงตอบอย่างตรงไปตรงมา เฉียบคม แต่มีประโยชน์"""}]
    }

    payload = {"contents": chat_history, "systemInstruction": system_instruction}
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    result_json = response.json()
    return result_json['candidates'][0]['content']['parts'][0]['text']


if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=False)

