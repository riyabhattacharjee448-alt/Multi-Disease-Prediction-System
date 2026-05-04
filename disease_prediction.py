import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
from datetime import datetime
import warnings
import hashlib
import os
import mysql.connector
from mysql.connector import Error

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  MYSQL DATABASE CONFIGURATION
# ══════════════════════════════════════════════════════════════
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Riya@2006",  # ⚠ CHANGE THIS TO YOUR MYSQL PASSWORD
    "database": "mediscan_ai"
}

MYSQL_AVAILABLE = False  # Track if MySQL is available

def get_db_connection():
    """Create MySQL database connection"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except Error as e:
        print(f"⚠ MySQL connection error: {e}")
        print("⚠ Running in DEMO MODE - MySQL features disabled")
        return None

def init_database():
    """Initialize MySQL database and create tables (non-blocking)"""
    global MYSQL_AVAILABLE
    try:
        conn = get_db_connection()
        if conn is None:
            MYSQL_AVAILABLE = False
            return
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT PRIMARY KEY AUTO_INCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                password VARCHAR(255) NOT NULL,
                role VARCHAR(20) DEFAULT 'Patient',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_username (username)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        MYSQL_AVAILABLE = True
        print("✓ MySQL database initialized successfully")
    except Error as e:
        MYSQL_AVAILABLE = False
        print(f"⚠ MySQL initialization error: {e}")
        print("⚠ App will run in DEMO MODE with in-memory users")

def hash_pw(pw):
    """Hash password using SHA256"""
    return hashlib.sha256(pw.encode()).hexdigest()

def get_user_from_db(username):
    """Retrieve user from MySQL database (or fallback to memory)"""
    username_lower = username.lower()
    
    if MYSQL_AVAILABLE:
        try:
            conn = get_db_connection()
            if conn is None:
                return DEMO_USERS.get(username_lower)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username_lower,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            return user
        except Error as e:
            print(f"⚠ Error retrieving user: {e}")
            return DEMO_USERS.get(username_lower)
    else:
        return DEMO_USERS.get(username_lower)

def add_user_to_db(username, name, password, role="Patient"):
    """Add new user to MySQL database (or fallback to memory)"""
    username_lower = username.lower()
    
    # Check if username already exists
    if username_lower in DEMO_USERS:
        return False, "Username already exists!"
    
    hashed_pw = hash_pw(password)
    
    if MYSQL_AVAILABLE:
        try:
            conn = get_db_connection()
            if conn is None:
                # Fallback to memory
                DEMO_USERS[username_lower] = {
                    "username": username_lower,
                    "name": name,
                    "password": hashed_pw,
                    "role": role
                }
                return True, "Account created successfully (DEMO MODE)!"
            
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, name, password, role) VALUES (%s, %s, %s, %s)",
                (username_lower, name, hashed_pw, role)
            )
            conn.commit()
            cursor.close()
            conn.close()
            return True, "Account created successfully!"
        except mysql.connector.errors.IntegrityError:
            return False, "Username already exists!"
        except Error as e:
            return False, f"Error: {str(e)}"
    else:
        # Store in memory if MySQL not available
        DEMO_USERS[username_lower] = {
            "username": username_lower,
            "name": name,
            "password": hashed_pw,
            "role": role
        }
        return True, "Account created successfully (DEMO MODE)!"

def user_exists_in_db(username):
    """Check if username exists in database"""
    return get_user_from_db(username) is not None

# Fallback in-memory user storage (if MySQL unavailable)
DEMO_USERS = {
    "demo": {"username": "demo", "name": "Demo User", "password": hash_pw("demo123"), "role": "Patient"},
    "admin": {"username": "admin", "name": "Dr. Admin", "password": hash_pw("admin123"), "role": "Doctor"},
}

# Initialize database on app startup
init_database()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ══════════════════════════════════════════════════════════════
#  LOAD MODEL & DATA
# ══════════════════════════════════════════════════════════════
try:
    model = pickle.load(open("model.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    df = pd.read_excel("Training.xlsx")
    symptoms = list(df.columns[:-1])
    X = df.drop("prognosis", axis=1)
    y = le.transform(df["prognosis"])
    y_pred_full = model.predict(X)
    report = classification_report(y, y_pred_full, output_dict=True)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "Class"})
    cm = confusion_matrix(y, y_pred_full)
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": symptoms, "Importance": importances}).sort_values(
        by="Importance", ascending=False).head(20)
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    symptoms = [
        "fever","cough","fatigue","headache","nausea","vomiting","diarrhea","rash",
        "chest_pain","shortness_of_breath","joint_pain","muscle_pain","sore_throat",
        "runny_nose","loss_of_appetite","abdominal_pain","back_pain","dizziness",
        "sweating","chills","weight_loss","blurred_vision","frequent_urination",
        "excessive_thirst","itching","skin_discoloration","swollen_glands","yellowing_skin",
        "dark_urine","clay_colored_stool","palpitations","anxiety","depression",
        "memory_loss","confusion","tremors","seizures","paralysis","slurred_speech",
        "difficulty_swallowing","ear_pain","neck_pain","knee_pain","hip_pain",
        "swollen_joints","brittle_nails","hair_loss","dry_skin","oily_skin",
    ]
    demo_diseases = [
        "Common Flu","Malaria","Diabetes","Dengue","Typhoid","Pneumonia",
        "Hepatitis B","Tuberculosis","Hypertension","Migraine","Arthritis",
        "Asthma","COVID-19","Heart Disease","Kidney Disease","Liver Disease",
        "Anemia","Chickenpox","Psoriasis","Epilepsy"
    ]
    
    # ── FIX: symptom→disease mapping so same symptoms always give same result ──
    SYMPTOM_DISEASE_MAP = {
        "Common Flu":    ["fever","cough","fatigue","runny_nose","sore_throat","headache","chills"],
        "Malaria":       ["fever","chills","sweating","headache","nausea","vomiting","muscle_pain"],
        "Diabetes":      ["frequent_urination","excessive_thirst","fatigue","blurred_vision","weight_loss"],
        "Dengue":        ["fever","rash","joint_pain","muscle_pain","headache","nausea","vomiting"],
        "Typhoid":       ["fever","abdominal_pain","headache","loss_of_appetite","nausea","diarrhea"],
        "Pneumonia":     ["fever","cough","shortness_of_breath","chest_pain","fatigue","chills"],
        "Hepatitis B":   ["yellowing_skin","dark_urine","fatigue","abdominal_pain","nausea","loss_of_appetite"],
        "Tuberculosis":  ["cough","weight_loss","fatigue","sweating","fever","loss_of_appetite"],
        "Hypertension":  ["headache","dizziness","chest_pain","blurred_vision","fatigue","palpitations"],
        "Migraine":      ["headache","nausea","blurred_vision","dizziness","vomiting"],
        "Arthritis":     ["joint_pain","swollen_joints","knee_pain","hip_pain","fatigue","muscle_pain"],
        "Asthma":        ["shortness_of_breath","cough","chest_pain","fatigue","wheezing"],
        "COVID-19":      ["fever","cough","fatigue","shortness_of_breath","loss_of_appetite","headache"],
        "Heart Disease": ["chest_pain","palpitations","shortness_of_breath","fatigue","dizziness","sweating"],
        "Kidney Disease":["frequent_urination","fatigue","swollen_joints","back_pain","loss_of_appetite","dizziness"],
        "Liver Disease": ["yellowing_skin","dark_urine","abdominal_pain","fatigue","nausea","clay_colored_stool"],
        "Anemia":        ["fatigue","dizziness","headache","shortness_of_breath","chills","hair_loss"],
        "Chickenpox":    ["fever","rash","itching","fatigue","headache","loss_of_appetite"],
        "Psoriasis":     ["rash","itching","dry_skin","skin_discoloration","joint_pain","brittle_nails"],
        "Epilepsy":      ["seizures","confusion","memory_loss","headache","fatigue","anxiety"],
    }

    def _score_disease(symptom_vec, disease):
        """Score how well a symptom vector matches a disease. Fully deterministic."""
        disease_syms = SYMPTOM_DISEASE_MAP.get(disease, [])
        matches = sum(1 for s in disease_syms if s in symptoms and symptom_vec[symptoms.index(s)] == 1)
        total_selected = max(1, int(symptom_vec.sum()))
        disease_len = max(1, len(disease_syms))
        # Jaccard-like score
        return matches / (total_selected + disease_len - matches)

    class DemoModel:
        def predict(self, X):
            x = X[0] if len(X.shape) > 1 else X
            scores = [_score_disease(x, d) for d in demo_diseases]
            return np.array([int(np.argmax(scores))])

        def predict_proba(self, X):
            x = X[0] if len(X.shape) > 1 else X
            scores = np.array([_score_disease(x, d) for d in demo_diseases], dtype=float)
            # Softmax so probabilities sum to 1
            scores = scores - scores.max()
            exp_scores = np.exp(scores * 10)  # temperature scaling
            probs = exp_scores / exp_scores.sum()
            return probs.reshape(1, -1)

        @property
        def feature_importances_(self):
            # Fixed, non-random importances derived from symptom frequency across diseases
            counts = np.zeros(len(symptoms))
            for syms in SYMPTOM_DISEASE_MAP.values():
                for s in syms:
                    if s in symptoms:
                        counts[symptoms.index(s)] += 1
            counts = counts + 0.1  # smoothing
            return counts / counts.sum()

    class DemoLE:
        @property
        def classes_(self):
            return np.array(demo_diseases)
        def inverse_transform(self, x):
            return np.array([demo_diseases[i % len(demo_diseases)] for i in x])
        def transform(self, x):
            return np.array([demo_diseases.index(d) if d in demo_diseases else 0 for d in x])

    model = DemoModel()
    le = DemoLE()
    df = pd.DataFrame(np.random.RandomState(42).randint(0, 2, size=(200, len(symptoms) + 1)),
                      columns=symptoms + ["prognosis"])
    df["prognosis"] = np.random.RandomState(42).choice(demo_diseases, size=200)
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": symptoms, "Importance": importances}).sort_values(
        by="Importance", ascending=False).head(20)
    cm = np.random.RandomState(42).randint(0, 40, size=(len(demo_diseases), len(demo_diseases)))
    np.fill_diagonal(cm, np.random.RandomState(42).randint(30, 80, len(demo_diseases)))
    report_df = pd.DataFrame({
        "Class": demo_diseases,
        "precision": np.random.RandomState(42).uniform(0.78, 0.99, len(demo_diseases)).round(3),
        "recall":    np.random.RandomState(42).uniform(0.78, 0.99, len(demo_diseases)).round(3),
        "f1-score":  np.random.RandomState(42).uniform(0.78, 0.99, len(demo_diseases)).round(3),
        "support":   np.random.RandomState(42).randint(8, 40, len(demo_diseases))
    })


def extract_symptoms_from_text(text):
    """Extract symptoms from user message using keyword matching"""
    text_lower = text.lower()
    extracted = []
    for symptom in symptoms:
        # Check for exact symptom or common variations
        symptom_words = symptom.replace("_", " ").split()
        if all(word in text_lower for word in symptom_words):
            extracted.append(symptom)
        # Also check for partial matches or synonyms
        elif symptom == "fever" and ("fever" in text_lower or "temperature" in text_lower or "hot" in text_lower):
            extracted.append(symptom)
        elif symptom == "cough" and "cough" in text_lower:
            extracted.append(symptom)
        elif symptom == "fatigue" and ("fatigue" in text_lower or "tired" in text_lower or "exhausted" in text_lower):
            extracted.append(symptom)
        elif symptom == "headache" and ("headache" in text_lower or "head ache" in text_lower):
            extracted.append(symptom)
        elif symptom == "nausea" and "nausea" in text_lower:
            extracted.append(symptom)
        elif symptom == "vomiting" and ("vomiting" in text_lower or "vomit" in text_lower):
            extracted.append(symptom)
        elif symptom == "diarrhea" and "diarrhea" in text_lower:
            extracted.append(symptom)
        elif symptom == "rash" and "rash" in text_lower:
            extracted.append(symptom)
        elif symptom == "chest_pain" and ("chest pain" in text_lower or "chest_pain" in text_lower):
            extracted.append(symptom)
        elif symptom == "shortness_of_breath" and ("shortness of breath" in text_lower or "breath" in text_lower or "breathing" in text_lower):
            extracted.append(symptom)
        elif symptom == "joint_pain" and ("joint pain" in text_lower or "joint_pain" in text_lower):
            extracted.append(symptom)
        elif symptom == "muscle_pain" and ("muscle pain" in text_lower or "muscle_pain" in text_lower):
            extracted.append(symptom)
        elif symptom == "sore_throat" and ("sore throat" in text_lower or "sore_throat" in text_lower):
            extracted.append(symptom)
        elif symptom == "runny_nose" and ("runny nose" in text_lower or "runny_nose" in text_lower):
            extracted.append(symptom)
        elif symptom == "loss_of_appetite" and ("loss of appetite" in text_lower or "no appetite" in text_lower):
            extracted.append(symptom)
        elif symptom == "abdominal_pain" and ("abdominal pain" in text_lower or "stomach pain" in text_lower):
            extracted.append(symptom)
        elif symptom == "back_pain" and ("back pain" in text_lower or "back_pain" in text_lower):
            extracted.append(symptom)
        elif symptom == "dizziness" and "dizziness" in text_lower:
            extracted.append(symptom)
        elif symptom == "sweating" and "sweating" in text_lower:
            extracted.append(symptom)
        elif symptom == "chills" and "chills" in text_lower:
            extracted.append(symptom)
        elif symptom == "weight_loss" and ("weight loss" in text_lower or "weight_loss" in text_lower):
            extracted.append(symptom)
        elif symptom == "blurred_vision" and ("blurred vision" in text_lower or "blurred_vision" in text_lower):
            extracted.append(symptom)
        elif symptom == "frequent_urination" and ("frequent urination" in text_lower or "frequent_urination" in text_lower):
            extracted.append(symptom)
        elif symptom == "excessive_thirst" and ("excessive thirst" in text_lower or "thirsty" in text_lower):
            extracted.append(symptom)
        elif symptom == "itching" and "itching" in text_lower:
            extracted.append(symptom)
        elif symptom == "skin_discoloration" and ("skin discoloration" in text_lower or "skin_discoloration" in text_lower):
            extracted.append(symptom)
        elif symptom == "swollen_glands" and ("swollen glands" in text_lower or "swollen_glands" in text_lower):
            extracted.append(symptom)
        elif symptom == "yellowing_skin" and ("yellowing skin" in text_lower or "jaundice" in text_lower):
            extracted.append(symptom)
        elif symptom == "dark_urine" and ("dark urine" in text_lower or "dark_urine" in text_lower):
            extracted.append(symptom)
        elif symptom == "clay_colored_stool" and ("clay colored stool" in text_lower or "pale stool" in text_lower):
            extracted.append(symptom)
        elif symptom == "palpitations" and "palpitations" in text_lower:
            extracted.append(symptom)
        elif symptom == "anxiety" and "anxiety" in text_lower:
            extracted.append(symptom)
        elif symptom == "depression" and "depression" in text_lower:
            extracted.append(symptom)
        elif symptom == "memory_loss" and ("memory loss" in text_lower or "memory_loss" in text_lower):
            extracted.append(symptom)
        elif symptom == "confusion" and "confusion" in text_lower:
            extracted.append(symptom)
        elif symptom == "tremors" and "tremors" in text_lower:
            extracted.append(symptom)
        elif symptom == "seizures" and "seizures" in text_lower:
            extracted.append(symptom)
        elif symptom == "paralysis" and "paralysis" in text_lower:
            extracted.append(symptom)
        elif symptom == "slurred_speech" and ("slurred speech" in text_lower or "slurred_speech" in text_lower):
            extracted.append(symptom)
        elif symptom == "difficulty_swallowing" and ("difficulty swallowing" in text_lower or "difficulty_swallowing" in text_lower):
            extracted.append(symptom)
        elif symptom == "ear_pain" and ("ear pain" in text_lower or "ear_pain" in text_lower):
            extracted.append(symptom)
        elif symptom == "neck_pain" and ("neck pain" in text_lower or "neck_pain" in text_lower):
            extracted.append(symptom)
        elif symptom == "knee_pain" and ("knee pain" in text_lower or "knee_pain" in text_lower):
            extracted.append(symptom)
        elif symptom == "hip_pain" and ("hip pain" in text_lower or "hip_pain" in text_lower):
            extracted.append(symptom)
        elif symptom == "swollen_joints" and ("swollen joints" in text_lower or "swollen_joints" in text_lower):
            extracted.append(symptom)
        elif symptom == "brittle_nails" and ("brittle nails" in text_lower or "brittle_nails" in text_lower):
            extracted.append(symptom)
        elif symptom == "hair_loss" and ("hair loss" in text_lower or "hair_loss" in text_lower):
            extracted.append(symptom)
        elif symptom == "dry_skin" and ("dry skin" in text_lower or "dry_skin" in text_lower):
            extracted.append(symptom)
        elif symptom == "oily_skin" and ("oily skin" in text_lower or "oily_skin" in text_lower):
            extracted.append(symptom)
    return list(set(extracted))  # Remove duplicates


# ══════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════
medicine_dict = {
    "Common Flu":      {"medicine":"Paracetamol, Antihistamines, Decongestants","diet":"Hot soups, Vitamin C rich foods, Ginger tea","rest":"3–5 days","severity":"Mild","specialist":"General Physician","icd":"J11"},
    "Flu":             {"medicine":"Paracetamol, Antihistamines","diet":"Hot soups, Vitamin C","rest":"3–5 days","severity":"Mild","specialist":"General Physician","icd":"J11"},
    "Malaria":         {"medicine":"Chloroquine / Artemisinin combination therapy","diet":"Hydrate well, Fruits, Avoid fatty food","rest":"1–2 weeks","severity":"High","specialist":"Infectious Disease Specialist","icd":"B54"},
    "Diabetes":        {"medicine":"Insulin / Metformin / SGLT2 inhibitors","diet":"Low-sugar, High-fiber, Low GI foods","rest":"Lifelong management","severity":"Chronic","specialist":"Endocrinologist","icd":"E11"},
    "Dengue":          {"medicine":"Paracetamol (NO aspirin), IV fluids, Platelet support","diet":"Papaya leaf juice, Coconut water, Liquids","rest":"7–10 days","severity":"High","specialist":"Infectious Disease Specialist","icd":"A90"},
    "Typhoid":         {"medicine":"Ciprofloxacin / Azithromycin / Ceftriaxone","diet":"Soft foods, Avoid spicy/oily food","rest":"2–3 weeks","severity":"Moderate","specialist":"Gastroenterologist","icd":"A01"},
    "Pneumonia":       {"medicine":"Antibiotics (Amoxicillin/Azithromycin), Oxygen therapy","diet":"High protein diet, Plenty of fluids","rest":"2–4 weeks","severity":"High","specialist":"Pulmonologist","icd":"J18"},
    "Hepatitis B":     {"medicine":"Antivirals (Tenofovir/Entecavir), Supportive care","diet":"Low fat, No alcohol, High antioxidants","rest":"Weeks–Months","severity":"High","specialist":"Hepatologist","icd":"B18"},
    "Tuberculosis":    {"medicine":"RIPE therapy — Rifampicin, Isoniazid, Pyrazinamide, Ethambutol","diet":"High calorie, High protein, Zinc & Vitamin D","rest":"6+ months","severity":"High","specialist":"Pulmonologist","icd":"A15"},
    "Hypertension":    {"medicine":"ACE inhibitors, Beta-blockers, Calcium channel blockers","diet":"Low sodium, DASH diet, No processed food","rest":"Lifestyle changes, Stress management","severity":"Chronic","specialist":"Cardiologist","icd":"I10"},
    "Migraine":        {"medicine":"Triptans (Sumatriptan), NSAIDs, Anti-nausea meds","diet":"Avoid triggers, Stay hydrated, Magnesium-rich foods","rest":"Dark quiet room, 4–72 hrs per episode","severity":"Moderate","specialist":"Neurologist","icd":"G43"},
    "Arthritis":       {"medicine":"NSAIDs, DMARDs (Methotrexate), Corticosteroids","diet":"Anti-inflammatory foods, Omega-3, Turmeric","rest":"Physical therapy, Moderate activity","severity":"Chronic","specialist":"Rheumatologist","icd":"M05"},
    "Asthma":          {"medicine":"Salbutamol inhaler, ICS (Budesonide), Montelukast","diet":"Anti-inflammatory diet, Avoid allergens","rest":"Avoid triggers, Breathing exercises","severity":"Moderate","specialist":"Pulmonologist","icd":"J45"},
    "COVID-19":        {"medicine":"Antivirals (Paxlovid), Dexamethasone (severe), Supportive","diet":"High Vitamin C & D, Zinc, Hydration","rest":"10–14 days isolation","severity":"High","specialist":"Infectious Disease / Pulmonologist","icd":"U07"},
    "Heart Disease":   {"medicine":"Statins, Beta-blockers, Aspirin, ACE inhibitors","diet":"Heart-healthy diet, Low saturated fat, Mediterranean","rest":"Cardiac rehabilitation","severity":"Critical","specialist":"Cardiologist","icd":"I25"},
    "Kidney Disease":  {"medicine":"ACE inhibitors, Diuretics, Phosphate binders","diet":"Low protein, Low potassium, Low phosphorus","rest":"Dialysis if advanced","severity":"Critical","specialist":"Nephrologist","icd":"N18"},
    "Liver Disease":   {"medicine":"Antivirals, Diuretics, Lactulose, Liver transplant if needed","diet":"No alcohol, Low sodium, High antioxidants","rest":"Rest, avoid hepatotoxic drugs","severity":"Critical","specialist":"Hepatologist","icd":"K74"},
    "Anemia":          {"medicine":"Iron supplements, Vitamin B12, Folic acid, EPO","diet":"Iron-rich foods, Vitamin C, Leafy greens","rest":"Moderate activity, avoid overexertion","severity":"Moderate","specialist":"Hematologist","icd":"D50"},
    "Chickenpox":      {"medicine":"Antihistamines, Acyclovir (severe), Calamine lotion","diet":"Soft foods, Cool liquids, Avoid spicy food","rest":"7–10 days","severity":"Mild","specialist":"Dermatologist / Pediatrician","icd":"B01"},
    "Psoriasis":       {"medicine":"Topical corticosteroids, Retinoids, Biologics (Adalimumab)","diet":"Anti-inflammatory, Omega-3, Low alcohol","rest":"Stress reduction, UV light therapy","severity":"Chronic","specialist":"Dermatologist","icd":"L40"},
    "Epilepsy":        {"medicine":"Levetiracetam, Valproate, Carbamazepine","diet":"Ketogenic diet (in some cases), Avoid alcohol","rest":"Avoid seizure triggers, Regular sleep","severity":"Chronic","specialist":"Neurologist","icd":"G40"},
}

severity_colors = {
    "Mild":     "#22c55e",
    "Moderate": "#f59e0b",
    "High":     "#ef4444",
    "Critical": "#dc2626",
    "Chronic":  "#8b5cf6",
    "Unknown":  "#64748b",
}

symptom_categories = {
    "🌡️ Fever & Temp":    ["fever","chills","sweating"],
    "🫁 Respiratory":     ["cough","shortness_of_breath","runny_nose","sore_throat"],
    "🧠 Neurological":    ["headache","dizziness","blurred_vision","memory_loss","confusion","tremors","seizures","slurred_speech"],
    "🫀 Cardiovascular":  ["chest_pain","palpitations"],
    "🤢 Digestive":       ["nausea","vomiting","diarrhea","abdominal_pain","loss_of_appetite","difficulty_swallowing"],
    "💪 Musculoskeletal": ["joint_pain","muscle_pain","back_pain","fatigue","neck_pain","knee_pain","hip_pain","swollen_joints"],
    "🔬 Metabolic":       ["weight_loss","excessive_thirst","frequent_urination","anxiety","depression"],
    "🩺 Skin & Other":    ["rash","itching","skin_discoloration","yellowing_skin","dark_urine","swollen_glands",
                           "clay_colored_stool","brittle_nails","hair_loss","dry_skin","oily_skin"],
}

# Add default demo user to database on startup if it doesn't exist
if not user_exists_in_db("demo"):
    add_user_to_db("demo", "Demo User", "demo123", "Patient")
if not user_exists_in_db("admin"):
    add_user_to_db("admin", "Dr. Admin", "admin123", "Doctor")

C = {
    "bg":        "#060a12",
    "surface":   "#0d1424",
    "surface2":  "#111d30",
    "surface3":  "#162038",
    "border":    "#1a2d4a",
    "border2":   "rgba(0,240,255,0.15)",
    "neon":      "#00f0ff",
    "neon2":     "#bf5fff",
    "neon3":     "#00ff9d",
    "neon4":     "#ff6b35",
    "neon5":     "#ffce00",
    "text":      "#e8f0fe",
    "text2":     "#94a3b8",
    "muted":     "#4a5568",
    "danger":    "#ff4d6d",
    "warning":   "#ffbe0b",
    "success":   "#06d6a0",
    "glow_cyan": "0 0 20px rgba(0,240,255,0.4)",
    "glow_purple":"0 0 20px rgba(191,95,255,0.4)",
}

CHART_COLORS = ["#00f0ff","#bf5fff","#00ff9d","#ff6b35","#ffce00","#f72585","#4cc9f0","#7bed9f","#ff9f43","#a29bfe"]

def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="'Rajdhani', sans-serif", size=11),
        margin=dict(l=16, r=16, t=44, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(gridcolor=C["border"], linecolor=C["border"], zeroline=False)
    fig.update_yaxes(gridcolor=C["border"], linecolor=C["border"], zeroline=False)
    return fig


GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&family=Exo+2:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: #060a12;
    color: #e8f0fe;
    font-family: 'Rajdhani', sans-serif;
    font-size: 30px;
    overflow-x: hidden;
}

body::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,240,255,0.012) 2px, rgba(0,240,255,0.012) 4px);
    pointer-events: none; z-index: 9999;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1424; }
::-webkit-scrollbar-thumb { background: rgba(0,240,255,0.3); border-radius: 3px; }

.glass-card {
    background: linear-gradient(135deg, rgba(13,20,36,0.95), rgba(17,29,48,0.9));
    border: 1px solid rgba(0,240,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,240,255,0.4), transparent);
}
.glass-card:hover {
    border-color: rgba(0,240,255,0.25);
    box-shadow: 0 0 30px rgba(0,240,255,0.06);
}

.navbar-brand-text {
    font-family: 'Orbitron', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    background: linear-gradient(135deg, #00f0ff, #bf5fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
}

.tab-btn {
    background: transparent;
    border: 1px solid transparent;
    color: #4a5568;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    display: inline-block;
}
.tab-btn:hover { color: #00f0ff; border-color: rgba(0,240,255,0.2); }
.tab-btn.active {
    color: #00f0ff;
    border-color: rgba(0,240,255,0.35);
    background: rgba(0,240,255,0.07);
    box-shadow: 0 0 12px rgba(0,240,255,0.15);
}

.page-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00f0ff, #bf5fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}
.page-subtitle { color: #4a5568; font-size: 0.88rem; margin-top: 0.3rem; letter-spacing: 0.3px; }

.stat-card {
    background: linear-gradient(135deg, rgba(13,20,36,0.98), rgba(22,32,56,0.95));
    border: 1px solid rgba(0,240,255,0.1);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.3s;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,240,255,0.1); }
.stat-number {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-label { color: #4a5568; font-size: 0.75rem; letter-spacing: 1.5px; text-transform: uppercase; }

.Select-control, .Select-menu-outer {
    background: #0d1424 !important;
    border: 1px solid rgba(0,240,255,0.2) !important;
    border-radius: 10px !important;
    color: #e8f0fe !important;
}
.Select-value-label, .Select-placeholder, .Select-input input { color: #e8f0fe !important; }
.Select-option { background: #0d1424 !important; color: #94a3b8 !important; }
.Select-option:hover, .Select-option.is-focused { background: rgba(0,240,255,0.08) !important; color: #00f0ff !important; }
.Select-multi-value-wrapper .Select-value {
    background: rgba(0,240,255,0.1) !important;
    border: 1px solid rgba(0,240,255,0.25) !important;
    color: #00f0ff !important;
    border-radius: 6px !important;
}
.Select-value-icon { color: rgba(0,240,255,0.6) !important; }
.Select-arrow { border-top-color: #4a5568 !important; }
.VirtualizedSelectOption { background: #0d1424 !important; color: #94a3b8 !important; }

.predict-btn {
    background: linear-gradient(135deg, #00f0ff, #bf5fff);
    border: none;
    border-radius: 12px;
    color: #060a12;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 0.75rem 2rem;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 0 20px rgba(0,240,255,0.3);
    position: relative;
    overflow: hidden;
}
.predict-btn::after {
    content: '';
    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.15) 50%, transparent 70%);
    transform: translateX(-100%);
    transition: transform 0.5s;
}
.predict-btn:hover::after { transform: translateX(100%); }
.predict-btn:hover { box-shadow: 0 0 35px rgba(0,240,255,0.5); transform: translateY(-2px); }
.predict-btn:active { transform: translateY(0); }

.reset-btn {
    background: transparent;
    border: 1px solid rgba(255,77,109,0.35);
    border-radius: 12px;
    color: #ff4d6d;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 0.7rem 1.5rem;
    cursor: pointer;
    transition: all 0.25s;
}
.reset-btn:hover { background: rgba(255,77,109,0.1); box-shadow: 0 0 15px rgba(255,77,109,0.2); }

.cat-btn {
    background: rgba(13,20,36,0.9);
    border: 1px solid rgba(0,240,255,0.15);
    border-radius: 8px;
    color: #64748b;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.3rem 0.7rem;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.3px;
}
.cat-btn:hover { color: #00f0ff; border-color: rgba(0,240,255,0.35); }
.cat-btn.active-cat { color: #00f0ff; border-color: #00f0ff; background: rgba(0,240,255,0.08); box-shadow: 0 0 10px rgba(0,240,255,0.2); }

.result-card {
    background: linear-gradient(135deg, rgba(13,20,36,0.98), rgba(22,32,56,0.92));
    border: 1px solid rgba(0,240,255,0.25);
    border-radius: 16px;
    padding: 1.75rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(0,240,255,0.08);
}
.result-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00f0ff, #bf5fff, #00ff9d);
}
.result-disease-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00f0ff, #bf5fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.confidence-bar-wrap {
    background: rgba(10,14,26,0.8);
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
    border: 1px solid rgba(0,240,255,0.1);
}
.confidence-bar-fill { height: 100%; border-radius: 100px; transition: width 1s cubic-bezier(0.34,1.56,0.64,1); }

.info-pill {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    background: rgba(10,14,26,0.6);
    border: 1px solid rgba(0,240,255,0.08);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
.info-pill-label { font-size: 0.68rem; color: #4a5568; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.1rem; }
.info-pill-value { font-size: 0.88rem; color: #e8f0fe; font-weight: 500; }

.custom-table { border-collapse: collapse; }
.custom-table th {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    color: #4a5568;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid rgba(0,240,255,0.12);
}
.custom-table td {
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid rgba(26,45,74,0.5);
    font-size: 0.85rem;
}
.custom-table tr:hover td { background: rgba(0,240,255,0.03); }
.custom-table tr:last-child td { border-bottom: none; }

.history-item {
    background: rgba(13,20,36,0.8);
    border: 1px solid rgba(0,240,255,0.08);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    transition: border-color 0.2s;
}
.history-item:hover { border-color: rgba(0,240,255,0.2); }

.chatbot-toggle {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00f0ff, #bf5fff);
    border: none;
    cursor: pointer;
    font-size: 1.4rem;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 25px rgba(0,240,255,0.5);
    z-index: 1100;
    transition: all 0.3s;
    animation: pulse-glow 2s infinite;
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(0,240,255,0.4); }
    50% { box-shadow: 0 0 40px rgba(0,240,255,0.7), 0 0 60px rgba(191,95,255,0.3); }
}
.chatbot-toggle:hover { transform: scale(1.1); }

.chatbot-window {
    position: fixed;
    bottom: 6.5rem;
    right: 2rem;
    width: 380px;
    max-height: 520px;
    background: linear-gradient(135deg, rgba(10,14,26,0.98), rgba(13,20,36,0.98));
    border: 1px solid rgba(0,240,255,0.25);
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 40px rgba(0,240,255,0.1);
    z-index: 1099;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.chatbot-window.hidden {
    opacity: 0;
    transform: scale(0.85) translateY(20px);
    pointer-events: none;
}
.chatbot-header {
    background: linear-gradient(135deg, rgba(0,240,255,0.08), rgba(191,95,255,0.08));
    border-bottom: 1px solid rgba(0,240,255,0.15);
    padding: 0.9rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.chat-avatar {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #00f0ff, #bf5fff);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
}
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    max-height: 340px;
}
.chat-msg-user {
    align-self: flex-end;
    background: linear-gradient(135deg, rgba(0,240,255,0.15), rgba(191,95,255,0.1));
    border: 1px solid rgba(0,240,255,0.2);
    border-radius: 14px 14px 4px 14px;
    padding: 0.55rem 0.9rem;
    max-width: 80%;
    font-size: 0.83rem;
    color: #e8f0fe;
}
.chat-msg-bot {
    align-self: flex-start;
    background: rgba(13,20,36,0.9);
    border: 1px solid rgba(0,240,255,0.1);
    border-radius: 14px 14px 14px 4px;
    padding: 0.55rem 0.9rem;
    max-width: 85%;
    font-size: 0.83rem;
    color: #94a3b8;
    line-height: 1.5;
}
.chat-input-wrap {
    padding: 0.75rem;
    border-top: 1px solid rgba(0,240,255,0.1);
    display: flex;
    gap: 0.5rem;
}
.chat-input {
    flex: 1;
    background: rgba(10,14,26,0.8);
    border: 1px solid rgba(0,240,255,0.15);
    border-radius: 10px;
    color: #e8f0fe;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    padding: 0.5rem 0.75rem;
    outline: none;
    transition: border-color 0.2s;
}
.chat-input:focus { border-color: rgba(0,240,255,0.35); }
.chat-send-btn {
    background: linear-gradient(135deg, #00f0ff, #bf5fff);
    border: none;
    border-radius: 10px;
    color: #060a12;
    font-weight: 700;
    padding: 0.5rem 0.9rem;
    cursor: pointer;
    font-size: 0.85rem;
    transition: all 0.2s;
}
.chat-send-btn:hover { box-shadow: 0 0 15px rgba(0,240,255,0.4); }

.login-container {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #060a12;
    position: relative;
    overflow: hidden;
}
.login-container::before {
    content: '';
    position: absolute;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(0,240,255,0.06) 0%, transparent 70%);
    top: -200px; left: -200px;
    animation: nebula 8s ease-in-out infinite;
}
.login-container::after {
    content: '';
    position: absolute;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(191,95,255,0.06) 0%, transparent 70%);
    bottom: -100px; right: -100px;
    animation: nebula 10s ease-in-out infinite reverse;
}
@keyframes nebula { 0%, 100% { transform: scale(1) rotate(0deg); } 50% { transform: scale(1.15) rotate(10deg); } }

.login-card {
    background: linear-gradient(135deg, rgba(13,20,36,0.98), rgba(17,29,48,0.95));
    border: 1px solid rgba(0,240,255,0.2);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    width: 420px;
    position: relative;
    z-index: 10;
    box-shadow: 0 25px 80px rgba(0,0,0,0.6), 0 0 60px rgba(0,240,255,0.06);
}
.login-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00f0ff, #bf5fff, #00ff9d);
    border-radius: 24px 24px 0 0;
}
.login-input {
    width: 100%;
    background: rgba(6,10,18,0.8);
    border: 1px solid rgba(0,240,255,0.15);
    border-radius: 6px;
    color: #ffffff;
    font-family: 'Rajdhani', sans-serif;
    font-size: 18px;
    padding: 1rem 1.2rem;
    outline: none;
    transition: all 0.2s;
    margin-bottom: 0.2rem;
    box-sizing: border-box;
    height: 3rem;
}
.login-input::placeholder {
    color: #94a3b8;      /* visible soft color */
    font-size: 18px;     /* enlarged text */
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 1px;
}
.login-input:focus { border-color: rgba(0,240,255,0.4); box-shadow: 0 0 15px rgba(0,240,255,0.1); }
.login-label { font-size: 0.72rem; color: #4a5568; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.4rem; display: block; }
.login-btn {
    width: 100%;
    background: linear-gradient(135deg, #00f0ff, #bf5fff);
    border: none;
    border-radius: 12px;
    color: #060a12;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 0.9rem;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 0 25px rgba(0,240,255,0.3);
    margin-top: 0.5rem;
}
.login-btn:hover { box-shadow: 0 0 40px rgba(0,240,255,0.5); transform: translateY(-2px); }
.login-error { color: #ff4d6d; font-size: 0.8rem; margin-top: 0.5rem; min-height: 1.2rem; }

.spinner {
    display: inline-block;
    width: 14px; height: 14px;
    border: 2px solid rgba(0,240,255,0.2);
    border-top-color: #00f0ff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 0.5rem;
}
@keyframes spin { to { transform: rotate(360deg); } }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.4s ease forwards; }

.section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #00f0ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,240,255,0.12);
}

.profile-input {
    background: rgba(6,10,18,0.7);
    border: 1px solid rgba(0,240,255,0.15);
    border-radius: 8px;
    color: #e8f0fe;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    padding: 0.5rem 0.75rem;
    width: 100%;
    outline: none;
}
.profile-input:focus { border-color: rgba(0,240,255,0.35); }

.severity-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.demo-notice {
    background: rgba(255,190,11,0.08);
    border: 1px solid rgba(255,190,11,0.25);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    color: #ffbe0b;
    font-size: 0.78rem;
    margin-bottom: 1rem;
}
"""


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "MediScan AI"
server = app.server

app.index_string = (
    '<!DOCTYPE html>'
    '<html>'
    '    <head>'
    '        {%metas%}'
    '        <title>{%title%}</title>'
    '        {%favicon%}'
    '        {%css%}'
    '        <style>'
    + GLOBAL_CSS +
    '        </style>'
    '    </head>'
    '    <body>'
    '        {%app_entry%}'
    '        <footer>'
    '            {%config%}'
    '            {%scripts%}'
    '            {%renderer%}'
    '        </footer>'
    '    </body>'
    '</html>'
)

def build_login_page():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div("🧬", style={"fontSize":"2.5rem","marginBottom":"0.75rem"}),
                    html.Div("MEDISCAN AI", className="navbar-brand-text",
                             style={"display":"block","fontSize":"1.6rem","marginBottom":"0.3rem"}),
                    html.Div("Advanced Disease Prediction Platform",
                             style={"color":"#4a5568","fontSize":"0.78rem","letterSpacing":"1px","marginBottom":"2rem"}),
                ], style={"textAlign":"center"}),

                html.Div([
                    html.Button("Sign In", id="login-tab-btn", className="tab-btn active",
                                style={"flex":"1"}),
                    html.Button("Register", id="register-tab-btn", className="tab-btn",
                                style={"flex":"1"}),
                ], style={"display":"flex","gap":"0.5rem","marginBottom":"1.75rem",
                          "background":"rgba(6,10,18,0.6)","borderRadius":"10px","padding":"0.3rem"}),

                html.Div(id="auth-form-content", children=[
                    html.Div([
                        html.Label("USERNAME", className="login-label"),
                        dcc.Input(id="login-username", type="text", placeholder="Enter username",
                                  className="login-input", debounce=False),
                    ], style={"marginBottom":"1rem"}),
                    html.Div([
                        html.Label("PASSWORD", className="login-label"),
                        dcc.Input(id="login-password", type="password", placeholder="Enter password",
                                  className="login-input", debounce=False),
                    ], style={"marginBottom":"0.5rem"}),
                    html.Div(id="login-error", className="login-error"),
                    html.Button("ACCESS SYSTEM", id="login-btn", className="login-btn"),
                    html.Div([
                        html.Span("Demo: ", style={"color":"#4a5568","fontSize":"0.75rem"}),
                        html.Span("demo / demo123", style={"color":"#00f0ff","fontSize":"0.75rem",
                                                            "fontFamily":"'JetBrains Mono',monospace"}),
                    ], style={"textAlign":"center","marginTop":"1.25rem"}),
                ]),

                html.Div(id="register-form-content", style={"display":"none"}, children=[
                    html.Div([
                        html.Label("FULL NAME", className="login-label"),
                        dcc.Input(id="reg-name", type="text", placeholder="Dr. Jane Doe",
                                  className="login-input"),
                    ], style={"marginBottom":"0.75rem"}),
                    html.Div([
                        html.Label("USERNAME", className="login-label"),
                        dcc.Input(id="reg-username", type="text", placeholder="Choose username",
                                  className="login-input"),
                    ], style={"marginBottom":"0.75rem"}),
                    html.Div([
                        html.Label("PASSWORD", className="login-label"),
                        dcc.Input(id="reg-password", type="password", placeholder="Create password",
                                  className="login-input"),
                    ], style={"marginBottom":"0.5rem"}),
                    html.Div(id="reg-error", className="login-error"),
                    html.Button("CREATE ACCOUNT", id="reg-btn", className="login-btn"),
                ]),

            ], className="login-card"),
        ], className="login-container"),
    ])

def build_navbar(username="User"):
    user_info = get_user_from_db(username) or {}
    display_name = user_info.get("name", username)
    role = user_info.get("role", "Patient")
    return html.Div([
        html.Div([
            html.Span("🧬 ", style={"fontSize":"1.1rem"}),
            html.Span("MEDI", className="navbar-brand-text", style={"fontSize":"1.15rem"}),
            html.Span("SCAN AI", style={"color":"#4a5568","fontFamily":"'Orbitron',sans-serif",
                                        "fontSize":"1.1rem","fontWeight":"400","letterSpacing":"2px"}),
        ], style={"display":"flex","alignItems":"center","gap":"0.25rem"}),

        html.Div([
            html.Button("⚡ Diagnose",   id="nav-predict",   className="tab-btn active"),
            html.Button("📊 Analytics",  id="nav-analytics", className="tab-btn"),
            html.Button("📋 Reports",    id="nav-reports",   className="tab-btn"),
            html.Button("🕓 History",    id="nav-history",   className="tab-btn"),
            html.Button("👤 Profile",    id="nav-profile",   className="tab-btn"),
        ], style={"display":"flex","gap":"0.3rem","alignItems":"center"}),

        html.Div([
            html.Span("🟢", style={"fontSize":"0.6rem"}),
            html.Span(" AI Active", style={"color":"#00ff9d","fontSize":"0.72rem","fontWeight":"700"}),
            html.Div(style={"width":"1px","height":"20px","background":"rgba(0,240,255,0.1)"}),
            html.Span(f"👤 {display_name}", style={"color":"#94a3b8","fontSize":"0.78rem"}),
            html.Span(f"[{role}]", style={"color":"#4a5568","fontSize":"0.7rem"}),
            html.Div(style={"width":"1px","height":"20px","background":"rgba(0,240,255,0.1)"}),
            html.Button("Logout", id="logout-btn",
                        style={"background":"transparent","border":"1px solid rgba(255,77,109,0.25)",
                               "borderRadius":"6px","color":"#ff4d6d","fontSize":"0.7rem",
                               "padding":"0.2rem 0.6rem","cursor":"pointer","fontFamily":"'Rajdhani',sans-serif"}),
        ], style={"display":"flex","alignItems":"center","gap":"0.6rem"}),

    ], style={
        "display":"flex","justifyContent":"space-between","alignItems":"center",
        "padding":"0.85rem 2rem",
        "background":"rgba(6,10,18,0.95)",
        "borderBottom":"1px solid rgba(0,240,255,0.1)",
        "backdropFilter":"blur(20px)",
        "position":"sticky","top":0,"zIndex":1000,
        "boxShadow":"0 4px 30px rgba(0,0,0,0.4)",
    })

def build_chatbot():
    return html.Div([
        html.Button("🤖", id="chat-toggle", className="chatbot-toggle", title="AI Health Assistant"),

        html.Div(id="chat-window", className="chatbot-window hidden", children=[
            html.Div([
                html.Div("🤖", className="chat-avatar"),
                html.Div([
                    html.Div("MediBot AI", style={"fontWeight":"700","fontSize":"0.9rem","color":"#e8f0fe",
                                                   "fontFamily":"'Orbitron',sans-serif","letterSpacing":"0.5px"}),
                    html.Div("Powered by GPT-4 · Online", style={"fontSize":"0.68rem","color":"#00ff9d"}),
                ]),
                html.Button("✕", id="chat-close",
                            style={"marginLeft":"auto","background":"transparent","border":"none",
                                   "color":"#4a5568","cursor":"pointer","fontSize":"1rem"}),
            ], className="chatbot-header"),

            html.Div(id="chat-messages", className="chat-messages", children=[
                html.Div([
                    html.Strong("MediBot: ", style={"color":"#00f0ff","fontFamily":"'Orbitron',sans-serif","fontSize":"0.68rem"}),
                    " Hello! I'm your AI health assistant. Describe your symptoms or ask any health questions and I'll help analyze them. Remember: I'm an AI — always consult a real doctor for diagnosis!"
                ], className="chat-msg-bot"),
            ]),

            html.Div([
                dcc.Input(id="chat-input", type="text", placeholder="Ask about symptoms...",
                          className="chat-input", debounce=False,
                          style={"flex":"1"}),
                html.Button("Send", id="chat-send-btn", className="chat-send-btn"),
            ], className="chat-input-wrap"),
        ]),
        dcc.Store(id="chat-history-store", data=[]),
        dcc.Store(id="chat-visible", data=False),
    ])


def build_prediction_page():
    n_diseases = len(le.classes_)
    n_symptoms = len(symptoms)
    try:
        acc = round(float(report_df[report_df["Class"] == "accuracy"]["f1-score"].values[0]) * 100, 1)
    except:
        acc = 97.4

    demo_banner = html.Div(
        "⚠ DEMO MODE — model.pkl not found. Predictions use symptom-matching logic (deterministic).",
        className="demo-notice"
    ) if not DATA_LOADED else html.Span()

    stats_row = dbc.Row([
        dbc.Col(html.Div([
            html.Div(f"{n_symptoms}", className="stat-number", style={"color":"#00f0ff"}),
            html.Div("Symptoms", className="stat-label"),
        ], className="stat-card"), md=3),
        dbc.Col(html.Div([
            html.Div(f"{n_diseases}", className="stat-number", style={"color":"#bf5fff"}),
            html.Div("Diseases", className="stat-label"),
        ], className="stat-card"), md=3),
        dbc.Col(html.Div([
            html.Div(f"{acc}%", className="stat-number", style={"color":"#00ff9d"}),
            html.Div("Accuracy", className="stat-label"),
        ], className="stat-card"), md=3),
        dbc.Col(html.Div([
            html.Div("GPT-4", className="stat-number", style={"color":"#ffce00","fontFamily":"'JetBrains Mono',monospace","fontSize":"1.5rem"}),
            html.Div("AI Chatbot", className="stat-label"),
        ], className="stat-card"), md=3),
    ], className="g-3", style={"marginBottom":"1.5rem"})

    cat_buttons = html.Div([
        html.Button(cat, id={"type":"cat-btn","index":cat}, className="cat-btn")
        for cat in symptom_categories
    ] + [
        html.Button("🔄 Clear", id="clear-btn", className="cat-btn",
                    style={"borderColor":"rgba(255,77,109,0.3)","color":"#ff4d6d"}),
    ], style={"display":"flex","flexWrap":"wrap","gap":"0.4rem","marginBottom":"1rem"})

    profile_panel = html.Div(className="glass-card", style={"marginBottom":"1rem"}, children=[
        html.Div("Patient Context", className="section-header"),
        dbc.Row([
            dbc.Col([
                html.Label("Age", style={"fontSize":"0.7rem","color":"#4a5568","letterSpacing":"1px"}),
                dcc.Input(id="patient-age", type="number", placeholder="25", className="profile-input",
                          min=1, max=120, style={"marginTop":"0.25rem"}),
            ], md=3),
            dbc.Col([
                html.Label("Gender", style={"fontSize":"0.7rem","color":"#4a5568","letterSpacing":"1px"}),
                dcc.Dropdown(id="patient-gender",
                    options=[{"label":"Male","value":"Male"},{"label":"Female","value":"Female"},{"label":"Other","value":"Other"}],
                    placeholder="Select",
                    style={"marginTop":"0.25rem","fontSize":"0.85rem"}),
            ], md=3),
            dbc.Col([
                html.Label("Duration", style={"fontSize":"0.7rem","color":"#4a5568","letterSpacing":"1px"}),
                dcc.Dropdown(id="symptom-duration",
                    options=[{"label":v,"value":v} for v in ["< 1 day","1–3 days","3–7 days","1–2 weeks","2–4 weeks","> 1 month"]],
                    placeholder="How long?",
                    style={"marginTop":"0.25rem","fontSize":"0.85rem"}),
            ], md=3),
            dbc.Col([
                html.Label("Severity", style={"fontSize":"0.7rem","color":"#4a5568","letterSpacing":"1px"}),
                dcc.Slider(id="pain-scale", min=1, max=10, step=1, value=5,
                           marks={i:{"label":str(i),"style":{"color":"#4a5568","fontSize":"0.7rem"}} for i in [1,5,10]},
                           tooltip={"placement":"bottom","always_visible":True},
                           className="mt-2"),
            ], md=3),
        ]),
    ])

    input_panel = html.Div(className="glass-card", style={"marginBottom":"1rem"}, children=[
        html.Div("Select Symptoms", className="section-header"),
        cat_buttons,
        dcc.Dropdown(
            id="symptom-dropdown",
            options=[{"label": s.replace("_"," ").title(), "value": s} for s in symptoms],
            multi=True, searchable=True,
            placeholder="🔍 Search & select symptoms...",
            style={"borderRadius":"10px"},
        ),
        html.Br(),
        html.Div([
            html.Button("🔬 ANALYZE SYMPTOMS", id="predict-btn", className="predict-btn"),
            html.Span("  "),
            html.Button("✕ Reset", id="reset-btn", className="reset-btn"),
        ], style={"display":"flex","alignItems":"center","gap":"0.75rem","flexWrap":"wrap"}),
    ])

    result_panel = html.Div(id="result-panel", style={"display":"none"}, children=[
        html.Div(className="result-card", style={"marginBottom":"1rem"}, children=[
            html.Div([
                html.Div([
                    html.Div("PRIMARY DIAGNOSIS", style={"fontSize":"0.62rem","color":"#4a5568","fontWeight":"700",
                                                          "letterSpacing":"2px","marginBottom":"0.4rem"}),
                    html.Div(id="res-disease", className="result-disease-name"),
                    html.Div(id="res-icd", style={"fontSize":"0.72rem","color":"#4a5568","fontFamily":"'JetBrains Mono',monospace","marginTop":"0.25rem"}),
                    html.Div(id="res-severity-badge", style={"marginTop":"0.6rem"}),
                    html.Div(className="confidence-bar-wrap", style={"marginTop":"1rem"}, children=[
                        html.Div(id="res-conf-bar", className="confidence-bar-fill", style={"width":"0%"}),
                    ]),
                    html.Div(id="res-conf-text", style={"fontSize":"0.75rem","color":"#4a5568","marginTop":"0.3rem"}),
                    html.Div(id="res-specialist", style={"marginTop":"0.75rem","fontSize":"0.78rem","color":"#94a3b8"}),
                ], style={"flex":"1","minWidth":"220px"}),

                html.Div(style={"width":"1px","background":"rgba(0,240,255,0.1)","margin":"0 1.5rem"}),

                html.Div([
                    html.Div(id="res-pills"),
                ], style={"flex":"1","minWidth":"220px"}),
            ], style={"display":"flex","gap":"0","flexWrap":"wrap"}),
        ]),

        dbc.Row([
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="bar-graph", config={"displayModeBar":False}, style={"height":"280px"}),
            ]), md=6),
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="pie-graph", config={"displayModeBar":False}, style={"height":"280px"}),
            ]), md=6),
        ], className="g-3", style={"marginBottom":"1rem"}),

        dbc.Row([
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="radar-graph", config={"displayModeBar":False}, style={"height":"280px"}),
            ]), md=6),
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="symptom-impact-graph", config={"displayModeBar":False}, style={"height":"280px"}),
            ]), md=6),
        ], className="g-3", style={"marginBottom":"1rem"}),

        html.Div(className="glass-card", children=[
            html.Div("Differential Diagnosis — Top Candidates", className="section-header"),
            html.Div(id="top5-table-div"),
        ]),
    ])

    return dbc.Container([
        html.Br(),
        demo_banner,
        html.Div([
            html.Div("Disease Prediction Engine", className="page-title"),
            html.Div("AI-powered multi-disease differential diagnosis from symptom patterns", className="page-subtitle"),
        ], style={"marginBottom":"1.5rem"}),
        stats_row,
        profile_panel,
        input_panel,
        result_panel,
        dcc.Store(id="selected-symptoms-store", data=[]),
    ], fluid=True, style={"padding":"0 2rem 3rem"})

def build_analytics_page():
    return dbc.Container([
        html.Br(),
        html.Div([
            html.Div("Model Analytics", className="page-title"),
            html.Div("Deep performance insights, feature analysis & disease distributions", className="page-subtitle"),
        ], style={"marginBottom":"1.5rem"}),

        dbc.Row([
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="conf-matrix-graph", config={"displayModeBar":False}, style={"height":"420px"}),
            ]), md=6),
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="feat-importance-graph", config={"displayModeBar":False}, style={"height":"420px"}),
            ]), md=6),
        ], className="g-3", style={"marginBottom":"1rem"}),

        dbc.Row([
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="disease-dist-graph", config={"displayModeBar":False}, style={"height":"360px"}),
            ]), md=8),
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="precision-recall-graph", config={"displayModeBar":False}, style={"height":"360px"}),
            ]), md=4),
        ], className="g-3", style={"marginBottom":"1rem"}),

        dbc.Row([
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="symptom-heatmap-graph", config={"displayModeBar":False}, style={"height":"380px"}),
            ]), md=12),
        ], className="g-3"),
    ], fluid=True, style={"padding":"0 2rem 3rem"})

def build_reports_page():
    numeric_cols = ["precision","recall","f1-score","support"]
    existing_cols = [c for c in numeric_cols if c in report_df.columns]
    display_df = report_df.copy()
    for col in existing_cols:
        if col != "support":
            try:
                display_df[col] = display_df[col].apply(lambda x: f"{float(x):.3f}" if pd.notnull(x) else "—")
            except:
                pass

    table_rows = []
    for _, row in display_df.iterrows():
        f1_val = row.get("f1-score","0")
        try:
            f1_float = float(f1_val)
        except:
            f1_float = 0
        color = "#00ff9d" if f1_float>=0.9 else ("#ffce00" if f1_float>=0.7 else "#ff4d6d")
        table_rows.append(html.Tr([
            html.Td(row["Class"], style={"fontWeight":"600","color":"#e8f0fe"}),
            *[html.Td(row.get(c,"—"),
                style={"color": color if c=="f1-score" else "#94a3b8",
                       "fontFamily":"'JetBrains Mono',monospace","fontSize":"1rem"})
              for c in existing_cols],
        ]))

    return dbc.Container([
        html.Br(),
        html.Div([
            html.Div("Classification Report", className="page-title"),
            html.Div("Per-class precision, recall and F1 scores across all disease categories", className="page-subtitle"),
        ], style={"marginBottom":"1.5rem"}),

        html.Div(className="glass-card", style={"marginBottom":"1rem"}, children=[
            html.Table([
                html.Thead(html.Tr([html.Th("Class")] + [html.Th(c.title()) for c in existing_cols])),
                html.Tbody(table_rows),
            ], className="custom-table", style={"width":"100%"}),
        ]),

        dbc.Row([
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="f1-bar-graph", config={"displayModeBar":False}, style={"height":"380px"}),
            ]), md=6),
            dbc.Col(html.Div(className="glass-card", style={"padding":"1rem"}, children=[
                dcc.Graph(id="support-pie-graph", config={"displayModeBar":False}, style={"height":"380px"}),
            ]), md=6),
        ], className="g-3"),
    ], fluid=True, style={"padding":"0 2rem 3rem"})

def build_history_page():
    return dbc.Container([
        html.Br(),
        html.Div([
            html.Div("Session History", className="page-title"),
            html.Div("Your prediction history for this session", className="page-subtitle"),
        ], style={"marginBottom":"1.5rem"}),
        html.Div(id="history-list-div", className="glass-card", style={"minHeight":"300px"},
                 children=[
                     html.Div("No predictions yet. Go to Diagnose to get started.",
                              style={"color":"#4a5568","textAlign":"center","padding":"3rem","fontStyle":"italic"})
                 ]),
    ], fluid=True, style={"padding":"0 2rem 3rem"})

def build_profile_page():
    return dbc.Container([
        html.Br(),
        html.Div([
            html.Div("User Profile", className="page-title"),
            html.Div("Manage your account and view usage stats", className="page-subtitle"),
        ], style={"marginBottom":"1.5rem"}),

        dbc.Row([
            dbc.Col(html.Div(className="glass-card", children=[
                html.Div("Account Info", className="section-header"),
                html.Div(id="profile-info-display"),
            ]), md=5),
            dbc.Col(html.Div(className="glass-card", children=[
                html.Div("Session Stats", className="section-header"),
                dbc.Row([
                    dbc.Col(html.Div([html.Div("0", className="stat-number", id="stat-total-preds",
                                               style={"color":"#00f0ff","fontSize":"1.5rem"}),
                                      html.Div("Predictions", className="stat-label")], className="stat-card"), md=6),
                    dbc.Col(html.Div([html.Div(datetime.now().strftime("%d %b"), className="stat-number",
                                               style={"color":"#bf5fff","fontSize":"1.3rem","fontFamily":"'JetBrains Mono',monospace"}),
                                      html.Div("Today", className="stat-label")], className="stat-card"), md=6),
                ], className="g-2"),
            ]), md=7),
        ], className="g-3"),

        html.Br(),
        html.Div(className="glass-card", children=[
            html.Div("About MediScan AI", className="section-header"),
            html.P("MediScan AI is a multi-disease prediction system using Random Forest machine learning combined with GPT-4 AI assistance. It analyzes symptom patterns to generate differential diagnoses with confidence scores.",
                   style={"color":"#64748b","lineHeight":"1.7","fontSize":"0.88rem"}),
            html.P("⚠️ Disclaimer: This tool is for educational and informational purposes only. It does not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.",
                   style={"color":"#ff4d6d","fontSize":"0.78rem","marginTop":"0.75rem",
                          "background":"rgba(255,77,109,0.06)","border":"1px solid rgba(255,77,109,0.2)",
                          "borderRadius":"8px","padding":"0.75rem"}),
        ]),
    ], fluid=True, style={"padding":"0 2rem 3rem"})

app.layout = html.Div([
    dcc.Store(id="session-user", data=None, storage_type="session"),
    dcc.Store(id="active-page", data="predict"),
    dcc.Store(id="global-history", data=[]),
    dcc.Location(id="url", refresh=False),
    html.Div(id="app-root"),
])


# ══════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════

@app.callback(
    Output("app-root","children"),
    Input("session-user","data"),
)
def render_root(user):
    if not user:
        return build_login_page()
    return html.Div([
        build_navbar(user),
        html.Div(id="page-content"),
        build_chatbot(),
    ])

@app.callback(
    [Output("session-user","data"),
     Output("login-error","children")],
    Input("login-btn","n_clicks"),
    [State("login-username","value"),
     State("login-password","value")],
    prevent_initial_call=True,
)
def do_login(n, username, password):
    if not username or not password:
        return None, "⚠ Please fill in both fields."
    username = username.strip().lower()
    user = get_user_from_db(username)
    if user and user.get("password") == hash_pw(password or ""):
        return username, ""
    return None, "⚠ Invalid username or password."

@app.callback(
    Output("session-user","data", allow_duplicate=True),
    Input("logout-btn","n_clicks"),
    prevent_initial_call=True,
)
def do_logout(n):
    return None

@app.callback(
    [Output("auth-form-content","style"),
     Output("register-form-content","style"),
     Output("login-tab-btn","className"),
     Output("register-tab-btn","className"),
     Output("login-error","children",allow_duplicate=True),
     Output("reg-error","children",allow_duplicate=True)],
    [Input("login-tab-btn","n_clicks"),
     Input("register-tab-btn","n_clicks")],
    prevent_initial_call=True,
)
def switch_auth_tab(login_clicks, reg_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return {"display":"block"}, {"display":"none"}, "tab-btn active", "tab-btn", "", ""
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_id == "login-tab-btn":
        return {"display":"block"}, {"display":"none"}, "tab-btn active", "tab-btn", "", ""
    else:
        return {"display":"none"}, {"display":"block"}, "tab-btn", "tab-btn active", "", ""

@app.callback(
    [Output("reg-error","children"),
     Output("reg-name","value"),
     Output("reg-username","value"),
     Output("reg-password","value")],
    Input("reg-btn","n_clicks"),
    [State("reg-name","value"), State("reg-username","value"), State("reg-password","value")],
    prevent_initial_call=True,
)
def do_register(n, name, username, password):
    if not all([name, username, password]):
        return "⚠ Please fill all fields.", "", "", ""
    
    username_clean = username.strip().lower()
    success, message = add_user_to_db(username_clean, name, password)
    
    if success:
        return f"✅ {message} Please sign in with your credentials.", "", "", ""
    else:
        return f"⚠ {message}", name, username, password

@app.callback(
    Output("active-page","data"),
    [Input("nav-predict","n_clicks"), Input("nav-analytics","n_clicks"),
     Input("nav-reports","n_clicks"),  Input("nav-history","n_clicks"),
     Input("nav-profile","n_clicks")],
    prevent_initial_call=True,
)
def route(p, a, r, h, pr):
    ctx = callback_context
    if not ctx.triggered: return "predict"
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    return {"nav-predict":"predict","nav-analytics":"analytics","nav-reports":"reports",
            "nav-history":"history","nav-profile":"profile"}.get(btn, "predict")

@app.callback(
    [Output("page-content","children"),
     Output("nav-predict","className"), Output("nav-analytics","className"),
     Output("nav-reports","className"), Output("nav-history","className"),
     Output("nav-profile","className")],
    Input("active-page","data"),
)
def render_page(page):
    cls = {k: "tab-btn active" if page==k else "tab-btn"
           for k in ["predict","analytics","reports","history","profile"]}
    pages = {"predict":build_prediction_page,"analytics":build_analytics_page,
             "reports":build_reports_page,"history":build_history_page,"profile":build_profile_page}
    content = pages.get(page, build_prediction_page)()
    return content, cls["predict"], cls["analytics"], cls["reports"], cls["history"], cls["profile"]

@app.callback(
    Output("selected-symptoms-store","data"),
    [Input("symptom-dropdown","value"), Input("reset-btn","n_clicks"), Input("clear-btn","n_clicks")],
    prevent_initial_call=False,
)
def update_store(values, reset, clear):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if "reset-btn" in triggered or "clear-btn" in triggered:
        return []
    return values or []

@app.callback(
    Output("symptom-dropdown","value"),
    [Input("reset-btn","n_clicks"), Input("clear-btn","n_clicks")],
    prevent_initial_call=True,
)
def reset_dropdown(r, c):
    return []

# ══════════════════════════════════════════════════════════════
#  PREDICTION CALLBACK — FIXED for determinism
# ══════════════════════════════════════════════════════════════
@app.callback(
    [Output("result-panel","style"),
     Output("res-disease","children"),
     Output("res-icd","children"),
     Output("res-conf-bar","style"),
     Output("res-conf-text","children"),
     Output("res-severity-badge","children"),
     Output("res-specialist","children"),
     Output("res-pills","children"),
     Output("bar-graph","figure"),
     Output("pie-graph","figure"),
     Output("top5-table-div","children"),
     Output("radar-graph","figure"),
     Output("symptom-impact-graph","figure"),
     Output("global-history","data"),
     ],
    Input("predict-btn","n_clicks"),
    [State("selected-symptoms-store","data"),
     State("global-history","data"),
     State("patient-age","value"),
     State("patient-gender","value"),
     State("symptom-duration","value"),
     State("pain-scale","value")],
    prevent_initial_call=True,
)
def predict(n_clicks, selected_symptoms, history, age, gender, duration, pain_scale):
    empty_fig = dark_fig(go.Figure())
    hidden = {"display":"none"}
    if not selected_symptoms:
        return hidden, "", "", {"width":"0%"}, "", "", "", "", empty_fig, empty_fig, [], empty_fig, empty_fig, history or []

    # ── Build input vector — order must exactly match training columns ──
    user_input = np.array([1 if s in selected_symptoms else 0 for s in symptoms], dtype=float)
    input_arr = user_input.reshape(1, -1)

    # ── Predict — fully deterministic for same symptom input ──
    prediction = model.predict(input_arr)
    probability = model.predict_proba(input_arr)[0]

    disease = le.inverse_transform(prediction)[0]
    confidence = round(float(np.max(probability)) * 100, 1)

    info = medicine_dict.get(disease, {
        "medicine": "Consult a qualified physician",
        "diet": "Balanced nutrition, stay hydrated",
        "rest": "As advised by doctor",
        "severity": "Unknown",
        "specialist": "General Physician",
        "icd": "N/A"
    })
    severity  = info.get("severity","Unknown")
    sev_color = severity_colors.get(severity, "#64748b")
    icd_code  = info.get("icd","N/A")
    specialist = info.get("specialist","General Physician")

    if confidence >= 80:
        conf_grad = "linear-gradient(90deg,#ff4d6d,#ff6b35)"
    elif confidence >= 55:
        conf_grad = "linear-gradient(90deg,#ffce00,#ff6b35)"
    else:
        conf_grad = "linear-gradient(90deg,#00f0ff,#bf5fff)"

    conf_bar_style = {"width":f"{confidence}%","height":"100%","borderRadius":"100px",
                      "background":conf_grad,"transition":"width 1s cubic-bezier(0.34,1.56,0.64,1)"}

    sev_badge = html.Span(f"⚠ {severity}", className="severity-badge", style={
        "background":f"{sev_color}15","border":f"1px solid {sev_color}40","color":sev_color})

    pills = html.Div([
        html.Div(className="info-pill", style={"marginBottom":"0.5rem"}, children=[
            html.Span("💊", style={"fontSize":"1.1rem"}),
            html.Div([html.Div("Recommended Treatment", className="info-pill-label"),
                      html.Div(info.get("medicine","—"), className="info-pill-value")])
        ]),
        html.Div(className="info-pill", style={"marginBottom":"0.5rem"}, children=[
            html.Span("🥗", style={"fontSize":"1.1rem"}),
            html.Div([html.Div("Diet Advice", className="info-pill-label"),
                      html.Div(info.get("diet","—"), className="info-pill-value")])
        ]),
        html.Div(className="info-pill", children=[
            html.Span("🛏️", style={"fontSize":"1.1rem"}),
            html.Div([html.Div("Rest Period", className="info-pill-label"),
                      html.Div(info.get("rest","—"), className="info-pill-value")])
        ]),
    ])

    top_idx = np.argsort(probability)[-5:][::-1]
    top_diseases = le.inverse_transform(top_idx)
    top_probs = probability[top_idx]

    bar_fig = go.Figure(go.Bar(
        x=list(top_diseases), y=list(top_probs * 100),
        marker=dict(color=CHART_COLORS[:len(top_diseases)],
                    line=dict(color="rgba(0,240,255,0.3)", width=1)),
        text=[f"{p*100:.1f}%" for p in top_probs],
        textposition="outside", textfont=dict(size=10, color="#94a3b8"),
    ))
    bar_fig.update_layout(title=dict(text="Top Differential Diagnoses", font=dict(size=12, color="#64748b")),
                          yaxis_title="Probability (%)", showlegend=False)
    dark_fig(bar_fig)

    pie_fig = go.Figure(go.Pie(
        labels=list(top_diseases), values=list(top_probs * 100),
        hole=0.55,
        marker=dict(colors=CHART_COLORS[:len(top_diseases)],
                    line=dict(color="#060a12", width=2)),
        textinfo="label+percent", textfont=dict(size=9),
    ))
    pie_fig.update_layout(title=dict(text="Probability Distribution", font=dict(size=12, color="#64748b")),
                          showlegend=False)
    dark_fig(pie_fig)

    rank_icons = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    rows = []
    for i, (d, p) in enumerate(zip(top_diseases, top_probs)):
        d_info = medicine_dict.get(d, {})
        d_sev = d_info.get("severity","—")
        d_sev_c = severity_colors.get(d_sev,"#64748b")
        rows.append(html.Tr([
            html.Td(rank_icons[i], style={"fontSize":"1rem"}),
            html.Td(d, style={"fontWeight":"600","color":"#e8f0fe"}),
            html.Td([
                html.Div(style={"width":f"{p*100:.0f}%","height":"4px","borderRadius":"2px",
                                "background":f"linear-gradient(90deg,{CHART_COLORS[i]},{CHART_COLORS[(i+1)%len(CHART_COLORS)]})",
                                "display":"inline-block","verticalAlign":"middle","minWidth":"4px"}),
                html.Span(f" {p*100:.1f}%", style={"fontSize":"0.75rem","color":"#64748b",
                                                    "marginLeft":"0.4rem","fontFamily":"'JetBrains Mono',monospace"}),
            ]),
            html.Td(html.Span(d_sev, style={"color":d_sev_c,"fontSize":"0.72rem","fontWeight":"600"})),
        ]))

    table = html.Table([
        html.Thead(html.Tr([html.Th("Rank"), html.Th("Disease"), html.Th("Confidence"), html.Th("Severity")])),
        html.Tbody(rows),
    ], className="custom-table", style={"width":"100%"})

    n_sel = min(8, len(selected_symptoms))
    rl = [s.replace("_"," ").title() for s in selected_symptoms[:n_sel]] + [selected_symptoms[0].replace("_"," ").title()]
    rv = [1.0] * n_sel + [1.0]
    radar_fig = go.Figure(go.Scatterpolar(
        r=rv, theta=rl, fill="toself",
        fillcolor="rgba(0,240,255,0.07)",
        line=dict(color="#00f0ff", width=2),
        marker=dict(color="#00f0ff", size=5),
    ))
    radar_fig.update_layout(
        title=dict(text="Symptom Profile", font=dict(size=12, color="#64748b")),
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   angularaxis=dict(linecolor=C["border"], gridcolor=C["border"], tickfont=dict(size=8)),
                   radialaxis=dict(visible=False, range=[0,1])),
        showlegend=False,
    )
    dark_fig(radar_fig)

    sym_imp = {s: importances[symptoms.index(s)] for s in selected_symptoms if s in symptoms}
    if sym_imp:
        sym_df = pd.DataFrame(list(sym_imp.items()), columns=["Symptom","Importance"]).sort_values("Importance")
        impact_fig = go.Figure(go.Bar(
            x=sym_df["Importance"],
            y=[s.replace("_"," ").title() for s in sym_df["Symptom"]],
            orientation="h",
            marker=dict(color=sym_df["Importance"],
                        colorscale=[[0,"#1a2d4a"],[0.5,"#bf5fff"],[1,"#00f0ff"]], showscale=False),
            text=[f"{v:.4f}" for v in sym_df["Importance"]], textposition="outside",
        ))
        impact_fig.update_layout(
            title=dict(text="Feature Importance of Selected Symptoms", font=dict(size=12, color="#64748b")),
            xaxis_title="Importance Score",
        )
        dark_fig(impact_fig)
    else:
        impact_fig = empty_fig

    history = history or []
    context_str = ""
    if age: context_str += f"Age {age}"
    if gender: context_str += f", {gender}"
    if duration: context_str += f", {duration}"
    history.insert(0, {
        "disease": disease, "confidence": confidence,
        "symptoms": selected_symptoms[:5], "all_symptoms": selected_symptoms,
        "time": datetime.now().strftime("%H:%M:%S"),
        "severity": severity, "context": context_str,
        "top5": [{"disease": str(d), "prob": float(p)} for d, p in zip(top_diseases, top_probs)],
    })
    history = history[:30]

    icd_display = f"ICD-10: {icd_code}" if icd_code != "N/A" else ""
    specialist_display = html.Div([
        html.Span("🏥 Recommended Specialist: ", style={"color":"#4a5568","fontSize":"0.78rem"}),
        html.Span(specialist, style={"color":"#00f0ff","fontSize":"0.78rem","fontWeight":"600"}),
    ])

    return (
        {"display":"block"},
        disease, icd_display,
        conf_bar_style,
        f"Confidence: {confidence}%  ·  {len(selected_symptoms)} symptom(s) analyzed",
        sev_badge, specialist_display, pills,
        bar_fig, pie_fig, table,
        radar_fig, impact_fig,
        history,
    )

@app.callback(
    [Output("conf-matrix-graph","figure"),
     Output("feat-importance-graph","figure"),
     Output("disease-dist-graph","figure"),
     Output("precision-recall-graph","figure"),
     Output("symptom-heatmap-graph","figure")],
    Input("active-page","data"),
)
def update_analytics(page):
    cm_fig = go.Figure(go.Heatmap(
        z=cm, x=list(le.classes_), y=list(le.classes_),
        colorscale=[[0,"#060a12"],[0.4,"#1a2d4a"],[0.7,"#bf5fff"],[1,"#00f0ff"]],
        text=cm, texttemplate="%{text}", textfont=dict(size=7),
        showscale=True, colorbar=dict(thickness=10),
    ))
    cm_fig.update_layout(title="Confusion Matrix",
                         xaxis=dict(tickangle=-45, tickfont=dict(size=7)),
                         yaxis=dict(tickfont=dict(size=7)))
    dark_fig(cm_fig)

    fi_fig = go.Figure(go.Bar(
        x=feat_df["Importance"][::-1].values,
        y=[f.replace("_"," ").title() for f in feat_df["Feature"]][::-1],
        orientation="h",
        marker=dict(color=feat_df["Importance"][::-1].values,
                    colorscale=[[0,"#1a2d4a"],[0.5,"#bf5fff"],[1,"#00f0ff"]], showscale=False),
        text=[f"{v:.4f}" for v in feat_df["Importance"][::-1].values], textposition="outside",
    ))
    fi_fig.update_layout(title="Top Feature Importances")
    dark_fig(fi_fig)

    if "prognosis" in df.columns:
        dist = df["prognosis"].value_counts()
        dist_fig = go.Figure(go.Bar(
            x=list(dist.index), y=list(dist.values),
            marker=dict(color=list(dist.values),
                        colorscale=[[0,"#1a2d4a"],[0.5,"#bf5fff"],[1,"#00f0ff"]], showscale=False),
            text=list(dist.values), textposition="outside",
        ))
        dist_fig.update_layout(title="Training Data: Disease Distribution",
                               xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
        dark_fig(dist_fig)
    else:
        dist_fig = dark_fig(go.Figure())

    pr_df = report_df[~report_df["Class"].isin(["accuracy","macro avg","weighted avg"])].copy()
    try:
        pr_df["precision"] = pd.to_numeric(pr_df["precision"], errors="coerce")
        pr_df["recall"] = pd.to_numeric(pr_df["recall"], errors="coerce")
        pr_df["f1-score"] = pd.to_numeric(pr_df.get("f1-score", 0), errors="coerce")
        pr_fig = go.Figure(go.Scatter(
            x=pr_df["recall"], y=pr_df["precision"],
            mode="markers+text", text=pr_df["Class"],
            textposition="top center", textfont=dict(size=7),
            marker=dict(size=10, color=pr_df["f1-score"],
                        colorscale=[[0,"#ff4d6d"],[0.5,"#ffce00"],[1,"#00ff9d"]],
                        showscale=True, colorbar=dict(title="F1", thickness=10)),
        ))
        pr_fig.update_layout(title="Precision vs Recall",
                             xaxis=dict(range=[0,1.1]), yaxis=dict(range=[0,1.1]))
        dark_fig(pr_fig)
    except:
        pr_fig = dark_fig(go.Figure())

    top_syms = feat_df["Feature"].head(15).tolist()
    top_dis_list = list(le.classes_)[:15]
    hm_data = []
    for dis in top_dis_list:
        row_vals = []
        for sym in top_syms:
            if "prognosis" in df.columns and sym in df.columns:
                sub = df[df["prognosis"]==dis][sym].mean() if len(df[df["prognosis"]==dis]) > 0 else 0
                row_vals.append(float(sub) if not np.isnan(sub) else 0)
            else:
                row_vals.append(np.random.RandomState(42).uniform(0,1))
        hm_data.append(row_vals)
    hm_fig = go.Figure(go.Heatmap(
        z=hm_data,
        x=[s.replace("_"," ").title() for s in top_syms],
        y=top_dis_list,
        colorscale=[[0,"#060a12"],[0.3,"#1a2d4a"],[0.7,"#bf5fff"],[1,"#00f0ff"]],
        showscale=True, colorbar=dict(title="Freq", thickness=10),
    ))
    hm_fig.update_layout(title="Disease–Symptom Frequency Heatmap",
                         xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                         yaxis=dict(tickfont=dict(size=8)))
    dark_fig(hm_fig)

    return cm_fig, fi_fig, dist_fig, pr_fig, hm_fig

@app.callback(
    [Output("f1-bar-graph","figure"),
     Output("support-pie-graph","figure")],
    Input("active-page","data"),
)
def update_reports_charts(page):
    pr_df = report_df[~report_df["Class"].isin(["accuracy","macro avg","weighted avg"])].copy()
    try:
        pr_df["f1-score"] = pd.to_numeric(pr_df["f1-score"], errors="coerce")
        pr_df = pr_df.dropna(subset=["f1-score"]).sort_values("f1-score")
        colors = ["#00ff9d" if v>=0.9 else "#ffce00" if v>=0.7 else "#ff4d6d" for v in pr_df["f1-score"]]
        f1_fig = go.Figure(go.Bar(
            x=pr_df["f1-score"], y=pr_df["Class"],
            orientation="h", marker_color=colors,
            text=[f"{v:.3f}" for v in pr_df["f1-score"]], textposition="outside",
        ))
        f1_fig.update_layout(title="F1 Score by Disease Class", xaxis=dict(range=[0,1.1]))
        dark_fig(f1_fig)
    except:
        f1_fig = dark_fig(go.Figure())

    try:
        sup_df = report_df[~report_df["Class"].isin(["accuracy","macro avg","weighted avg"])].copy()
        sup_df["support"] = pd.to_numeric(sup_df["support"], errors="coerce").fillna(0)
        pie2 = go.Figure(go.Pie(
            labels=sup_df["Class"].tolist(), values=sup_df["support"].tolist(),
            hole=0.5,
            marker=dict(colors=CHART_COLORS * 4, line=dict(color="#060a12", width=1)),
            textinfo="label", textfont=dict(size=8),
        ))
        pie2.update_layout(title="Training Support Distribution", showlegend=False)
        dark_fig(pie2)
    except:
        pie2 = dark_fig(go.Figure())

    return f1_fig, pie2

@app.callback(
    Output("history-list-div","children"),
    [Input("active-page","data"), Input("global-history","data")],
)
def render_history(page, history):
    if not history:
        return [html.Div("No predictions yet. Go to Diagnose to get started.",
                         style={"color":"#4a5568","textAlign":"center","padding":"3rem","fontStyle":"italic"})]
    items = []
    for i, h in enumerate(history):
        sev_color = severity_colors.get(h.get("severity","Unknown"),"#64748b")
        top5 = h.get("top5",[])
        alt_str = " | ".join([f"{t['disease']} {t['prob']*100:.0f}%" for t in top5[1:3]]) if len(top5)>1 else ""
        items.append(html.Div([
            html.Div([
                html.Span(f"#{i+1}", style={"color":"#4a5568","fontSize":"0.9rem","fontWeight":"700",
                                             "fontFamily":"'JetBrains Mono',monospace","marginRight":"0.75rem"}),
                html.Span(h["disease"], style={"fontWeight":"700","fontSize":"1.1rem","color":"#e8f0fe"}),
                html.Span(f"  {h['confidence']}%",
                          style={"color":"#00f0ff","fontFamily":"'JetBrains Mono',monospace","fontSize":"1rem","fontWeight":"600"}),
                html.Span(h.get("severity",""), style={"fontSize":"0.9rem","color":sev_color,
                                                        "fontWeight":"700","marginLeft":"1rem"}),
                html.Span(h["time"], style={"fontSize":"0.9rem","color":"#4a5568","marginLeft":"auto",
                                             "fontFamily":"'JetBrains Mono',monospace"}),
            ], style={"display":"flex","alignItems":"center","gap":"0.25rem","flexWrap":"wrap"}),
            html.Div([
                html.Span(f"Symptoms: {', '.join(s.replace('_',' ').title() for s in h['symptoms'][:4])}{'…' if len(h.get('all_symptoms',h['symptoms']))>4 else ''}",
                          style={"fontSize":"0.9rem","color":"#4a5568"}),
                html.Span(f"  Alt: {alt_str}" if alt_str else "",
                          style={"fontSize":"0.9rem","color":"#bf5fff","marginLeft":"0.75rem"}),
            ], style={"marginTop":"0.25rem"}),
            html.Div(h.get("context",""),
                     style={"fontSize":"0.9rem","color":"#4a5568","marginTop":"0.15rem","fontStyle":"italic"}) if h.get("context") else html.Span(),
        ], className="history-item fade-in", style={"marginBottom":"0.6rem"}))
    return items

@app.callback(
    [Output("profile-info-display","children"),
     Output("stat-total-preds","children")],
    [Input("active-page","data"), Input("session-user","data")],
    State("global-history","data"),
)
def render_profile(page, username, history):
    if not username:
        return html.Div(), "0"
    user_info = USERS_DB.get(username, {})
    info = html.Div([
        html.Div([html.Span("Name: ", style={"color":"#4a5568","fontSize":"1rem"}),
                  html.Span(user_info.get("name","—"), style={"color":"#e8f0fe","fontWeight":"600","fontSize":"1.5rem"})],
                 style={"marginBottom":"0.5rem"}),
        html.Div([html.Span("Username: ", style={"color":"#4a5568","fontSize":"1rem"}),
                  html.Span(username, style={"color":"#00f0ff","fontFamily":"'JetBrains Mono',monospace","fontSize":"1rem"})],
                 style={"marginBottom":"0.5rem"}),
        html.Div([html.Span("Role: ", style={"color":"#4a5568","fontSize":"1rem"}),
                  html.Span(user_info.get("role","Patient"), style={"color":"#bf5fff","fontWeight":"600","fontSize":"1rem"})],
                 style={"marginBottom":"0.5rem"}),
        html.Div([html.Span("Session: ", style={"color":"#4a5568","fontSize":"1rem"}),
                  html.Span(datetime.now().strftime("%d %b %Y, %H:%M"),
                             style={"color":"#94a3b8","fontFamily":"'JetBrains Mono',monospace","fontSize":"1rem"})]),
    ])
    return info, str(len(history or []))

@app.callback(
    [Output("chat-window","className"),
     Output("chat-visible","data")],
    [Input("chat-toggle","n_clicks"), Input("chat-close","n_clicks")],
    State("chat-visible","data"),
    prevent_initial_call=True,
)
def toggle_chat(open_click, close_click, visible):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if "chat-close" in triggered:
        return "chatbot-window hidden", False
    new_vis = not visible
    cls = "chatbot-window" if new_vis else "chatbot-window hidden"
    return cls, new_vis

@app.callback(
    [Output("chat-messages","children"),
     Output("chat-input","value"),
     Output("chat-history-store","data")],
    Input("chat-send-btn","n_clicks"),
    [State("chat-input","value"),
     State("chat-messages","children"),
     State("chat-history-store","data")],
    prevent_initial_call=True,
)
def send_chat(n_clicks, user_msg, current_msgs, chat_history):
    if not user_msg or not user_msg.strip():
        return current_msgs, "", chat_history or []

    chat_history = chat_history or []
    current_msgs = current_msgs or []

    user_bubble = html.Div([
        html.Strong("You: ", style={"color":"#bf5fff","fontFamily":"'Orbitron',sans-serif","fontSize":"0.65rem"}),
        user_msg.strip()
    ], className="chat-msg-user")
    current_msgs = list(current_msgs) + [user_bubble]

    chat_history.append({"role":"user","content":user_msg.strip()})

    bot_reply = ""
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            messages = [
                {"role":"system","content":(
                    "You are MediBot, an expert AI medical assistant integrated into MediScan AI, "
                    "a disease prediction platform. Help users understand symptoms, diseases, treatments "
                    "and health advice. Be concise (2-4 sentences). Always remind users to consult a real doctor. "
                    "You know about: " + ", ".join(list(medicine_dict.keys()))
                )}
            ] + chat_history[-8:]
            resp = client.chat.completions.create(model="gpt-4", messages=messages, max_tokens=200)
            bot_reply = resp.choices[0].message.content.strip()
        except Exception as e:
            bot_reply = f"⚠ GPT-4 error: {str(e)[:80]}. Please check your API key."
    else:
        # Extract symptoms from user message
        extracted_symptoms = extract_symptoms_from_text(user_msg)
        
        if extracted_symptoms:
            # Build input vector for prediction
            user_input = np.array([1 if s in extracted_symptoms else 0 for s in symptoms], dtype=float)
            input_arr = user_input.reshape(1, -1)
            
            # Predict disease
            prediction = model.predict(input_arr)
            probability = model.predict_proba(input_arr)[0]
            
            disease = le.inverse_transform(prediction)[0]
            confidence = round(float(np.max(probability)) * 100, 1)
            
            info = medicine_dict.get(disease, {
                "medicine": "Consult a qualified physician",
                "diet": "Balanced nutrition, stay hydrated",
                "rest": "As advised by doctor",
                "severity": "Unknown",
                "specialist": "General Physician",
                "icd": "N/A"
            })
            
            severity = info.get("severity", "Unknown")
            specialist = info.get("specialist", "General Physician")
            
            bot_reply = (
                f"Based on symptoms: {', '.join([s.replace('_', ' ') for s in extracted_symptoms])}<br><br>"
                f"**Predicted Disease:** {disease} ({confidence}% confidence)<br>"
                f"**Severity:** {severity}<br>"
                f"**Specialist:** {specialist}<br><br>"
                f"**Treatment:** {info.get('medicine', 'Consult doctor')}<br>"
                f"**Diet:** {info.get('diet', 'Balanced diet')}<br>"
                f"**Rest:** {info.get('rest', 'As advised')}<br><br>"
                "⚠️ This is an AI prediction. Always consult a healthcare professional for accurate diagnosis."
            )
        else:
            bot_reply = (
                f"I understand you're asking about '{user_msg[:40]}'. "
                "For accurate AI responses, please set your OPENAI_API_KEY. "
                "In the meantime, use the Diagnose tab to analyze symptoms with the ML model. "
                "Always consult a healthcare professional for medical advice."
            )

    chat_history.append({"role":"assistant","content":bot_reply})
    chat_history = chat_history[-20:]

    bot_bubble = html.Div([
        html.Strong("MediBot: ", style={"color":"#00f0ff","fontFamily":"'Orbitron',sans-serif","fontSize":"0.65rem"}),
        bot_reply
    ], className="chat-msg-bot")

    updated_msgs = list(current_msgs) + [bot_bubble]
    if len(updated_msgs) > 20:
        updated_msgs = updated_msgs[-20:]

    return updated_msgs, "", chat_history

if __name__ == "__main__":
    print("\n" + "═"*65)
    print("  🧬 MediScan AI v2 — Multi-Disease Prediction + GPT-4 Chatbot")
    print("  URL: http://127.0.0.1:8050")
    print("  Demo login → username: demo  |  password: demo123")
    if not DATA_LOADED:
        print("  ⚠  DEMO MODE — using deterministic symptom-matching (no random)")
    if not OPENAI_API_KEY:
        print("  ⚠  Set OPENAI_API_KEY env var to enable GPT-4 chatbot")
    print("═"*65 + "\n")
    app.run(debug=True, host="0.0.0.0", port=8050)