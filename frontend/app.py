"""
frontend/app.py
--------------------
Real-time diabetes risk monitoring dashboard.
Reads live data from FastAPI → PostgreSQL pipeline.
Supports English and Uzbek (O'zbek) UI languages.
"""

import os
import time
import tempfile
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://fastapi:8000")
DAGSTER_INTERNAL_URL = os.getenv("DAGSTER_INTERNAL_URL", "http://dagster-webserver:3000")
REFRESH_INTERVAL = 60  # seconds

st.set_page_config(
    page_title="Diabetes Risk Monitor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Translations
# ──────────────────────────────────────────────
TRANSLATIONS = {
    "uz": {
        "brand_name": "🩺 Diabet Monitoringi",
        "login_subtitle": "Diabet xavfi monitoringi tizimiga kirish",
        "login_password_label": "Parol",
        "login_button": "🔓 Kirish",
        "login_error": "❌ Parol noto'g'ri",
        "logout_button": "🔒 Chiqish",
        "status_label": "Holat:",
        "status_up": "🟢 Tizim ishlayapti",
        "status_down": "🔴 Tizim ishlamayapti",
        "last_update": "Oxirgi yangilanish:",
        "records_slider": "Yuklanadigan yozuvlar soni",
        "class_filter": "Sinf bo'yicha filtr",
        "filter_all": "Barchasi",
        "filter_pos": "Diabet xavfi bor (1)",
        "filter_neg": "Diabet xavfi yo'q (0)",
        "auto_refresh": "Avtomatik yangilash (60s)",
        "refresh_btn": "🔄 Hozir yangilash",
        "sidebar_info": (
            "📅 Jarayon jadvali:<br>"
            "🕘 Har soatda, 09:00–17:00<br>"
            "Dushanba–Juma"
        ),
        "hero_title": "🩺 Diabet xavfi monitoringi",
        "live_badge": "JONLI",
        "hero_sub": "Real vaqt rejimida monitoring",
        "badge_schedule": "⏱ Har soatda · 09:00–17:00 Dush–Juma",
        "badge_auto": "⚙️ Avtomatik jarayon orqali boshqariladi",
        "no_data": "Ma'lumot topilmadi. Pipeline ishlab turganiga ishonch hosil qiling.",
        "kpi_total": "Jami yozuvlar",
        "kpi_diabetic": "Diabet xavfi (%)",
        "kpi_glucose": "O'rtacha glukoza",
        "kpi_bmi": "O'rtacha BMI",
        "kpi_hba1c": "O'rtacha HbA1c",
        "delta_high": "yuqori",
        "delta_normal": "normal",
        "delta_up_high": "↑ yuqori",
        "delta_obese": "↑ ortiqcha vazn",
        "tab1": "📈 Umumiy ko'rinish",
        "tab2": "🔬 Xususiyatlar tahlili",
        "tab3": "📋 Bemorlar ro'yxati",
        "tab4": "⚙️ Jarayon",
        "tab5": "🧮 Xavfni baholash",
        "class_neg": "Diabet yo'q",
        "class_pos": "Diabet xavfi bor",
        "class_dist_title": "Sinflar taqsimoti",
        "timeline_title": "Soatlar bo'yicha yozuvlar",
        "risk_dist_title": "Xavf darajasi taqsimoti",
        "risk_label": "Xavf darajasi",
        "class_label": "Sinf",
        "boundary_annot": "Chegara (0.5)",
        "risk_gauge_title": "O'rtacha diabet xavfi",
        "feature_dist_title": "Xususiyatlarning natija bo'yicha taqsimoti",
        "yes_no": ["Yo'q", "Bor"],
        "corr_title": "Xususiyatlar korrelyatsiya matritsasi",
        "pearson_title": "Pearson korrelyatsiyasi",
        "scatter_section": "Glukoza va HbA1c nisbati",
        "scatter_title": "Glukoza vs HbA1c (doira o'lchami = xavf darajasi)",
        "patients_title": "So'nggi bemorlar ro'yxati",
        "risk_threshold_label": "Xavf darajasi ≥",
        "age_range_label": "Yosh oralig'i",
        "columns_label": "Ustunlar",
        "records_found": "Jami **{n}** ta yozuv topildi",
        "csv_download": "⬇️ CSV yuklab olish",
        "pipeline_services_title": "Jarayon xizmatlari",
        "status_running": "Ishlayapti",
        "status_stopped": "Ishlamayapti",
        "data_freshness_title": "Ma'lumotlar yangiligi",
        "latest_record": "Oxirgi yozuv: **{ts}** ({m} daqiqa oldin)",
        "hourly_title": "Soatlik yozuvlar (so'nggi 24 partiya)",
        "hour_label": "Soat",
        "records_label": "Yozuvlar",
        "export_title": "Hisobot eksport",
        "export_info": "Joriy asosiy ko'rsatkichlar va grafiklarning PDF ko'rinishini yarating — hamkasblar bilan ulashish uchun qulay.",
        "pdf_generate_btn": "📄 PDF hisobot yaratish",
        "pdf_spinner": "Hisobot tayyorlanmoqda...",
        "pdf_download_btn": "⬇️ PDF hisobotni yuklab olish",
        "footer_line1": "BTEC Level 6 Diploma in Digital Technologies &middot; Independent Project",
        "footer_updated": "Oxirgi yangilanish",
        "predict_title": "Bemor uchun diabet xavfini baholash",
        "predict_intro": "Bemorning ko'rsatkichlarini kiriting",
        "predict_field_age": "Yosh",
        "predict_field_gender": "Jins",
        "predict_gender_male": "Erkak",
        "predict_gender_female": "Ayol",
        "predict_field_bmi": "BMI (tana-vazn indeksi)",
        "predict_field_glucose": "Glukoza (mg/dL)",
        "predict_field_bp": "Qon bosimi (mmHg)",
        "predict_field_hba1c": "HbA1c (%)",
        "predict_field_insulin": "Insulin (μU/mL)",
        "predict_field_skin": "Teri qatlami qalinligi (mm)",
        "predict_field_pregnancies": "Homiladorliklar soni",
        "predict_button": "🧮 Xavfni hisoblash",
        "predict_result_title": "Natija",
        "predict_risk_label": "Diabet xavfi darajasi",
        "predict_class_pos": "⚠️ Yuqori xavf — diabet ehtimoli mavjud",
        "predict_class_neg": "✅ Past xavf — diabet ehtimoli kam",
        "predict_model_used": "Model: {model}",
        "predict_error": "Bashorat olishda xatolik: model hali tayyor emas. Avval ML modelni o'qitish kerak.",
    },
    "en": {
        "brand_name": "🩺 Diabetes Monitor",
        "login_subtitle": "Sign in to the Diabetes Risk Monitoring System",
        "login_password_label": "Password",
        "login_button": "🔓 Sign in",
        "login_error": "❌ Incorrect password",
        "logout_button": "🔒 Sign out",
        "status_label": "Status:",
        "status_up": "🟢 System online",
        "status_down": "🔴 System offline",
        "last_update": "Last updated:",
        "records_slider": "Number of records to load",
        "class_filter": "Filter by class",
        "filter_all": "All",
        "filter_pos": "Diabetic (1)",
        "filter_neg": "Non-diabetic (0)",
        "auto_refresh": "Auto-refresh (60s)",
        "refresh_btn": "🔄 Refresh now",
        "sidebar_info": (
            "📅 Pipeline schedule:<br>"
            "🕘 Hourly, 09:00–17:00<br>"
            "Monday–Friday"
        ),
        "hero_title": "🩺 Diabetes Risk Monitor",
        "live_badge": "LIVE",
        "hero_sub": "Real-time monitoring",
        "badge_schedule": "⏱ Hourly · 09:00–17:00 Mon–Fri",
        "badge_auto": "⚙️ Managed by automated pipeline",
        "no_data": "No data available. Make sure the pipeline is running.",
        "kpi_total": "Total records",
        "kpi_diabetic": "Diabetic (%)",
        "kpi_glucose": "Avg glucose",
        "kpi_bmi": "Avg BMI",
        "kpi_hba1c": "Avg HbA1c",
        "delta_high": "high",
        "delta_normal": "normal",
        "delta_up_high": "↑ high",
        "delta_obese": "↑ obese",
        "tab1": "📈 Overview",
        "tab2": "🔬 Feature Analysis",
        "tab3": "📋 Patient Records",
        "tab4": "⚙️ Pipeline",
        "tab5": "🧮 Risk Calculator",
        "class_neg": "Non-diabetic",
        "class_pos": "Diabetic",
        "class_dist_title": "Class distribution",
        "timeline_title": "Records by hour",
        "risk_dist_title": "Risk score distribution",
        "risk_label": "Risk score",
        "class_label": "Class",
        "boundary_annot": "Threshold (0.5)",
        "risk_gauge_title": "Average diabetes risk",
        "feature_dist_title": "Feature distribution by outcome",
        "yes_no": ["No", "Yes"],
        "corr_title": "Feature correlation matrix",
        "pearson_title": "Pearson correlation",
        "scatter_section": "Glucose vs HbA1c",
        "scatter_title": "Glucose vs HbA1c (bubble size = risk score)",
        "patients_title": "Latest patient records",
        "risk_threshold_label": "Risk score ≥",
        "age_range_label": "Age range",
        "columns_label": "Columns",
        "records_found": "Found **{n}** records",
        "csv_download": "⬇️ Download CSV",
        "pipeline_services_title": "Pipeline services",
        "status_running": "Running",
        "status_stopped": "Offline",
        "data_freshness_title": "Data freshness",
        "latest_record": "Latest record: **{ts}** ({m} min ago)",
        "hourly_title": "Records per hour (last 24 batches)",
        "hour_label": "Hour",
        "records_label": "Records",
        "export_title": "Export report",
        "export_info": "Generate a PDF snapshot of the current key metrics and charts — handy for sharing with stakeholders.",
        "pdf_generate_btn": "📄 Generate PDF report",
        "pdf_spinner": "Building report...",
        "pdf_download_btn": "⬇️ Download PDF report",
        "footer_line1": "BTEC Level 6 Diploma in Digital Technologies &middot; Independent Project",
        "footer_updated": "Last refreshed",
        "predict_title": "Patient Diabetes Risk Assessment",
        "predict_intro": "Enter the patient's measurements — the trained ML model will estimate diabetes risk.",
        "predict_field_age": "Age",
        "predict_field_gender": "Gender",
        "predict_gender_male": "Male",
        "predict_gender_female": "Female",
        "predict_field_bmi": "BMI (Body Mass Index)",
        "predict_field_glucose": "Glucose (mg/dL)",
        "predict_field_bp": "Blood pressure (mmHg)",
        "predict_field_hba1c": "HbA1c (%)",
        "predict_field_insulin": "Insulin (μU/mL)",
        "predict_field_skin": "Skin thickness (mm)",
        "predict_field_pregnancies": "Number of pregnancies",
        "predict_button": "🧮 Calculate risk",
        "predict_result_title": "Result",
        "predict_risk_label": "Diabetes risk score",
        "predict_class_pos": "⚠️ High risk — diabetes likely",
        "predict_class_neg": "✅ Low risk — diabetes unlikely",
        "predict_model_used": "Model: {model}",
        "predict_error": "Could not get a prediction: the model isn't trained yet.",
    },
    "ru": {
        "brand_name": "🩺 Монитор диабета",
        "login_subtitle": "Вход в систему мониторинга риска диабета",
        "login_password_label": "Пароль",
        "login_button": "🔓 Войти",
        "login_error": "❌ Неверный пароль",
        "logout_button": "🔒 Выйти",
        "status_label": "Статус:",
        "status_up": "🟢 Система работает",
        "status_down": "🔴 Система не работает",
        "last_update": "Последнее обновление:",
        "records_slider": "Количество загружаемых записей",
        "class_filter": "Фильтр по классу",
        "filter_all": "Все",
        "filter_pos": "Риск диабета (1)",
        "filter_neg": "Без риска диабета (0)",
        "auto_refresh": "Автообновление (60с)",
        "refresh_btn": "🔄 Обновить сейчас",
        "sidebar_info": (
            "📅 График процесса:<br>"
            "🕘 Ежечасно, 09:00–17:00<br>"
            "Понедельник–пятница"
        ),
        "hero_title": "🩺 Мониторинг риска диабета",
        "live_badge": "ОНЛАЙН",
        "hero_sub": "Мониторинг в реальном времени",
        "badge_schedule": "⏱ Ежечасно · 09:00–17:00 Пн–Пт",
        "badge_auto": "⚙️ Управляется автоматическим процессом",
        "no_data": "Данные не найдены. Убедитесь, что пайплайн запущен.",
        "kpi_total": "Всего записей",
        "kpi_diabetic": "Риск диабета (%)",
        "kpi_glucose": "Средняя глюкоза",
        "kpi_bmi": "Средний ИМТ",
        "kpi_hba1c": "Средний HbA1c",
        "delta_high": "высокий",
        "delta_normal": "норма",
        "delta_up_high": "↑ высокий",
        "delta_obese": "↑ ожирение",
        "tab1": "📈 Обзор",
        "tab2": "🔬 Анализ показателей",
        "tab3": "📋 Список пациентов",
        "tab4": "⚙️ Процесс",
        "tab5": "🧮 Оценка риска",
        "class_neg": "Нет диабета",
        "class_pos": "Риск диабета",
        "class_dist_title": "Распределение классов",
        "timeline_title": "Записи по часам",
        "risk_dist_title": "Распределение уровня риска",
        "risk_label": "Уровень риска",
        "class_label": "Класс",
        "boundary_annot": "Граница (0.5)",
        "risk_gauge_title": "Средний риск диабета",
        "feature_dist_title": "Распределение показателей по результату",
        "yes_no": ["Нет", "Да"],
        "corr_title": "Матрица корреляции показателей",
        "pearson_title": "Корреляция Пирсона",
        "scatter_section": "Глюкоза и HbA1c",
        "scatter_title": "Глюкоза vs HbA1c (размер = уровень риска)",
        "patients_title": "Последние записи пациентов",
        "risk_threshold_label": "Уровень риска ≥",
        "age_range_label": "Возрастной диапазон",
        "columns_label": "Столбцы",
        "records_found": "Найдено **{n}** записей",
        "csv_download": "⬇️ Скачать CSV",
        "pipeline_services_title": "Сервисы процесса",
        "status_running": "Работает",
        "status_stopped": "Не работает",
        "data_freshness_title": "Актуальность данных",
        "latest_record": "Последняя запись: **{ts}** ({m} мин назад)",
        "hourly_title": "Записи по часам (последние 24 партии)",
        "hour_label": "Час",
        "records_label": "Записи",
        "export_title": "Экспорт отчёта",
        "export_info": "Создать PDF-снимок текущих ключевых показателей и графиков — удобно для обмена с коллегами.",
        "pdf_generate_btn": "📄 Создать PDF-отчёт",
        "pdf_spinner": "Подготовка отчёта...",
        "pdf_download_btn": "⬇️ Скачать PDF-отчёт",
        "footer_line1": "BTEC Level 6 Diploma in Digital Technologies &middot; Independent Project",
        "footer_updated": "Последнее обновление",
        "predict_title": "Оценка риска диабета для пациента",
        "predict_intro": "Введите показатели пациента — обученная модель ML рассчитает риск диабета.",
        "predict_field_age": "Возраст",
        "predict_field_gender": "Пол",
        "predict_gender_male": "Мужской",
        "predict_gender_female": "Женский",
        "predict_field_bmi": "ИМТ (индекс массы тела)",
        "predict_field_glucose": "Глюкоза (мг/дЛ)",
        "predict_field_bp": "Давление (мм рт. ст.)",
        "predict_field_hba1c": "HbA1c (%)",
        "predict_field_insulin": "Инсулин (мкЕд/мл)",
        "predict_field_skin": "Толщина кожной складки (мм)",
        "predict_field_pregnancies": "Количество беременностей",
        "predict_button": "🧮 Рассчитать риск",
        "predict_result_title": "Результат",
        "predict_risk_label": "Уровень риска диабета",
        "predict_class_pos": "⚠️ Высокий риск — вероятен диабет",
        "predict_class_neg": "✅ Низкий риск — диабет маловероятен",
        "predict_model_used": "Модель: {model}",
        "predict_error": "Не удалось получить прогноз: модель еще не обучена.",
    },
}

FEATURE_NAMES = {
    "uz": {
        "glucose": "Glukoza", "bmi": "BMI", "hba1c": "HbA1c",
        "blood_pressure": "Qon bosimi", "insulin": "Insulin", "age": "Yosh",
    },
    "en": {
        "glucose": "Glucose", "bmi": "BMI", "hba1c": "HbA1c",
        "blood_pressure": "Blood pressure", "insulin": "Insulin", "age": "Age",
    },
    "ru": {
        "glucose": "Глюкоза", "bmi": "ИМТ", "hba1c": "HbA1c",
        "blood_pressure": "Давление", "insulin": "Инсулин", "age": "Возраст",
    },
}

COLUMN_LABELS = {
    "uz": {
        "id": "ID", "age": "Yosh", "gender": "Jins", "bmi": "BMI",
        "glucose": "Glukoza", "hba1c": "HbA1c", "blood_pressure": "Qon bosimi",
        "diabetes_risk": "Xavf darajasi", "label": "Sinf", "recorded_at": "Vaqt",
    },
    "en": {
        "id": "ID", "age": "Age", "gender": "Gender", "bmi": "BMI",
        "glucose": "Glucose", "hba1c": "HbA1c", "blood_pressure": "Blood pressure",
        "diabetes_risk": "Risk score", "label": "Class", "recorded_at": "Recorded at",
    },
    "ru": {
        "id": "ID", "age": "Возраст", "gender": "Пол", "bmi": "ИМТ",
        "glucose": "Глюкоза", "hba1c": "HbA1c", "blood_pressure": "Давление",
        "diabetes_risk": "Уровень риска", "label": "Класс", "recorded_at": "Время записи",
    },
}

LANG_OPTIONS = {"O'zbek": "uz", "English": "en", "Русский": "ru"}

# Language selector lives at the very top of the sidebar so it applies to everything below.
with st.sidebar:
    lang_choice = st.selectbox("🌐 Til / Language", options=list(LANG_OPTIONS.keys()), index=0)
LANG = LANG_OPTIONS[lang_choice]


def t(key: str) -> str:
    return TRANSLATIONS[LANG][key]


# ──────────────────────────────────────────────
# Custom CSS — modern dark dashboard
# ──────────────────────────────────────────────
st.markdown("""
<style>
  /* Sidebar */
  [data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #1e293b; }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #f1f5f9 !important; }

  /* Global background */
  .stApp {
    background: radial-gradient(1200px 600px at 10% 0%, #16213e 0%, #0a0f1e 45%, #0a0f1e 100%);
    color: #e2e8f0;
  }
  .main .block-container { padding-top: 1.5rem; max-width: 1300px; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(160deg, #1e293b 0%, #161f33 100%);
    border: 1px solid #2d3a52;
    border-radius: 12px;
    padding: 16px !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    transition: transform 0.15s ease, border-color 0.15s ease;
  }
  [data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-color: #38bdf8;
  }
  [data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.8rem; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f8fafc; font-size: 2rem; font-weight: 700; }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.85rem; }

  /* Headers */
  h1 { color: #f8fafc !important; font-size: 1.6rem !important; }
  h2 { color: #e2e8f0 !important; font-size: 1.15rem !important; font-weight: 600 !important; }
  h3 { color: #cbd5e1 !important; }

  hr { border-color: #1e293b; margin: 1.2rem 0; }

  /* Status badge */
  .status-live {
    display: inline-block;
    background: #064e3b;
    color: #6ee7b7;
    border: 1px solid #059669;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-left: 8px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    70%  { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
  }

  /* Risk colors */
  .risk-high { color: #f87171; font-weight: 700; }
  .risk-low  { color: #6ee7b7; font-weight: 700; }

  /* Table */
  [data-testid="stDataFrame"] { border: 1px solid #2d3a52; border-radius: 10px; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    color: #94a3b8; font-weight: 600; padding: 10px 18px;
    border-radius: 10px 10px 0 0; background: #131c30;
  }
  .stTabs [aria-selected="true"] {
    color: #38bdf8 !important; background: #1e293b !important;
    border-bottom: 2px solid #38bdf8 !important;
  }

  /* Expander */
  .streamlit-expanderHeader { color: #94a3b8 !important; }

  /* Info box */
  .info-box {
    background: #132235;
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 10px 14px;
    color: #94a3b8;
    font-size: 0.85rem;
    margin: 8px 0;
  }

  /* Hero header */
  .hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 55%, #0f172a 100%);
    border: 1px solid #2d3a52;
    border-radius: 16px;
    padding: 26px 30px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
    box-shadow: 0 8px 30px rgba(30,58,138,0.25);
  }
  .hero h1 {
    margin: 0 !important;
    font-size: 2.1rem !important;
    background: linear-gradient(90deg, #f8fafc, #93c5fd, #38bdf8);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
  }
  .hero p {
    margin: 6px 0 0 0;
    color: #94a3b8;
    font-size: 0.92rem;
  }
  .hero-badges { display: flex; gap: 10px; flex-wrap: wrap; }

  .pipe-badge {
    display: inline-block;
    background: #1e293b;
    color: #93c5fd;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.03em;
  }

  /* Footer */
  .app-footer {
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid #1e293b;
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    line-height: 1.7;
  }
  .app-footer a { color: #38bdf8; text-decoration: none; font-weight: 600; }

  .svc-up   { color: #6ee7b7; font-weight: 700; }
  .svc-down { color: #f87171; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Login gate
# ──────────────────────────────────────────────
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "diabetes2024")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        f"<h1 style='text-align:center; margin-top:3rem;'>{t('brand_name')}</h1>"
        f"<p style='text-align:center; color:#94a3b8;'>{t('login_subtitle')}</p>",
        unsafe_allow_html=True,
    )
    _, login_col, _ = st.columns([1, 1, 1])
    with login_col:
        with st.form("login_form"):
            pwd = st.text_input(t("login_password_label"), type="password")
            submitted = st.form_submit_button(t("login_button"), use_container_width=True)
            if submitted:
                if pwd == DASHBOARD_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error(t("login_error"))
    st.stop()


# ──────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_patients(limit: int = 500, label: int | None = None) -> pd.DataFrame:
    try:
        params = {"limit": limit}
        if label is not None:
            params["label"] = label
        r = requests.get(f"{API_BASE}/patients", params=params, timeout=5)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        return df
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_stats() -> dict:
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Stats API error: {e}")
        return {}


def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────
# Plot helpers — dark theme
# ──────────────────────────────────────────────
CHART_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#1e293b",
    "font": {"color": "#94a3b8", "family": "Inter, sans-serif", "size": 12},
    "xaxis": {"gridcolor": "#334155", "linecolor": "#334155"},
    "yaxis": {"gridcolor": "#334155", "linecolor": "#334155"},
    "margin": {"t": 36, "r": 12, "b": 32, "l": 40},
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
}

POS_COLOR = "#f87171"   # red — positive
NEG_COLOR = "#6ee7b7"   # green — negative


def apply_theme(fig):
    fig.update_layout(**CHART_THEME)
    return fig


def risk_gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": "#f8fafc"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": "#38bdf8"},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 35],  "color": "#064e3b"},
                {"range": [35, 60], "color": "#713f12"},
                {"range": [60, 100],"color": "#450a0a"},
            ],
            "threshold": {
                "line": {"color": "#f87171", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
        title={"text": title, "font": {"color": "#94a3b8", "size": 14}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=240,
                      margin={"t": 20, "r": 10, "b": 10, "l": 10})
    return fig


# ──────────────────────────────────────────────
# PDF report
# ──────────────────────────────────────────────
def build_pdf_report(df: pd.DataFrame, stats: dict) -> bytes:
    """Build a summary PDF report with KPIs and key charts (light theme for print)."""
    light_theme = {
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"color": "#1e293b", "family": "Helvetica, sans-serif", "size": 12},
        "xaxis": {"gridcolor": "#e2e8f0", "linecolor": "#cbd5e1"},
        "yaxis": {"gridcolor": "#e2e8f0", "linecolor": "#cbd5e1"},
        "margin": {"t": 48, "r": 16, "b": 40, "l": 48},
    }

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Diabetes Risk Dashboard - Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── KPI table ──
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Key Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    kpis = {
        "Total records": f"{stats.get('total_records', len(df)):,}",
        "Diabetic %": f"{stats.get('diabetic_percentage', round(df['label'].mean() * 100, 1))}%",
        "Avg glucose (mg/dL)": stats.get("avg_glucose", round(df["glucose"].mean(), 1)),
        "Avg BMI": stats.get("avg_bmi", round(df["bmi"].mean(), 1)),
        "Avg HbA1c (%)": stats.get("avg_hba1c", round(df["hba1c"].mean(), 2)),
    }
    for k, v in kpis.items():
        pdf.cell(70, 8, k, border=1)
        pdf.cell(0, 8, str(v), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Charts ──
    label_counts = df["label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]
    label_counts["class"] = label_counts["label"].map({0: "Negative", 1: "Positive"})
    fig_pie = px.pie(
        label_counts, values="count", names="class",
        color="class",
        color_discrete_map={"Positive": "#ef4444", "Negative": "#10b981"},
        hole=0.5, title="Class Distribution",
    )
    fig_pie.update_layout(**light_theme)

    fig_hist = px.histogram(
        df, x="diabetes_risk", color="label",
        color_discrete_map={0: "#10b981", 1: "#ef4444"},
        nbins=30, barmode="overlay", opacity=0.75,
        title="Risk Score Distribution",
        labels={"diabetes_risk": "Risk Score", "label": "Class"},
    )
    fig_hist.update_layout(**light_theme)

    num_cols = ["glucose", "bmi", "hba1c", "blood_pressure", "insulin", "age", "diabetes_risk", "label"]
    corr = df[num_cols].corr()
    fig_corr = px.imshow(
        corr, color_continuous_scale="RdBu_r", aspect="auto",
        title="Feature Correlation Matrix", zmin=-1, zmax=1,
    )
    fig_corr.update_layout(**light_theme, height=400)

    for fig in (fig_pie, fig_hist, fig_corr):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.write_image(tmp.name, format="png", width=700, height=400, scale=2)
            pdf.image(tmp.name, x=10, w=190)
        os.unlink(tmp.name)
        pdf.ln(2)

    return bytes(pdf.output())


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## {t('brand_name')}")
    st.markdown("---")

    alive = api_health()
    status_text = t("status_up") if alive else t("status_down")
    st.markdown(f"**{t('status_label')}** {status_text}")
    st.markdown(f"**{t('last_update')}** {datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")

    record_limit = st.slider(t("records_slider"), 100, 5000, 1000, step=100)
    filter_label = st.selectbox(
        t("class_filter"),
        options=[t("filter_all"), t("filter_pos"), t("filter_neg")],
    )
    label_map = {t("filter_all"): None, t("filter_pos"): 1, t("filter_neg"): 0}
    selected_label = label_map[filter_label]

    st.markdown("---")
    auto_refresh = st.checkbox(t("auto_refresh"), value=False)
    if st.button(t("refresh_btn"), use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.78rem; color:#64748b; line-height:1.6'>{t('sidebar_info')}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button(t("logout_button"), use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.cache_data.clear()
    st.rerun()


# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <div>
        <h1>{t('hero_title')} <span class="status-live">{t('live_badge')}</span></h1>
        <p>{t('hero_sub')}</p>
      </div>
      <div class="hero-badges">
        <span class="pipe-badge">{t('badge_schedule')}</span>
        <span class="pipe-badge">{t('badge_auto')}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

df = fetch_patients(limit=record_limit, label=selected_label)
stats = fetch_stats()

if df.empty:
    st.warning(t("no_data"))
    st.stop()

# ── KPI row ──────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

total = stats.get("total_records", len(df))
diabetic_pct = stats.get("diabetic_percentage", round(df["label"].mean() * 100, 1))
avg_glucose = stats.get("avg_glucose", round(df["glucose"].mean(), 1))
avg_bmi = stats.get("avg_bmi", round(df["bmi"].mean(), 1))
avg_hba1c = stats.get("avg_hba1c", round(df["hba1c"].mean(), 2))

k1.metric(t("kpi_total"), f"{total:,}")
k2.metric(t("kpi_diabetic"), f"{diabetic_pct}%",
          delta=t("delta_high") if diabetic_pct > 40 else t("delta_normal"),
          delta_color="inverse")
k3.metric(t("kpi_glucose"), f"{avg_glucose} mg/dL",
          delta=t("delta_up_high") if avg_glucose > 140 else t("delta_normal"),
          delta_color="inverse" if avg_glucose > 140 else "normal")
k4.metric(t("kpi_bmi"), f"{avg_bmi}",
          delta=t("delta_obese") if avg_bmi > 30 else t("delta_normal"),
          delta_color="inverse" if avg_bmi > 30 else "normal")
k5.metric(t("kpi_hba1c"), f"{avg_hba1c}%",
          delta=t("delta_up_high") if avg_hba1c > 6.5 else t("delta_normal"),
          delta_color="inverse" if avg_hba1c > 6.5 else "normal")

st.markdown("---")

# ── Tabs ─────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([t("tab1"), t("tab2"), t("tab3"), t("tab4"), t("tab5")])

# ═══════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════
with tab1:
    col_gauge, col_pie, col_timeline = st.columns([1, 1, 2])

    with col_gauge:
        avg_risk = df["diabetes_risk"].mean()
        st.plotly_chart(risk_gauge(avg_risk, t("risk_gauge_title")), use_container_width=True)

    with col_pie:
        label_counts = df["label"].value_counts().reset_index()
        label_counts.columns = ["label", "count"]
        label_counts["class"] = label_counts["label"].map({0: t("class_neg"), 1: t("class_pos")})
        fig_pie = px.pie(
            label_counts, values="count", names="class",
            color="class",
            color_discrete_map={t("class_pos"): POS_COLOR, t("class_neg"): NEG_COLOR},
            hole=0.55,
            title=t("class_dist_title"),
        )
        fig_pie.update_traces(textfont_color="#e2e8f0")
        apply_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_timeline:
        df_time = df.copy()
        df_time["hour"] = df_time["recorded_at"].dt.floor("h")
        timeline = (
            df_time.groupby(["hour", "label"])
            .size().reset_index(name="count")
        )
        timeline["class"] = timeline["label"].map({0: t("class_neg"), 1: t("class_pos")})
        fig_line = px.area(
            timeline, x="hour", y="count", color="class",
            color_discrete_map={t("class_pos"): POS_COLOR, t("class_neg"): NEG_COLOR},
            title=t("timeline_title"),
        )
        apply_theme(fig_line)
        fig_line.update_layout(legend_title="")
        st.plotly_chart(fig_line, use_container_width=True)

    # Risk score distribution
    fig_hist = px.histogram(
        df, x="diabetes_risk", color="label",
        color_discrete_map={0: NEG_COLOR, 1: POS_COLOR},
        nbins=40, barmode="overlay", opacity=0.75,
        title=t("risk_dist_title"),
        labels={"diabetes_risk": t("risk_label"), "label": t("class_label")},
    )
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#f97316",
                       annotation_text=t("boundary_annot"))
    apply_theme(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 2 — Feature Analysis
# ═══════════════════════════════════════════════
with tab2:
    features = ["glucose", "bmi", "hba1c", "blood_pressure", "insulin", "age"]
    feature_names = FEATURE_NAMES[LANG]

    # Box plots grid
    st.markdown(f"## {t('feature_dist_title')}")
    cols = st.columns(3)
    for i, feat in enumerate(features):
        fig_box = px.box(
            df, x="label", y=feat, color="label",
            color_discrete_map={0: NEG_COLOR, 1: POS_COLOR},
            labels={"label": "", feat: feature_names[feat]},
            title=feature_names[feat],
            points="outliers",
        )
        fig_box.update_xaxes(tickvals=[0, 1], ticktext=t("yes_no"))
        fig_box.update_layout(showlegend=False, height=260)
        apply_theme(fig_box)
        cols[i % 3].plotly_chart(fig_box, use_container_width=True)

    # Correlation heatmap
    st.markdown(f"## {t('corr_title')}")
    num_cols = features + ["diabetes_risk", "label"]
    corr = df[num_cols].corr()
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=t("pearson_title"),
        zmin=-1, zmax=1,
    )
    fig_corr.update_layout(height=420)
    apply_theme(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter: Glucose vs HbA1c
    st.markdown(f"## {t('scatter_section')}")
    fig_scatter = px.scatter(
        df.sample(min(500, len(df))),
        x="glucose", y="hba1c",
        color="label",
        color_discrete_map={0: NEG_COLOR, 1: POS_COLOR},
        size="diabetes_risk",
        size_max=14,
        opacity=0.65,
        labels={"glucose": feature_names["glucose"], "hba1c": feature_names["hba1c"], "label": t("class_label")},
        title=t("scatter_title"),
        hover_data=["age", "bmi", "diabetes_risk"],
    )
    apply_theme(fig_scatter)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3 — Patient Records
# ═══════════════════════════════════════════════
with tab3:
    st.markdown(f"## {t('patients_title')}")

    col_labels = COLUMN_LABELS[LANG]
    all_cols = ["id", "age", "gender", "bmi", "glucose", "hba1c",
                 "blood_pressure", "diabetes_risk", "label", "recorded_at"]
    default_cols = ["id", "age", "gender", "glucose", "bmi", "hba1c",
                     "diabetes_risk", "label", "recorded_at"]

    search_cols = st.columns([2, 1, 1, 1])
    with search_cols[0]:
        risk_threshold = st.slider(t("risk_threshold_label"), 0.0, 1.0, 0.0, 0.05)
    with search_cols[1]:
        age_min, age_max = st.slider(t("age_range_label"), 20, 80, (20, 80))
    with search_cols[2]:
        show_cols = st.multiselect(
            t("columns_label"),
            options=all_cols,
            default=default_cols,
            format_func=lambda c: col_labels.get(c, c),
        )

    filtered = df[
        (df["diabetes_risk"] >= risk_threshold) &
        (df["age"] >= age_min) &
        (df["age"] <= age_max)
    ]

    st.markdown(t("records_found").format(n=len(filtered)))

    def style_label(val):
        if val == 1:
            return "color: #f87171; font-weight: 600"
        return "color: #6ee7b7"

    display_df = filtered[show_cols].sort_values(
        "recorded_at" if "recorded_at" in show_cols else show_cols[0],
        ascending=False
    ).head(200)
    display_df = display_df.rename(columns=col_labels)

    label_col = col_labels["label"]
    st.dataframe(
        display_df.style.map(style_label, subset=[label_col] if label_col in display_df.columns else []),
        use_container_width=True,
        height=420,
    )

    # Download button
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        t("csv_download"),
        data=csv,
        file_name=f"diabetes_records_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════
# TAB 4 — Pipeline Status
# ═══════════════════════════════════════════════
with tab4:
    st.markdown(f"## {t('pipeline_services_title')}")

    services = [
        {"name": "FastAPI",    "url": f"{API_BASE}/health",              "port": "8000"},
        {"name": "Dagster",    "url": f"{DAGSTER_INTERNAL_URL}/dagster/server_info", "port": "3000"},
        {"name": "PostgreSQL", "url": None,                                "port": "5432"},
    ]

    s_cols = st.columns(len(services))
    for col, svc in zip(s_cols, services):
        if svc["url"]:
            try:
                r = requests.get(svc["url"], timeout=2)
                ok = r.status_code == 200
            except Exception:
                ok = False
        else:
            ok = not df.empty  # postgres inferred from data presence
        icon = "🟢" if ok else "🔴"
        status = t("status_running") if ok else t("status_stopped")
        col.metric(svc["name"], f"{icon} {status}", f":{svc['port']}")

    st.markdown("---")
    st.markdown(f"## {t('data_freshness_title')}")
    if not df.empty:
        latest_record = df["recorded_at"].max()
        minutes_ago = (datetime.now() - latest_record.replace(tzinfo=None)).seconds // 60
        st.markdown(t("latest_record").format(ts=latest_record.strftime('%Y-%m-%d %H:%M:%S'), m=minutes_ago))

        hourly = (
            df.groupby(df["recorded_at"].dt.floor("h"))
            .size().reset_index(name="count")
        )
        fig_bar = px.bar(
            hourly.tail(24), x="recorded_at", y="count",
            title=t("hourly_title"),
            labels={"recorded_at": t("hour_label"), "count": t("records_label")},
            color_discrete_sequence=["#38bdf8"],
        )
        apply_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown(f"## {t('export_title')}")
    st.markdown(
        f'<div class="info-box">{t("export_info")}</div>',
        unsafe_allow_html=True
    )
    if st.button(t("pdf_generate_btn")):
        with st.spinner(t("pdf_spinner")):
            pdf_bytes = build_pdf_report(df, stats)
        st.download_button(
            t("pdf_download_btn"),
            data=pdf_bytes,
            file_name=f"diabetes_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )


# ═══════════════════════════════════════════════
# TAB 5 — Risk Calculator (ML prediction)
# ═══════════════════════════════════════════════
with tab5:
    st.markdown(f"## {t('predict_title')}")
    st.markdown(f'<div class="info-box">{t("predict_intro")}</div>', unsafe_allow_html=True)

    gender_options = {t("predict_gender_female"): "Female", t("predict_gender_male"): "Male"}

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_age = st.number_input(t("predict_field_age"), min_value=1, max_value=120, value=45)
            p_gender_label = st.selectbox(t("predict_field_gender"), options=list(gender_options.keys()))
            p_bmi = st.number_input(t("predict_field_bmi"), min_value=10.0, max_value=70.0, value=28.0, step=0.1)
        with c2:
            p_glucose = st.number_input(t("predict_field_glucose"), min_value=40.0, max_value=400.0, value=110.0, step=1.0)
            p_bp = st.number_input(t("predict_field_bp"), min_value=30.0, max_value=200.0, value=80.0, step=1.0)
            p_hba1c = st.number_input(t("predict_field_hba1c"), min_value=3.0, max_value=16.0, value=5.5, step=0.1)
        with c3:
            p_insulin = st.number_input(t("predict_field_insulin"), min_value=0.0, max_value=900.0, value=80.0, step=1.0)
            p_skin = st.number_input(t("predict_field_skin"), min_value=0.0, max_value=100.0, value=25.0, step=1.0)
            p_pregnancies = st.number_input(t("predict_field_pregnancies"), min_value=0, max_value=15, value=0, step=1)

        predict_submitted = st.form_submit_button(t("predict_button"), use_container_width=True)

    if predict_submitted:
        payload = {
            "age": int(p_age),
            "gender": gender_options[p_gender_label],
            "bmi": p_bmi,
            "glucose": p_glucose,
            "blood_pressure": p_bp,
            "hba1c": p_hba1c,
            "insulin": p_insulin,
            "skin_thickness": p_skin,
            "pregnancies": int(p_pregnancies),
        }
        try:
            r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            r.raise_for_status()
            result = r.json()

            st.markdown(f"### {t('predict_result_title')}")
            r1, r2 = st.columns([1, 1])
            with r1:
                st.plotly_chart(
                    risk_gauge(result["diabetes_risk"], t("predict_risk_label")),
                    use_container_width=True,
                )
            with r2:
                if result["label"] == 1:
                    st.markdown(f'<div class="risk-high" style="font-size:1.3rem; margin-top:2rem;">{t("predict_class_pos")}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low" style="font-size:1.3rem; margin-top:2rem;">{t("predict_class_neg")}</div>', unsafe_allow_html=True)
                st.caption(t("predict_model_used").format(model=result["model_name"]))
        except Exception:
            st.error(t("predict_error"))


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-footer">
      {t('footer_line1')}<br>
      {t('footer_updated')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """,
    unsafe_allow_html=True,
)
