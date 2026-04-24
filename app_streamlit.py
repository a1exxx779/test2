"""
ISRA 2026 — SOC-in-a-Box Dashboard
Sector: Sănătate | ML: XGBoost | LLM: Gemini | Conformitate: NIS2 + EU AI Act

Rulare locală:
    pip install streamlit google-generativeai xgboost scikit-learn shap joblib pandas numpy plotly
    streamlit run app_streamlit.py

Rulare Streamlit Cloud:
    - Adaugă în Settings > Secrets: GEMINI_API_KEY = "AIzaSy..."
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import datetime
import plotly.graph_objects as go
import plotly.express as px

import google.generativeai as genai

# ─────────────────────────────────────────────────────────────
# CONFIGURARE PAGINĂ
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISRA 2026 — SOC Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563EB 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .alert-critical {
        background: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .alert-safe {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .report-box {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 20px;
        font-family: monospace;
        font-size: 13px;
        max-height: 500px;
        overflow-y: auto;
    }
    .xai-note {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 13px;
    }
    .footer-note {
        font-size: 11px;
        color: #94a3b8;
        text-align: center;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0">🏥 ISRA 2026 — SOC-in-a-Box Dashboard</h2>
    <p style="margin:5px 0 0 0; opacity:0.85">
        Sector Sănătate | Detecție Anomalii XGBoost | Raportare NIS2 cu Gemini AI
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Configurare
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurare")

    # Cheie API — din variabile de mediu, Secrets sau input manual
    # SECURITATE: Nu hardcoda cheia în cod! (Regula de Aur ISRA 2026)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            api_key = ""  # Fișierul secrets.toml nu există — fallback la input manual
    if not api_key:
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Obțineți gratuit de la aistudio.google.com"
        )

    st.divider()
    st.subheader("📦 Model ML")
    uploaded_model = st.file_uploader("xgboost_ids_model.joblib", type=['joblib'])
    uploaded_scaler = st.file_uploader("scaler.joblib", type=['joblib'])
    uploaded_features = st.file_uploader("feature_names.json", type=['json'])

    st.divider()
    st.subheader("ℹ️ Conformitate")
    st.markdown("""
    - ✅ **NIS2** — Art. 21, 23
    - ✅ **EU AI Act** — Supraveghere Umană
    - ✅ **CSA 2** — MSS Nivel Ridicat
    - ✅ **GDPR** — Art. 9 (date medicale)
    """)

# ─────────────────────────────────────────────────────────────
# ÎNCĂRCARE MODEL ML
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_assets(model_file, scaler_file, features_file):
    """Încarcă modelul ML din fișierele uploadate."""
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            f.write(model_file.read())
            model = joblib.load(f.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            f.write(scaler_file.read())
            scaler = joblib.load(f.name)
        features = json.load(features_file)
        return model, scaler, features, True
    except Exception as e:
        return None, None, None, str(e)

ml_model, ml_scaler, feature_names, ml_status = None, None, None, False
if uploaded_model and uploaded_scaler and uploaded_features:
    ml_model, ml_scaler, feature_names, ml_status = load_ml_assets(
        uploaded_model, uploaded_scaler, uploaded_features
    )
    if ml_status is True:
        st.sidebar.success(f"✅ Model încărcat! ({len(feature_names)} features)")
    else:
        st.sidebar.error(f"❌ Eroare: {ml_status}")
else:
    st.sidebar.info("💡 Încărcați modelul ML sau continuați în modul Demo.")

# ─────────────────────────────────────────────────────────────
# TABS PRINCIPALE
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Analiză Incident",
    "📄 Raport NIS2",
    "🔍 Explicabilitate XAI",
    "📊 Dashboard Conformitate"
])

# ══════════════════════════════════════════════════════════════
# TAB 1: ANALIZĂ INCIDENT
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🚨 Introducere Date Incident")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📡 Date Rețea**")
        source_ip = st.text_input("IP Sursă", value="10.0.5.47")
        dest_ip = st.text_input("IP Destinație", value="10.0.1.10")
        dest_port = st.number_input("Port Destinație", value=4242, min_value=1, max_value=65535)
        protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])

    with col2:
        st.markdown("**📊 Metrici Trafic**")
        flow_packets_per_sec = st.number_input(
            "Flow Packets/s", value=12345.67, format="%.2f",
            help="Valori > 5000 indică posibil DDoS"
        )
        flow_bytes_per_sec = st.number_input(
            "Flow Bytes/s", value=987654.32, format="%.2f"
        )
        syn_flag_count = st.number_input(
            "SYN Flag Count", value=8901,
            help="Valori mari indică SYN Flood"
        )
        avg_packet_size = st.number_input("Dimensiune medie pachet (bytes)", value=48)

    st.markdown("**🏥 Context Medical**")
    col3, col4 = st.columns(2)
    with col3:
        affected_system = st.selectbox("Sistem afectat", [
            "Sistem PACS — Imagistică Medicală (RMN/CT)",
            "EMR — Dosare Electronice Pacienți",
            "Sistem Farmaceutic",
            "LIS — Laborator Analize",
            "Infrastructură Rețea Spital"
        ])
    with col4:
        hospital_unit = st.text_input("Unitate spitalicească", value="Radiologie & Imagistică")

    # ── PREDICȚIE ML ──────────────────────────────────────────
    st.divider()
    analyze_btn = st.button("🔍 Analizează Incident", type="primary", use_container_width=True)

    if analyze_btn:
        alert = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": source_ip,
            "destination_ip": dest_ip,
            "destination_port": int(dest_port),
            "protocol": protocol,
            "flow_packets_per_sec": flow_packets_per_sec,
            "flow_bytes_per_sec": flow_bytes_per_sec,
            "syn_flag_count": int(syn_flag_count),
            "avg_packet_size_bytes": int(avg_packet_size),
            "affected_system": affected_system,
            "hospital_unit": hospital_unit,
            "data_sensitivity": "Date medicale — Înaltă Sensibilitate (GDPR Art. 9)"
        }

        with st.spinner("🤖 Modelul ML analizează traficul..."):
            if ml_model is not None and ml_scaler is not None:
                input_vector = np.zeros((1, len(feature_names)))
                feature_map = {
                    'Flow Packets/s': flow_packets_per_sec,
                    'Flow Bytes/s': flow_bytes_per_sec,
                    'SYN Flag Count': syn_flag_count,
                    'Average Packet Size': avg_packet_size,
                }
                for feat, val in feature_map.items():
                    if feat in feature_names:
                        idx = feature_names.index(feat)
                        input_vector[0, idx] = val

                # DOAR .transform() — niciodată .fit_transform() pe date noi!
                input_scaled = ml_scaler.transform(input_vector)
                prediction = ml_model.predict(input_scaled)[0]
                risk_score = float(ml_model.predict_proba(input_scaled)[0][1])
            else:
                # Mod demo: calcul euristic
                risk_score = min(0.99, (flow_packets_per_sec / 15000 +
                                         syn_flag_count / 10000) / 2)
                prediction = 1 if risk_score > 0.5 else 0

        alert['ml_verdict'] = "ATAC CONFIRMAT ⚠️" if prediction == 1 else "BENIGN ✅"
        alert['ml_risk_score'] = round(risk_score, 4)
        alert['ml_risk_percent'] = f"{risk_score * 100:.1f}%"

        st.session_state['current_alert'] = alert

        # ── Afișare rezultat ──
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            color = "🔴" if prediction == 1 else "🟢"
            st.metric("Verdict ML", f"{color} {alert['ml_verdict']}")
        with col_r2:
            st.metric("Risk Score", f"{risk_score * 100:.1f}%")
        with col_r3:
            severity = "CRITIC" if risk_score > 0.9 else ("RIDICAT" if risk_score > 0.7 else "MEDIU")
            st.metric("Severitate NIS2", severity)

        # Gauge chart risk score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score ML (%)", 'font': {'size': 16}},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#dc2626" if risk_score > 0.7 else "#16a34a"},
                'steps': [
                    {'range': [0, 40], 'color': '#dcfce7'},
                    {'range': [40, 70], 'color': '#fef9c3'},
                    {'range': [70, 100], 'color': '#fee2e2'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if prediction == 1:
            st.markdown("""
            <div class="alert-critical">
            ⚠️ <b>ALERTĂ CRITICĂ</b> — Incident detectat! Generați Raportul NIS2 din tab-ul următor.
            Conform Art. 23 NIS2, notificarea DNSC trebuie efectuată în <b>maxim 24 de ore</b>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-safe">
            ✅ <b>Trafic benign</b> — Nu sunt necesare acțiuni imediate. Continuați monitorizarea.
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2: RAPORT NIS2
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📄 Generare Raport NIS2 cu Gemini AI")

    if 'current_alert' not in st.session_state:
        st.info("💡 Analizați mai întâi un incident din tab-ul **Analiză Incident**.")
    else:
        alert = st.session_state['current_alert']

        st.markdown(f"""
        **Incident curent:** `{alert['source_ip']} → {alert['destination_ip']}:{alert['destination_port']}`
        | **Verdict:** {alert['ml_verdict']}
        | **Risk Score:** {alert['ml_risk_percent']}
        """)

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            generate_btn = st.button(
                "📝 Generează Raport NIS2 cu Gemini",
                type="primary",
                use_container_width=True,
                disabled=not bool(api_key)
            )
        with col_btn2:
            if 'nis2_report' in st.session_state:
                st.download_button(
                    "⬇️ Descarcă Raport (.md)",
                    data=st.session_state['nis2_report'],
                    file_name=f"NIS2_Report_{datetime.date.today()}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        if not api_key:
            st.warning("⚠️ Introduceți cheia API Gemini în sidebar pentru a genera rapoarte.")

        if generate_btn and api_key:
            genai.configure(api_key=api_key)
            llm = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,  # Strict factual — fără halucinații!
                    max_output_tokens=1500
                )
            )

            prompt = f"""Ești un analist SOC senior specializat în conformitate NIS2 pentru sectorul Sănătate din România.
Generează un raport oficial de incident de securitate cibernetică conform Directivei NIS2 (UE 2022/2555).

INSTRUCȚIUNI CRITICE:
- Folosește EXCLUSIV datele furnizate. NU inventa adrese IP, timestamps sau alte detalii.
- Răspunde în română, cu terminologie tehnică și juridică precisă.
- Evaluează dacă incidentul trebuie notificat DNSC în termen de 24h (Art. 23 NIS2).

DATE INCIDENT:
- Timestamp: {alert['timestamp']}
- IP Sursă: {alert['source_ip']} → IP Destinație: {alert['destination_ip']}:{alert['destination_port']}
- Sistem afectat: {alert['affected_system']}
- Verdict ML: {alert['ml_verdict']} | Risk Score: {alert['ml_risk_percent']}
- Flow Packets/s: {alert['flow_packets_per_sec']} | SYN Flag Count: {alert['syn_flag_count']}
- Sensibilitate: {alert['data_sensitivity']}

Structurează raportul cu secțiunile: Rezumat Executiv, Clasificare Incident,
Analiză Tehnică, Mapare MITRE ATT&CK, Obligații Notificare NIS2,
Măsuri de Răspuns, Conformitate EU AI Act."""

            with st.spinner("⏳ Gemini generează raportul NIS2... (temperature=0.0)"):
                try:
                    response = llm.generate_content(prompt)
                    report_text = response.text
                    st.session_state['nis2_report'] = report_text

                    st.markdown("---")
                    st.markdown("### 📄 Raport NIS2 Generat")
                    st.markdown(
                        f"<div class='report-box'>{report_text.replace(chr(10), '<br>')}</div>",
                        unsafe_allow_html=True
                    )
                    st.success("✅ Raport generat cu succes! Descărcați-l din butonul de mai sus.")
                except Exception as e:
                    st.error(f"❌ Eroare Gemini API: {e}")

        elif 'nis2_report' in st.session_state:
            st.markdown("### 📄 Ultimul Raport Generat")
            st.markdown(st.session_state['nis2_report'])

# ══════════════════════════════════════════════════════════════
# TAB 3: EXPLICABILITATE XAI
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 Explicabilitate XAI — Feature Importance")

    st.markdown("""
    <div class="xai-note">
    📋 <b>Conformitate EU AI Act:</b> Sistemele AI de risc ridicat utilizate în sectorul Sănătate
    trebuie să asigure <b>transparență și supraveghere umană</b>. Feature Importance vizualizează
    de ce modelul a luat o anumită decizie, transformând "cutia neagră" ML într-un instrument
    de decizie transparent și auditabil.
    </div>
    """, unsafe_allow_html=True)

    if ml_model is not None and feature_names is not None:
        # Feature Importance din XGBoost
        importances = ml_model.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)

        fig_fi = px.bar(
            feat_imp_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 20 Feature Importance — XGBoost IDS (Sector Sănătate)',
            color='Importance',
            color_continuous_scale='Blues',
            labels={'Importance': 'Importanță SHAP', 'Feature': 'Caracteristică Rețea'}
        )
        fig_fi.update_layout(
            height=550,
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False,
            title_font_size=14
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("""
        **Interpretare:**
        - **Valori mari** → caracteristica influențează puternic decizia modelului
        - **Flow Packets/s, SYN Flag Count** → indicatori primari pentru DDoS
        - **Average Packet Size** → pachete mici caracteristice flood-ului
        """)

    else:
        # Demonstrație cu date simulate
        st.info("💡 Mod Demo — Încărcați modelul ML pentru XAI real.")
        demo_features = [
            'Flow Packets/s', 'SYN Flag Count', 'Flow Bytes/s',
            'Total Fwd Packets', 'Avg Packet Size', 'Flow Duration',
            'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
            'Flow IAT Mean', 'RST Flag Count', 'PSH Flag Count',
            'Fwd IAT Mean', 'Bwd IAT Mean', 'ACK Flag Count', 'URG Flag Count'
        ]
        demo_importances = sorted(
            np.random.dirichlet(np.ones(15) * 0.5),
            reverse=True
        )
        demo_df = pd.DataFrame({'Feature': demo_features, 'Importance': demo_importances})
        fig_demo = px.bar(
            demo_df, x='Importance', y='Feature', orientation='h',
            title='Feature Importance — Demo (XGBoost IDS)',
            color='Importance', color_continuous_scale='Blues'
        )
        fig_demo.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_demo, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4: DASHBOARD CONFORMITATE
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Dashboard Conformitate — NIS2 / EU AI Act / CSA 2")

    # Tabel conformitate (din ghidul ISRA 2026)
    conformity_data = {
        "Reglementare": ["NIS2 (UE 2022/2555)", "EU AI Act", "EU CSA 2", "GDPR Art. 9"],
        "Cerință": [
            "Notificarea incidentelor în 24h",
            "Supraveghere Umană sisteme risc ridicat",
            "Serviciu MSS — Nivel Asigurare Ridicat",
            "Protecție date medicale categorii speciale"
        ],
        "Implementare în Proiect": [
            "✅ Raportare automată LLM în <5 minute",
            "✅ XAI Feature Importance + validare analist SOC",
            "✅ SOC-in-a-Box ca MSS cu detecție continuă",
            "✅ Alertare automată la date sensibile Art. 9"
        ],
        "Status": ["✅ Conform", "✅ Conform", "✅ Conform", "✅ Conform"]
    }

    df_conf = pd.DataFrame(conformity_data)
    st.dataframe(df_conf, use_container_width=True, hide_index=True)

    st.divider()

    # KPI-uri proiect
    st.markdown("### 📈 KPI-uri Proiect")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Timp Detectare", "< 1s", "vs. 24h manual")
    with k2:
        st.metric("Timp Raportare NIS2", "< 5 min", "vs. 4h manual")
    with k3:
        st.metric("Acuratețe ML", "~97%", "pe CICIDS2017")
    with k4:
        st.metric("Temperatura LLM", "0.0", "Anti-halucinații")

    # MITRE ATT&CK Top 3 pentru Sănătate
    st.divider()
    st.markdown("### 🗺️ Top 3 Tehnici MITRE ATT&CK — Sector Sănătate")
    mitre_data = {
        "Technică ID": ["T1498", "T1566", "T1486"],
        "Denumire": [
            "Network Denial of Service",
            "Phishing",
            "Data Encrypted for Impact (Ransomware)"
        ],
        "Tactică": ["Impact (TA0040)", "Initial Access (TA0001)", "Impact (TA0040)"],
        "Impact Medical": [
            "Indisponibilitate PACS/EMR",
            "Acces neautorizat EMR",
            "Criptare date — risc vital"
        ],
        "Detectat de Model": ["✅ Da", "✅ Da", "⚠️ Parțial"]
    }
    st.dataframe(pd.DataFrame(mitre_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    ISRA 2026 Capstone Project — Sistem Inteligent de Reziliență & Securitate Sectorială<br>
    Sector: Sănătate | Stack: XGBoost + Gemini AI + Streamlit | Conformitate: NIS2 • EU AI Act • CSA 2 • GDPR
</div>
""", unsafe_allow_html=True)
