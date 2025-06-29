import streamlit as st
from google import genai
from google.genai import types
from transformers import pipeline
from huggingface_hub import login
from streamlit_elements import elements, nivo, mui
import requests
import os
from collections import Counter  

# Tokens
hf_token = os.getenv("HF_TOKEN")
genai_token = os.getenv("G_TOKEN")
maps_token = os.getenv("MAP_TOKEN")

# Authenticate
login(token=hf_token)
client = genai.Client(api_key=genai_token)

# Logo
logo_path = "https://huggingface.co/spaces/Elite13/mental-health/resolve/main/logo.png"
st.logo(logo_path, size="large")

# Find Psychiatrists Function
def find_psychiatrists(location, token):
    endpoint = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"psychiatrist near {location}",
        "key": token
    }
    try:
        res = requests.get(endpoint, params=params)
        res.raise_for_status()
        data = res.json()
        return data.get("results", [])[:5]
    except Exception as e:
        st.error(f"Failed to fetch psychiatrists: {e}")
        return []

# Label Mapping
label_map = {
    "LABEL_0": 'anxiety',
    "LABEL_1": 'depression',
    "LABEL_2": 'bipolar',
    "LABEL_3": 'normal',
    "LABEL_4": 'personality disorder',
    "LABEL_5": 'stress',
    "LABEL_6": 'suicidal'
}

# Load Model
model_name = "Elite13/bert-finetuned-mental-health"
classifier = pipeline("text-classification", model=model_name, tokenizer=model_name)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []
if "accepted" not in st.session_state:
    st.session_state.accepted = False
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "psychiatrist_results" not in st.session_state:
    st.session_state.psychiatrist_results = []
if "last_location" not in st.session_state:
    st.session_state.last_location = ""

# Title
st.markdown("<h1 style='text-align: center;'>Mental Health Chatbot</h1>", unsafe_allow_html=True)
st.markdown("---")

# Terms & Conditions
if not st.session_state.accepted:
    st.markdown("<h2 style ='text-align:center;'>Terms and Conditions</h2>", unsafe_allow_html=True)
    st.markdown("""
        **This is a University Project.**
        - The application is part of academic coursework or research.
        - Data entered here may be used for project evaluation or improvement.
        - No personal data will be stored or shared beyond academic purposes.
    """)
    if st.button("✅ I Agree"):
        st.session_state.accepted = True
        st.rerun()

# Main App
if st.session_state.accepted:
    
    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        
    # Chat Input
    if prompt := st.chat_input("Say something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.total_messages += 1
        chat_texts = [
            types.Content(role=msg["role"], parts=[types.Part(text=msg["content"])])
            for msg in st.session_state.messages
        ]

        # Gemini Response
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=(
                "You are a kind and supportive friend. Speak casually like a real human, "
                "offering empathy, emotional comfort, and helpful suggestions in a non-judgmental tone."
            )),
            contents=chat_texts
        )
        reply = response.text
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

        # Emotion Detection
        with st.spinner("Calculating Emotion..."):
            results = classifier(prompt, return_all_scores=True)
            top_pred = max(results[0], key=lambda x: x['score'])
            emotion = label_map[top_pred['label']]
            confidence = top_pred['score']
            st.session_state.emotion_log.append(emotion)
            st.markdown(f"🧠 **Detected Emotion:** `{emotion}` ({confidence:.2f} confidence)")

        st.session_state.analysis_done = False
        st.session_state.psychiatrist_results = []

    # View Analysis Button
    if st.session_state.total_messages > 0 and len(st.session_state.emotion_log) > 0 and not st.session_state.analysis_done:
        st.markdown("---")
        if st.button("📊 View Analysis "):
            emotion_counts = Counter(st.session_state.emotion_log)
            total = st.session_state.total_messages
            percentages = {e: (c / total) * 100 for e, c in emotion_counts.items()}

            st.markdown("### Emotion Breakdown")
            for e, p in percentages.items():
                st.markdown(f"- **{e.capitalize()}**: {p:.2f}%")

            # Pie Chart
            nivo_data = [{"id": e, "label": e.capitalize(), "value": c} for e, c in emotion_counts.items()]
            with elements("nivo_pie"):
                with mui.Box(sx={"height": 500}):
                    nivo.Pie(
                        data=nivo_data,
                        margin={"top": 40, "right": 80, "bottom": 80, "left": 80},
                        innerRadius=0.5, padAngle=0.7, cornerRadius=3,
                        activeOuterRadiusOffset=8, borderWidth=1,
                        borderColor={"from": "color", "modifiers": [["darker", 0.2]]},
                        arcLinkLabelsSkipAngle=10, arcLabelsSkipAngle=10,
                        arcLabelsTextColor={"from": "color", "modifiers": [["darker", 2]]},
                        legends=[{
                            "anchor": "bottom", "direction": "row", "translateY": 56,
                            "itemWidth": 100, "itemHeight": 18, "symbolSize": 18,
                            "symbolShape": "circle", "itemTextColor": "#999",
                            "effects": [{"on": "hover", "style": {"itemTextColor": "#000"}}]
                        }],
                        theme={"background": "#fff", "textColor": "#333",
                               "tooltip": {"container": {"background": "#fff", "color": "#333"}}}
                    )

            # Summary Report
            st.markdown("---")
            st.markdown("### Summary Report")
            summary = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(system_instruction=(
                    "You are a mental health chatbot summarizing emotional state based on the user's chats. "
                    "Include key emotions, percentages, and patterns in a warm, encouraging tone."
                )),
                contents=st.session_state.emotion_log
            )
            st.markdown(summary.text)

            st.session_state.analysis_done = True

    # Psychiatrist Finder - Always Visible Once Analysis Done
    if st.session_state.analysis_done:
        st.markdown("---")
        st.markdown("### 🩺 Find Psychiatrists Near You")

        with st.form("find_psych_form"):
            location = st.text_input("Enter your City or Area:")
            submitted = st.form_submit_button("🔍 Find Psychiatrists")

        if submitted and location:
            with st.spinner("Searching nearby psychiatrists..."):
                results = find_psychiatrists(location, maps_token)
                st.session_state.psychiatrist_results = results
                st.session_state.last_location = location

        if st.session_state.psychiatrist_results:
            st.markdown(f"### 🏥 Psychiatrists near {st.session_state.last_location.title()}:")
            for place in st.session_state.psychiatrist_results:
                name = place.get("name")
                address = place.get("formatted_address")
                rating = place.get("rating", "N/A")
                st.markdown(f"- **{name}**, 📍 {address}, ⭐ {rating} rating")
