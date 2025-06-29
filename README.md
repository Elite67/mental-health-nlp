# 🧠 Mental Health Chat Analyzer (NLP-Based)

This project is a university research tool designed to analyze user conversations and detect potential mental health conditions such as **anxiety**, **depression**, **bipolar disorder**, **stress**, and more — using a fine-tuned BERT model. Users interact with a human-like chatbot powered by **Gemini** (Google Generative AI), while the backend monitors emotional signals across the conversation.

## 🔬 Objective

To build an AI-powered conversational system that:
- Engages in empathetic dialogue with users
- Analyzes emotional patterns in real-time
- Classifies mental health conditions based on input text

## 📊 Try the Demo

👉 [Launch on Hugging Face Spaces 🚀](https://huggingface.co/spaces/Elite13/mental-health)

## 📁 Datasets Used

1. **Reddit Mental Health Classification Dataset**  
   → [kamruzzaman-asif/reddit-mental-health-classification](https://huggingface.co/datasets/kamruzzaman-asif/reddit-mental-health-classification)  
   ~100K Reddit posts, trimmed to 80K

2. **Mental_Health_Condition_Classification by sai1908**  
   → [sai1908/Mental_Health_Condition_Classification](https://huggingface.co/datasets/sai1908/Mental_Health_Condition_Classification)

---

## 🤖 Model

- `bert-base-uncased` (fine-tuned for mental health classification)
- Hosted on Hugging Face: [Elite13/bert-finetuned-mental-health](https://huggingface.co/Elite13/bert-finetuned-mental-health)

### 🏆 Model Performance
| Metric            | Value     |
|------------------|-----------|
| Accuracy          | 0.9656    |
| Validation Loss   | 0.1513    |
| Training Loss     | 0.0483    |
| Epochs            | 3         |
| FLOPs             | 25.6T     |
| GPUs Used         | 2x Tesla T4 |

---

## 💬 Chatbot Layer

- Frontend: Built using **Streamlit**
- Conversational engine: **Gemini Flash 2.0** via Google GenAI API
- Emotion classifier: Fine-tuned BERT model on Hugging Face

---

## 🧪 Local Development

### 🔧 Setup
```bash
git clone https://github.com/Elite67/mental-health-nlp.git
cd mental-health-nlp
pip install -r requirements.txt
streamlit run app.py
