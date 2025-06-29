# 🧠 Mental Health Text Classification using BERT

This project focuses on identifying mental health conditions such as anxiety, depression, bipolar disorder, and more using natural language inputs. It uses a fine-tuned BERT model to classify user statements and is integrated into a chatbot interface for human-like conversation.

---

## 📌 Project Highlights

- ✅ Fine-tuned BERT model (`bert-base-uncased`) on Reddit mental health posts  
- ✅ 7-class mental health classification (e.g., anxiety, depression, stress, etc.)  
- ✅ Cleaned and processed over 80,000 posts  
- ✅ Integrated with Gemini for conversational front-end  
- ✅ Hosted on Hugging Face Spaces using Streamlit  

---

## 📂 Dataset Used

- **Source**: [Reddit Mental Health Dataset](https://huggingface.co/datasets/kamruzzaman-asif/reddit-mental-health-classification)  
- **Size**: 100,000 posts (trimmed to 80,000 for balanced class distribution)  
- **Labels**:
  - anxiety  
  - depression  
  - bipolar  
  - stress  
  - suicidal  
  - personality disorder  
  - normal

---

## 🤖 Model Overview

- **Model Architecture**: BERT base (`bert-base-uncased`)
- **Fine-tuned Model**: [`Elite13/bert-finetuned-mental-health`](https://huggingface.co/Elite13/bert-finetuned-mental-health)
- **Evaluation Accuracy**: `96.55%`
- **Eval Loss**: `0.15`
- **Training Loss**: `0.19`
- **Epochs**: 3  
- **Hardware Used**: Dual Tesla T4 GPUs (Kaggle)

---

## 🚀 Try the Streamlit App

Interact with the mental health chatbot live:

🔗 [Try the App](https://huggingface.co/spaces/Elite13/mental-health)

---

## 🔽 Download or Use the Fine-Tuned Model

Use this model directly via Hugging Face Transformers:

🔗 [Elite13/bert-finetuned-mental-health](https://huggingface.co/Elite13/bert-finetuned-mental-health)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("Elite13/bert-finetuned-mental-health")
tokenizer = AutoTokenizer.from_pretrained("Elite13/bert-finetuned-mental-health")
