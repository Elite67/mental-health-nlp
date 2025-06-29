# Mental Health Text Classification using BERT

This project is designed to identify mental health conditions such as anxiety, depression, bipolar disorder, and others using natural language processing (NLP). A BERT-based classifier is trained on Reddit mental health posts and integrated with a human-like conversational chatbot interface.

## Project Overview

- Fine-tuned `bert-base-uncased` model
- Multi-class classification for 7 mental health categories
- Trained on a cleaned and balanced Reddit dataset
- Front-end chat interface built with Streamlit and Gemini API
- Model and demo hosted on Hugging Face

## Dataset Details

- **Source**: [Reddit Mental Health Dataset](https://huggingface.co/datasets/kamruzzaman-asif/reddit-mental-health-classification)
- **Size**: 100,000 posts (trimmed to 80,000 for class balance)
- **Classes**:
  - anxiety
  - depression
  - bipolar
  - stress
  - suicidal
  - personality disorder
  - normal

## Model Summary

- **Base Model**: BERT (`bert-base-uncased`)
- **Fine-Tuned Model**: [`Elite13/bert-finetuned-mental-health`](https://huggingface.co/Elite13/bert-finetuned-mental-health)
- **Evaluation Accuracy**: 96.56%
- **Evaluation Loss**: 0.151
- **Training Loss**: 0.190
- **Epochs**: 3
- **Training Environment**: Dual Tesla T4 GPUs on Kaggle

## Try the Web App

You can interact with the chatbot and see real-time predictions:

[Streamlit Demo (Hosted on Hugging Face Spaces)](https://huggingface.co/spaces/Elite13/mental-health)

## Use the Fine-Tuned Model

Clone and use the model directly via Hugging Face Transformers:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Elite13/bert-finetuned-mental-health")
model = AutoModelForSequenceClassification.from_pretrained("Elite13/bert-finetuned-mental-health")
