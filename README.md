# Mental Health Sentiment Analyzer using BERT Fine-Tuning

A production-ready NLP project that leverages BERT fine-tuning to classify mental health-related text into multiple sentiment categories such as Depression, Anxiety, Stress, Suicidal, Normal etc.

This project demonstrates an end-to-end pipeline  from dataset handling and model training to deployment using an interactive Streamlit web app.

## 💼 Problem Statement


Traditional sentiment analysis (positive/negative/neutral) is insufficient for mental health contexts.

> This project fills that gap by classifying text into psychologically meaningful categories, enabling deeper insights into emotional states

## 🚀 Project Overview

In this project, we:

- Fine-tuned a BERT-based transformer model on a mental health dataset
- Built a modular inference pipeline
- Deployed the model using a clean and interactive UI with Streamlit
  

## 📊  Dataset


- **Source:** [Kaggle  Mental Health Sentiment Dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
- **Task:** Multi-class classification


## 💡 Why This Matters

Most sentiment models only classify text as *positive/negative*.
This system goes deeper  identifying **mental health states** like:

* Depression
* Anxiety
* Stress
* Suicidal
* Bipolar
* Personality Disorder
* Normal

This makes it closer to **real-world NLP applications** in healthcare and wellbeing.


## Model Overview

* Model: `bert-base-uncased` (fine-tuned)
* Framework: PyTorch + Hugging Face Transformers
* Task: Multi-class text classification
* Max sequence length: 200

**Key Strength:**
Captures contextual meaning in emotionally complex sentences something traditional ML models struggle with.


## 🏗️ System Design

```id="p7j3x1"
User Input → Tokenizer → BERT Model → Logits → Label Decoder → Output
```

### Key Components:

* **Tokenizer + Model Loader** (auto-loaded from saved artifacts)
* **Inference Class (`SentimentModel`)** for clean predictions
* **Label Encoder** for mapping outputs
* **Streamlit UI** for interaction



## Application UI

![UI Preview](https://github.com/user-attachments/assets/1e33fb3b-a151-44fd-8264-6e3df1a53d90)  



![UI Preview Normal](https://github.com/user-attachments/assets/4d7a3a3d-8760-4bd3-bb99-346da9d2d28b)  


![UI Preview Depression](https://github.com/user-attachments/assets/491d6c3d-d8cb-45d8-886d-30b2e5b89667)  


![UI Preview Sucidial](https://github.com/user-attachments/assets/69c322d5-5edb-49fd-b299-803b3f592ae5)  


![UI Preview Anxiety](https://github.com/user-attachments/assets/583f555f-dca7-498e-92fb-9829d3ae0262)


![UI Preview Stress](https://github.com/user-attachments/assets/446d4388-2596-49d2-8fea-d1f1b6f6bf09)


![UI Preview Personality Disorder](https://github.com/user-attachments/assets/2fff5dee-7294-45aa-86e1-ad2cc0d88a2b)  




## 📁 Project Structure

```id="x2k9d4"
.
├── app.py                    # Streamlit app UI layer
├── utils/
│   └── model_utils.py       # Inference pipeline
├── saved_model/             # Model files and tokenizer with encoder
├── notebooks/               # Training 
└── requirements.txt
└── README.md
```


## ⚙️ Run Locally

```bash id="d4k2l1"
git clone https://github.com/your-username/
mental-health-sentiment-analyzer.git
```

```bash cd mental-health-sentiment-analyzer
pip install -r requirements.txt
streamlit run app.py
```




## 👤 Author

**Syeda Arifa Batool**  
SE @ Karachi University | AI & ML Practitioner | Applying technology to create real-world value 📈



## 🔗 Connect with Me

- **LinkedIn:** [Syeda Arifa Batool](https://www.linkedin.com/in/arifa-batool/)  
- **Kaggle:** [Syeda Arifa Batool](https://www.kaggle.com/arifa-batool/)  
- **Email:** [thearifabatool@gmail.com](mailto:thearifabatool@gmail.com)

⭐ If you find this project useful, feel free to star the repository!