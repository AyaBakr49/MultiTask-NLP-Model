# 🧠 Multi-Task NLP Model

A Multi-Task Natural Language Processing (NLP) application built with **PyTorch** and **Streamlit**. The model is trained on multiple GLUE benchmark tasks using a shared Bidirectional RNN architecture with task-specific classification heads.

## 🚀 Features

- Sentiment Analysis (SST-2)
- Paraphrase Detection (MRPC)
- Duplicate Question Detection (QQP)
- Natural Language Inference (MNLI)
- Interactive Streamlit web interface
- Single shared model for multiple NLP tasks

---

## 🏗️ Model Architecture

- Shared Embedding Layer (GloVe Initialized)
- Bidirectional RNN
- Task-Specific Classification Heads
- Multi-Task Learning

---

## 📊 Supported Tasks

| Task | Description |
|------|-------------|
| SST-2 | Sentiment Classification |
| MRPC | Microsoft Research Paraphrase Corpus |
| QQP | Quora Question Pairs |
| MNLI | Multi-Genre Natural Language Inference |

---

## 📁 Project Structure

```
.
├── app.py
├── best_multitask_model.pt
├── word2idx.json
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🖥️ Using the App

1. Select an NLP task.
2. Enter the required text.
3. Click **Predict**.
4. View the prediction and confidence scores.

---

## 🧠 Model Details

- Framework: PyTorch
- Embedding Size: 100
- Maximum Sequence Length: 50
- Architecture: Bidirectional RNN
- Multi-Task Learning
- Task-Specific Output Heads

---

## 📦 Required Files

Make sure these files are located in the same directory:

- `best_multitask_model.pt`
- `word2idx.json`
- `app.py`

---

## 📚 Technologies Used

- Python
- PyTorch
- Streamlit
- NumPy

---

## 👩‍💻 Author

Aya Bakr

---

## 📄 License

This project is developed for educational and research purposes.
