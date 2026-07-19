"""
Streamlit app for the Multi-Task NLP model (SST2 / MRPC / QQP / MNLI).

Files needed in the same folder as this app.py:
- best_multitask_model.pt   (trained model weights)
- word2idx.json              (vocabulary mapping)

Run locally with:
    streamlit run app.py
"""

import re
import json

import numpy as np
import torch
import torch.nn as nn
import streamlit as st


# ---------------------------------------------------------------------------
# Config — must match training config
# ---------------------------------------------------------------------------
MAX_LEN = 50
EMBED_DIM = 100

MODEL_PATH = "best_multitask_model.pt"
VOCAB_PATH = "word2idx.json"

TASK_INFO = {
    "SST2 (Sentiment)": {
        "id": 0,
        "num_inputs": 1,
        "labels": {0: "Negative 😠", 1: "Positive 😊"},
    },
    "MRPC (Paraphrase Detection)": {
        "id": 1,
        "num_inputs": 2,
        "labels": {0: "Not a paraphrase", 1: "Paraphrase"},
    },
    "QQP (Duplicate Questions)": {
        "id": 2,
        "num_inputs": 2,
        "labels": {0: "Different questions", 1: "Duplicate questions"},
    },
    "MNLI (Natural Language Inference)": {
        "id": 3,
        "num_inputs": 2,
        "labels": {0: "Entailment", 1: "Neutral", 2: "Contradiction"},
    },
}


# ---------------------------------------------------------------------------
# Tokenizer + text -> index conversion (must match training)
# ---------------------------------------------------------------------------
def simple_tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())


def text_to_indices(text, word2idx):
    tokens = simple_tokenizer(text)
    ids = [word2idx.get(tok, 1) for tok in tokens]  # 1 = <UNK>

    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))  # 0 = <PAD>
    else:
        ids = ids[:MAX_LEN]

    return ids


# ---------------------------------------------------------------------------
# Model definition (must match training exactly)
# ---------------------------------------------------------------------------
class MultiTaskBiRNN(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        # random init here — real weights are loaded from checkpoint after
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.birnn = nn.RNN(
            embed_dim,
            128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.4)

        self.sst_head = nn.Linear(256, 2)
        self.mrpc_head = nn.Linear(256, 2)
        self.qqp_head = nn.Linear(256, 2)
        self.mnli_head = nn.Linear(256, 3)

    def forward(self, input_ids, task_ids):
        x = self.embedding(input_ids)
        output, hidden = self.birnn(x)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        h = torch.cat((forward_hidden, backward_hidden), dim=1)
        h = self.dropout(h)

        batch_size = input_ids.size(0)
        logits = torch.zeros(batch_size, 3, device=input_ids.device)

        sst_mask = task_ids == 0
        mrpc_mask = task_ids == 1
        qqp_mask = task_ids == 2
        mnli_mask = task_ids == 3

        if sst_mask.any():
            logits[sst_mask, :2] = self.sst_head(h[sst_mask])
        if mrpc_mask.any():
            logits[mrpc_mask, :2] = self.mrpc_head(h[mrpc_mask])
        if qqp_mask.any():
            logits[qqp_mask, :2] = self.qqp_head(h[qqp_mask])
        if mnli_mask.any():
            logits[mnli_mask] = self.mnli_head(h[mnli_mask])

        return logits


# ---------------------------------------------------------------------------
# Cached loaders (loaded once, reused across interactions)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    return word2idx


@st.cache_resource
def load_model(vocab_size):
    model = MultiTaskBiRNN(vocab_size, EMBED_DIM)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Task NLP Model", page_icon="🧠", layout="centered")

st.title("🧠 Multi-Task NLP Model")
st.caption("A single BiRNN model shared across SST-2, MRPC, QQP, and MNLI tasks")

word2idx = load_vocab()
model = load_model(len(word2idx))

task_choice = st.selectbox("Choose a task:", list(TASK_INFO.keys()))
task_config = TASK_INFO[task_choice]

st.divider()

if task_config["num_inputs"] == 1:
    text_input = st.text_area("Enter a sentence:", height=100,
                               placeholder="e.g. This movie was absolutely wonderful!")
    combined_text = text_input
else:
    label_pair = (
        ("Sentence 1:", "Sentence 2:") if "MRPC" in task_choice
        else ("Question 1:", "Question 2:") if "QQP" in task_choice
        else ("Premise:", "Hypothesis:")
    )
    text1 = st.text_area(label_pair[0], height=80)
    text2 = st.text_area(label_pair[1], height=80)
    combined_text = f"{text1} <sep> {text2}" if text1 and text2 else ""

if st.button("Predict", type="primary", use_container_width=True):
    if not combined_text.strip():
        st.warning("Please fill in the required text field(s) first.")
    else:
        ids = text_to_indices(combined_text, word2idx)
        input_tensor = torch.tensor([ids], dtype=torch.long)
        task_tensor = torch.tensor([task_config["id"]], dtype=torch.long)

        with torch.no_grad():
            logits = model(input_tensor, task_tensor)
            num_classes = len(task_config["labels"])
            probs = torch.softmax(logits[0, :num_classes], dim=0).numpy()
            pred_class = int(np.argmax(probs))

        st.success(f"**Prediction:** {task_config['labels'][pred_class]}")

        st.subheader("Confidence scores")
        for class_id, label in task_config["labels"].items():
            st.progress(float(probs[class_id]), text=f"{label}: {probs[class_id]:.1%}")

st.divider()
st.caption(
    "Model: Bidirectional RNN with GloVe-initialized embeddings, "
    "trained jointly on four GLUE tasks with task-specific classification heads."
)
