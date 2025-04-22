from transformers import AutoTokenizer, AutoModel
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Sentences we want sentence embeddings, we could use pyvi, underthesea, RDRSegment to segment words
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(
    "bkai-foundation-models/vietnamese-bi-encoder"
)
model = AutoModel.from_pretrained("bkai-foundation-models/vietnamese-bi-encoder")


def encode(question):
    # Tokenize sentences
    encoded_input = tokenizer(
        question, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return sentence_embeddings


import numpy as np


def searchSimilarity(encoded_q1, encoded_q2):
    # Compute similarity
    similarity = np.dot(encoded_q1, encoded_q2) / (
        np.linalg.norm(encoded_q1) * np.linalg.norm(encoded_q2)
    )
    return similarity.item()
