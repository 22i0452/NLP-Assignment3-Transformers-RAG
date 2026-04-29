# NLP Assignment 3: Transformers + RAG
Name: Shazil
Roll No: 22i-0452

## Project Overview

This project builds a Transformer-based NLP pipeline for Amazon review understanding and RAG-enhanced explanation generation.

The system has three main parts:

1. Encoder-only Transformer

   * Predicts review sentiment
   * Predicts a derived review feature
   * Produces review embeddings

2. Retrieval Module

   * Uses saved encoder embeddings
   * Retrieves similar reviews using cosine similarity

3. Decoder-only Transformer

   * Generates short explanations
   * Uses review text, predicted labels, and retrieved reviews

Both Transformer models were implemented from scratch.

## Restrictions Followed

The following modules were not used:

nn.Transformer
nn.MultiheadAttention
nn.TransformerEncoder

Attention, multi-head attention, encoder blocks, decoder blocks, and causal masking were implemented manually.

## Dataset

Amazon review data was used from three categories:

sports
cellphones
home

Due to hardware limitations, a smaller subset was used.

Final dataset:

Total samples: 5998
Train: 4198
Validation: 900
Test: 900

## Labels

Sentiment mapping:

1-2 stars -> negative
3 stars -> neutral
4-5 stars -> positive

Second derived feature:

review_length_class = short / medium / long

## Results

Encoder test results:

Sentiment Accuracy: 47.0%
Sentiment Macro F1: 29.71%
Length Accuracy: 96.33%
Length Macro F1: 96.24%

Retrieval results:

Top-1 same sentiment rate: 73.56%
Average top-1 cosine similarity: 0.9815
Top-k: 3

Decoder and RAG results:

RAG Perplexity: 7.7282
No-RAG Perplexity: 60.8991

The RAG model achieved lower perplexity than the no-retrieval baseline.

## Repository Structure

data/
processed_reviews.csv
train.csv
val.csv
test.csv

models/
encoder_initial.pt
encoder_trained.pt
decoder_trained.pt

results/
vocab.json
encoder_metrics.json
learning_curves.png
train_embeddings.npy
train_metadata.csv
retrieval_examples.json
retrieval_analysis.csv
decoder_vocab.json
perplexity.json
generated_examples.json
rag_ablation.csv

i220452-NLP-Assignment3.ipynb
report.pdf
README.md

## How to Run

Open the notebook:

i220452-NLP-Assignment3.ipynb

Run all cells from top to bottom.

## Commit Phases

Phase 1: setup repository and preprocess dataset
Phase 2: build vocabulary and dataloaders
Phase 3: implement encoder transformer model
Phase 4: train encoder and save review embeddings
Phase 5: add cosine similarity retrieval module
Phase 6: implement decoder transformer and RAG generation
Phase 7: finalize report and runnable notebook
