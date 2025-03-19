# AIOlimpia-NLP

## Starter

Install the requirements:

```bash
pip install -r requirements.txt
```

**Deadline:** 18.03.2025.

## Questions
* python + pytorch, can we use other libraries? Transformers, Langchain, models from huggingface?
  * The more they see the better, but show the from scratch solution too.
* RNN, LSTM?
  * No.
* GPT or other LLM with API?
  * No, somebody else will do that.
* What is the QA part (5) really about?
  * We don't really know for sure.
* Text classification only with NN or also traditional ML?
  * The more they see the better.
* Simple chatbots?
  * We don't really know for sure.

## Syllabus
1. Word Embeddings (Word2Vec, GloVe)
2. Transformer Basics (Attention Mechanism)
3. Text Classification
4. Introduction to Pre-trained NLP Models (e.g., BERT, GPT)
5. Question Answering with Pre-trained Models
6. Introduction to Large Language Models (LLMs) (e.g., GPT-4)
7. Building Simple Chatbots Using NLP
8. Model Fine-Tuning: Methods and Limitations (LoRA, Adapters, etc.)
9. LLM Agents Basics

## Detailed TODO list

- [x] Word Embeddings
  - [x] Text preprocessing
    - [x] Tokenization etc.
  - [x] Embeddings
    - [x] Words to numbers (so we can process them with NN)
    - [x] Words to vectors
      - [x] Intuition: king - man + woman = queen
    - [x]  How to get these vectors - short description of training algorithm and practical example for usage
    - [x]  Word2Vec, GloVe
- [ ] Transformer Basics
  - [ ] Input, output
  - [ ] Encoder + decoder
  - [ ] Attention Mechanism etc. (Bahdanau Attention(?))
  - [ ] Dynamic embeddings
  - [ ] Good parallelization capacity
- [ ] Introduction to Pre-trained NLP Models 
  - [ ] Encoder-only: BERT
  - [ ] Decoder-only: GPT
- [ ] Text Classification
  - [ ] Tf-idf?
  - [ ] Supervised-learning
    - [ ] Examples for labels and texts
      - [ ] Spam, review rating, sentiment analysis, NER
        - [ ] Named Entity Recognition (NER): Identify proper names like "Neumann" (person) or "Budapest" (city)
        - [ ] Part-of-Speech (POS) Tagging: Label words as nouns, verbs, adjectives, etc.
    - [ ] Evaluation: Accuracy, precision, recall, F1, confusion matrix
  - [ ] Naïve Bayes
  - [ ] Decision tree with features??
  - [ ] Text embeddings
  - [ ] KNN
  - [ ] Neural networks (logistic regression?)
    - [ ] Learning methods
- [ ] Question Answering with Pre-trained Models
  - [ ] Fine-tuning for question answering
    - [ ] RLHF
- [ ] Introduction to Large Language Models (LLMs) (e.g., GPT-4)
  - [ ] Scaling law
  - [ ] Few-shot, zero-shot, and chain-of-thought prompting
    - [ ] Instruction tuning
  - [ ] Biases
  - [ ] LLM models: GPT, LLama etc.
- [ ] Building Simple Chatbots Using NLP
  - [ ] Rule-based vs. NN-based chatbots
  - [ ] Previous context in conversation
- [ ] Model Fine-Tuning: Methods and Limitations (LoRA, Adapters, etc.)
  - [ ] Why we want it?
  - [ ] How: full vs. PEFT
- [ ] LLM Agents Basics
  - [ ] Tool usage and retrieval-augmented generation (RAG)

## File Structures

```
├── exercises
├── img
|     └── 01_embeddings
├── study_materials
|     └── 01_embeddings.ipynb
├── LICENSE
├── README
└── Syllabus-2025-Final.pdf
```
TODO: külön mappa feladatoknak

