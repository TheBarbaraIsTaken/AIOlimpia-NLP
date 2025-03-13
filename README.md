# AIOlimpia-NLP

**Deadline:** 18.03.2025.


## Kérdések
* python + pytorch
  * más könyvtár használható?
  * Transformers, Langchain, huggingface-ről leszedni modelleket?
* RNN, LSTM?
* Kell gpt API vagy valami más modell?
* QA rész az amire gondolok?
* Text classification csak NN vagy ML módszerek is?
* simple chatbots?

## Syllabus
* Word Embeddings (Word2Vec, GloVe)
* Transformers Basics (Attention Mechanism)
* Text Classification
* Introduction to Pre-trained NLP Models (e.g., BERT, GPT)
* Question Answering with Pre-trained Models
* Introduction to Large Language Models (LLMs) (e.g., GPT-4)
* Building Simple Chatbots Using NLP
* Model Fine-Tuning: Methods and Limitations (LoRA, Adapters, etc.)
* LLM Agents Basics

## Detailed TODO list

- [ ] Word Embeddings
  - [ ] Tokenization
  - [ ] Words to numbers (so we can process them with NN)
  - [ ] Words to vectors
    - [ ] Intuition: king - man + woman = queen
  - [ ]  How to get these vectors - short description of training algorithm and practical example for usage
    - [ ]  Word2Vec, GloVe
  - [ ]  Dense and sparse representations
- [ ] Transformers Basics
  - [ ] Input, output
  - [ ] Encoder + decoder
  - [ ] Attention Mechanism etc.
  - [ ] Good parallelization capacity
- [ ] Introduction to Pre-trained NLP Models 
  - [ ] Encoder-only: BERT
  - [ ] Decoder-only: GPT
  - [ ] Embeddings
- [ ] Text Classification
  - [ ] Supervised-learning
    - [ ] Examples for labels and texts
      - [ ] Spam, review rating, sentiment analysis, NER
    - [ ] Evaluation: Accuracy, precision, recall, confusion matrix
  - [ ] Naïve Bayes
  - [ ] KNN
  - [ ] Decision tree with features??
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
- [ ] Model Fine-Tuning: Methods and Limitations (LoRA, Adapters, etc.)
  - [ ] Why we want it?
  - [ ] How: full vs. PEFT
- [ ] LLM Agents Basics
  - [ ] Tool usage and retrieval-augmented generation (RAG)

