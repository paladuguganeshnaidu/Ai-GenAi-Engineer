# GEN-AI Roadmap

## Repository Structure

```text
Learning/
├── Machine_Learning/
│   ├── Concepts/
│   └── Projects/
├── Neural_Networks/
│   ├── Concepts/
│   └── Projects/
├── GPU_Engineering/Concepts/
└── NLP/Concepts/

Data/
├── Raw/
└── ResumeShortlister/
	├── Training/
	├── Testing/
	└── Resumes/

Learning/Machine_Learning/Projects/ResumeShortlister/
├── Models/
├── Results/
├── Tests Results/
└── *.py

Tools/AI_Assistants/
```

Use the matching `Learning/<subject>/Concepts` folder for lessons, keep experiments in that subject's `Projects` folder, and store all datasets under `Data/`. Start with [Learning/Machine_Learning/Projects/ResumeShortlister/Readme.md](Learning/Machine_Learning/Projects/ResumeShortlister/Readme.md) for the complete ML workflow.

### Environment variables

The API examples read secrets from environment variables:

```powershell
$env:NEWS_API_KEY = "your-news-api-key"
$env:DETECT_LANGUAGE_API_KEY = "your-detect-language-api-key"
```

This document combines two learning paths into a single structured plan:

- Track A: AI Engineer with NLP and Computer Vision
- Track B: GenAI and LLM Engineer

The roadmap is organized by learning stage, not by calendar, so it stays focused on skills, projects, and portfolio outcomes.

## Shared Foundations

Build these fundamentals before branching deep into either track:

- Python syntax, OOP, functions, file I/O, and virtual environments
- Git and GitHub workflow
- Linear algebra for machine learning
- Probability, statistics, and descriptive analysis
- SQL fundamentals: `SELECT`, `JOIN`, `GROUP BY`
- Introductory machine learning with scikit-learn
- Basic experiment tracking and reproducibility habits

## Hardware and Lab Practice

Use NVIDIA DGX Spark as the core lab environment for both tracks:

- Set up the operating system, Jupyter access, and GPU validation
- Run small tensor operations and training jobs on GPU
- Compare CPU and GPU behavior for model training and inference
- Record benchmark results, memory behavior, and implementation notes
- Use Spark for local model development, fine-tuning, testing, and deployment experiments

## Track A: AI Engineer Roadmap

### Core Deep Learning

- Neural network fundamentals: perceptron, forward pass, backward pass, activations, and loss functions
- PyTorch fundamentals: tensors, autograd, `nn.Module`, optimizers, and training loops
- CNN fundamentals: convolution, pooling, stride, padding, and MNIST classification
- Transfer learning, augmentation, dropout, and batch normalization

### NLP Foundations

- Text preprocessing: tokenization, stemming, lemmatization, and embeddings
- RNN, LSTM, and GRU concepts for sequence modeling
- Transformer architecture: self-attention, multi-head attention, encoder-decoder structure
- Hugging Face workflows: pipelines, tokenizers, and pretrained language models

### Computer Vision Deep Dive

- OpenCV pipelines for filtering, edge detection, and contour analysis
- Object detection theory: YOLO, R-CNN family, anchor boxes, IoU, and mAP
- Hands-on object detection on a custom dataset
- Segmentation basics with U-Net concepts

### Advanced NLP and Fine-Tuning

- Fine-tune BERT or RoBERTa for classification and NER
- LoRA and QLoRA concepts for parameter-efficient fine-tuning
- Fine-tune a 7B-class open model for a domain-specific NLP task
- Document model behavior, setup decisions, and evaluation clearly

### MLOps and Deployment

- Docker for ML applications
- FastAPI for inference APIs
- MLflow for experiment tracking and model versioning
- Logging, monitoring, and deployment readiness
- Deploy both CV and NLP models as working APIs

### Flagship Capstone for Track A

- Build a multimodal document intelligence pipeline
- Combine OCR, object detection, NLP extraction, and classification
- Train, evaluate, deploy, and document the full system

## Track B: GenAI and LLM Engineer Roadmap

### Transformer and LLM Fundamentals

- Neural network refresher focused on what supports transformer training
- Transformer architecture: self-attention, positional encoding, decoder-only models
- Tokenization and embeddings: BPE, embedding spaces, semantic similarity
- Run open LLMs locally and measure throughput, memory use, and responsiveness

### Prompting and RAG

- Prompt engineering: zero-shot, few-shot, structured outputs, and system prompts
- Vector search with embeddings and cosine similarity
- RAG architecture: chunking, retrieval, reranking, and answer generation
- Build an end-to-end RAG chatbot using local models and local retrieval

### Frameworks and Agentic AI

- LangChain for chains, tools, memory, and output parsing
- LlamaIndex for indexing and query workflows
- Agentic design patterns: ReAct, tool use, and multi-step reasoning
- Build a tool-using agent with search, calculation, and file lookup

### Advanced Agent Systems and MCP

- LangGraph for stateful workflows
- Multi-agent orchestration with researcher, writer, and reviewer roles
- Model Context Protocol fundamentals: servers, clients, tools, and resources
- Build a custom MCP server and connect it to local model workflows

### Fine-Tuning and Optimization

- LoRA and QLoRA for parameter-efficient fine-tuning
- Fine-tune a 7B to 30B open model for a custom task
- Quantization concepts: FP4, INT8, and inference tradeoffs
- Benchmark pre- and post-quantization inference behavior
- Integrate a fine-tuned model into a RAG or agent pipeline

### Production Concerns

- Cost management across models and prompts
- Context window design and retrieval quality
- Prompt caching and batching strategies
- Latency, hallucination rate, and evaluation discipline

### Flagship Capstone for Track B

- Build a production-grade agentic system
- Combine RAG, tool use, MCP, multi-agent orchestration, and a fine-tuned model
- Package the system as a polished, documented application

## Certification Path

Recommended certifications and learning programs:

- Google AI Professional Certificate
- NVIDIA Deep Learning Institute certificate or course
- AWS Certified Machine Learning Engineer - Associate
- Google Cloud Generative AI Leader
- IBM Generative AI Engineering Professional Certificate
- Databricks Generative AI Engineer Associate or Claude Certified Architect
- GitHub Copilot GH-300

## Portfolio Checkpoints

### Track A Portfolio

- Sentiment analysis project
- Object detection project
- Fine-tuned NLP project with LoRA
- Deployed CV and NLP APIs
- Multimodal capstone project

### Track B Portfolio

- RAG chatbot
- Tool-using agent
- Multi-agent system with MCP
- Fine-tuned domain-specific LLM
- Flagship agentic capstone

## Practice Habits

- Build hands-on code regularly
- Solve DSA problems consistently
- Keep GitHub documentation updated
- Record experiments, metrics, and design choices
- Write short notes for each project so the portfolio is easy to review later

## How the Tracks Work Together

The two tracks share the same foundation but diverge in depth:

- Track A is stronger when the goal is classical ML, NLP, CV, deployment, and model training workflows
- Track B is stronger when the goal is LLM applications, RAG, agentic systems, MCP, and local fine-tuning
- Both tracks benefit from the same lab discipline, portfolio quality, and documentation style
- If both are pursued together, keep the foundations shared and let the specialization projects prove depth in each area

## Suggested Presentation Style For GitHub

- Keep each project repository focused on one problem
- Add architecture diagrams, evaluation notes, and demo media
- Explain dataset choice, model choice, tradeoffs, and results
- Call out Spark usage clearly in setup and training notes
- Make the README easy to scan with headings, tables, and short bullet points
