# Schema Mapping Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://schema-mapping-platform-apbvzvujnb8gthfwdiuf63.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Source%20Code-black?logo=github)](https://github.com/Haani76/schema-mapping-platform)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Haani76/schema-mapping-ner)


## What is this project?

Imagine your company receives data files from hundreds of different customers. Each customer names their spreadsheet columns differently — one calls it `cust_id`, another calls it `client_code`, another calls it `account_number`. They all mean the same thing: a unique identifier for a customer. But your system doesn't know that automatically.

Traditionally, a human analyst would have to manually look at each file and say "okay, this column maps to Customer ID, this one maps to Revenue, this one maps to Date." This is slow, expensive, error-prone, and completely unscalable when you're dealing with thousands of files.

**This platform solves that problem automatically using AI.**

It reads column names from any uploaded spreadsheet or database, understands what each column *means* semantically (not just what it's called), and maps it to the correct standardized field — all in milliseconds, with a confidence score attached to every decision.

---

## Real World Example

A customer uploads a CSV file with these columns:
```
cust_no | rev_q3 | purchase_dt | qty_ordered | contact_mail
```

A human would instantly recognize:
- `cust_no` = Customer ID
- `rev_q3` = Revenue
- `purchase_dt` = Date
- `qty_ordered` = Quantity
- `contact_mail` = Email

Our AI does exactly the same thing — automatically, instantly, at scale.

---

## Why does this matter?

- **Data onboarding** at SaaS companies is a massive bottleneck. Every new customer brings differently formatted data.
- **Manual mapping** costs analyst hours and introduces human error.
- **This platform** reduces that process from days to seconds, with 99%+ accuracy.
- Low-confidence predictions are automatically flagged for human review — so the system knows what it doesn't know.

---

## Key Concepts Explained (For Non-Technical Readers)

### What is NER (Named Entity Recognition)?
NER is an AI technique that identifies and classifies pieces of text into predefined categories. In traditional NER, you might identify "London" as a LOCATION or "Apple" as an ORGANIZATION in a sentence. We use the same technique but applied to column headers — identifying "cust_id" as a CUSTOMER_ID or "rev_q3" as REVENUE.

### What is BERT?
BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art AI language model developed by Google. It understands language context deeply — not just individual words but how words relate to each other. We fine-tune (specialize) BERT specifically for our column-mapping task. Think of it like hiring a highly educated person and then giving them specific training for your industry.

### What is Fine-Tuning?
A pre-trained model like BERT already understands language from reading billions of web pages. Fine-tuning means we take that pre-trained knowledge and further train it on our specific task — column header classification. It's like a doctor (general knowledge) specializing into a cardiologist (domain-specific expertise).

### What is Confidence Scoring?
Every prediction our model makes comes with a confidence score between 0 and 1. A score of 0.99 means the model is 99% sure. A score of 0.45 means it's uncertain. We use a threshold of 0.85 — predictions above 85% confidence are automatically mapped, predictions below are sent to a human reviewer. This is how the system knows its own limits.

### What is Auto-Map vs Human Review?
- **Auto-Map**: The model is confident enough (above 85%) to make the mapping decision automatically, no human needed.
- **Human Review**: The model is uncertain and flags the column for a human analyst to manually review. This prevents wrong mappings from slipping through.

### What is an API?
An API (Application Programming Interface) is a way for different software systems to talk to each other. Our FastAPI server exposes the AI model as a web service — any application can send a column name and receive back a prediction in milliseconds, without needing to know anything about AI or machine learning.

### What is MLflow?
MLflow is a tool that tracks all our AI experiments. Every time we train the model, it records what settings we used, how accurate the model was, and saves the best version. Think of it as a detailed lab notebook for AI experiments — so we can always go back and see exactly what produced the best results.

### What is Docker?
Docker packages our entire application — the AI model, the API server, all dependencies — into a self-contained box called a container. This means the application runs identically on any computer or cloud server, eliminating "it works on my machine" problems.

### What is Data Drift?
Over time, the column names customers use might change in ways the model hasn't seen before. Data drift is when the real-world data starts looking different from what the model was trained on, causing accuracy to drop. Our monitoring system detects this automatically by watching confidence scores over time and raising an alert if they drop significantly.

### What is SageMaker?
Amazon SageMaker is AWS's cloud platform for training and deploying machine learning models at scale. Instead of running on a single laptop, SageMaker lets us train on powerful cloud GPUs and serve predictions to millions of users simultaneously.

### What is pgvector?
pgvector is a database extension that stores "embeddings" — numerical representations of text that capture semantic meaning. We use it to store column header embeddings so we can find similar columns across different customer datasets using mathematical similarity search.

---

## Architecture
```
Raw CSV Upload
      ↓
Column Name Extraction
      ↓
BERT NER Model (Fine-tuned)
      ↓
Confidence Score Generated
      ↓
    /     \
≥ 0.85   < 0.85
  ↓          ↓
Auto-Map   Human Review Queue
      ↓
Canonical Schema Field Assigned
      ↓
Monitoring & Drift Detection
```

---

## Features

- BERT-based NER model fine-tuned for semantic column inference
- Confidence scoring with configurable thresholds
- Auto-map vs human review routing logic
- FastAPI REST endpoints for real-time inference
- MLflow experiment tracking and model versioning
- Prediction monitoring with drift detection
- Docker containerization for production deployment
- SageMaker-ready training pipeline

---

## Project Structure
```
schema-mapping-platform/
├── configs/            # Configuration management
├── src/
│   ├── data/           # Data generation and loading
│   ├── models/         # BERT NER model definition
│   ├── training/       # Training engine with MLflow
│   ├── inference/      # Predictor with confidence scoring
│   ├── monitoring/     # Prediction monitoring and drift detection
│   └── api/            # FastAPI REST API
├── data/
│   ├── raw/            # Raw input data
│   ├── processed/      # Processed data
│   └── training/       # Train/val/test splits
├── models/             # Saved model weights
├── notebooks/          # Jupyter notebooks
├── docker/             # Dockerfile
├── tests/              # Unit tests
├── logs/               # Monitoring logs
└── mlruns/             # MLflow tracking
```

---

## Supported Semantic Types

| Label | Example Columns | What it means |
|---|---|---|
| CUSTOMER_ID | customer_id, cust_id, client_code | Unique identifier for a customer |
| PRODUCT_ID | product_id, sku, item_code | Unique identifier for a product |
| REVENUE | revenue, sales, total_amount | Money earned from a transaction |
| DATE | order_date, created_at, sale_dt | Any date or timestamp field |
| QUANTITY | qty, units_sold, quantity | Number of items |
| LOCATION | city, region, postal_code | Geographic information |
| EMAIL | email, contact_email, email_address | Email address field |
| PHONE | phone, mobile, contact_number | Phone number field |
| NAME | customer_name, rep_name, full_name | Any person or company name |
| STATUS | order_status, is_active, flag | State or condition of a record |
| CATEGORY | product_category, item_type, segment | Classification or grouping field |

---

## Quick Start

### 1. Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Training Data
```bash
python -m src.data.data_generator
```

### 3. Train Model
```bash
python -m src.training.train
```

### 4. Start API
```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start MLflow UI
```bash
python -m mlflow ui --port 5000
```

### 6. Docker Deployment
```bash
docker-compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check — is the service running? |
| GET | /labels | List all supported semantic label types |
| POST | /predict | Single column prediction with confidence score |
| POST | /predict/batch | Batch prediction for multiple columns at once |

### Example Request
```json
POST /predict
{
  "column_name": "cust_id",
  "sample_value": "C-1234"
}
```

### Example Response
```json
{
  "column_name": "cust_id",
  "predicted_label": "CUSTOMER_ID",
  "confidence": 0.9903,
  "routing": "auto_map",
  "threshold_used": 0.85
}
```

---

## Model Performance

| Metric | Value |
|---|---|
| Validation F1 Score | 99.18% |
| Auto-map Rate | 91.7% |
| Mean Confidence | 91.4% |
| Training Epochs | 5 |
| Model Parameters | 108.9 Million |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch | Deep learning framework |
| HuggingFace Transformers | BERT model and tokenizer |
| FastAPI | REST API server |
| MLflow | Experiment tracking |
| Docker | Containerization |
| Amazon SageMaker | Cloud training and deployment |
| pgvector | Embedding similarity search |
| seqeval | NER evaluation metrics |

---

## Who is this for?

- **SaaS platforms** that onboard customer data files regularly
- **Data engineering teams** tired of manual schema mapping
- **Analytics platforms** that need to normalize data from multiple sources
- **Any business** that receives data from external partners in varying formats