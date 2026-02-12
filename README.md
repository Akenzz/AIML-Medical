# HealthDesk AI – Unified Medical Assistant Server

A comprehensive FastAPI-based unified server providing three integrated AI-powered medical services: disease prediction, intelligent medical chatbot, and patient consultation analysis.

## 📋 Project Overview

HealthDesk AI is a unified microservices server that combines multiple machine learning and generative AI capabilities to provide comprehensive healthcare support. It includes:

- **Symptom-based disease prediction** using machine learning
- **Multilingual medical chatbot** powered by RAG (Retrieval-Augmented Generation)
- **Intelligent consultation analysis** for severity assessment and summarization

## 🏗️ Architecture

The project is structured as a unified FastAPI server that mounts three separate services:

```
HealthDesk AI Server (Port 8000)
├── /predict-disease    → Symptom Checker API
├── /medical-bot        → Medical Chatbot API
└── /medreach           → Consultation Analysis API
```

## 📦 Services

### 1. **Predict Disease** (`/services/predict_disease/`)
Predicts the top 3 most likely diseases based on provided symptoms using a pre-trained machine learning model.

**Key Features:**
- Quick disease prediction from symptom list
- Returns probability scores for each prediction
- Uses LightGBM model trained on medical symptom data

**Endpoints:**
- `GET /predict-disease/` – Service information
- `POST /predict-disease/predict` – Predict diseases from symptoms

**Request Example:**
```json
{
  "symptoms": ["fever", "cough", "headache"]
}
```

**Response Example:**
```json
{
  "predictions": [
    {"disease": "Common Cold", "probability": 0.85},
    {"disease": "Flu", "probability": 0.72},
    {"disease": "COVID-19", "probability": 0.65}
  ]
}
```

### 2. **Medical Bot** (`/services/medical_bot/`)
An intelligent medical chatbot powered by RAG that retrieves relevant medical information and maintains conversation history. Supports multiple languages including English, Hindi, Urdu, and Kannada.

**Key Features:**
- **Retrieval-Augmented Generation (RAG)** using Pinecone vector store
- **Multilingual support** with automatic language detection (English, Hindi/Urdu in Roman script, Kannada in Roman script)
- **Conversation history** per user for contextual responses
- **Dual LLM support**: Google Gemini (main responses) + Groq LLaMA (language detection)
- **Medical PDF knowledge base** integration

**Endpoints:**
- `GET /medical-bot/` – Service status
- `POST /medical-bot/chat` – Chat with medical assistant

**Request Example:**
```json
{
  "question": "What are the symptoms of diabetes?",
  "user_id": "user123"
}
```

**Response Example:**
```json
{
  "answer": "Diabetes symptoms include...",
  "context": ["retrieved medical document 1", "retrieved medical document 2"],
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "detected_language": "English",
  "detected_script": "English",
  "english_translation": null
}
```

### 3. **Medreach** (`/services/medreach/`)
Analyzes patient consultation transcripts, provides medical summaries, and assesses severity levels for triage and emergency response prioritization.

**Key Features:**
- **Transcript analysis** using Google Gemini API
- **Severity assessment** with 4-level classification:
  - **Low**: Minor ailments (common cold, mild headache)
  - **Medium**: Non-urgent conditions (rash, controlled chronic issues)
  - **High**: Urgent conditions requiring prompt attention (high fever with confusion)
  - **Critical**: Life-threatening emergencies (chest pain, stroke signs, severe bleeding)
- **Intelligent summarization** of patient symptoms
- **Integration with Java backend** for database persistence

**Endpoints:**
- `GET /medreach/` – Service information
- `POST /medreach/submit` – Analyze consultation transcript

**Request Example:**
```json
{
  "transcript": "Patient reports chest pain radiating to the left arm..."
}
```

**Response Example:**
```json
{
  "summary": "Patient experiencing chest pain with left arm radiation",
  "severity": "Critical"
}
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- FastAPI and Uvicorn
- API keys for:
  - **Pinecone** (medical-bot)
  - **Google Generative AI** (medical-bot, medreach)
  - **Groq API** (medical-bot)
  - **LangChain** (optional, for tracing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd HealthDesk-AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file in the root directory with:
   ```
   # Pinecone Configuration
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=medical-bot

   # Google Generative AI
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_GEMINI_MODEL=gemini-2.5-flash-lite

   # Groq API
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL=llama-3.1-8b-instant

   # Java Backend (for Medreach)
   JAVA_BACKEND_URL=http://your-java-backend-url

   # LangChain (optional)
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=Medical-bot
   ```

4. **Run the unified server:**
   ```bash
   python app.py
   ```
   Or with Uvicorn directly:
   ```bash
   uvicorn app:app --reload --port 8000
   ```

## 📁 Project Structure

```
.
├── app.py                              # Main unified FastAPI server
├── requirements.txt                    # Python dependencies
├── services/
│   ├── predict_disease/
│   │   ├── app.py                      # Symptom checker API
│   │   ├── symptom_checker_model.joblib    # Pre-trained ML model
│   │   └── symptom_columns.joblib      # Model feature columns
│   ├── medical_bot/
│   │   ├── app.py                      # Medical chatbot with RAG
│   │   ├── src/
│   │   │   ├── helper.py               # PDF loading & text processing utilities
│   │   │   └── prompt.py               # System prompt for medical assistant
│   │   └── __init__.py
│   ├── medreach/
│   │   ├── app.py                      # Consultation analysis API
│   │   └── __init__.py
│   └── __init__.py
└── README.md                           # This file
```

## 🔧 Dependencies

Key packages used in this project:

**Core Framework:**
- `fastapi` – Modern API framework
- `uvicorn[standard]` – ASGI server

**ML & Data Processing:**
- `pandas` – Data manipulation
- `scikit-learn` – Machine learning algorithms
- `lightgbm` – Gradient boosting model
- `joblib` – Model serialization

**Generative AI & LLMs:**
- `langchain-google-genai` – Google Gemini integration
- `langchain-groq` – Groq API integration
- `langchain-community` – Additional LangChain tools
- `langchain-pinecone` – Pinecone vector store integration
- `langchain-huggingface` – Hugging Face embeddings
- `sentence-transformers` – Text embedding models
- `groq` – Groq API client
- `pinecone` – Vector database client

**Utilities:**
- `python-multipart` – Form data handling
- `requests` – HTTP client
- `python-dotenv` – Environment variable management

For complete dependencies, see [requirements.txt](requirements.txt).

## 🔐 Security Considerations

- Store API keys securely in `.env` file (never commit to version control)
- Implement authentication for production deployment
- Validate and sanitize all user inputs
- Use HTTPS in production
- Implement rate limiting for API endpoints

## 📚 API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🤝 Features

### Multilingual Support (Medical Bot)
The medical bot automatically detects and translates:
- **English**: Standard medical queries
- **Hindi/Urdu (Roman script)**: "bukhar hai", "sir dard", "pet dard"
- **Kannada (Roman script)**: "jwaravu ide", "tale novu", "hotte novu"
- **Native scripts**: Devanagari (देवनागरी), Arabic (عربي), Kannada (ಕನ್ನಡ)

### RAG Implementation (Medical Bot)
- Retrieves relevant medical documents from Pinecone vector store
- Combines retrieved context with LLM for accurate, sourced responses
- Maintains conversation history for contextual understanding
- Default retrieval: Top 2 most similar documents

### Severity Assessment (Medreach)
Intelligent 4-level severity classification based on:
- Symptom descriptions
- Clinical indicators
- Emergency red flags
- Medical urgency guidelines

## ⚙️ Configuration

### Medical Bot Configuration
- **Chunk Size**: 1,300 characters with 230-character overlap
- **Retrieval Type**: Similarity-based (k=2)
- **Primary LLM**: Google Gemini
- **Fallback LLM**: Groq LLaMA

### Predict Disease Configuration
- **Model Type**: LightGBM classifier
- **Output**: Top 3 predictions with probabilities

## 🤖 How It Works

### Workflow for Medical Bot

1. **User Query Input**: Patient sends a medical question
2. **Language Detection**: LLaMA detects the language and script
3. **Translation**: Convert to English if needed
4. **RAG Retrieval**: Pinecone retrieves 2 most relevant medical documents
5. **LLM Response**: Gemini generates response based on context + history
6. **Response Formatting**: Return answer with context, history, and language metadata

### Workflow for Medreach

1. **Transcript Upload**: Medical conversation transcript received
2. **Analysis**: Gemini analyzes transcript
3. **Summary Generation**: Extract key symptoms and findings
4. **Severity Assessment**: Classify as Low/Medium/High/Critical
5. **Response**: Return structured analysis

### Workflow for Predict Disease

1. **Symptom Input**: User provides list of symptoms
2. **Feature Vectorization**: Convert symptoms to model-compatible format
3. **Prediction**: LightGBM model generates probability scores
4. **Ranking**: Return top 3 diseases by probability

## 📊 Performance Notes

- Medical Bot: Depends on Pinecone latency and LLM API response time
- Predict Disease: ~50-100ms per prediction (model inference)
- Medreach: Depends on Google Gemini API response time
- Conversation history maintained in-memory (consider database for production)

## 🐛 Troubleshooting

**Model not found error:**
```
Error: Model or column file not found in services/predict_disease.
```
Ensure `symptom_checker_model.joblib` and `symptom_columns.joblib` exist in the predict_disease directory.

**API Key missing error:**
Check that all required environment variables are set in `.env` file.

**Pinecone connection error:**
Verify `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` are correct.

## 🚢 Deployment

### Docker Deployment
```bash
docker build -t healthdesk-ai .
docker run -p 8000:8000 --env-file .env healthdesk-ai
```

### Production Considerations
- Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)
- Implement database persistence for conversation history
- Add authentication and authorization
- Set up request logging and monitoring
- Implement rate limiting
- Use environment-specific configurations

## 💡 Example Use Cases

1. **Patient Self-Diagnosis**: Use medical bot to understand symptoms or predict disease
2. **Triage System**: Use Medreach to quickly assess patient severity
3. **Medical Support**: Chat-based guidance for common health concerns
4. **Clinical Training**: Educational tool for medical professionals
5. **Telehealth Integration**: Embed services into telemedicine platforms

## 📝 Version History

- **v1.2.0** – Unified server with all three services
- **v2.1.0** – Predict Disease (latest)
- **v1.3.0** – Medical Bot (latest)

## 📄 License

[Specify your license here]

## 👥 Contributing

[Add contribution guidelines here]

## 📧 Support

For issues, questions, or contributions, please [contact information/issue tracker].

---

**Built with ❤️ for better healthcare accessibility**
