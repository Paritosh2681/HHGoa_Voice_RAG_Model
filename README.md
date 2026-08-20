---
title: HH Goa Voice RAG Model
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
---

# OPERATOR — Voice-Driven AI Research Interface

> **HH Goa 2026 — Task 2 Submission**

OPERATOR is an interactive AI-powered research interface designed to make information discovery faster and more natural through **voice interaction, intelligent question answering, multilingual support, hybrid information retrieval, and transparent processing visualization**.

Instead of relying only on traditional text-based search, OPERATOR allows users to speak their questions and receive structured, understandable answers through an immersive research interface.

---

## 🚀 Key Features

### 🎙️ Voice-Based Interaction

* Speak directly to the system using a microphone.
* Converts spoken queries into text.
* Provides a conversational research experience.
* Displays voice/activity visualization while processing.
* Supports voice-driven interaction with the research pipeline.

### 🧠 Intelligent Question Answering

* Accepts natural-language questions.
* Processes user queries through the backend AI pipeline.
* Generates concise and understandable answers.
* Designed specifically for research and information discovery.

### 🔎 Hybrid Retrieval & RAG

OPERATOR uses a retrieval-augmented approach to improve information discovery.

The backend combines:

* Knowledge-base retrieval
* Semantic search
* Hybrid retrieval
* FAISS-based indexing
* AI-powered answer generation

This allows the system to retrieve relevant information before generating the final response.

### 🌐 Multilingual Support

* Supports queries and responses in multiple languages.
* Enables more accessible interaction.
* Designed to reduce language barriers in information discovery.
* Suitable for English and Indian-language interaction.

### 📊 Information & Data Visualization

The interface presents information through:

* Processing indicators
* Audio waveform visualization
* Research/data cards
* Query history
* Processing statistics
* Structured answer sections
* Retrieval and response information

### ⚡ Optimized Processing

The project includes performance optimization and benchmarking for the retrieval pipeline.

The repository contains:

* Retrieval benchmarks
* Latency measurements
* FAISS optimization experiments
* Kaggle experimentation
* Fast-path processing evaluation

---

## ⚡ Interactive Research Workflow

The system follows this workflow:

```text
User Question
      ↓
Voice Input
      ↓
Speech-to-Text
      ↓
Query Processing
      ↓
Hybrid Information Retrieval
      ↓
RAG / AI Processing
      ↓
Answer Generation
      ↓
Text + Voice Response
```

---

# 🎯 Problem Being Addressed

Traditional information-search interfaces often require users to:

* Type long queries
* Navigate multiple pages
* Read large amounts of information
* Switch between different tools
* Deal with language barriers
* Manually identify relevant information

This creates friction when users want quick and meaningful answers.

**OPERATOR** aims to provide a more natural interaction model where users can **ask questions conversationally and receive meaningful information through a single interface**.

---

# 💡 Our Approach

OPERATOR combines:

**Voice Interface + AI Processing + Hybrid Retrieval + RAG + Multilingual Interaction + Visual Feedback**

This creates an interface that behaves more like an intelligent research assistant rather than a conventional search page.

The system is designed around the following principle:

```text
ASK
 ↓
UNDERSTAND
 ↓
RETRIEVE
 ↓
PROCESS
 ↓
ANSWER
 ↓
EXPLORE
```

---

# 🖥️ Interface Highlights

The website contains multiple sections designed around an immersive research-console experience:

* Landing / introduction section
* Interactive voice query interface
* Voice activity visualization
* Query input and processing area
* AI-generated answer section
* Multilingual response area
* Research/data statistics
* Processing visualization
* Technical explanation section
* Interactive query suggestions

---

# 🛠️ Technology Stack

## Frontend

* React.js
* JavaScript
* HTML5
* CSS3
* Web Audio / voice interaction APIs

## Backend

* Python
* REST API
* AI / NLP processing
* Retrieval-Augmented Generation (RAG)
* Hybrid retrieval pipeline

## Information Retrieval

* FAISS
* Semantic search
* Hybrid retrieval
* Vector indexing
* Knowledge-base search

## AI / Processing

* Speech-to-Text
* Natural Language Processing
* Large Language Model / AI API
* Retrieval-Augmented Generation

## Development & Deployment

* Git
* GitHub
* VS Code
* Render
* Kaggle

---

# 📂 Project Structure

```text
OPERATOR/
│
├── assets/
│   └── # Images, icons, UI assets and other static resources
│
├── backend/
│   ├── # Voice processing pipeline
│   ├── # Query processing
│   ├── # RAG pipeline
│   ├── # Hybrid retrieval
│   └── # Backend API / AI services
│
├── data/
│   ├── # Knowledge-base data
│   ├── # Indexed data
│   └── # Retrieval datasets
│
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   ├── styles.css
│   │   └── ...
│   │
│   ├── public/
│   ├── package.json
│   └── ...
│
├── kaggle/
│   ├── # Kaggle notebooks
│   └── # Model and retrieval experiments
│
├── results/
│   ├── # Benchmark results
│   ├── # Latency measurements
│   ├── # Retrieval evaluation
│   └── # Performance reports
│
├── .env.example
├── .gitignore
├── README.md
├── render.yaml
└── requirements.txt
```

---

## 📁 Directory Overview

| Directory / File   | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `assets/`          | Images, icons and other static project assets            |
| `backend/`         | AI processing, voice pipeline, RAG and retrieval backend |
| `data/`            | Knowledge-base and indexed retrieval data                |
| `frontend/`        | React-based user interface                               |
| `kaggle/`          | Kaggle notebooks, experiments and optimization work      |
| `results/`         | Performance, latency and retrieval evaluation results    |
| `.env.example`     | Example environment-variable configuration               |
| `.gitignore`       | Files excluded from Git tracking                         |
| `render.yaml`      | Render deployment configuration                          |
| `requirements.txt` | Python backend dependencies                              |
| `README.md`        | Project documentation                                    |

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │        USER          │
                         │   Voice / Text Query │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FRONTEND       │
                         │   React Web Interface │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       BACKEND        │
                         │ Voice + Query Pipeline│
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         ▼                      ▼
                ┌─────────────────┐    ┌─────────────────┐
                │ Hybrid Retrieval│    │   AI / LLM      │
                │  + FAISS / RAG  │    │    Processing    │
                └────────┬────────┘    └────────┬────────┘
                         │                      │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   Answer Generation  │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         ▼                      ▼
                  ┌─────────────┐       ┌─────────────┐
                  │ Text Answer │       │Voice Answer │
                  └──────┬──────┘       └──────┬──────┘
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         ┌──────────────────────┐
                         │      OPERATOR        │
                         │    User Interface    │
                         └──────────────────────┘
```

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

## 2. Navigate to the Project

```bash
cd OPERATOR
```

## 3. Backend Setup

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

## 4. Environment Variables

Create your environment file:

```bash
cp .env.example .env
```

On Windows, you can manually copy `.env.example` to `.env`.

Add the required API keys and configuration values inside `.env`.

> **Never commit real API keys or secrets to GitHub.**

## 5. Frontend Setup

Navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

## 6. Open the Application

Open the local URL displayed in the terminal, usually:

```text
http://localhost:5173
```

---

# 🎤 How to Use

1. Open the OPERATOR website.
2. Allow microphone access when requested.
3. Activate the voice interface.
4. Ask a question naturally.
5. The system converts the speech into a query.
6. The backend processes the query.
7. Relevant information is retrieved using the retrieval pipeline.
8. The AI generates the final response.
9. The answer is displayed in the interface.
10. Users can interact with the response and explore the processing information.

---

# 🔥 What Makes OPERATOR Different?

## 1. Voice-First Interaction

The system is designed around natural voice interaction instead of treating voice as an additional feature.

---

## 2. Hybrid Retrieval

Instead of depending entirely on an LLM, OPERATOR uses retrieval mechanisms to locate relevant information before generating an answer.

This helps create a more research-oriented workflow.

---

## 3. RAG-Based Architecture

The system follows a Retrieval-Augmented Generation approach:

```text
User Query
    ↓
Query Understanding
    ↓
Relevant Data Retrieval
    ↓
Context Formation
    ↓
AI Generation
    ↓
Final Answer
```

---

## 4. Immersive Interface

The visual design represents an AI research console rather than a conventional chatbot.

The interface provides continuous visual feedback during processing.

---

## 5. Multilingual Accessibility

Users can interact using different languages, making information access more inclusive.

---

## 6. Transparent Processing

Instead of leaving users staring at a blank loading screen, the interface provides visual feedback about system activity and processing.

---

## 7. Performance-Oriented Design

The project includes dedicated benchmarking and optimization experiments to improve retrieval and response performance.

Performance analysis is maintained inside the `results/` directory.

---

# 🧪 Example Query

### User

> "What is the largest river in India?"

### OPERATOR

The system:

```text
Voice Input
    ↓
Speech Recognition
    ↓
Query Processing
    ↓
Information Retrieval
    ↓
AI Processing
    ↓
Answer
```

The resulting answer is presented through the interactive interface along with voice/activity visualization.

---

# 📊 Performance & Evaluation

The repository contains dedicated resources for evaluating system performance.

```text
kaggle/
   ↓
Experiments & Optimization
   ↓
Retrieval / Model Evaluation
   ↓
results/
   ↓
Performance Reports
```

The evaluation can include:

* Retrieval latency
* Query processing time
* FAISS performance
* Retrieval quality
* Fast-path performance
* Different retrieval modes
* System response latency

---

# 📈 Future Improvements

The project can be extended with:

* Real-time speech recognition
* More accurate multilingual speech processing
* Source citations for generated answers
* Document/PDF research
* Web-based research
* Personal knowledge bases
* Conversation history
* AI-generated summaries
* Voice-controlled navigation
* Offline speech recognition
* Advanced RAG architecture
* User authentication
* Mobile/PWA support
* Streaming AI responses
* Personalized research profiles

---

# 🔐 Privacy Considerations

The application should follow privacy-first principles:

* Microphone access should only be requested when required.
* Voice data should not be permanently stored without user consent.
* API keys should never be exposed in frontend source code.
* Sensitive information should not be logged unnecessarily.
* Environment variables should be stored securely.

---

# 🚀 Deployment

The project includes a `render.yaml` configuration for deployment using **Render**.

The deployment architecture can be represented as:

```text
                 GitHub Repository
                        │
                        ▼
                 Render Deployment
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
         Frontend              Backend
              │                   │
              └─────────┬─────────┘
                        ▼
                 OPERATOR System
```

---

# 👥 Team

**Hackathon:** - HH Goa 2026
**Task:** - Task 2
**Project:** - Voice-Driven AI Research Interface

### Team Members

* Member 1 — ABHISHEK JHA
* Member 2 — PARITOSH GAIDHANI
* Member 3 — ANURAJ CHAVAN


---

# 📜 License

This project is developed for **HH Goa 2026**.

---

# ⭐ Project Vision

> **"Ask naturally. Understand intelligently. Explore deeply."**

OPERATOR aims to transform information discovery from a traditional:

**Search → Click → Read**

workflow into a more natural:

**Speak → Understand → Retrieve → Process → Answer → Explore**

experience.

---

## 🏆 HH Goa 2026 — Task 2

Built with a focus on:

**Innovation • Voice AI • RAG • Retrieval • Accessibility • Performance • User Experience**
 #RAGInGoa.
