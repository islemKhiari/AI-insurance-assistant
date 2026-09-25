# 🤖 Maghrebia AI Insurance Recommendation Assistant

> An intelligent conversational assistant for personalized insurance recommendations, developed during a summer internship at **Assurances Maghrebia**.

## 📌 Project Overview

This project aims to develop an AI-powered conversational assistant that helps users identify their insurance needs and discover insurance products that may be relevant to their personal and professional profiles.

Instead of asking users to manually browse through different insurance products, the assistant conducts a natural conversation, collects relevant information about the user's profile, and uses a Machine Learning model to generate personalized insurance recommendations.

The solution combines:

- 💬 A conversational chatbot powered by **Qwen 2.5**
- 🧠 A **multi-label Machine Learning recommendation model**
- ⚙️ A **Flask REST API**
- 🌐 A web interface built with **HTML, CSS and JavaScript**
- 📊 A dataset containing **50,000 insurance-related profiles**
- 🔗 Integration between conversational AI and Machine Learning

---

## 🎯 Objectives

The main objectives of the project are to:

- Provide a simple conversational interface for users.
- Collect user information through natural language.
- Identify the user's insurance needs from their profile.
- Recommend relevant insurance products.
- Provide multiple recommendations with prediction scores.
- Make insurance products easier to understand and discover.
- Combine Large Language Models with traditional Machine Learning techniques.

---

## 🏗️ System Architecture

The application follows the following architecture:

```text
                    👤 User
                       │
                       ▼
              🌐 Web Interface
             HTML / CSS / JavaScript
                       │
                       ▼
              🤖 Ollama / Qwen 2.5
           Conversation & Profile Extraction
                       │
                       ▼
              📄 User Profile JSON
                       │
                       ▼
                 ⚙️ Flask API
                  /predict
                       │
                       ▼
              🔄 Data Preprocessing
                       │
                       ▼
             🧠 ML Recommendation Model
              Multi-Label Classification
                       │
                       ▼
             📊 Insurance Predictions
                       │
                       ▼
              💬 Recommendations
                 displayed to user




🛠️ Technologies
Python
Scikit-learn
Pandas
NumPy
Flask
Flask-CORS
Ollama
Qwen 2.5
HTML5
CSS3
JavaScript


## 🚀 How to Run

Run the following commands in separate terminals:

### 1. Start the Flask backend

```bash
cd ml_backend
py app.py
2. Start Ollama
ollama serve
3. Start the frontend

From the project root folder:

py -m http.server 8000

Then open the application in your browser:

http://localhost:8000




👩‍💻 Internship Context

This project was developed during a summer internship at Assurances Maghrebia, with a focus on Artificial Intelligence, Machine Learning, conversational AI, and personalized insurance recommendations.
