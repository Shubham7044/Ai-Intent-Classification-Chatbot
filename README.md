```markdown
# 🤖 AI Intent Classification Chatbot (NLP + ML + Streamlit)

An end-to-end AI chatbot that classifies user intents from natural language commands using NLP (TF-IDF) and Machine Learning (Logistic Regression). The system leverages a hybrid rule-based + ML approach for robust real-world behavior and exposes confidence scores for interpretability. It is deployed as both a CLI tool and an interactive Streamlit web application.

---

## ✨ Features

- 🧠 **Intent Classification (NLP + ML):** TF-IDF vectorization combined with a Logistic Regression classifier.
- 🔀 **Hybrid Inference:** Rule-based shortcuts for critical commands, augmented by ML models for general queries.
- 📊 **Confidence Scores:** Prediction confidence accompanies each classified intent to support uncertainty handling.
- 🌐 **Web UI:** Interactive Streamlit app enabling live demos and user interaction.
- 🧪 **CLI Predictor:** Quick command-line testing interface.
- 📁 **Real-World Dataset:** Supports JSON-formatted intent corpora for training and inference.
- 🧩 **Production-minded UX:** Low-confidence predictions prompt friendly rephrasing requests for better accuracy.

---

## 📥 Dataset

The dataset is sourced from the [Intent Corpus on Kaggle](https://www.kaggle.com/datasets/lorencpetr/chatbot-intent-classification), provided in JSON format.

### Example JSON structure:
```json
{
  "name": "intent-corpus-basic",
  "sentences": [
    {"text": "stop the game", "intent": "stop", "training": true},
    {"text": "turn volume up", "intent": "volumeUp", "training": true}
  ]
}
```

### Usage:

- Place your JSON file inside the `data/` directory using the path:  
  `data/intent-corpus-basic.json`
- You can swap in a richer intent corpus without modifying the training or inference pipelines.

---

## ⚙️ Setup & Installation

### 1) Create a Python virtual environment (optional but recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2) Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

Run the training script to preprocess data, train the logistic regression model, and save it:

```bash
python src/train.py
```

You will see a classification report and a message indicating:

```
✅ Model trained and saved as chatbot_model.pkl
```

---

## ▶️ Run the CLI Predictor

Launch a quick interactive session to test intent classification from the terminal:

```bash
python src/predict.py
```

### Sample CLI Interaction:

```
🤖 Intent Predictor with Confidence (type 'exit' to quit)
You: turn volume up
Predicted Intent: volumeUp (confidence: 1.00)

You: stop the game
Predicted Intent: stop (confidence: 1.00)

You: what games are available
Predicted Intent: listOfGames (confidence: 0.35)
```

---

## 🌐 Run the Web App (Streamlit)

Start the Streamlit app for a user-friendly interface with real-time intent prediction:

```bash
streamlit run app.py
```

### UI Highlights:

- Predicted intent displayed as a badge
- Confidence score (e.g. `0.29`, `1.00`)
- Friendly contextual response (e.g. “Stopping the current action.”)

### Example UI Outputs:

| Input                    | Predicted Intent | Confidence | Notes                               |
|--------------------------|------------------|------------|-----------------------------------|
| stop the game            | stop             | 1.00       | Rule-based shortcut handled safely |
| what games are available | listOfGames      | ~0.35      | ML guess with moderate confidence  |
| list all games           | listOfGames      | Low        | Possible rule-based fallback needed|

---

## 🧪 Example Inputs to Try

| Intent Category | Example User Inputs                                   |
|-----------------|-----------------------------------------------------|
| Volume          | "turn volume up", "increase the volume", "turn the volume down" |
| Control         | "stop", "stop the game"                             |
| Games           | "list all games", "what games are available", "play another game" |
| Yes / No        | "yes", "no", "sure", "nah"                          |

---

## 🧩 How It Works (Quick Overview)

1. **Preprocess**: Text is lowercased and cleaned.
2. **Vectorize**: Convert text to TF-IDF features (unigrams + bigrams, stopwords filtered).
3. **Model**: Logistic Regression classifies intents.
4. **Hybrid Layer**: Rule-based shortcuts catch critical commands (e.g. "stop").
5. **Inference**: Predict intent and provide confidence score.
6. **UI**: Streamlit presents results in real time with responses.

---

## 📈 Results & Observations

- Achieves ~80%+ accuracy on held-out samples; varies depending on corpus balance.
- Rule-based shortcuts guarantee perfect classification on safety-critical commands like `stop`.
- Confidence scores effectively expose prediction uncertainty, enabling UX improvements such as asking users to rephrase when confidence is low.

---

## 🔮 Future Improvements

- Add confusion matrix and per-intent precision/recall metrics in the web UI.
- Implement data augmentation to boost performance on low-frequency (rare) intents.
- Experiment with Linear SVM classifiers for improved short-text intent recognition.
- Develop a FastAPI endpoint for integration with automation tools (e.g., n8n).
- Introduce logging and a feedback loop for continuous improvement of the model.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/my-feature`).
3. Make your changes with clear commit messages.
4. Test your code thoroughly.
5. Submit a pull request describing your changes.

Please adhere to the existing code style and ensure documentation is updated as needed.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ leveraging NLP and ML techniques for intent classification.*
```
