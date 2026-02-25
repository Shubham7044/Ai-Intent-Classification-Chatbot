import streamlit as st
import joblib
from src.preprocess import clean_text

# Load trained model
model = joblib.load("chatbot_model.pkl")

def rule_based_intent(text: str):
    t = text.lower().strip()
    if "volume" in t and ("up" in t or "increase" in t):
        return "volumeUp"
    if "volume" in t and ("down" in t or "decrease" in t):
        return "volumeDown"
    if "stop" in t:
        return "stop"
    if t in ["yes", "yeah", "yep", "sure", "ok", "okay"]:
        return "yesNode"
    if t in ["no", "nope", "nah"]:
        return "noNode"
    return None

def predict_intent_with_confidence(text: str):
    # Rule-based shortcut (100% confidence)
    rb = rule_based_intent(text)
    if rb:
        return rb, 1.00

    cleaned = clean_text(text)

    # ML prediction
    intent = model.predict([cleaned])[0]

    # Confidence (if classifier supports probabilities)
    clf = model.named_steps.get("clf")
    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
        confidence = float(probs.max())
    else:
        confidence = None

    return intent, confidence

RESPONSES = {
    "yesNode": "Got it! 👍",
    "noNode": "Okay, no problem.",
    "volumeUp": "Turning the volume up 🔊",
    "volumeDown": "Turning the volume down 🔉",
    "stop": "Stopping the current action.",
    "totalStop": "All actions stopped.",
    "listOfGames": "Here are the available games 🎮",
    "playOtherGame": "Switching to another game.",
    "gamesCounter": "Here’s how many games you’ve played.",
    "timesPlayed": "Here’s your play count."
}

# ---- Streamlit UI ----
st.set_page_config(page_title="AI Intent Classification Chatbot", page_icon="🤖")
st.title("🤖 AI Intent Classification Chatbot")

st.write("Type a message (e.g., 'turn volume up', 'stop the game'):")

user_input = st.text_input("Your message")

if user_input:
    intent, conf = predict_intent_with_confidence(user_input)

    st.markdown(f"**Predicted Intent:** `{intent}`")
    if conf is not None:
        st.caption(f"Confidence: {conf:.2f}")

    st.success(RESPONSES.get(intent, "I understood your intent!"))