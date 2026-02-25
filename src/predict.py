import joblib
from preprocess import clean_text

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
    # Rule-based shortcut
    rb = rule_based_intent(text)
    if rb:
        return rb, 1.00  # 100% confidence for rule-based matches

    cleaned = clean_text(text)

    # ML prediction
    intent = model.predict([cleaned])[0]

    # Confidence (probability) if supported
    confidence = None
    clf = model.named_steps.get("clf")

    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
        confidence = float(probs.max())
    else:
        confidence = None

    return intent, confidence

if __name__ == "__main__":
    print("🤖 Intent Predictor with Confidence (type 'exit' to quit)")

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == "exit":
            print("Bye 👋")
            break

        intent, conf = predict_intent_with_confidence(user_input)

        if conf is not None:
            print(f"Predicted Intent: {intent} (confidence: {conf:.2f})")
        else:
            print(f"Predicted Intent: {intent}")