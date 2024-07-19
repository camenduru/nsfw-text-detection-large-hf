import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

MODEL_URL = "MichalMlodawski/nsfw-text-detection-large"
TITLE = "🖼️🔍 Image Prompt Safety Classifier 🛡️"
DESCRIPTION = "✨ Enter an image generation prompt to classify its safety level! ✨"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_URL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_URL)

# Define class names with emojis and detailed descriptions
CLASS_NAMES = {
    0: "✅ SAFE - This prompt is appropriate and harmless.",
    1: "⚠️ QUESTIONABLE - This prompt may require further review.",
    2: "🚫 UNSAFE - This prompt is likely to generate inappropriate content."
}

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return CLASS_NAMES[predicted_class]

# Define Gradio interface
def gradio_interface(text):
    classification = classify_text(text)
    return f"🏷️ Classification: {classification}"

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=3, placeholder="🖊️ Enter your image generation prompt here..."),
    outputs=gr.Textbox(label="🔎 Classification Result"),
    title=TITLE,
    description=DESCRIPTION,
    examples=[
        ["A beautiful sunset over a calm ocean"],
        ["An inappropriate scene involving explicit content"]
    ],
    theme=gr.themes.Soft(primary_hue="blue"),
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()