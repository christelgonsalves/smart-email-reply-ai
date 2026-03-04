import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Page Configuration
st.set_page_config(
    page_title="Smart Email Auto-Reply Generator",
    page_icon="📧",
    layout="wide"
)

# Clean Styling
st.markdown("""
<style>
.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Load Model (Cached)
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Header
st.title("📧 Smart Email Auto-Reply Generator")
st.markdown("### AI-powered assistant using T5 (FLAN-T5) architecture")
st.markdown("---")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    email_text = st.text_area("📥 Paste the received email here:", height=250)

with col2:
    st.sidebar.header("⚙ Customization Panel")

    tone = st.sidebar.selectbox(
        "Select Tone:",
        ["Formal", "Friendly", "Apology", "Professional"]
    )

    length = st.sidebar.slider("Reply Length:", 80, 300, 150)

    temperature = st.sidebar.slider("Creativity Level:", 0.3, 1.0, 0.7)

    signature = st.sidebar.text_input("Your Signature Name:", "Christel")

# Generate Button
if st.button("🚀 Generate Smart Reply"):

    if email_text.strip() == "":
        st.warning("Please enter an email first.")
    else:

        email_text = email_text.strip()

        # 1️⃣ Email Category Detection
        category_prompt = f"""
Classify the following email into one category:
Complaint, Leave Request, Meeting, Interview, General Inquiry

Email:
{email_text}

Category:
"""
        cat_ids = tokenizer.encode(category_prompt, return_tensors="pt", truncation=True)
        cat_output = model.generate(cat_ids, max_length=20)
        category = tokenizer.decode(cat_output[0], skip_special_tokens=True)

        st.success(f"📌 Detected Email Type: {category}")

        # 2️⃣ Generate Proper Reply (Improved Prompt)
        prompt = f"""
You are a professional email assistant.

Write a proper reply to the email below.
Do NOT repeat or rewrite the original email.
Respond as if you are the receiver.
Provide a meaningful and context-aware response.

Tone: {tone}

Received Email:
{email_text}

Write only the reply email:
"""

        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)

        outputs = model.generate(
            input_ids,
            max_length=length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            num_beams=4
        )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

        final_reply = reply + f"\n\nBest regards,\n{signature}"

        st.markdown("### ✉ Generated Reply")
        st.text_area("", final_reply, height=250)

        # 3️⃣ Generate Subject Line
        subject_prompt = f"""
Generate a short and appropriate subject line for the following email reply:

{reply}

Subject:
"""

        subject_ids = tokenizer.encode(subject_prompt, return_tensors="pt", truncation=True)
        subject_output = model.generate(subject_ids, max_length=20)
        generated_subject = tokenizer.decode(subject_output[0], skip_special_tokens=True)

        st.markdown("### 📌 Suggested Subject Line")
        st.text_input("", generated_subject)


