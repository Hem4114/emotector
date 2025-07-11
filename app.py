import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
import os

from PIL import Image, ImageDraw

def make_image_circular(img_path):
    img = Image.open(img_path).convert("RGBA")
    np_img = np.array(img)

    h, w = img.size
    alpha = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.ellipse((0, 0, h, w), fill=255)

    np_alpha = np.array(alpha)
    np_img[:, :, 3] = np_alpha
    circ_img = Image.fromarray(np_img)

    return circ_img

# Load and make the logo circular
logo_path = "D:\Top Notch\emotector.png"  # Update this to your full path if needed
circular_logo = make_image_circular(logo_path)

# --- Streamlit Setup ---
st.set_page_config(page_title="Emotector", layout="centered")

# --- Session State for Username ---
if 'username' not in st.session_state:
    st.session_state.username = ""

# --- Welcome Page ---
if st.session_state.username == "":
    
    # Display circular logo and app title
    st.image(circular_logo, width=120)
    st.markdown("<h1 style='color:#1f77b4;'>Emotector</h1>", unsafe_allow_html=True)
    st.caption("AI-powered Emotion Detection from Facial Images")

    username = st.text_input("👋 Hello! What's your name?", placeholder="Eg: Dustin")
    if username:
        st.session_state.username = username
        st.rerun()

# --- Main App ---
else:
    st.image(circular_logo, width=120)
    st.markdown("<h1 style='color:#1f77b4;'>Emotector</h1>", unsafe_allow_html=True)
    st.caption(f"Welcome, **{st.session_state.username}**! Let's detect some emotions.")

    # Load model and face detector
    model = load_model("emotion_model.h5")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    emotions = ['😠 Angry', '🤢 Disgust', '😨 Fear', '😊 Happy', '😢 Sad', '😲 Surprise', '😐 Neutral']
    bar_colors = ['#e74c3c', '#9b59b6', '#8e44ad', '#2ecc71', '#3498db', '#f1c40f', '#95a5a6']
    emotion_messages = {
        'Angry': "It's okay to feel angry. Take a deep breath and try to calm down.",
        'Disgust': "If something feels off, it's okay to walk away. Protect your peace.",
        'Fear': "You are stronger than your fears. Face them one step at a time.",
        'Happy': "That's wonderful! Keep smiling and spread the joy. You look cute while smiling!",
        'Sad': "Don't be sad. Everything happens for a reason. Brighter days are coming.",
        'Surprise': "Whoa! Something unexpected, huh? Enjoy the excitement!",
        'Neutral': "Just chilling, huh? Sometimes peace is all we need."
    }

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Resize to mobile-friendly size
        max_width = 300
        ratio = max_width / float(image.width)
        new_height = int(image.height * ratio)
        resized_img = image.resize((max_width, new_height))
        st.image(resized_img, caption="Uploaded Image", use_container_width=False)

        # Face detection
        img_array = np.array(image.convert('L'))
        faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            st.warning("⚠️ No face detected in the image.")
        else:
            (x, y, w, h) = faces[0]
            face = img_array[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.reshape(1, 48, 48, 1) / 255.0

            predictions = model.predict(face)[0]
            emotion_index = np.argmax(predictions)
            predicted_emotion = emotions[emotion_index]
            plain_emotion = predicted_emotion.split(" ")[1]
            message = emotion_messages.get(plain_emotion, "")

            # Display result
            st.markdown("### 🎭 Detected Emotion:")
            st.subheader(predicted_emotion)
            st.markdown(f"<p style='font-size: 16px; color: #555;'>{message}</p>", unsafe_allow_html=True)

            if plain_emotion == "Happy":
                st.balloons()

            # Confidence chart
            st.markdown("### 📊 Prediction Chart:")
            fig, ax = plt.subplots(figsize=(6, 3), facecolor='none')
            fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')

            bars = ax.bar(emotions, predictions * 100, color=bar_colors)
            ax.set_ylabel("Confidence (%)", fontsize=10, color='white')
            ax.set_ylim(0, 100)
            ax.set_title("Emotion Prediction Chart", fontsize=12, color='white')
            ax.set_xticks(range(len(emotions)))
            ax.set_xticklabels(emotions, rotation=45, ha="right", fontsize=9, color='white')
            ax.tick_params(axis='y', labelsize=8, colors='white')

            for bar, score in zip(bars, predictions * 100):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f'{score:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='white'
                )

            st.pyplot(fig)

            # Save uploaded image temporarily
            img_path = f"{st.session_state.username}_img.jpg"
            image.save(img_path)

            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(200, 10, txt="Emotector", ln=True, align='C')
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Hello {st.session_state.username},", ln=True)
            pdf.cell(200, 10, txt=f"Detected Emotion: {plain_emotion}", ln=True)
            pdf.multi_cell(0, 10, txt=message)
            pdf.ln(5)
            pdf.image(img_path, w=120)

            # Generate dynamic PDF name
            clean_name = st.session_state.username.strip().replace(" ", "_").lower()
            pdf_output = f"{clean_name}_emotector.pdf"
            pdf.output(pdf_output)

            # Download button
            with open(pdf_output, "rb") as f:
                st.download_button("📄 Download Your Report", f, file_name=pdf_output, mime="application/pdf")

            # Cleanup
            os.remove(img_path)
            os.remove(pdf_output)

    # --- Rating Section ---
    st.markdown("---")
    st.markdown("### 🌟 We'd love your rating!")
    
    rating = st.slider("Rate your experience with Emotector:", 1, 5, 3)
    
    if st.button("Submit Rating"):
        with open("feedback.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {st.session_state.username}\n")
            f.write(f"Rating: {rating}/5\n")
            f.write("-" * 40 + "\n")
        st.success("Thanks for your rating! 🙏")

# --- Footer ---
st.markdown(
    "<hr style='margin-top:30px; margin-bottom:10px; border: 0; border-top: 1px solid #ccc;'>",
    unsafe_allow_html=True
)
st.markdown("<center>Powered by Coffee ☕ & Streamlit — Hem Bhatt 🫶 </center>", unsafe_allow_html=True)
