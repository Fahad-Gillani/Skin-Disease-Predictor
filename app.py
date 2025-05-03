import streamlit as st
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from PIL import Image
import time
import os

# ====== Streamlit Page Config ======
st.set_page_config(page_title="Skin Disease Predictor", page_icon="🩺", layout="centered")

# ====== CSS Styling ======
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        h1 {
            color: #ff4b4b;
            text-align: center;
            font-size: 2.5rem;
        }
        .stFileUploader {
            border: 2px dashed #ff4b4b;
            padding: 10px;
            border-radius: 10px;
        }
        .stImage {
            display: flex;
            justify-content: center;
        }
        .stSuccess {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Skin Disease Predictor")
st.write("Upload an image of the affected skin area, and our AI will analyze it.")

# ====== Google Drive Model Download ======
@st.cache_resource
def load_trained_model():
    model_path = "cnn_model.h5"
    
    if not os.path.exists(model_path):  # Check if model already exists
        with st.spinner("🔄 Downloading model... Please wait!"):
            file_id = "1M_pW_xgz_ZXdbCpKnXHlibTU9F-HI-t9"  # Extracted from Drive link
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    
    return load_model(model_path)

# Load model
model = load_trained_model()

# ====== Image Preprocessing ======
def preprocess_image(image, target_size=(224, 224)):  
    img = Image.open(image)   
    img = img.convert("RGB")   
    img = img.resize(target_size)  
    img = np.array(img) / 255.0   
    img = np.expand_dims(img, axis=0)  
    return img

# ====== Image Upload ======
image = st.file_uploader("📸 Upload Image Here", type=["jpg", "jpeg", "png"])

if image is not None:
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)
    
    with st.spinner("🔍 Analyzing the image..."):
        time.sleep(2)  

    img = preprocess_image(image)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0] 

    # ====== Class Labels ======
    class_labels = {
        0: "Acne and Rosacea", 1: "Actinic Keratosis", 2: "Atopic Dermatitis", 
        3: "Cellulitis", 4: "Eczema", 5: "Exanthems", 6: "Herpes", 7: "Pigmentation",
        8: "Lupus", 9: "Melanoma", 10: "Poison Ivy", 11: "Psoriasis", 12: "Seborrheic Keratoses",
        13: "Systemic Disease", 14: "Tinea", 15: "Urticaria", 16: "Vascular Tumors", 
        17: "Vasculitis", 18: "Warts"
    }

    # ====== Display Result ======
    st.success(f"🩹 **Predicted Disease:** {class_labels[predicted_class]}")
        # ====== Suggested Medicines ======
    medicine_suggestions = {
        0: ["Benzoyl Peroxide", "Clindamycin", "Adapalene", "Tretinoin", "Azelaic Acid", "Doxycycline", "Isotretinoin"],
        1: ["Fluorouracil", "Imiquimod", "Diclofenac", "Ingenol mebutate", "Efudex", "Picato", "Solaraze"],
        2: ["Hydrocortisone", "Tacrolimus", "Pimecrolimus", "Antihistamines", "Dupilumab", "Cetrizine", "Eucerin Cream"],
        3: ["Cephalexin", "Clindamycin", "Dicloxacillin", "Vancomycin", "Linezolid", "Penicillin", "Amoxicillin"],
        4: ["Hydrocortisone", "Mometasone", "Antihistamines", "Tacrolimus", "Coal Tar", "Cetrizine", "Eucerin"],
        5: ["Paracetamol", "Ibuprofen", "Diphenhydramine", "Calamine Lotion", "Cetirizine", "Hydrocortisone", "Acyclovir"],
        6: ["Acyclovir", "Valacyclovir", "Famciclovir", "Abreva", "Penciclovir", "Ibuprofen", "Lidocaine gel"],
        7: ["Hydroquinone", "Tretinoin", "Azelaic Acid", "Corticosteroids", "Niacinamide", "Vitamin C", "Kojic Acid"],
        8: ["Hydroxychloroquine", "Prednisone", "Methotrexate", "Azathioprine", "Belimumab", "Mycophenolate", "NSAIDs"],
        9: ["Dacarbazine", "Interferon-alpha", "Nivolumab", "Pembrolizumab", "Ipilimumab", "BRAF Inhibitors", "MEK Inhibitors"],
        10: ["Hydrocortisone", "Calamine Lotion", "Diphenhydramine", "Oatmeal Bath", "Prednisone", "Cetirizine", "Antihistamines"],
        11: ["Topical Steroids", "Coal Tar", "Vitamin D Analogues", "Methotrexate", "Biologics", "Salicylic Acid", "Cyclosporine"],
        12: ["Salicylic Acid", "Urea Cream", "Cryotherapy", "Topical Retinoids", "Hydrocortisone", "Laser Therapy", "Electrosurgery"],
        13: ["Prednisone", "Methotrexate", "Cyclophosphamide", "Azathioprine", "Mycophenolate", "Rituximab", "NSAIDs"],
        14: ["Clotrimazole", "Miconazole", "Terbinafine", "Fluconazole", "Griseofulvin", "Ketoconazole", "Tolnaftate"],
        15: ["Antihistamines", "Corticosteroids", "Omalizumab", "Montelukast", "Cetirizine", "Loratadine", "Hydroxyzine"],
        16: ["Propranolol", "Prednisone", "Vincristine", "Interferon", "Laser Therapy", "Atenolol", "Timolol"],
        17: ["Prednisone", "Cyclophosphamide", "Azathioprine", "Methotrexate", "Mycophenolate", "Rituximab", "NSAIDs"],
        18: ["Salicylic Acid", "Cryotherapy", "Imiquimod", "Cantharidin", "Trichloroacetic Acid", "Laser Treatment", "Podophyllin"]
    }

    suggested_meds = medicine_suggestions.get(predicted_class, ["No medicines available."])

    st.markdown("### 💊 Suggested Medicines:")
    for med in suggested_meds:
        st.markdown(f"- {med}")

    # ====== Medical Disclaimer ======
    st.markdown(
        "<p style='color:red; font-weight:bold;'>⚠️ Note: Use medicines only as prescribed by a certified dermatologist.</p>",
        unsafe_allow_html=True
    )
