#========================================
#-------------- Importer ----------------
#========================================

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from scipy.ndimage import center_of_mass, shift
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

#=============================================
#-------------- Pre-procesing ----------------
#=============================================

# Jag börjar med att förbereda för hanteringen av bilden som kommer ritas genom att skapa en funktion som ska 
# hantera bilden så som dom gjort i datasetet MNIST, alltså att de har gjort den grayscale, plockat ut de pixlar
# som är skrivna på (bounding box), resizat de till 20x20 för att sedan lägga det i en tom 28x28 bild och sist ändra så att mittpunkten
# av pixlarna blir mittpunkten av bilden.
# Jag har även lagt till padding efter bounding boxen för att inte sifforna skulle råkas kapa utan att det är lite utrymme att spela på vid skalningen
# Jag la också till en dilation på 2x2 om bilden skulle skalas ner, detta eftersom jag märkte att min modell predikterade mycket sämre när jag skrev strora siffor
# än när jag skrev små. Jag kollade och såg att det var mycket mindre ink på de större bilderna än de små och det var då jag la till dilationen. 

def canvas_to_mnist(img_rgba):
    """Hantering av canvas bild till rätt format för modellen"""

    # Konvertera RGBA -> grayscale
    gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)

    # invertar eftersom MNIST har vit bakgrund och svart siffra
    gray = 255 - gray

    # binarize (lite tolerant)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # bounding box
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # Lägg padding runt bounding box för att inte klippa för tight satt på 3 för mindre klippte vissa siffor och mer skapade problem med små 9or och 8or som då tolkades fel
    pad = 3 
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(binary.shape[0] - 1, y1 + pad); x1 = min(binary.shape[1] - 1, x1 + pad)

    # Region of Interest: tar ut det området jag kartlagt med bounding boxen plus paddingen
    roi = gray[y0:y1+1, x0:x1+1]

    # Skala om så att den blir 20x20
    height, width = roi.shape
    scale = 20.0 / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    roi_resized = cv2.resize(roi, new_size, interpolation=cv2.INTER_AREA)

    # Placera i en tom 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - roi_resized.shape[0]) // 2
    x_offset = (28 - roi_resized.shape[1]) // 2
    canvas[
        y_offset:y_offset + roi_resized.shape[0],
        x_offset:x_offset + roi_resized.shape[1]
    ] = roi_resized

    # Dilation endast om vi skalar NER (stor ritning -> tunna streck)
    if scale < 1.0:
        kernel = np.ones((2, 2), np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=1)

    # Nomralisera intensiteten så att stora och små siffor tolkas lika
    canvas = canvas.astype(np.float32)
    if canvas.max() > 0:
        canvas /= canvas.max()
        canvas *= 255
    canvas = canvas.astype(np.uint8)

    # Gör en "center of mass alignment" så att bläckets tyngdpunkt blir centrerad på bilden
    cy, cx = center_of_mass(canvas)
    shift_y = 14 - cy
    shift_x = 14 - cx
    canvas = shift(canvas, (shift_y, shift_x), mode='constant')

    return canvas

# Gör en funktion för ladda in modellen - detta för att jag då kan använda @st.cache_resource vilket ökar prestandan då den sparar det i cachen.
@st.cache_resource
def load_model(): 
    """Laddar modellen till cachen för snabbare hantering"""
    return joblib.load('rbf_svc_clf_proba_best.joblib')

#===============================================
#-------------- Streamlit appen ----------------
#===============================================

st.title("Skriv en siffra och låt modellen gissa")
st.markdown("""Här nere kan du rita en siffra mellan 0-9 och modellen gissar vilken siffra du skrev.
            Ju finare du skriver desto lättare kommer modellen kunna gissa. """)

# Gör en canvas som siffran ska skrivas i:
canvas_result = st_canvas(
    fill_color='black',
    stroke_width=(15), #Hade 5 tidigare men alla andra i presentationen hade mycket större penna så jag testar det också 
    stroke_color='black',
    background_color='white',
    update_streamlit=True,
    width=500,
    height=500,
    drawing_mode='freedraw',
    display_toolbar=True,
    key='canvas',
)

# Konvertera bilden som gjordes i canvasen så att den matchar MNIST datasetet genom funktionen canvas_to_mnist
mnist_img = canvas_to_mnist(canvas_result.image_data)

# Laddar modellen och reshapear bilden och konventerar den till float
model = load_model()
X = mnist_img.reshape(1, -1).astype(np.float32) 

# Predikteringen och dess stats
proba = model.predict_proba(X)[0]
y_pred = proba.argmax()
confidence = proba[y_pred]

# Initiera historik en gång per session
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# Stapeldiagrammet & Top 3 för gissningarna och dess säkerhet samt knappar för spara resultatet till tabell eller tömma tabellen
if int(mnist_img.sum()) > 0:
    if confidence > 0.55:
        st.markdown(f'Modellen gissade på {y_pred} med {confidence:.2%} säkerhet')

        st.image(mnist_img, caption="Efter preprocessing", clamp=True)

        col1, col2 = st.columns([2,1])

        # Stapeldiagram
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
            ax.bar(np.arange(10), proba)
            ax.set_ylim(0, 1)
            ax.set_xticks(np.arange(10))
            ax.set_title(f"Fördelningen av gissningarna")
            st.pyplot(fig, use_container_width=False)

        # Top 3 med dess säkerhet
        with col2:
            top_idx = np.argsort(proba)[::-1][:3]
            st.markdown("**Top 3:**")
            for i in top_idx:
                st.write(f"**{i}** med {proba[i]:.2%} säkerhet")

            # Knappar: spara / töm
            c_save, c_clear = st.columns(2)

        # Sparknappen
        with c_save:
            if st.button("Spara resultat"):
                row = {
                    "Tid": datetime.now().strftime("%H:%M:%S"),
                    "Pred": int(top_idx[0]), "Säkerhet pred": float(proba[top_idx[0]]),
                    "Top 2": int(top_idx[1]), "Säkerhet Top 2": float(proba[top_idx[1]]),
                    "Top 3": int(top_idx[2]), "Säkerhet Top 3": float(proba[top_idx[2]]),
                }
                st.session_state.pred_history.append(row)

        # Töm knappen
        with c_clear:
            if st.button("Töm historik"):
                st.session_state.pred_history = []

    else:
        st.write("Modellen är för osäker för att kunna prediktera. Kontrollera att du verkligen skrivit en siffra mellan 0-9")

# tabellen för att kunna spara resultatet man väljer att spara
st.subheader("Sparade resultat")
if st.session_state.pred_history:
    df = pd.DataFrame(st.session_state.pred_history).iloc[::-1].reset_index(drop=True)
    st.dataframe(df, use_container_width=True)
else:
    st.caption("Inga sparade resultat ännu.")