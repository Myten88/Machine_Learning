import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from scipy.ndimage import center_of_mass, shift

# Jag börjar med att förbereda för hanteringen av bilden som kommer ritas genom att skapa en funktion som ska 
# hantera bilden så som dom gjort i datasetet MNIST, alltså att de har gjort den grayscale, plockat ut de pixlar
# som är skrivda på, resizat de till 20x20 för att sedan lägga det i en tom 28x28 bild och sist ändra så att mittpunkten
# av pixlarna blir mittpunkten av bilden.

def canvas_to_mnist(img_rgba):
    # Konvertera RGBA -> grayscale
    gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)

    # invert: MNIST har vit bakgrund, svart siffra
    gray = 255 - gray

    # binarize (lite tolerant)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # bounding box
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    roi = binary[y0:y1+1, x0:x1+1]

    # Skala om så att den blir 20x20
    h, w = roi.shape
    scale = 20.0 / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    roi_resized = cv2.resize(roi, new_size, interpolation=cv2.INTER_AREA)

    # Placera i en tom 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - roi_resized.shape[0]) // 2
    x_offset = (28 - roi_resized.shape[1]) // 2
    canvas[
        y_offset:y_offset + roi_resized.shape[0],
        x_offset:x_offset + roi_resized.shape[1]
    ] = roi_resized

    # Gör en "center of mass alignment"
    cy, cx = center_of_mass(canvas)
    shift_y = 14 - cy
    shift_x = 14 - cx
    canvas = shift(canvas, (shift_y, shift_x), mode='constant')

    return canvas


# Här börjar själva Streamlit-appen: 
st.title("Skriv en siffra och låt modellen gissa")
st.markdown("""Här nere kan du rita en siffra mellan 0-9 och låta modellen gissa vilken siffra du skrev.
            Ju finare du skriver desto lättare kommer modellen kunna gissa. """)

# Gör en canvas som siffran ska skrivas i:
canvas_result = st_canvas(
    fill_color='black',
    stroke_width=(5),
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




