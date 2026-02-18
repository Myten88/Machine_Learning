import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.title("Skriv en siffra och låt modellen gissa")
st.markdown("""Här nere kan du rita en siffra mellan 0-9 och låta modellen gissa vilken siffra du skrev.
            Ju finare du skriver desto lättare kommer modellen kunna gissa. """)

canvas_result = st_canvas()
st_canvas(
    stroke_width=(1, 25, 3),
    stroke_color='black',
    update_streamlit=True,
    drawing_mode='freedraw',
    key='canvas',
)



