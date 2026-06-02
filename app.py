import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Smart Tyre Wear AI",
    page_icon="🚗",
    layout="wide"
)

# ---------- CSS UI ----------

st.markdown("""
<style>

.main{
background-color:#0E1117;
color:white;
}

.title{
font-size:42px;
font-weight:700;
color:#00E5FF;
text-align:center;
}

.subtitle{
font-size:18px;
color:#BBBBBB;
text-align:center;
}

.prediction-box{
background:#1C2333;
padding:25px;
border-radius:18px;
box-shadow:0px 0px 15px rgba(0,229,255,0.3);
}

.metric-card{
background:#151B28;
padding:20px;
border-radius:15px;
text-align:center;
}

</style>
""",unsafe_allow_html=True)

# ---------- LOAD MODEL ----------

model=tf.keras.models.load_model(
'tire_model.h5'
)

class_names = sorted([

'0 km','10000 km','100 km','10500 km','11000 km',
'11500 km','12000 km','1250 km','13000 km','13500 km',
'14000 km','15000 km','1500 km','15500 km','16500 km',
'17500 km','1750 km','18500 km','19000 km','2000 Km',
'20500 km','21000 km','23000 km','23500 km','24500 km',
'25000 km','250 km','26000 km','26500 km','27500 km',
'28000 km','32000 km','3250 km','3750 km','38000 km',
'39000 km','4000 km','41500 km','42500 km','4250 Km',
'43500 km','44500 km','4500 km','4750 km','49000 km',
'50000 km','50 km','51000 km','52000 km','53000 km',
'57000 km','5750 Km','60000 km','6000 km','61000 km',
'6250 Km','66000 km','7250 km','75000 km','7500 km',
'79000 km','80000 km'

])

# ---------- HEADER ----------

st.markdown(
'<p class="title">🚗 Smart Tyre Wear Prediction System</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Deep Learning Based Tyre Condition Analysis for Automotive Industry</p>',
unsafe_allow_html=True
)

st.divider()

# ---------- LAYOUT ----------

left,right=st.columns([1,1])

with left:

    uploaded=st.file_uploader(
        "Upload Tyre Image",
        type=['jpg','jpeg','png']
    )

    if uploaded:

        image=Image.open(uploaded)

        st.image(
            image,
            caption="Uploaded Tyre Image",
            use_container_width=True
        )

with right:

    if uploaded:

        img=image.resize((224,224))

        img=np.array(img)/255.0

        img=np.expand_dims(img,axis=0)

        prediction=model.predict(img)

        score=np.argmax(prediction)

        confidence=float(np.max(prediction)*100)

        predicted_class=class_names[score]

        st.markdown(
        f"""
        <div class="prediction-box">

        <h2 style='color:#00E5FF'>
        Prediction Result
        </h2>

        <h1 style='color:white'>
        {predicted_class}
        </h1>

        </div>
        """,
        unsafe_allow_html=True
        )

        st.progress(confidence/100)

        col1,col2=st.columns(2)

        with col1:

            st.markdown(
            f"""
            <div class="metric-card">

            <h3>Confidence</h3>

            <h1 style='color:#00FF95'>
            {confidence:.2f}%
            </h1>

            </div>
            """,
            unsafe_allow_html=True
            )

        with col2:

            st.markdown(
            f"""
            <div class="metric-card">

            <h3>AI Model</h3>

            <h1 style='color:#FF9800'>
            MobileNetV2
            </h1>

            </div>
            """,
            unsafe_allow_html=True
            )

st.divider()

st.caption(
"SC2026 Project • Automotive AI • CNN Transfer Learning • Streamlit Deployment"
)
