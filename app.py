import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Icon & title of the page :
img = Image.open("1.jpg")
st.set_page_config(page_title="Face Detection",page_icon=img,layout="wide")

# Hide Menu_Bar & Footer :
hide_menu_style = """
    <style>
    #MainMenu {visibility : hidden;}
    footer {visibility : hidden;}
    </style>
"""
st.markdown(hide_menu_style , unsafe_allow_html=True)

# Set the background image :

Background_image = """

<style>
[data-testid="stAppViewContainer"] > .main
{
background-image: url("https://img.freepik.com/free-vector/gradient-red-color-background-modern-design-abstract_343694-2156.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=ais");

background-size : 100%
background-position : top left;

background-position: center;
background-size: cover;
background-repeat : repeat;
background-repeat: round;


background-attachment : local;

background-image: url("https://img.freepik.com/free-vector/gradient-red-color-background-modern-design-abstract_343694-2156.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=ais");
background-position: right bottom;
background-repeat: no-repeat;
}  

[data-testid="stHeader"]
{
background-color : rgba(0,0,0,0);
}

</style>                                
"""

st.markdown(Background_image,unsafe_allow_html=True)
    
st.sidebar.markdown("<p style='text-align: center; color: white;font-family:Stencil;font-size:55px'>FACE DETECTION</p>", unsafe_allow_html=True)
st.sidebar.image("https://img.freepik.com/free-vector/man-face-scan-biometric-digital-technology_24908-56401.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=ais",width=300)
st.sidebar.image("https://img.freepik.com/free-vector/tiny-people-scientists-identify-womans-emotions-from-voice-face-emotion-detection-emotional-state-recognizing-emo-sensor-technology-concept_335657-2442.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=ais",width = 300)

st.markdown("""
            <marquee behavior="scroll" scrollamount="10" style='color:#470505;font-family:Castellar;background-color:#F19595;font-size:30px'> Face detection refers to the process of automatically detecting and localizing faces in images or video streams, typically using algorithms and machine learning techniques. The primary objective of face detection is to identify and extract facial regions accurately and efficiently from visual data. Face detection algorithms typically analyze patterns, structures, and characteristics of faces, such as skin tone, texture, shape, and arrangement of facial features (eyes, nose, mouth, etc.) </marquee>
            
            """,unsafe_allow_html=True)
  
c1,c2 = st.columns([3,7])
with c1:
    st.image("https://img.freepik.com/premium-vector/isometric-unlocking-smartphone-with-biometric-facial-identification-biometric-identification-facial-recognition-system-concept-vector-illustration-business-infographic-banner_589019-2569.jpg?size=626&ext=jpg&ga=GA1.2.2087154549.1663432512&semt=ais",width=200)
with c2:
    st.markdown("<p style= 'color: black; font-family:Hobo Std;font-size:20px'> Face detection is a computer vision technology that involves identifying and locating human faces within digital images or video frames. It is an essential component of numerous applications, including facial recognition, image analysis, biometrics and more. Face detection plays a vital role in numerous fields, improving automation, security, and human-computer interaction by accurately identifying and localizing faces in images and videos.</p>", unsafe_allow_html=True)


Pre_Trained_Dataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

file_uploader = st.file_uploader("Choose the file",type = ['jpg','png','jpeg'])

if file_uploader is not None:
    
    file_path = file_uploader.name 
    
    image = Image.open(file_uploader)
    
    image = np.array(image)
    figure = plt.figure()
    
    plt.imshow(image)
    
    plt.axis("off")
    
    st.pyplot(figure)
    
    st.markdown("<h6 style='font-family:Footlight MT Light;color:#470541;text-align:center;font-size:20px'>Image Uploaded Successfully...</h6>",unsafe_allow_html=True)

st.balloons()

if file_uploader:
    st.markdown("<h6 style='color:#350227;text-align:center;font-size:25px;font-family:Footlight MT Light'>Processing..</h6>",unsafe_allow_html=True)

def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = Pre_Trained_Dataset.detectMultiScale(gray)
for x,y,w,h in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2) 
    
result = Image.fromarray(image)

a1,a2 = st.columns([5,5])

with a1:
    pre = st.button("Preview",key='1')
    if pre:
        st.image(result)
with a2:
    btn = st.button("Download")
    if btn:
        st.markdown(get_image_download_link(result,file_path,'Download Image'),unsafe_allow_html=True)
    