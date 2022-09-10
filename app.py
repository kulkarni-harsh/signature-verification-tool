import numpy as np
import cv2
import streamlit as st

from functions import preprocess_test_image
st.set_page_config(layout="wide")

headerSite=st.container()
introduction=st.container()
implementation=st.container()
dataset=st.container()
testing=st.container()
credits=st.container()

with headerSite:
    st.title('Signature Verification Tool')

with introduction:
    st.subheader('Introduction')
    st.write("""
        A conventional method of verifying an individual's indentity is their Signatures.
        Someone with malicious intent may try to forge an individual's signature.
        This tool can check whether the signature is forged or not.
    """)

with implementation:
    st.subheader('Implementation Details')
    st.markdown("""
        The research paper I studied was [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)\n 
        Here Siamese Network is introduced which is a couple of identical Convolutional Neural Network. An Anchor Image (Reference Image) is passed through one CNN and the validation image (which may or may not be forged) is passed through the other. Feature Vectors coming out these CNNs are then compared with Siamese Distance function which if then passed to a Single cell having sigmoid function. 
        
        Below is a diagram depicting the structure of Siamese Network.
    """)
    st.image('assets/Siamese Network Research Paper diagram.png')
    st.markdown("""
        * **Anchor Image** -- Original Signature of an individual
        * **Validation Image** -- Signature which may or may not be forged.
    """)
    st.markdown("""
        For simplicity I am resizing all images to 100 x 200 px (height x width). 
        All images are converted to grayscale and the grayscale values are standardised.
    """)
    st.markdown("""
        Below is my version of Siamese model developed using Tensorflow.    
    """)
    st.image('assets/model_plot.png')

with dataset:
    st.subheader('Dataset Sources')
    st.markdown("""
        I have combined two different datasets
        * [ICDAR 2009 Signature Verification Competition (SigComp2009)](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2009_Signature_Verification_Competition_(SigComp2009))
        * [ICDAR 2011 Signature Verification Competition (SigComp2011)](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011))
        Code for combining dataset as per their different nomenclature is provided on my [Github repo](https://github.com/kulkarni-harsh/signature-verification-tool)
    """)
    

with testing:
    st.subheader("Test the Signature Verification Tool")
    anchor_image_col,validation_image_col=st.columns(2)
    anchor_file=anchor_image_col.file_uploader('Upload Anchor Image',['png','jpg'])
    validation_file=validation_image_col.file_uploader('Upload Validation Image',['png','jpg'])
    
    if anchor_file is not None:
        # Convert the file to an opencv image.
        anchor_bytes = np.asarray(bytearray(anchor_file.read()), dtype=np.uint8)
        anchor_image = cv2.imdecode(anchor_bytes, 0)
        anchor_image=preprocess_test_image(anchor_image,4,50)
        anchor_image_col.image(anchor_image)
        anchor_image_col.write(type(anchor_image))

    if validation_file is not None:
        # Convert the file to an opencv image.
        validation_bytes = np.asarray(bytearray(validation_file.read()), dtype=np.uint8)
        validation_image = cv2.imdecode(validation_bytes, 0)
        validation_image=preprocess_test_image(validation_image,4,50)
        validation_image_col.image(validation_image)
    
