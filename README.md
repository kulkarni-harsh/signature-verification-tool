
# Signature Verification Tool
A conventional method of verifying an individual's indentity is their Signatures.
This tool can check whether the signature is forged or not.

Signature Verification Tool based on Siamese Network.
The Network is constructed using Tensorflow and frontend is made using Streamlit.

## Research Paper
The research paper I studied was [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

Here Siamese Network is introduced which is a couple of identical Convolutional Neural Network. An Anchor Image (Reference Image) is passed through one CNN and the validation image (which may or may not be forged) is passed through the other. Feature Vectors coming out these CNNs are then compared with Siamese Distance function which if then passed to a Single cell having sigmoid function. 

Below is a diagram depicting the structure of Siamese Network.


For simplicity I am resizing all images to 100 x 200 px (height x width). 
All images are converted to grayscale and the grayscale values are standardised.

## Dataset
I have combined two different datasets
* [ICDAR 2009 Signature Verification Competition (SigComp2009)](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2009_Signature_Verification_Competition_(SigComp2009))
* [ICDAR 2011 Signature Verification Competition (SigComp2011)](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011))
Code for combining dataset as per their different nomenclature is provided in notebooks folder.