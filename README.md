[Website Link](https://kulkarni-harsh-signature-verification-tool-app-7r4du7.streamlit.app/)

# [Signature Verification Tool](https://kulkarni-harsh-signature-verification-tool-app-7r4du7.streamlit.app/)
A conventional method of verifying an individual's indentity is their Signatures.
This tool can check whether the signature is forged or not.

Signature Verification Tool based on Siamese Network.
The Network is constructed using Tensorflow and frontend is made using Streamlit.
The entire application is hosted on Streamlit Cloud.

## Research Paper
The research paper I studied was [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

Here Siamese Network is introduced which is a couple of identical Convolutional Neural Network. An Anchor Image (Reference Image) is passed through one CNN and the validation image (which may or may not be forged) is passed through the other. Feature Vectors coming out these CNNs are then compared with Siamese Distance function which if then passed to a Single cell having sigmoid function. 

Below is a diagram depicting the structure of Siamese Network.
![model_plot](https://user-images.githubusercontent.com/70194206/189485451-64370502-0f7c-4ed9-a882-22aa83cd60df.png)

For simplicity I am resizing all images to 100 x 200 px (height x width). 
All images are converted to grayscale and the grayscale values are standardised.

## Dataset
I have combined two different datasets
* [ICDAR 2009 Signature Verification Competition (SigComp2009)](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2009_Signature_Verification_Competition_(SigComp2009))
* [ICDAR 2011 Signature Verification Competition (SigComp2011)](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)) -- Only Dutch Signatures

* Code for combining dataset as per their nomenclatures is provided in notebooks folder.

## Screenshots
![image](https://user-images.githubusercontent.com/70194206/189485387-6832d19f-0b09-4a79-97ca-b390c5f577b5.png)
![image](https://user-images.githubusercontent.com/70194206/189485431-1f368325-399d-41ce-ad33-49935281bc06.png)

The performance of this model can be further improved by training on the entire dataset (rather than taking a sample from it) and by increasing the number of epochs but due to resource constraints, it cannot be further improved.
