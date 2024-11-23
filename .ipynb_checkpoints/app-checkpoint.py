import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('chest_xray_model.keras')

# Define a function to preprocess the uploaded image
def preprocess_image(image):
      image = image.convert("RGB")
      image = np.array(image.resize((224,224)))
      image = image.reshape([224,224,3])
      image = image[:,:,2]
      image = np.array(image)
      image = image.astype('float32')
      image = image/np.max(image)
      image = image.reshape(1,224,224,1)
      return image
    # Resize the image to the input shape required by the model
    #image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    # Convert the image to array
    #img_array = np.array(image) / 255.0
    # Add batch dimension
    #img_array = np.expand_dims(img_array, axis=0)
    #return img_array

# Define the main app
def main():
    st.title("Chest X-Ray Classifier")
    st.write("This app classifies chest X-ray images into Cancer, Pneumonia, or Normal.")
    
    # File uploader for the X-ray image
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        st.write("Classifying...")
        
        # Preprocess the image and make prediction
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        class_labels = ['Cancer', 'Pneumonia', 'Normal']
        predicted_class = class_labels[np.argmax(prediction)]
        
        st.write(f"Prediction: **{predicted_class}**")
        st.write("Prediction probabilities:", dict(zip(class_labels, prediction[0])))

if __name__ == "__main__":
    main()
