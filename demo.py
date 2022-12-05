import tensorflow as tf
import pandas as pd
import gradio as gr

artists = pd.read_csv(r'data\artists.csv')
artists_alphabetic = artists.sort_values('name').reset_index(drop=True)
artists_dict = dict(zip(artists_alphabetic.index, artists_alphabetic.name))
labels = list(artists_dict.values())

model = tf.keras.models.load_model('saved_model/my_model')

def classify_image(input):
    input = input.reshape((-1, 224, 224, 3))
    input = tf.keras.applications.resnet50.preprocess_input(input)
    prediction = model.predict(input).flatten()
    confidences = {labels[i]: float(prediction[i]) for i in range(50)}
    return confidences

demo = gr.Interface(
    fn=classify_image, 
    inputs=gr.Image(shape=(224, 224)), 
    outputs=gr.Label(num_top_classes=3)
    )
             
demo.launch(share=True)