#Import Libraries
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from utils import head, set_bg, equal_text, body, example, about, explanation_of_movements, transform_output
from tensorflow.keras.preprocessing import image_dataset_from_directory
#from deep_painting_app.explore_data import random_painting, pick_up_one_painting_per_class
#from deep_painting_app.data_processing import load_and_divide_dataset, classes_names_to_dict, give_class_name
import requests
import random
import math
import numpy as np
import plotly.express as px

def give_class_name(vector, class_names_dict):
    """
    return the artistic movement given the associated vector in class_names_dict
    vector can be a list ou a np.array
    class_names_dict should be set with the classes_names_to_dict from this file librairy
    """
    for class_name, v in class_names_dict.items():
        if v == list(vector):
            return class_name

def classes_names_to_dict(dataset):
    """
    return a dict of the classes and corresponding vectors
    dataset type: tensorflow.python.data.ops.dataset_ops.BatchDataset
    https://www.tensorflow.org/jvm/api_docs/java/org/tensorflow/op/data/BatchDataset
    """

    # raise an error if dataset is not a tensorflow BatchDataset
    if not dataset.__class__.__name__ == 'BatchDataset':
        raise TypeError("This function has been written for tensorflow BatchDataset\n\
           Please check https://www.tensorflow.org/jvm/api_docs/java/org/tensorflow/op/data/BatchDataset")

    classes_names = dataset.class_names
    identity_matrix = np.identity(len(classes_names)).tolist()

    classes_dict = dict(zip(classes_names, identity_matrix))
    return classes_dict

def random_painting(path, img_height=180, img_width=180):
    """
    return a tuple from a dataset of images: a random image(numpy ndarray) and its class(string)
    The database must be divided into folders (one folder per class).
    arguments:
    * path: path of the dataset (by default: current path)
    * image size: img_height and img_width. By default 180 for both
    """

    # raise an error if path is not a string
    if not type(path) is str:
        raise TypeError("path must be a string")

    # random seed value for image_dataset_from_directory
    random.seed()
    seed = random.randint(0,100)

    # load dataset from path
    dataset = image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        shuffle = True,
        seed = seed,
        image_size = (img_height, img_width),
        color_mode='rgb',
        batch_size=1)

    #iterates through dataset, set one painting and its class, randomly stop
    class_names_dict = classes_names_to_dict(dataset)
    for x, y in dataset.as_numpy_iterator():
        painting, painting_class = x[0], give_class_name(y[0], class_names_dict)
        if random.randint(0,1000) > 700:
            break

    return painting, painting_class


def pick_up_one_painting_per_class(path, img_height=180, img_width=180):
    """
    pick-up randomly one image per class and return a dictionnary.
    key: class(string) - value: image(numpy ndarray)
    The database must be divided into folders (one folder per class).
    arguments:
    * path: path of the dataset (by default: current path)
    * image size: img_height and img_width. By default 180 for both
    """
    # random seed value for image_dataset_from_directory
    random.seed()
    seed = random.randint(0,100)

    # load full dataset from path
    dataset = image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        batch_size=1,
        seed = seed,
        image_size = (img_height, img_width))

    # dict of class names and corresponding vectors
    class_names_dict = classes_names_to_dict(dataset)

    # intitate a dictionnary: key = artistic mvt, value = list of paintings
    paintings_per_class = {c: [] for c in dataset.class_names}

    # iterate the dataset and choose the first encountered painting, for each class
    # the dataset has been load randomly already
    for cl in dataset.class_names:
        for painting, cla in dataset.as_numpy_iterator():
            if cl == give_class_name(cla[0], class_names_dict):
                paintings_per_class[give_class_name(cla[0], class_names_dict)] = painting[0]
                break
    return paintings_per_class


#Opens and displays the image
def get_opened_image(image):
    return Image.open(image)

#Title
head()

#Sets background image
#set_bg('data/black_background.jpg')


#Shows examples of images for each class
example()

path = '../raw_data/orgImg_small'
imgs = pick_up_one_painting_per_class(path)
figure, axs = plt.subplots(1, 6, figsize=(20,20))
i = 0
for cl in imgs:
    axs[i].imshow(imgs[cl]/255)
    axs[i].set_title(equal_text(cl))
    axs[i].set_axis_off()
    i += 1
st.pyplot(figure)
explanation_of_movements()

#Body
body()
#Uploads the image file
image_file = st.file_uploader('Upload an image', type = ['png', 'jpg',
                                                         'jpeg', 'pdf'])

deep_painting_url = 'https://deeppainting3-ynhfw4pdza-an.a.run.app/predict/image'

if image_file:
    #Sets the image
    st.markdown(f'<h2 style="font-size:30px;">{"Results:"}</h2>',
                unsafe_allow_html=True)
    r = requests.post(url = deep_painting_url,files={'file':image_file})
    predicted_movement = r.json()['movement']
    predicted_probabilty = r.json()['confidence']
    st.markdown(f'<h2 style="font-size:30px;margin-bottom:-35px">{f"The Deep Painting App classifies this image as {predicted_movement} ."}</h2>',
                unsafe_allow_html=True)
    api_df = transform_output(r.json())
    fig = px.bar(api_df, x='Confidence', y='Movement',orientation='h', color = 'Movement')
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible = False)
    st.plotly_chart(fig)
    image = get_opened_image(image_file)
    with st.expander("Selected Image", expanded = True):
        st.image(image, use_column_width = True)




#THE GUESSING GAME
if 'random_image' not in st.session_state:
    st.session_state['random_image'] = random_painting(path)
    print("Init")

def form2_callback():
    st.session_state['random_image'] = random_painting(path)
    print("callback")

with st.sidebar:
    with st.expander('About'): #About section
        about()

    st.title('Guess the Movement')
    fig, ax = plt.subplots()
    ax.imshow(st.session_state['random_image'][0]/255)
    ax.set_axis_off()
    st.pyplot(fig)

    label = st.session_state['random_image'][1]
    label = equal_text(label)


    with st.form(key ='Form1'):
        movement = st.radio("What do you think?",
                    ('High Renaissance', 'Impressionism', 'Northern Renaissance',
                     'Post Impressionism', 'Rococo', 'Ukiyo-e'))

        submitted = st.form_submit_button(label = 'Submit')

    if submitted:
        if movement == label:
            st.write('Well done!')
        else:
            st.write(f'Ooops! It was {label} .')

        #with st.form(key = 'Form2'):
        btn = st.button("Next", on_click= form2_callback)
