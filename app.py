import functions
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from keras.models import load_model


# config streamlit
st.set_page_config(
    page_title="irm cerveau",
    layout="wide",
    page_icon="🧠"
)
st.set_option('deprecation.showPyplotGlobalUse', False)

# application // visualisation
st.title("Visualisation IRM cérébrale")
st.sidebar.title("📌 Menu")
npy_file = st.sidebar.file_uploader("Sélectionnez un fichier .npy", type=['npy', 'img'], accept_multiple_files=False)

if npy_file is not None:

    npy_file = np.load(npy_file)
    img_type = st.selectbox('''Type d'image''', ('FLAIR', 'T1CE', 'T2'))

    if img_type == 'FLAIR':
        channel = 0
    elif img_type == 'T1CE':
        channel = 1
    else:
        channel = 2
    
    col1, col2, col3 = st.columns(3)

    # slider
    n_slices1 = npy_file.shape[2]
    slice_i1 = col1.slider('Coupe Axiale', 0, n_slices1, int(n_slices1 / 2))

    n_slices2 = npy_file.shape[0]
    slice_i2 = col2.slider('Coupe Coronale', 0, n_slices2, int(n_slices2 / 2))

    n_slices3 = npy_file.shape[1]
    slice_i3 = col3.slider('Coupe Sagittale', 0, n_slices3, int(n_slices3 / 2))

    # plot
    fig, ax = plt.subplots()
    plt.axis('off')
    fig = functions.plotImage(npy_file, slice_i1, channel)

    fig1, ax1 = plt.subplots()
    plt.axis('off')
    fig1 = functions.plotImageCor(npy_file, slice_i2, channel)

    fig2, ax2 = plt.subplots()
    plt.axis('off')
    fig2 = functions.plotImageSag(npy_file, slice_i3, channel)

    plot = col1.pyplot(fig)
    plot = col2.pyplot(fig1)
    plot = col3.pyplot(fig2)

    # application // creation masque avec modele unet
    seg = st.sidebar.checkbox('Segmentation UNET 3D')

    if seg:

        col4, col5, col6 = st.columns(3)

        model = load_model('drive/MyDrive/model/brats.hdf5', compile = False)

        npy_file_input = np.expand_dims(npy_file, axis=0)
        test_prediction = model.predict(npy_file_input)
        test_prediction_argmax = np.argmax(test_prediction, axis=4)[0,:,:,:]


        fig, ax = functions.plotRedMask(fig, ax, npy_file, slice_i1, 'AX')
        fig1, ax1 = functions.plotRedMask(fig1, ax1, npy_file, slice_i2, 'CR')
        fig2, ax2 = functions.plotRedMask(fig2, ax2, npy_file, slice_i3, 'SG')


        plot = col1.pyplot(fig)
        plot = col2.pyplot(fig1)
        plot = col3.pyplot(fig2)
