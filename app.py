import matplotlib
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

def plotImage(img_vol, slice_i, channel):
    selected_slice = img_vol[:, :, slice_i, channel]
    ax.imshow(selected_slice, 'gray', interpolation='none')

    return fig

def plotImageSag(img_vol, slice_i, channel):
    selected_slice3 = img_vol[:, slice_i, :, channel]

    rotateIm = list(reversed(list(zip(*selected_slice3))))
    ax2.imshow(rotateIm, 'gray', interpolation='none')

    return fig2

def plotImageCor(img_vol, slice_i, channel):
    selected_slice2 = img_vol[slice_i, :, :, channel]
    rotateIm = list(reversed(list(zip(*selected_slice2))))
    ax1.imshow(rotateIm, 'gray', interpolation='none')

    return fig1

def plotMask(img_vol, slice_i):
    selected_slice = img_vol[:, :, slice_i]
    ax.imshow(selected_slice, interpolation='none')

    return fig

def plotMaskSag(img_vol, slice_i):
    selected_slice = img_vol[:, slice_i, :]
    rotateIm = list(reversed(list(zip(*selected_slice))))
    ax2.imshow(rotateIm, interpolation='none')

    return fig2

def plotMaskCor(img_vol, slice_i):
    selected_slice = img_vol[slice_i, :, :]
    rotateIm = list(reversed(list(zip(*selected_slice))))
    ax1.imshow(rotateIm, interpolation='none')

    return fig1

def plotRedMask(fig, ax, img, slice_i, view):

    if view == 'AX':
        tm90 = test_prediction_argmax[:, :, slice_i]
        tm90[tm90 >= 1] = 1
        masked = np.ma.masked_where(tm90 == 0, tm90)

        cmapm = matplotlib.colors.ListedColormap(["red", "red", "red"], name='from_list', N=None)
        ax.imshow(masked, cmap=cmapm, interpolation='none', alpha=0.3)
        ax.contour(tm90, colors='red', linewidths=1.0)

    if view == 'CR':
        tm90 = test_prediction_argmax[slice_i, :, :]
        tm90[tm90 >= 1] = 1
        masked = np.ma.masked_where(tm90 == 0, tm90)

        cmapm = matplotlib.colors.ListedColormap(["red", "red", "red"], name='from_list', N=None)
        rotMasked = list(reversed(list(zip(*masked))))
        ax.imshow(rotMasked, cmap=cmapm, interpolation='none', alpha=0.3)
        rot_tm90 = list(reversed(list(zip(*tm90))))
        ax.contour(rot_tm90, colors='red', linewidths=1.0)

    if view == 'SG':
        tm90 = test_prediction_argmax[:, slice_i, :]
        tm90[tm90 >= 1] = 1
        masked = np.ma.masked_where(tm90 == 0, tm90)

        cmapm = matplotlib.colors.ListedColormap(["red", "red", "red"], name='from_list', N=None)
        rotMasked = list(reversed(list(zip(*masked))))
        ax.imshow(rotMasked, cmap=cmapm, interpolation='none', alpha=0.3)
        rot_tm90 = list(reversed(list(zip(*tm90))))
        ax.contour(rot_tm90, colors='red', linewidths=1.0)


    return fig, ax

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
    fig = plotImage(npy_file, slice_i1, channel)

    fig1, ax1 = plt.subplots()
    plt.axis('off')
    fig1 = plotImageCor(npy_file, slice_i2, channel)

    fig2, ax2 = plt.subplots()
    plt.axis('off')
    fig2 = plotImageSag(npy_file, slice_i3, channel)

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


        fig, ax = plotRedMask(fig, ax, npy_file, slice_i1, 'AX')
        fig1, ax1 = plotRedMask(fig1, ax1, npy_file, slice_i2, 'CR')
        fig2, ax2 = plotRedMask(fig2, ax2, npy_file, slice_i3, 'SG')


        plot = col1.pyplot(fig)
        plot = col2.pyplot(fig1)
        plot = col3.pyplot(fig2)
