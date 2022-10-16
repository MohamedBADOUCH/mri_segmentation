import matplotlib
import numpy as np


global ax, ax1, ax2, ax3, fig, fig1, fig2, fig3, test_prediction_argmax

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