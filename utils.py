
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow.python.ops import math_ops as tfmath_ops
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
import glob
from matplotlib.patches import Ellipse
import shutil
import pandas as pd
import pickle
import time
import subprocess as sp
import math


import random
from sklearn.decomposition import PCA
from scipy import ndimage
import scipy
import seaborn as sns

tfk = tfp.math.psd_kernels

def Make_path_batch(
    batch=40,
    tmax=30,
    lt=5,
    seed=None
    ):
    """
    Samples x(t), y(t) from a GP
    args:
        batch: number of samples
        tmax: length of samples
        lt: GP length scale
    returns:
        traj: nparray (batch, tmax, 2)
    """

    ilt = -0.5/(lt*lt)
    T = np.arange(tmax)

    Sigma = np.exp( ilt * (T.reshape(-1,1) - T.reshape(1,-1))**2)
    Mu = np.zeros(tmax)

    np.random.seed(seed)

    traj = np.random.multivariate_normal(Mu, Sigma, (batch, 2))
    traj = np.transpose(traj, (0,2,1))

    return traj


def Make_Video_batch(tmax=50,
    px=32,
    py=32,
    lt=5,
    batch=40,
    seed=1,
    r=3
    ):
    """
    params:
        tmax: number of frames to generate
        px: horizontal resolution
        py: vertical resolution
        lt: length scale
        batch: number of videos
        seed: rng seed
        r: radius of ball in pixels
    
    returns:
        traj0: (batch, tmax, 2) numpy array
        vid_batch: (batch, tmax, px, py) numpy array
    """


    traj0 = Make_path_batch(batch=batch, tmax=tmax, lt=lt)

    traj = traj0.copy()

    # convert trajectories to pixel dims
    traj[:,:,0] = traj[:,:,0] * (px/5) + (0.5*px)
    traj[:,:,1] = traj[:,:,1] * (py/5) + (0.5*py)

    rr = r*r

    def pixelate_frame(xy):
        """
        takes a single x,y pixel point and converts to binary image
        with ball centered at x,y.
        """
        x = xy[0]
        y = xy[1]

        sq_x = (np.arange(px) - x)**2
        sq_y = (np.arange(py) - y)**2

        sq = sq_x.reshape(1,-1) + sq_y.reshape(-1,1)

        image = 1*(sq < rr)

        return image

    
    def pixelate_series(XY):
        vid = map(pixelate_frame, XY)
        vid = [v for v in vid]
        return np.asarray(vid)


    vid_batch = [pixelate_series(traj_i) for traj_i in traj]

    vid_batch = np.asarray(vid_batch)

    return traj0, vid_batch

def play_video(vid_batch, j=0):
    """
    vid_batch: batch*tmax*px*py batch of videos
    j: int, which elem of batch to play
    """

    _, ax = plt.subplots(figsize=(5,5))
    plt.ion()

    for i in range(vid_batch.shape[1]):
        ax.clear()
        ax.imshow(vid_batch[j,i,:,:])
        plt.pause(0.1)


def build_video_batch_graph(tmax=50,
    px=32,
    py=32,
    lt=5,
    batch=1,
    seed=1,
    r=3,
    dtype=tf.float32):

    assert px==py, "video batch graph assumes square frames"

    rr = r*r

    ilt = tf.constant(-0.5/(lt**2), dtype=dtype)

    K = tf.range(tmax, dtype=dtype)
    K = (tf.reshape(K, (tmax, 1)) - tf.reshape(K, (1, tmax)))**2

    # print((K*ilt).get_shape())
    # sys.exit()


    K = tf.exp(K*ilt) + 0.00001*tf.eye(tmax, dtype=dtype)
    chol_K = tf.Variable(tf.linalg.cholesky(K), trainable=False)

    ran_Z = tf.random.normal((tmax, 2*batch))

    paths = tf.matmul(chol_K, ran_Z)
    paths = tf.reshape(paths, (tmax, batch, 2))
    paths = tf.transpose(paths, (1,0,2))

    # assumes px = py
    paths = paths*0.2*px + 0.5*px
    # paths[:,:,0] = paths[:,:,0]*0.2*px + 0.5*px
    # paths[:,:,1] = paths[:,:,1]*0.2*py + 0.5*py

    vid_batch = []

    tf_px = tf.range(px, dtype=dtype)
    tf_py = tf.range(py, dtype=dtype)

    for b in range(batch):
        frames_tmax = []
        for t in range(tmax):
            lx = tf.reshape((tf_px - paths[b,t,0])**2, (px, 1))
            ly = tf.reshape((tf_py - paths[b,t,1])**2, (1, py))
            frame = lx + ly < rr
            frames_tmax.append(tf.reshape(frame, (1,1,px,py)))
        vid_batch.append(tf.concat(frames_tmax, 1))
        

    vid_batch = [tfmath_ops.cast(vid, dtype=dtype) for vid in vid_batch]
    vid_batch = tf.concat(vid_batch, 0)

    return vid_batch


def MSE_rotation(X, Y, VX=None, full_cholesky=False):
    """
    Given X, rotate it onto Y
    args:
        X: np array (batch, tmax, 2)
        Y: np array (batch, tmax, 2)
        VX: variance of X values (batch, tmax, 2)

    returns:
        X_rot: rotated X (batch, tmax, 2)
        W: nparray (2, 2)
        B: nparray (2, 1)
        MSE: ||X_rot - Y||^2
        VX_rot: rotated cov matrices (default zeros)
    """

    batch, tmax, _ = X.shape

    X = X.reshape((batch*tmax, 2))


    X = np.hstack([X, np.ones((batch*tmax, 1))])

    
    Y = Y.reshape(batch*tmax, 2)

    W, MSE, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    try:
        MSE = MSE[0] + MSE[1]
    except:
        MSE = np.nan

    X_rot = np.matmul(X, W)
    X_rot = X_rot.reshape(batch, tmax, 2)


    VX_rot = np.zeros((batch, tmax, 2, 2))
    if VX is not None:
        if full_cholesky:
            VX = post_process_full_cholesky(VX, tmax)
        W_rot = W[:2,:]
        W_rot_t = np.transpose(W[:2,:])
        for b in range(batch):
            for t in range(tmax):
                VX_i = np.diag(VX[b,t,:])
                VX_i = np.matmul(W_rot, VX_i)
                VX_i = np.matmul(VX_i, W_rot_t)
                VX_rot[b,t,:,:] = VX_i

    return X_rot, W, MSE, VX_rot


def post_process_full_cholesky(arr, tmax):
    """
    Go from (batch, tmax, tmax) to (batch, tmax, 2)
    :param arr:
    :return:
    """

    var_x = np.diagonal(np.matmul(np.tril(arr[:, :, :tmax]), np.transpose(np.tril(arr[:, :, :tmax]),
                        axes=[0, 2, 1])), axis1=1, axis2=2)
    var_y = np.diagonal(np.matmul(np.tril(arr[:, :, tmax:]), np.transpose(np.tril(arr[:, :, tmax:]),
                        axes=[0, 2, 1])), axis1=1, axis2=2)
    return np.stack([var_x, var_y], axis=2)


def plot_latents(truevids, 
                 truepath, 
                 reconvids=None, 
                 reconpath=None, 
                 reconvar=None, 
                 ax=None, 
                 nplots=4, 
                 paths=None):
    """
    Plots an array of input videos and reconstructions.
    args:
        truevids: (batch, tmax, px, py) np array of videos
        truepath: (batch, tmax, 2) np array of latent positions
        reconvids: (batch, tmax, px, py) np array of videos
        reconpath: (batch, tmax, 2) np array of latent positions
        reconvar: (batch, tmax, 2, 2) np array, cov mat 
        ax: (optional) list of lists of axes objects for plotting
        nplots: int, number of rows of plot, each row is one video
        paths: (batch, tmax, 2) np array optional extra array to plot

    returns:
        fig: figure object with all plots

    """

    if ax is None:
        _, ax = plt.subplots(nplots,3, figsize=(6, 8))
    
    for axi in ax:
        for axj in axi:
            axj.clear()

    _, tmax, _, _ = truevids.shape


    # get axis limits for the latent space
    xmin = np.min([truepath[:nplots,:,0].min(), 
                   reconpath[:nplots,:,0].min()]) -0.1
    xmin = np.min([xmin, -2.5])
    xmax = np.max([truepath[:nplots,:,0].max(), 
                   reconpath[:nplots,:,0].max()]) +0.1
    xmax = np.max([xmax, 2.5])

    ymin = np.min([truepath[:nplots,:,1].min(), 
                   reconpath[:nplots,:,1].min()]) -0.1
    ymin = np.min([ymin, -2.5])
    ymax = np.max([truepath[:nplots,:,1].max(), 
                   reconpath[:nplots,:,1].max()]) +0.1
    ymax = np.max([xmax, 2.5])

    def make_heatmap(vid):
        """
        args:
            vid: tmax, px, py
        returns:
            flat_vid: px, py
        """
        vid = np.array([(t+4)*v for t,v in enumerate(vid)])
        flat_vid = np.max(vid, 0)*(1/(4+tmax))
        return flat_vid
    
    if reconvar is not None:
        E = np.linalg.eig(reconvar[:nplots,:,:,:])
        H = np.sqrt(E[0][:,:,0])
        W = np.sqrt(E[0][:,:,1])
        A = np.arctan2(E[1][:,:,0,1], E[1][:,:,0,0])*(360/(2*np.pi))

    def plot_set(i):
        # i is batch element = plot column

        # top row of plots is true data heatmap
        tv = make_heatmap(truevids[i,:,:,:])
        ax[0][i].imshow(1-tv, origin='lower', cmap='Greys')
        ax[0][i].axis('off')


        # middle row is trajectories
        ax[1][i].plot(truepath[i,:,0], truepath[i,:,1])
        ax[1][i].set_xlim([xmin, xmax])
        ax[1][i].set_ylim([ymin, ymax])
        ax[1][i].scatter(truepath[i,-1,0], truepath[i,-1,1])


        if reconpath is not None:
            ax[1][i].plot(reconpath[i,:,0], reconpath[i,:,1])
            ax[1][i].scatter(reconpath[i,-1,0], reconpath[i,-1,1])

        if paths is not None:
            ax[1][i].plot(paths[i,:,0], paths[i,:,1])
            ax[1][i].scatter(paths[i,-1,0], paths[i,-1,1])
        
        if reconvar is not None:
            ells = [Ellipse(xy=reconpath[i,t,:], 
                            width=W[i,t], 
                            height=H[i,t], 
                            angle=A[i,t]) for t in range(tmax)]
            for e in ells:
                ax[1][i].add_artist(e)
                e.set_clip_box(ax[1][i].bbox)
                e.set_alpha(0.25)
                e.set_facecolor('C1')

        # Third row is reconstructed video
        if reconvids is not None:
            rv = make_heatmap(reconvids[i,:,:,:])
            ax[2][i].imshow(1-rv, origin='lower', cmap='Greys')
            ax[2][i].axis('off')
    
    
    for i in range(nplots):
        plot_set(i)
    
    return ax
    

def make_checkpoint_folder(base_dir, expid=None, extra=""):
    """
    Makes a folder and sub folders for pics and results
    Args:
        base_dir: the root directory where new folder will be made
        expid: optional extra sub dir inside base_dir
    """

    # make a "root" dir to store all checkpoints
    # homedir = os.getenv("HOME")
    # base_dir = homedir+"/GPVAE_checkpoints/"

    if expid is not None:
        base_dir = base_dir + expid + "/"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    
    # now make a unique folder inside the root for this experiments
    filenum = str(len(os.listdir(base_dir))) + "_"+extra+"__on__"

    T = dt.now()

    filetime = str(T.day)+"_"+str(T.month)+"_"+str(T.year) + "__at__"
    filetime += str(T.hour)+"_"+str(T.minute)+"_"+str(T.second)

    # main folder
    checkpoint_folder = base_dir
    os.makedirs(checkpoint_folder, exist_ok=True)

    # pictures folder
    pic_folder = checkpoint_folder + "pics/"
    os.makedirs(pic_folder, exist_ok=True)

    # pickled results files
    res_folder = checkpoint_folder + "res/"
    os.makedirs(res_folder, exist_ok=True)

    # source code
    src_folder = checkpoint_folder + "sourcecode/"
    os.makedirs(src_folder, exist_ok=True)
    old_src_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    src_files = os.listdir(old_src_dir)
    print("\n\nCopying source Code to "+src_folder)
    for f in src_files:
        if ".py" in f:
            src_file = old_src_dir + f
            shutil.copy2(src_file, src_folder)
            print(src_file)
    print("\n")

    # predictions folder, for plotting purposes
    preds_folder = checkpoint_folder + "preds"
    os.makedirs(preds_folder, exist_ok=True)

    
    return checkpoint_folder + "/"


class pandas_res_saver:
    """
    Takes a file and a list of col names to initialise a
    pandas array. Then accepts extra rows to be added
    and occasionally written to disc.
    """
    def __init__(self, res_file, colnames):
        # reload old results frame
        if os.path.exists(res_file):
            if list(pd.read_pickle(res_file).columns)==colnames:
                print("res_file: recovered ")
                self.data = pd.read_pickle(res_file)
                self.res_file = res_file
            else:
                print("res_file: old exists but not same, making new ")
                self.res_file = res_file + "_" + str(time.time())
                self.data = pd.DataFrame(columns=colnames)
        else:
            print("res_file: new")
            self.res_file = res_file
            self.data = pd.DataFrame(columns=colnames)
            
        self.ncols = len(colnames)
        self.colnames = colnames
    
    def __call__(self, new_data, n_steps=10):
        new_data = np.asarray(new_data).reshape((-1, self.ncols))
        new_data = pd.DataFrame(new_data, columns=self.colnames)
        self.data = pd.concat([self.data, new_data])

        if self.data.shape[0]%n_steps == 0:
            self.data.to_pickle(self.res_file)
            print("Saved results to file: "+self.res_file)
    

def call_bash(cmd):
    process = sp.Popen(cmd.split(), stdout=sp.PIPE)
    output, error = process.communicate()


def dict_to_flags(mydict):
    cmd = ""
    for k, v in mydict.items():
        cmd += " --" + k + " " + str(v)
    return cmd


def gauss_cross_entropy(mu1, var1, mu2, var2):
    """
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1:  mean of expectation (batch, tmax, 2) tf variable
        var1: var  of expectation (batch, tmax, 2) tf variable
        mu2:  mean of integrand (batch, tmax, 2) tf variable
        var2: var of integrand (batch, tmax, 2) tf variable

    returns:
        cross_entropy: (batch, tmax, 2) tf variable
    """

    term0 = 1.8378770664093453  # log(2*pi)
    term1 = tf.log(var2)
    term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2

    cross_entropy = -0.5 * (term0 + term1 + term2)

    return cross_entropy


def generate_rotated_MNIST(save_path, N=400, nr_angles=16, valid_set_size=0.1, drop_rate=0.25, digits=[3, 6],
                           latent_dim_object_vector=8, shuffle_data=True, seed=0):
    """
    Generate rotated MNIST data from Casale's paper.

    Saves train, validation and test sets as pickle files.
    Each dataset is a Pyhton dictionary with keys: ['images', 'auxiliary data'].
    Auxiliary data consists of image id, rotation angle and PCA embedding vector.

    :param save_path: path for saving the generated data
    :param N: number of MNIST images of specified digits to use
    :param nr_angles: number of angles between [0, 2pi) considered
    :param valid_set_size: size of validation set
    :param drop_rate: how much images to drop
    :param digit: which digit to consider
    :param shuffle_data: whether or not to shuffle data. Might be important since if we pass
        all angles of the same digit in same batch, kernel matrices carry more information that model could exploit.
        Note that for Michael's extrapolatingGPVAE idea, data should not be shuffled, since there independent GPs are
        fitted for each image.
    :param latent_dim_object_vector: dimension of latent dimension of object vectors
    :param seed: random seed, for reproducibility
    """

    random.seed(seed)
    angles = np.linspace(0, 360, nr_angles + 1)[:-1]

    # load MNIST data
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train = x_train[..., np.newaxis] / 255.0

    # TODO: should MNIST images be binarized here?

    # filter out images with correct digit
    digits_df = []
    for digit in digits:
        x_train_digit = x_train[(y_train == digit)]
        print('Number of images with digit {}: {}'.format(digit, len(x_train_digit)))

        # subsample N images
        indices = random.sample(list(range(x_train_digit.shape[0])), N)
        digits_df.append(x_train_digit[indices, :, :, 0])  # (N, 28, 28)

    x_train = np.concatenate(digits_df)

    # PCA
    pca_df = x_train.copy().reshape((x_train.shape[0], -1))
    pca = PCA(n_components=latent_dim_object_vector)
    pca_df = pca.fit_transform(pca_df)
    print("Explained variance ratio PCA: {}".format(pca.explained_variance_ratio_))

    # save pca_df to pickle (for init of object vectors)
    digit_ending = "".join([str(x) for x in digits])
    with open(save_path + 'pca_ov_init{}_{}.p'.format(digit_ending, latent_dim_object_vector), 'wb') as ov_init_pickle:
        pickle.dump(pca_df, ov_init_pickle)

    # rotate images
    def rotate_image(image, image_id, angles, pca_embedding):
        images = []

        aux_data = np.array([tuple([image_id, math.radians(angle)] + list(pca_embedding)) for angle in angles])

        for i in range(len(angles)):
            images.append(ndimage.rotate(image, angles[i], reshape=False))

        images = np.stack(images)
        images = images[..., np.newaxis]

        return images, aux_data

    images, aux_data = [], []

    assert len(digits) * N == x_train.shape[0]
    for i in range(len(digits) * N):
        images_rot, aux_data_i = rotate_image(x_train[i, :, :], i, angles, pca_df[i, :].copy())
        images.append(images_rot)
        aux_data.append(aux_data_i)

    images = np.concatenate(images)  # (N * len(angles), 28, 28, 1)
    aux_data = np.concatenate(aux_data)  # (N * len(angles), 10)

    # train/test and eval split
    images_, aux_data_, eval_images_, eval_aux_data_ = [], [], [], []
    N_digit = int(len(images) / len(digits))
    N_eval = int(N_digit * (1 - valid_set_size))
    for i in range(len(digits)):
        images_.append(images[i * N_digit:i * N_digit + N_eval])
        aux_data_.append(aux_data[i * N_digit:i * N_digit + N_eval])
        eval_images_.append(images[i * N_digit + N_eval:(i + 1) * N_digit])
        eval_aux_data_.append(aux_data[i * N_digit + N_eval:(i + 1) * N_digit])

    images, aux_data, eval_images, eval_aux_data = np.concatenate(images_), np.concatenate(aux_data_), \
                                                   np.concatenate(eval_images_), np.concatenate(eval_aux_data_)

    # shuffle eval data
    if shuffle_data:
        eval_idx = random.sample(list(range(len(eval_images))), len(eval_images))
        eval_images, eval_aux_data = eval_images[eval_idx], eval_aux_data[eval_idx]

    # train and test split
    test_angle = random.sample(list(angles), 1)[0]
    mask = (aux_data[:, 1] == math.radians(test_angle))
    train_images, train_aux_data, test_images, test_aux_data = images[~mask], aux_data[~mask], \
                                                               images[mask], aux_data[mask]
    print("Test angle: {}".format(test_angle))

    # drop some images
    if shuffle_data:
        idx_train = random.sample(list(range(len(train_images))), int(len(train_images) * (1 - drop_rate)))
        idx_test = random.sample(list(range(len(test_images))), int(len(test_images) * (1 - drop_rate)))
    else:
        idx_train = list(range(int(len(train_images) * (1 - drop_rate))))
        idx_test = list(range(int(len(test_images) * (1 - drop_rate))))

        idx_train_not_in_test = list(range(int(len(train_images) * (1 - drop_rate)), len(train_images)))
        train_not_in_test_images = train_images[idx_train_not_in_test]
        train_not_in_test_aux_data = train_aux_data[idx_train_not_in_test]

    train_images, train_aux_data = train_images[idx_train], train_aux_data[idx_train]
    test_images, test_aux_data = test_images[idx_test], test_aux_data[idx_test]

    print('Size of training data: {}'.format(len(train_images)))
    print('Size of validation data: {}'.format(len(eval_images)))
    print('Size of test data: {}'.format(len(test_images)))

    if not shuffle_data:
        print('Size of training data without test ids: {}'.format(len(train_not_in_test_images)))

    # save to pickle files
    train_dict = {'images': train_images, 'aux_data': train_aux_data}
    eval_dict = {'images': eval_images, 'aux_data': eval_aux_data}
    test_dict = {'images': test_images, 'aux_data': test_aux_data}

    if not shuffle_data:
        train_not_in_test_dict = {'images': train_not_in_test_images, 'aux_data': train_not_in_test_aux_data}

    ending = "_not_shuffled_{}.p".format(latent_dim_object_vector) if not shuffle_data else "_{}.p".format(latent_dim_object_vector)
    ending = digit_ending + ending
    print(ending)

    with open(save_path + 'train_data' + ending, 'wb') as train_pickle:
        pickle.dump(train_dict, train_pickle)
    with open(save_path + 'eval_data' + ending, 'wb') as eval_pickle:
        pickle.dump(eval_dict, eval_pickle)
    with open(save_path + 'test_data' + ending, 'wb') as test_pickle:
        pickle.dump(test_dict, test_pickle)

    if not shuffle_data:
        with open(save_path + 'train_not_in_test_data' + ending, 'wb') as train_pickle:
            pickle.dump(train_not_in_test_dict, train_pickle)


def plot_mnist(arr, recon_arr, title, nr_images=8, seed=0):
    """

    :param arr:
    :param recon_arr:
    :param title:
    :param nr_images:
    :param seed:
    :return:
    """
    random.seed(seed)
    assert nr_images % 8 == 0

    indices = random.sample(list(range(len(arr))), nr_images)
    plt.figure(figsize=(10, 10*int(nr_images/8)))
    plt.suptitle(title)
    for i in range(int(nr_images*2)):
        plt.subplot(int(nr_images / 2), 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if i % 2 == 0:
            plt.imshow(arr[indices[i // 2]][:, :, 0], cmap='gray')
            plt.xlabel("Ground truth, id: {}".format(indices[i // 2]))
        else:
            plt.imshow(recon_arr[indices[i // 2]][:, :, 0], cmap='gray')
            plt.xlabel("Recon image, id: {}".format(indices[i // 2]))
    # plt.tight_layout()
    plt.draw()


def generate_init_inducing_points(train_data_path, n=5, nr_angles=16, seed_init=0, remove_test_angle=None,
                                  PCA=False, M=8, seed=0):
    """
    Generate initial inducing points for rotated MNIST data.
    For each angle we sample n object vectors from empirical distribution of PCA embeddings of training data.

    :param n: how many object vectors per each angle to sample
    :param nr_angles: number of angles between [0, 2pi)
    :param remove_test_angle: if None, test angle is kept in inducing point set. Else if index between 0 and nr_angles-1
        is passed, this angle is removed (to investigate possible data leakage through inducing points).
    :param PCA: whether or not to use PCA initialization
    :param M: dimension of GPLVM vectors
    """

    random.seed(seed)

    data = pickle.load(open(train_data_path, "rb"))
    data = data['aux_data']

    angles = np.linspace(0, 2 * np.pi, nr_angles + 1)[:-1]
    inducing_points = []

    if n < 1:
        indices = random.sample(list(range(nr_angles)), int(n*nr_angles))
        n = 1
    else:
        indices = range(nr_angles)

    for i in indices:

        # skip test angle
        if i == remove_test_angle:
            continue

        # for reproducibility
        seed = seed_init + i

        if PCA:
            obj_vectors = []
            for pca_ax in range(2, 2 + M):
                # sample from empirical dist of PCA embeddings
                obj_vectors.append(scipy.stats.gaussian_kde(data[:, pca_ax]).resample(int(n), seed=seed))

            obj_vectors = np.concatenate(tuple(obj_vectors)).T
        else:
            obj_vectors = np.random.normal(0, 1.5, int(n)*M).reshape(int(n), M)

        obj_vectors = np.hstack((np.full((int(n), 1), angles[i]), obj_vectors))  # add angle to each inducing point
        inducing_points.append(obj_vectors)

    inducing_points = np.concatenate(tuple(inducing_points))
    id_col = np.array([list(range(len(inducing_points)))]).T
    inducing_points = np.hstack((id_col, inducing_points))
    return inducing_points


def visualize_kernel_matrices(aux_data_arr, batch_size=32, N=1, K_obj_normalized=True,
                              amplitude=1.0, length_scale=1.0):
    """
    Visualize heatmaps of kernel matrices.

    :param aux_data_arr:
    :param batch_size:
    :param N: number of batches to visualize
    :param K_obj_normalized: whether or not to normalize (between -1 and 1) object kernel matrix (linear kernel)
    :param amplitude:
    :param length_scale:
    """

    # define kernels
    kernel_view = tfk.ExpSinSquared(amplitude=amplitude, length_scale=length_scale, period=2 * np.pi)
    kernel_object = tfk.Linear()
    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)
    z = tf.placeholder(dtype=tf.float32)
    w = tf.placeholder(dtype=tf.float32)
    K_view = kernel_view.matrix(tf.expand_dims(x, axis=1), tf.expand_dims(y, axis=1))
    K_obj = kernel_object.matrix(z, w)
    if K_obj_normalized:
        obj_norm = 1 / tf.matmul(tf.math.reduce_euclidean_norm(z, axis=1, keepdims=True),
                                 tf.transpose(tf.math.reduce_euclidean_norm(z, axis=1, keepdims=True), perm=[1, 0]))
        K_obj = K_obj * obj_norm
    K_prod = K_view * K_obj

    # util function for heatmaps
    def heatmap(ax_, arr, title, vmin=0, vmax=1):
        ax = sns.heatmap(arr, vmin=vmin, vmax=vmax, center=0,
                         cmap=sns.diverging_palette(20, 220, n=200),
                         square=True, ax=ax_)
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45,
                           horizontalalignment='right')
        ax.set_title(title);

    for i in range(N):
        # generate kernel matrices
        batch = aux_data_arr[i * batch_size:(i + 1) * batch_size]
        with tf.Session() as sess:
            K_view_, K_obj_, K_prod_ = sess.run([K_view, K_obj, K_prod],
                                                {x: batch[:, 1], y: batch[:, 1], z: batch[:, 2:], w: batch[:, 2:]})
        # plot kernel matrices
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        heatmap(axes[0], K_view_, "View kernel. Batch: {}. Det: {}".format(i + 1, np.linalg.det(K_view_)))
        heatmap(axes[1], K_obj_, "Object kernel. Batch: {}. Det: {}".format(i + 1, np.linalg.det(K_obj_)), vmin=-1)
        heatmap(axes[2], K_prod_, "Product kernel. Batch: {}. Det: {}".format(i + 1, np.linalg.det(K_prod_)), vmin=-1)
    plt.show()


def import_rotated_mnist(MNIST_path, ending, batch_size, digits="3", N_t=None):
    """

    Support for loading of data and batching via tf.data.Dataset API.

    :param MNIST_path:
    :param ending:
    :param batch_size:
    :param N_t: How many angels in train set for each image in test set
                (since reGPVAE implementation is based on not_shuffled data).

    :return:
    """

    # TODO: here we load entire data in the memory. For MNIST that is fine, for larger datasets will have to
    #  implement it in more efficient way

    # train data
    train_data_dict = pickle.load(open(MNIST_path + 'train_data' + ending, 'rb'))
    if N_t is not None:
        flatten = lambda l: [item for sublist in l for item in sublist]
        digit_mask = [True] * N_t + [False] * (15 - N_t)

        mask = [random.sample(digit_mask, len(digit_mask)) for _ in range(int(len(train_data_dict['aux_data'])/15))]
        mask = flatten(mask)
        train_data_dict['images'] = train_data_dict['images'][mask]
        train_data_dict['aux_data'] = train_data_dict['aux_data'][mask]

        # add train images without test angles
        if N_t < 15:
            train_not_in_test_data_dict = pickle.load(open(MNIST_path + 'train_not_in_test_data' + ending, 'rb'))

            n = int(len(digits) * 270 * (15 - N_t) / N_t) * N_t

            mask = [random.sample(digit_mask, len(digit_mask)) for _ in range(int(len(train_not_in_test_data_dict['aux_data']) / 15))]
            mask = flatten(mask)

            train_data_dict['images'] = np.concatenate((train_data_dict['images'],
                                                        train_not_in_test_data_dict['images'][mask][:n, ]), axis=0)
            train_data_dict['aux_data'] = np.concatenate((train_data_dict['aux_data'],
                                                          train_not_in_test_data_dict['aux_data'][mask][:n, ]), axis=0)

    train_data_images = tf.data.Dataset.from_tensor_slices(train_data_dict['images'])
    train_data_aux_data = tf.data.Dataset.from_tensor_slices(train_data_dict['aux_data'])
    train_data = tf.data.Dataset.zip((train_data_images, train_data_aux_data)).batch(batch_size)

    # eval data
    eval_batch_size_placeholder = tf.compat.v1.placeholder(dtype=tf.int64, shape=())
    eval_data_dict = pickle.load(open(MNIST_path + 'eval_data' + ending, 'rb'))
    eval_data_images = tf.data.Dataset.from_tensor_slices(eval_data_dict['images'])
    eval_data_aux_data = tf.data.Dataset.from_tensor_slices(eval_data_dict['aux_data'])
    eval_data = tf.data.Dataset.zip((eval_data_images, eval_data_aux_data)).batch(eval_batch_size_placeholder)

    # test data
    test_batch_size_placeholder = tf.compat.v1.placeholder(dtype=tf.int64, shape=())
    test_data_dict = pickle.load(open(MNIST_path + 'test_data' + ending, 'rb'))
    test_data_images = tf.data.Dataset.from_tensor_slices(test_data_dict['images'])
    test_data_aux_data = tf.data.Dataset.from_tensor_slices(test_data_dict['aux_data'])
    test_data = tf.data.Dataset.zip((test_data_images, test_data_aux_data)).batch(test_batch_size_placeholder)

    # init iterator
    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    training_init_op = iterator.make_initializer(train_data)
    eval_init_op = iterator.make_initializer(eval_data)
    test_init_op = iterator.make_initializer(test_data)

    return iterator, training_init_op, eval_init_op, test_init_op, \
           train_data_dict, eval_data_dict, test_data_dict, eval_batch_size_placeholder, test_batch_size_placeholder


def print_trainable_vars(vars):
    total_parameters = 0
    print("\n\nTrainable variables:")
    for v in vars:
        print(v)
        shape = v.get_shape()
        var_params = 1
        for dim in shape:
            var_params *= dim.value
        total_parameters += var_params
    print("Number of train params: {}".format(total_parameters))


def parse_opt_regime(arr):
    for i in range(len(arr)):
        regime, nr_epochs = arr[i].split("-")
        arr[i] = (regime, int(nr_epochs))
    training_regime = [[regime[0]] * regime[1] for regime in arr]
    flatten = lambda l: [item for sublist in l for item in sublist]
    training_regime = flatten(training_regime)
    nr_epochs = len(training_regime)
    return nr_epochs, training_regime


def IndexedSlicesValue_to_numpy(a):
    """
    Convert tf.IndexedSlicesValue to numpy array.
    Needed for gradients of object_vectors in SVGPVAE.

    Source: https://code.ihub.org.cn/projects/124/repository/revisions/master/entry/tensorflow/python/ops/gradient_checker_v2.py

    :param a:
    :return:
    """
    arr = np.zeros(a.dense_shape)
    assert len(a.values) == len(a.indices), (
            "IndexedSlicesValue has %s value slices but %s indices\n%s" % (
        a.values, a.indices, a))
    for values_slice, index in zip(a.values, a.indices):
        assert 0 <= index < len(arr), ("IndexedSlicesValue has invalid index %s\n%s" % (index, a))
        arr[index] += values_slice
    return arr


def import_sprites(batch_size, sprites_path='SPRITES data/', batch_size_test_char=576):
    """

    Support for loading of data and batching via tf.data.Dataset API.

    :param batch_size:
    :param sprites_path:
    :param batch_size_test_char: batch size for prediction pipeline for test_character dataset. 576 is chosen since
                                21312 % 576 == 0, so that implemenation of splitting of test batch into
                                context and target frames is easier.

    :return: train_iterator,
             test_action_iterator,
             test_character_iterator
    """

    assert batch_size_test_char % 72 == 0

    filenames_train = glob.glob(sprites_path + "train/*.tfrecord")
    filenames_test_action = glob.glob(sprites_path + "test_action/*.tfrecord")
    filenames_test_char = glob.glob(sprites_path + "test_character/*.tfrecord")

    dataset_train = tf.data.TFRecordDataset(filenames_train)
    dataset_test_action = tf.data.TFRecordDataset(filenames_test_action)
    dataset_test_char = tf.data.TFRecordDataset(filenames_test_char)

    # example proto decode
    def _parse_function(example_proto):
        keys_to_features = {'frame': tf.FixedLenFeature((64, 64, 3), tf.float32),
                            'character_ID': tf.FixedLenFeature((), tf.int64, default_value=0),
                            'action_ID': tf.FixedLenFeature((), tf.int64, default_value=0)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['frame'], parsed_features['character_ID'], parsed_features['action_ID']

    dataset_train = dataset_train.map(_parse_function)
    dataset_test_action = dataset_test_action.map(_parse_function)
    dataset_test_char = dataset_test_char.map(_parse_function)

    dataset_train = dataset_train.batch(batch_size)
    dataset_test_action = dataset_test_action.batch(batch_size)
    dataset_test_char = dataset_test_char.batch(batch_size_test_char)

    # if shuffle_train:
    #     dataset_train = dataset_train.shuffle(buffer_size=batch_size)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    training_init_op = iterator.make_initializer(dataset_train)
    test_action_init_op = iterator.make_initializer(dataset_test_action)
    test_char_init_op = iterator.make_initializer(dataset_test_char)

    return iterator, training_init_op, test_action_init_op, test_char_init_op


def sprites_PCA_init(path_train_dict, m=15, L_action=6, L_character=16, seed=42):
    """
    Produces PCA initialization for GPLVM action vectors as well as for inducing points for SPRITES dataset.

    :param path_train_dict:
    :param m: number of inducing point per action. Total number of inducing points = m*72
    :param L_action: dimension of GPLVM action vectors
    :param L_character: dimension of character vectors (needs to coincide with the dimension of
                                                                the output layer in representation network).

    :return: GPLVM action init (72, L_action),
             inducing point init (72*m, L_action + L_character)

    """

    # read in entire training data
    train_dict = pickle.load(open(path_train_dict, 'rb'))
    train_frames, train_aux_data = train_dict['frames'], train_dict['aux_data']
    del train_dict

    # GPLVM action vectors: first compute average frame for each action, resulting in (72, 64*64*3) dataframe.
    # Then do PCA on it and keep only first L_action principal components, yielding (72, L_action) dataframe.
    action_ids_train = {i: [] for i in range(72)}
    for i in range(len(train_aux_data)):
        action_ids_train[train_aux_data[i, 1]].append(i)

    GPLVM_action = []
    for action, ids in action_ids_train.items():
        mean_frame = train_frames[ids].mean(axis=0).reshape(-1)
        GPLVM_action.append(mean_frame)

    pca_GPLVM = PCA(n_components=L_action)
    GPLVM_action = pca_GPLVM.fit_transform(np.array(GPLVM_action))

    # inducing points: for each GPLVM_action vector, sample m characters vectors from
    # PCA dataframe on the entire dataset (N_train, L_character)
    pca_global = PCA(n_components=L_character)
    train_frames_PCA = pca_global.fit_transform(train_frames.reshape(-1, 64 * 64 * 3))

    inducing_points = []
    for i in range(len(GPLVM_action)):
        char_vectors = []
        for pca_ax in range(L_character):
            # sample from empirical distribution of a given PCA feature
            pca_sample = scipy.stats.gaussian_kde(train_frames_PCA[:, pca_ax]).resample(m, seed=seed).reshape(-1)
            char_vectors.append(pca_sample)
        char_vectors = np.array(char_vectors).T

        # repeat GPLVM_action vector m times
        GPLVM_action_vec = GPLVM_action[i, :]
        GPLVM_action_vec = np.tile(GPLVM_action_vec, (m, 1))

        # hstack GPLVM action vector and character vectors
        inducing_points.append(np.hstack((GPLVM_action_vec, char_vectors)))

    inducing_points = np.concatenate(inducing_points)

    del train_frames
    del train_aux_data

    return GPLVM_action, inducing_points


def plot_sprites(arr, recon_arr, title, nr_images=8, seed=0):
    """

    :param arr:
    :param recon_arr:
    :param title:
    :param nr_images:
    :param seed:
    :return:
    """
    random.seed(seed)
    assert nr_images % 8 == 0

    indices = random.sample(list(range(len(arr))), nr_images)
    plt.figure(figsize=(10, 10*int(nr_images/8)))
    plt.suptitle(title)
    for i in range(int(nr_images*2)):
        plt.subplot(int(nr_images / 2), 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if i % 2 == 0:
            plt.imshow(arr[indices[i // 2]])
        else:
            plt.imshow(recon_arr[indices[i // 2]])
    # plt.tight_layout()
    plt.draw()


def aux_data_sprites_utils(batch_size, N, repeats):
    """
    Generates auxiliary arrays that are needed for construction of auxiliary data for SPRITES dataset.
    Generated arrays are later used in function aux_data_SVGPVAE_sprites.

    :param batch_size:
    :param N: nr. frames per character
    :param repeats: how many times is each summed character vector copied

    :return:
    """
    N_char = int(batch_size / N)  # nr. of unique characters
    segment_ids = np.array([[i] * N for i in range(N_char)]).reshape(-1)
    repeats_arr = [repeats for _ in range(N_char)]

    return segment_ids, repeats_arr


def forward_pass_pretraining_repr_NN(frames, labels, repr_NN, classification_layer, test_pipeline=False):
    """

    :param frames:
    :param labels:
    :param repr_NN: representation neural net
    :param classification_layer: classificiation layer, used only in the pretraining step
    :param test_pipeline:
    :return:
    """

    if not test_pipeline:
        # shuffle data in the batch. .shuffle did not work in import_sprites function...
        indices = tf.range(start=0, limit=tf.shape(frames)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        frames = tf.gather(frames, shuffled_indices)
        labels = tf.gather(labels, shuffled_indices)

    embeddings = repr_NN.repr_nn(frames)

    logits = classification_layer(embeddings)

    # labels_ = tf.one_hot(labels, 1000)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    if test_pipeline:
        preds = tf.argmax(logits, 1)
        _, acc = tf.compat.v1.metrics.accuracy(labels=labels, predictions=preds)

        return loss, acc

    else:
        return loss


def compute_bias_variance_mean_estimators(arr_batch, arr_full):
    """

    :param arr_batch: (B, L, m)
    :param arr_full: (L, m)
    :return:
    """

    # TODO: add support for calculation of variance

    B, L, m = len(arr_batch), len(arr_batch[0]), arr_batch[0][0].shape[0]
    assert L == len(arr_full)
    assert m == arr_full[0].shape[0]

    avg_arr_batch = [0] * L
    for l in range(L):
        for b in range(B):
            if b == 0:
                avg_arr_batch[l] = arr_batch[b][l]
            else:
                avg_arr_batch[l] += arr_batch[b][l]

    avg_arr_batch = [x / B for x in avg_arr_batch]

    bias = [np.sum(np.abs(x-y)) for x, y in zip(avg_arr_batch, arr_full)]

    return np.mean(bias)


def latent_samples_VAE_full_train(train_images, vae, clipping_qs=False):
    """
    Get latent samples for training data. For t-SNE plots :)

    :param train_images:
    :param vae:
    :param clipping_qs:
    :return:
    """

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(train_images)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    # SAMPLE
    epsilon = tf.random.normal(shape=tf.shape(qnet_mu), dtype=vae.dtype)
    latent_samples = qnet_mu + epsilon * tf.sqrt(qnet_var)

    return latent_samples


def latent_samples_SVGPVAE(train_images, train_aux_data, vae, svgp, clipping_qs=False):
    """
    Get latent samples for training data. For t-SNE plots :)

    :param train_images:
    :param train_aux_data:
    :param vae:
    :param svgp:
    :param clipping_qs:
    :return:
    """

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(train_images)
    L = tf.cast(qnet_mu.get_shape()[1], dtype=vae.dtype)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    p_m, p_v = [], []
    for l in range(qnet_mu.get_shape()[1]):  # iterate over latent dimensions
        p_m_l, p_v_l, _, _ = svgp.approximate_posterior_params(train_aux_data, qnet_mu[:, l], qnet_var[:, l])
        p_m.append(p_m_l)
        p_v.append(p_v_l)

    p_m = tf.stack(p_m, axis=1)
    p_v = tf.stack(p_v, axis=1)

    epsilon = tf.random.normal(shape=tf.shape(p_m), dtype=vae.dtype)
    latent_samples = p_m + epsilon * tf.sqrt(p_v)

    return latent_samples


if __name__=="__main__":

    # generate_init_inducing_points("MNIST data/train_data3.p", PCA=False)

    # ============= generating rotated MNIST data =============
    # generate_rotated_MNIST("MNIST data/", digits=[3, 6])
    # generate_rotated_MNIST("MNIST data/", digits=[1, 3, 6, 7, 9])
    # generate_rotated_MNIST("MNIST data/")
    # generate_rotated_MNIST('MNIST data/', shuffle_data=False, digits=[6])
    # generate_rotated_MNIST('MNIST data/', shuffle_data=False, digits=[3])
    # generate_rotated_MNIST('MNIST data/', shuffle_data=False, digits=[3, 6])
    # generate_rotated_MNIST('MNIST data/', shuffle_data=False, digits=[1, 3, 6, 7, 9])
    # generate_rotated_MNIST('MNIST data/', shuffle_data=True, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # generate_rotated_MNIST('MNIST data/', shuffle_data=False, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # generate_rotated_MNIST("MNIST data/", digits=[3], latent_dim_object_vector=4)
    # generate_rotated_MNIST("MNIST data/", digits=[3], latent_dim_object_vector=16)
    # generate_rotated_MNIST("MNIST data/", digits=[3], latent_dim_object_vector=32)
    # generate_rotated_MNIST("MNIST data/", digits=[3], latent_dim_object_vector=64)
    generate_rotated_MNIST("MNIST data/", digits=[3], latent_dim_object_vector=24)

