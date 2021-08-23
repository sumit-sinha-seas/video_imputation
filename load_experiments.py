import argparse
import time
import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_probability as tfp
from utils import gauss_cross_entropy
import matplotlib.animation as animation
from utils import Make_Video_batch, make_checkpoint_folder, pandas_res_saver, \
                  build_video_batch_graph, plot_latents, MSE_rotation
from utils_circles_grid import Make_circles, Make_squares, plot_circle, plot_square
from SVGPVAE_model import SVGP, build_SVGPVAE_elbo_graph
from GPVAE_Pearce_model import build_pearce_elbo_graphs

tfd = tfp.distributions
tfk = tfp.math.psd_kernels

def compute_average_mse(reconvid, truevid):
  Y = np.square(np.subtract(reconvid,truevid)).mean()
  return Y

def compute_average_psnr(reconvid, truevid):
  psnr = cv2.PSNR(reconvid, truevid)
  return psnr

def plot_frames(reconvid, base_dir, truevid, b=0, frames = 5):
  print(reconvid.shape, truevid.shape)
  N = reconvid.shape[1] // frames
  fig, ax = plt.subplots(2,frames)
  for i, a in enumerate(ax[0]):
    a.imshow(reconvid[b][N*i], cmap='gray')
    a.set_xlabel(f'Time = {N*i+i}')
  for i, a in enumerate(ax[1]):
    a.imshow(truevid[b][N*i], cmap='gray')
    a.set_xlabel(f'Time = {N*i+i}')
  fig.savefig(base_dir + 'frames.png')

def double_frame_rate(reconvid, interp):
    videos = []
    for i in range(reconvid.shape[0]):
        recovered_video = [val for pair in zip(reconvid[i], interp[i]) for val in pair]
        videos.append(recovered_video)
    return np.array(videos)

def plot_sequence(seq, name, name_dir, interval = 200):
    fig = plt.figure()
    ims = [[plt.imshow(d, cmap='gray', animated=True)] for d in seq]
    ims = np.array(ims)
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=False, repeat_delay=0)
    dpath = os.path.join(args.base_dir, 'animations', name_dir)
    os.makedirs(dpath, exist_ok=True)
    ani.save(os.path.join(dpath, f'{name}.mp4'))
    plt.close()
    # plt.show()

def interpolate_latent(full_p_mu):
  samples = full_p_mu
  # extrapolate
  xl1 = full_p_mu[:,-2]
  xl2 = full_p_mu[:,-1]
  m = (xl2[:,1]-xl1[:,1]) / (xl2[:,0] - xl1[:,0])
  b = xl2[:,1] - m*xl2[:,0]
  new_x = xl2[:,0] + (xl2[:,0] - xl1[:,0])
  new_y = m * new_x + b
  new_x = tf.reshape(new_x, (new_x.shape[0], 1))
  new_y = tf.reshape(new_x, (new_y.shape[0], 1))
  new_point = tf.concat((new_x, new_y), axis = 1)
  new_point = tf.reshape(new_point, (new_point.shape[0], 1, new_point.shape[1]))
  samples = tf.concat((samples, new_point), axis = 1)

  l1 = samples[:,:-1]
  l2 = samples[:,1:]

  #interpolate
  interp = (l1 + l2) / 2
  return interp

def encode_decode(vid_batch, train_vars, beta, svgp_x, svgp_y, clipping_qs=False, layers=[500]):
  f = lambda x : x.name
  vrs_labels = tuple(map(f, train_vars))
  vrs = dict(zip(vrs_labels, train_vars))
  batch, tmax, px, py = [int(s) for s in vid_batch.get_shape()]

  dt = vid_batch.dtype
  T = tf.range(tmax, dtype=dt) + 1.0  # to have range between 1-30 instead of 0-29
  batch_T = tf.concat([tf.reshape(T, (1, tmax)) for i in range(batch)], 0)

  # ENCODER NETWORK
  batch, tmax, px, py = vid_batch.get_shape()

  # first layer, flatten images to vectors
  h0 = tf.reshape(vid_batch, (batch*tmax, px*py))

  # loop over layers in given list
  for l in layers:
      i_dims = int(h0.get_shape()[-1])
      W = vrs["encW:0"]
      B = vrs["encB:0"]
      h0 = tf.matmul(h0, W) + B
      h0 = tf.nn.tanh(h0)

  output_dim = 2*(1 + 1)

  i_dims = int(h0.get_shape()[-1])
  W = vrs["encW_1:0"]
  B = vrs["encB_1:0"]
  h0 = tf.matmul(h0, W) + B

  h0 = tf.reshape(h0, (batch, tmax, output_dim))

  qnet_mu = h0[:, :, :2]
  qnet_var = tf.exp(h0[:, :, 2:])

  # clipping of VAE posterior variance
  if clipping_qs:
      qnet_var = tf.clip_by_value(qnet_var, 1e-6, 1e3)

  # approx posterior distribution
  p_m_x, p_v_x, mu_hat_x, A_hat_x = svgp_x.approximate_posterior_params(batch_T, y=qnet_mu[:, :, 0],
                                                                        noise=qnet_var[:, :, 0])
  p_m_y, p_v_y, mu_hat_y, A_hat_y = svgp_y.approximate_posterior_params(batch_T, y=qnet_mu[:, :, 1],
                                                                        noise=qnet_var[:, :, 1])

  # Inside-ELBO term (L_2 or L_3)
  inside_elbo_recon_x, inside_elbo_kl_x = svgp_x.variational_loss(batch_T, qnet_mu[:, :, 0],  qnet_var[:, :, 0],
                                                                  mu_hat=mu_hat_x, A_hat=A_hat_x)
  inside_elbo_recon_y, inside_elbo_kl_y = svgp_y.variational_loss(batch_T, qnet_mu[:, :, 1], qnet_var[:, :, 1],
                                                                  mu_hat=mu_hat_y, A_hat=A_hat_y)
  inside_elbo_recon = inside_elbo_recon_x + inside_elbo_recon_y
  inside_elbo_kl = inside_elbo_kl_x + inside_elbo_kl_y
  inside_elbo = inside_elbo_recon - inside_elbo_kl

  # added on 20.4., to investigate Cholesky vs diag conundrum
  gp_covariance_posterior_elemwise_mean_x = tf.reduce_mean(p_v_x, 0)
  gp_covariance_posterior_elemwise_mean_y = tf.reduce_mean(p_v_y, 0)

  full_p_mu = tf.stack([p_m_x, p_m_y], axis=2)
  full_p_var = tf.stack([tf.linalg.diag_part(p_v_x), tf.linalg.diag_part(p_v_y)], axis=2)

  # cross entropy term
  ce_term = gauss_cross_entropy(full_p_mu, full_p_var, qnet_mu, qnet_var)  # (batch, tmax, 2)
  ce_term = -tf.reduce_sum(ce_term, (1, 2))

  # latent samples
  epsilon = tf.random.normal(shape=(batch, tmax, 2), seed = 1)
  latent_samples = full_p_mu + epsilon * tf.sqrt(tf.clip_by_value(full_p_var, 1e-4, 1000))
  batch, tmax, _ = latent_samples.shape
  interp_mu = interpolate_latent(full_p_mu)
  interp_var = interpolate_latent(full_p_var)

  #interpolate
  interp = interp_mu + epsilon * tf.sqrt(tf.clip_by_value(interp_var, 1e-4, 1000))
  # ii = sess.run(interp)
  # print(ii.shape)
  # fig, ax = fig.subplots(1,1)
  
  # flatten all latents into one matrix (decoded in i.i.d fashion)
  h0 = tf.reshape(interp, (batch*tmax, 2))

  # loop over layers in given list
  for l in layers:
      i_dims = int(h0.shape[-1])
      W = vrs["decW:0"]
      B = vrs["decB:0"]
      h0 = tf.matmul(h0, W) + B
      h0 = tf.nn.tanh(h0)

  # final layer just outputs full video batch
  l = px*py
  i_dims = int(h0.shape[-1])
  W = vrs["decW_1:0"]
  B = vrs["decB_1:0"]
  h0 = tf.matmul(h0, W) + B

  pred_vid_batch_logits = tf.reshape(h0, (batch, tmax, px, py))
  pred_vid = tf.nn.sigmoid(pred_vid_batch_logits)
  recon_term = tf.nn.sigmoid_cross_entropy_with_logits(labels=vid_batch, logits=pred_vid_batch_logits)
  recon_term = tf.reduce_sum(-recon_term, (1, 2, 3))  # (batch)

  KL_term = ce_term + inside_elbo
  CPH_elbo = recon_term + beta * KL_term
  return pred_vid

def eval_decoder(sess, train_vars, latent_samples_shape, px, py, layers=[500]):

  latent_samples = tf.placeholder(tf.float32, shape = latent_samples_shape)
  # latent samples

  pred_vid = tf.nn.softmax(pred_vid_batch_logits)

  return pred_vid, latent_samples

def run_experiment(args):
    """
    Moving ball experiment.

    :param args:
    :return:
    """
    if args.save:
        # Make a folder to save everything
        extra = args.elbo + "_" + str(args.beta0)
        print(args.reload)
        if not args.reload:
          chkpnt_dir = make_checkpoint_folder(args.base_dir)
          pic_folder = chkpnt_dir + "pics/"
          res_file = chkpnt_dir + "res/ELBO_pandas"
          print("\nCheckpoint Directory:\n"+str(chkpnt_dir)+"\n")
        else:
          chkpnt_dir = args.base_dir
          pic_folder = os.path.join(chkpnt_dir, "pics")
          res_file = os.path.join(chkpnt_dir, "res", "ELBO_pandas")
          print("\nCheckpoint Directory:\n"+str(chkpnt_dir)+"\n")


    # Data synthesis settings
    batch = 35
    tmax = args.tmax
    px = 32
    py = 32
    r = 3
    vid_lt = args.vidlt
    m = args.m

    if args.elbo == 'VAE':
        # A GP prior with a RBF kernel and a very small length scale reduces to the standard Gaussian prior
        model_lt = 0.001
    else:
        model_lt = args.modellt

    assert model_lt == vid_lt or args.GP_joint or args.elbo == 'VAE', \
        "GP params of data and model should match. Except when \
         doing a joint optimization of GP parameters or when fitting normal VAE."

    # Load/create batches of reproducible videos
    if os.path.isfile(args.base_dir + "Test_Batches_{}_{}.pkl".format(vid_lt, tmax*2)):
        with open(args.base_dir + "Test_Batches_{}_{}.pkl".format(vid_lt, tmax*2), "rb") as f:
            Test_Batches = pickle.load(f)
    else:
        make_batch = lambda s: Make_Video_batch(tmax=tmax*2, px=px, py=py, lt=vid_lt, batch=batch, seed=s, r=r)
        Test_Batches = [make_batch(s) for s in range(10)]
        with open(args.base_dir + "Test_Batches_{}_{}.pkl".format(vid_lt, tmax*2), "wb") as f:
            pickle.dump(Test_Batches, f)

    # Initialise a plots
    # this plot displays a  batch of videos + latents + reconstructions
    if args.save or args.show_pics:
        fig, ax = plt.subplots(4, 4, figsize=(8, 8), constrained_layout=True)
        plt.ion()

    if args.squares_circles:
        truth_c, V_c = Make_circles(tmax=tmax); batch_V_c = np.tile(V_c, (batch,1,1,1))
        truth_sq, V_sq = Make_squares(tmax=tmax); batch_V_sq = np.tile(V_sq, (batch,1,1,1))

    # make sure everything is created in the same graph!
    graph = tf.Graph()
    with graph.as_default():

        # Make all the graphs
        beta = tf.compat.v1.placeholder(dtype=tf.float32, shape=())

        vid_batch = build_video_batch_graph(batch=batch, tmax=tmax, px=px, py=py, r=r, lt=vid_lt)

        if args.elbo in ['GPVAE_Pearce', 'VAE', 'NP']:
            elbo, rec, pkl, p_m, \
                p_v, q_m, q_v, pred_vid, \
                l_GP_x, l_GP_y, _ = build_pearce_elbo_graphs(vid_batch, beta, type_elbo=args.elbo, lt=model_lt,
                                                             GP_joint=args.GP_joint, GP_init=args.GP_init)

        else:  # SVGPVAE_Titsias, SVGPVAE_Hensman

            titsias = 'Titsias' in args.elbo
            fixed_gp_params = not args.GP_joint
            fixed_inducing_points = not args.ip_joint
            svgp_x_ = SVGP(titsias=titsias, num_inducing_points=m,
                           fixed_inducing_points=fixed_inducing_points,
                           tmin=1, tmax=tmax, vidlt=vid_lt, fixed_gp_params=fixed_gp_params, name='x',
                           jitter=args.jitter, ip_min=args.ip_min, ip_max=args.ip_max, GP_init=args.GP_init)
            svgp_y_ = SVGP(titsias=titsias, num_inducing_points=m, fixed_inducing_points=fixed_inducing_points,
                           tmin=1, tmax=tmax, vidlt=vid_lt, fixed_gp_params=fixed_gp_params, name='y',
                           jitter=args.jitter, ip_min=args.ip_min, ip_max=args.ip_max, GP_init=args.GP_init)

            elbo, rec, pkl, l3_elbo, ce_term,\
            p_m, p_v, q_m, q_v, pred_vid, l_GP_x, l_GP_y,\
            l3_elbo_recon, l3_elbo_kl, inducing_points_x, inducing_points_y, \
            gp_cov_full_mean_x, gp_cov_full_mean_y, latent_samples, _ = build_SVGPVAE_elbo_graph(vid_batch, beta,
                                                                                 svgp_x=svgp_x_, svgp_y=svgp_y_,
                                                                                 clipping_qs=args.clip_qs)

        # The actual loss functions
        loss = -tf.reduce_mean(elbo)
        e_elb = tf.reduce_mean(elbo)
        e_pkl = tf.reduce_mean(pkl)
        e_rec = tf.reduce_mean(rec)
        if 'SVGPVAE' in args.elbo:
            e_l3_elbo = tf.reduce_mean(l3_elbo)
            e_ce_term = tf.reduce_mean(ce_term)
            e_l3_elbo_recon = tf.reduce_mean(l3_elbo_recon)
            e_l3_elbo_kl = tf.reduce_mean(l3_elbo_kl)

        # Add optimizer ops to graph (minimizing neg elbo!), print out trainable vars
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.compat.v1.train.AdamOptimizer()
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        if args.clip_grad:
            gradients = tf.gradients(loss, train_vars)
            gradients = [tf.clip_by_value(grad, -100000.0, 100000.0) for grad in gradients]
            optim_step = optimizer.apply_gradients(grads_and_vars=zip(gradients, train_vars),
                                                   global_step=global_step)

        else:
            optim_step = optimizer.minimize(loss=loss,
                                            var_list=train_vars,
                                            global_step=global_step)

        print("\n\nTrainable variables:")
        for v in train_vars:
            print(v)

        # Initializer ops for the graph and saver
        init_op = tf.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()

        if args.save:
            # Results to be tracked and Pandas saver
            res_vars = [global_step,
                        loss,
                        e_elb,
                        e_rec,
                        e_pkl,
                        tf.math.reduce_min(q_v),
                        tf.math.reduce_max(q_v),
                        tf.math.reduce_min(p_v),
                        tf.math.reduce_max(p_v),
                        tf.math.reduce_min(q_m),
                        tf.math.reduce_max(q_m),
                        tf.math.reduce_min(p_m),
                        tf.math.reduce_max(p_m),
                        l_GP_x,
                        l_GP_y]
            if 'SVGPVAE' in args.elbo:
                res_vars += [e_l3_elbo,
                             e_ce_term,
                             e_l3_elbo_recon,
                             e_l3_elbo_kl,
                             inducing_points_x,
                             inducing_points_y,
                             gp_cov_full_mean_x,
                             gp_cov_full_mean_y]
            res_names= ["Step",
                        "Loss",
                        "Train ELBO",
                        "Train Reconstruction",
                        "Train Prior KL",
                        "min qs_var",
                        "max qs_var",
                        "min q_var",
                        "max q_var",
                        'min qs_mean',
                        'max qs_mean',
                        'min q_mean',
                        'max q_mean',
                        "l_GP_x",
                        "l_GP_y"]
            if 'SVGPVAE' in args.elbo:
                res_names += ['SVGP elbo',
                              'ce term',
                              'SVGP elbo recon',
                              'SVGP elbo KL',
                               'inducing_points_x',
                              'inducing_points_y',
                              'gp_cov_full_mean_x',
                              'gp_cov_full_mean_y']
            res_names += ["MSE",  "Beta",  "Time"]
            res_saver = pandas_res_saver(res_file, res_names)

        # Now let's start doing some computation!
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.ram)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            # Attempt to restore weights
            print(chkpnt_dir+'debug')
            try:
                saver.restore(sess, tf.train.latest_checkpoint(chkpnt_dir+'debug'))
                print("\n\nRestored Model Weights")
            except:
                sess.run(init_op)
                print("\n\nInitialised Model Weights")
    
            titsias = 'Titsias' in args.elbo
            fixed_gp_params = not args.GP_joint
            fixed_inducing_points = not args.ip_joint
            svgp_x_ = SVGP(titsias=titsias, num_inducing_points=m,
                           fixed_inducing_points=fixed_inducing_points,
                           tmin=1, tmax=tmax, vidlt=vid_lt, fixed_gp_params=fixed_gp_params, name='x',
                           jitter=args.jitter, ip_min=args.ip_min, ip_max=args.ip_max, GP_init=args.GP_init)
            svgp_y_ = SVGP(titsias=titsias, num_inducing_points=m, fixed_inducing_points=fixed_inducing_points,
                           tmin=1, tmax=tmax, vidlt=vid_lt, fixed_gp_params=fixed_gp_params, name='y',
                           jitter=args.jitter, ip_min=args.ip_min, ip_max=args.ip_max, GP_init=args.GP_init)
            TT, TD = Test_Batches[0]
            TT1 = TT[:,::2]
            TD1 = TD[:,::2]
            
            TT2 = TT[:,1::2]
            TD2 = TD[:,1::2]
            vid_batch = build_video_batch_graph(batch=batch, tmax=tmax, px=px, py=py, r=r, lt=vid_lt)
            beta = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
            ppred_vid = encode_decode(vid_batch, train_vars, beta, svgp_x_, svgp_y_)
            ppred_vid = sess.run(ppred_vid, {vid_batch: TD1, beta: 1})
            reconvid = sess.run(pred_vid, {vid_batch: TD1, beta: 1})
            reconvid2 = sess.run(pred_vid, {vid_batch: TD2, beta: 1})
            chk_video = double_frame_rate(reconvid, ppred_vid)
            print(chk_video.shape)
            plot_frames(chk_video[:,1::2], args.base_dir, reconvid2)
            g_s = 0
            for i, chk_batch in enumerate(chk_video):
                plot_sequence(chk_batch, name=str(i), name_dir=str(g_s))
                plot_sequence(reconvid[i], name=str(i), name_dir=f'originals_{str(g_s)}', interval = 100)
                if i == 2:
                    break
            mse = compute_average_mse(chk_video[:,1::2], reconvid2)
            psnr = compute_average_psnr(chk_video[:,1::2], reconvid2)
            print('MSE',mse)
            print('PSNR',psnr)


if __name__=="__main__":

    default_base_dir = os.getcwd()

    parser = argparse.ArgumentParser(description='Moving ball experiment')
    parser.add_argument('--steps', type=int, default=25000, help='Number of steps of Adam')
    parser.add_argument('--beta0', type=float, default=1, help='initial beta annealing value')
    parser.add_argument('--elbo', type=str, choices=['GPVAE_Pearce', 'VAE', 'NP', 'SVGPVAE_Hensman', 'SVGPVAE_Titsias'],
                        default='GPVAE_Pearce',
                        help='Structured Inf Nets ELBO or Neural Processes ELBO')
    parser.add_argument('--modellt', type=float, default=2, help='time scale of model to fit to data')
    parser.add_argument('--base_dir', type=str, default=default_base_dir, help='folder within a new dir is made for each run')
    parser.add_argument('--expid', type=str, default="debug", help='give this experiment a name')
    parser.add_argument('--ram', type=float, default=0.5, help='fraction of GPU ram to use')
    parser.add_argument('--seed', type=int, default=None, help='seed for rng')
    parser.add_argument('--tmax', type=int, default=30, help='length of videos')
    parser.add_argument('--m', type=int, default=15, help='number of inducing points')
    parser.add_argument('--GP_joint', action="store_true", help='GP hyperparams joint optimization.')
    parser.add_argument('--ip_joint', action="store_true", help='Inducing points joint optimization.')
    parser.add_argument('--clip_qs', action="store_true", help='Clip variance of inference network.')

    parser.add_argument('--show_pics', action="store_true", help='Show images during training.')
    parser.add_argument('--save', action="store_true", help='Save model metrics in Pandas df as well as images.')
    parser.add_argument('--squares_circles', action="store_true", help='Whether or not to plot squares and circles.')

    parser.add_argument('--ip_min', type=int, default=1, help='ip start')
    parser.add_argument('--ip_max', type=int, default=30, help='ip end')
    parser.add_argument('--jitter', type=float, default=1e-9, help='noise for GP operations (inverse, cholesky)')
    parser.add_argument('--clip_grad', action="store_true", help='Whether or not to clip gradients.')
    parser.add_argument('--vidlt', type=float, default=2, help='time scale for data generation')
    parser.add_argument('--GP_init', type=float, default=2,
                        help='Initial value for GP kernel length scale. Used when running --GP_joint .')
    parser.add_argument('--reload', action="store_true", help='Reloads model.')
    args = parser.parse_args()

    run_experiment(args)



