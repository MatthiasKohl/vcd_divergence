import tensorflow as tf
import tensorflow_probability as tfp
import argparse
import os
import time
import numpy as np


METHOD_NAMES = {
    1: 'VCD',
    # 2: 'VCD_noControlVar',
    # 3: 'standardKL',
    # 4: 'Hoffman',
}
DATA_NAMES = {
    1: 'mnist',
    # 2: 'fashionmnist',
}
OUT_NAMES = {
    1: '{}VAE_{}',
    # 2: '{}gaussMF_{}',
    # 3: '{}poissMF_{}',
    # 4: '{}logisticMF_{}',
}

# global parameters
ITERS = 200000
BATCH_SIZE = 100
# Stepsize parameters
RHO_MODEL_PARAMS = 0.0005
RHO_THETA = 0.0005
RHO_SIGMA = 0.00025
REDUCED_BY = 0.9
REDUCED_EVERY = 15000
# RMSprop parameters
TT = 1
KAPPA0 = 0.1
# Model params
ENCODER_NUM_UNITS = [200, 200]
# MCMC params (adapt/leapfrog)
ADAPT = 1
LF = 5
# metric values
with tf.device('/CPU:0'):
    stochasticDiv = tf.zeros(ITERS)
    ELBO_q = tf.zeros(ITERS)
    expLogLik_qt = tf.zeros(ITERS)
    first_term = tf.zeros(ITERS)
    telapsed = tf.zeros(ITERS)
    test_loglik = tf.zeros(ITERS)


# needed as activation for sigma encoder
@tf.function
def softplus_threshold(x):
    ee = 1e-4
    return tf.where(
        x >= 0,
        x + tf.math.log(1 + tf.math.exp(ee - x)),
        ee + tf.math.log(1 + tf.math.exp(x - ee))
    )


def load_data(p):
    if p['flag_binarize_data']:
        if p['data_id'] == 1:
            # TODO data types
            p['data'] = tf.data.Dataset.from_tensor_slices(np.loadtxt('dat/mnist/binarized_mnist_train.amat', dtype=np.float32))
            p['data_test'] = tf.data.Dataset.from_tensor_slices(np.loadtxt('dat/mnist/binarized_mnist_test.amat', dtype=np.float32))
        else:
            # skipping binarizing fashion MNIST data
            raise NotImplementedError('TODO')
        p['N'] = p['data'].shape[0]
        p['D'] = p['data'].shape[1]
        p['test_N'] = p['data_test'].shape[0]
    else:
        # skipping loading mnist_all as we always use binarized data
        raise NotImplementedError('TODO')


def main():
    parser = argparse.ArgumentParser(
        description='Port of matlab to tensorflow python code'
    )
    parser.add_argument(
        '--method_id', metavar='<method ID>',
        default=1, type=int, required=False,
        help='(1=VCD; 2=VCD(no controlVar); 3=standardKL; 4=Hoffman\'s). '
             'For now, only 1 is implemented'
    )
    parser.add_argument(
        '--data_id', metavar='<data ID>',
        default=1, type=int, required=False,
        help='(1=MNIST; 2=Fashion-MNIST). For now, only 1 will be tested'
    )
    parser.add_argument(
        '--model_id', metavar='<model ID>',
        default=1, type=int, required=False,
        help='(1=BernoulliVAE; 2=Gaussian MF; 3=Poisson MF; 4=Logistic MF). '
             'For now, only 1 will is implemented'
    )
    parser.add_argument(
        '--burn_iters', metavar='<Burn Iters>',
        default=0, type=int, required=False,
        help='Number of HMC burn-in iterations for MCMC'
    )
    parser.add_argument(
        '--sampling_iters', metavar='<Sampling Iters>',
        default=8, type=int, required=False,
        help='Number of HMC sampling iterations for MCMC, '
             'excluding burn-in iterations'
    )
    args = parser.parse_args()
    assert args.method_id in METHOD_NAMES, 'Unknown method ID {}'.format(args.method_id)
    assert args.data_id in DATA_NAMES, 'Unknown data ID {}'.format(args.data_id)
    assert args.model_id in OUT_NAMES, 'Unknown model ID {}'.format(args.model_id)
    assert args.burn_iters >= 0, 'Burn iters must be non-negative, got {}'.format(args.burn_iters)
    assert args.sampling_iters >= 0, 'Sampling iters must be non-negative, got {}'.format(args.sampling_iters)
    p = vars(args)
    p['method_name'] = METHOD_NAMES[args.method_id]
    p['data_name'] = DATA_NAMES[args.data_id]
    p['out_dir'] = os.path.join('out', 'burn{}_sampling{}'.format(args.burn_iters, args.sampling_iters))
    p['out_name'] = OUT_NAMES[args.model_id].format(p['data_name'], p['method_name'])
    tf.random.set_seed(1)
    p['dim_z'] = 50 if args.method_id != 1 else 10
    if args.method_id == 2:
        p['variance_lik'] = 0.01
    p['flag_control_variate'] = False
    if args.method_id == 1:
        p['flag_control_variate'] = True
        p['decay_control_variate'] = 0.9
        p['control_iters_wait'] = 3000
    p['flag_binarize_data'] = False if args.model_id != 1 and args.model_id != 4 else True
    p['flag_normalize_data'] = False if args.model_id != 2 else True
    
    load_data(p)

    # skipping normalizing data to [0, 1] as we always have binarized data
    # skipping gammaln as we don't use poissMF

    # TODO weight init (see netcreate.m)
    decoder = tf.keras.Sequential([
        tf.keras.Input(shape=(p['dim_z'],)),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(p['D'], activation='sigmoid')
    ])
    # skipping other parameters for VAE as we always use BernouilliVAE
    encoderMu = tf.keras.Sequential([
        tf.keras.Input(shape=(p['D'],)),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(p['dim_z'])
    ])
    encoderSigma = tf.keras.Sequential([
        tf.keras.Input(shape=(p['D'],)),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(p['dim_z']),
        tf.keras.layers.Activation(softplus_threshold)
    ])
    def update_state(x):
        return decoder(x)

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=update_state,
            num_leapfrog_steps=LF,
            step_size=.5 / p['dim_z']),
        num_adaptation_steps=p['burn_iters'] + p['sampling_iters'] if ADAPT else p['sampling_iters']
    )
    
    @tf.function
    def run_chain(z):
        return tfp.mcmc.sample_chain(
            num_results=p['sampling_iters'],
            num_burnin_steps=p['burn_iters'],
            current_state=z,
            kernel=adaptive_hmc,
            return_final_kernel_results=True
        )
    
    optimizer = tf.keras.optimizers.RMSprop()

    for it in range(ITERS):
        train_iter = iter(p['data'].shuffle().batch(BATCH_SIZE).prefetch())
        for batch in train_iter:
            t_start = time.time()
            with tf.GradientTape() as tape:
                mu = encoderMu(batch)
                sigma = encoderSigma(batch)
                eta = tf.random.normal(shape=(BATCH_SIZE, p['dim_z']))
                z = mu + eta * sigma
                logpxz = decoder(z)
                first_term[it] = logpxz.mean() + .5 * p['dim_z']
                zt = run_chain(z)
                # TODO extra outputs (in particular adapt step size for HMC)
                zt_dec = decoder(zt)
                diff = (zt - mu) / sigma
                diff2 = diff * diff
                f_zt = zt_dec + .5 * diff2.sum(axis=1)
                stochasticDiv[it] = -first_term[it] + f_zt.mean()
            gradientsMu = tape.gradient(stochasticDiv[it], encoderMu.trainable_variables)
            optimizer.apply_gradients(zip(gradientsMu, encoderMu.trainable_variables))
            gradientsSigma = tape.gradient(stochasticDiv[it], encoderSigma.trainable_variables)
            optimizer.apply_gradients(zip(gradientsSigma, encoderSigma.trainable_variables))
            entropy = .5 * p['dim_z'] * np.log(2 * np.pi) + tf.math.log(encoderSigma[-1].grad).sum(axis=1) + p['dim_z'] / 2
            ELBO_q[it] = logpxz.mean() + mean(entropy)
            telapsed[it] = time.time() - t_start


if __name__ == "__main__":
    main()
