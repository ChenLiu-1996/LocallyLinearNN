import argparse
import os
import random
import sys

import numpy as np
import sklearn.datasets
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.insert(0, import_dir + '/utils/')
from attribute_hashmap import AttributeHashmap
from seed import seed_everything

sys.path.insert(0, import_dir + '/nn/')
from toy_gans import GAN, WGAN, WGANGP


# Dataset iterator
def real_dist_generator(dataset_name: str, batch_size: int):
    if dataset_name == '25gaussians':

        dataset = []
        for _ in np.arange(np.ceil(batch_size / 25)):
            for x in np.arange(-2, 3):
                for y in np.arange(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 2.828  # stdev
        while True:
            np.random.shuffle(dataset)
            yield dataset[:batch_size]

    elif dataset_name == 'swissroll':

        while True:
            dataset = sklearn.datasets.make_swiss_roll(n_samples=batch_size,
                                                    noise=0.25)[0]
            dataset = dataset.astype('float32')[:, [0, 2]]
            dataset /= 7.5  # stdev plus a little
            yield dataset

    elif dataset_name == '8gaussians':

        scale = 2.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for _ in np.arange(batch_size):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev
            yield dataset


def generate_image(real_dist_gen,
                   model,
                   device,
                   ax: plt.Axes,
                   num_pts: int = 128,
                   pt_range: float = 3):

    points = np.zeros((num_pts, num_pts, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-pt_range, pt_range, num_pts)[:, None]
    points[:, :, 1] = np.linspace(-pt_range, pt_range, num_pts)[None, :]
    points = points.reshape((-1, 2))
    points = torch.from_numpy(points).to(device)

    x = y = np.linspace(-pt_range, pt_range, num_pts)

    with torch.no_grad():
        z = torch.randn((args.batch_size, 2)).to(device)
        fake_samples = model.forward_G(z).cpu().numpy()
        disc_map = model.forward_D(points).cpu().numpy()

    ax.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

    real_dist = real_dist_gen.__next__()
    ax.scatter(real_dist[:, 0], real_dist[:, 1], c='orange', marker='+')
    ax.scatter(fake_samples[:, 0], fake_samples[:, 1], c='green', marker='+')


def train(args):
    device = torch.device('cuda:%d' %
                          args.gpu_id if torch.cuda.is_available() else 'cpu')

    if args.gan_name == 'GAN':
        model = GAN(learning_rate=args.learning_rate,
                          device=device,
                          batch_size=args.batch_size,
                          linearity_lambda=args.linearity_lambda)
    elif args.gan_name == 'WGAN':
        model = WGAN(learning_rate=args.learning_rate,
                          device=device,
                          batch_size=args.batch_size,
                          linearity_lambda=args.linearity_lambda)
    elif args.gan_name == 'WGANGP':
        model = WGANGP(learning_rate=args.learning_rate,
                          device=device,
                          batch_size=args.batch_size,
                          linearity_lambda=args.linearity_lambda)

    real_dist_gen = real_dist_generator(dataset_name=args.dataset_name,
                                        batch_size=args.batch_size)

    fig = plt.figure(figsize=(6, 4 * args.num_figs))
    fig_idx = 1

    for iter_idx in tqdm(range(1, args.iters + 1)):
        # Train discriminator
        model.optimize_D(real_dist_gen=real_dist_gen)

        # Train generator
        model.optimize_G(real_dist_gen=real_dist_gen)

        if iter_idx % int(args.iters / args.num_figs) == 0:
            ax = fig.add_subplot(args.num_figs, 1, fig_idx)
            fig_idx += 1
            generate_image(real_dist_gen=real_dist_gen,
                           model=model,
                           device=device,
                           ax=ax)
            ax.set_title('epoch: %d' % iter_idx)
            fig.tight_layout()
            fig.savefig(args.fig_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset_name', type=str, default='8gaussians')
    parser.add_argument('--gan_name', type=str, default='GAN')
    parser.add_argument('--linearity_lambda', type=float, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--num_figs', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=1)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)

    seed_everything(args.random_seed)

    os.makedirs('../results/figures/', exist_ok=True)
    args.fig_save_path = '../results/figures/toy-%s-%s-lambda=%s.png' % (
        args.dataset_name, args.gan_name, args.linearity_lambda)
    train(args)
