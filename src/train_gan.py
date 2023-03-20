import argparse
import os
import sys
from typing import List, Tuple, Union

import numpy as np
import yaml
import torch
import torchvision
from gan_evaluator import GAN_Evaluator
from matplotlib import pyplot as plt
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.insert(0, import_dir + '/utils/')
from attribute_hashmap import AttributeHashmap
from log_utils import log
from seed import seed_everything

sys.path.insert(0, import_dir + '/nn/')
from linearity import linearity_constraint
from toy_gans import GAN, WGAN, WGANGP


def update_config_dirs(config: AttributeHashmap) -> AttributeHashmap:
    root_dir = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    for key in config.keys():
        if type(config[key]) is str and '$ROOT_DIR' in config[key]:
            config[key] = config[key].replace('$ROOT_DIR', root_dir)
    return config


def normalize(
        image: Union[np.array, torch.Tensor],
        dynamic_range: List[float] = [0, 1]) -> Union[np.array, torch.Tensor]:
    assert len(dynamic_range) == 2

    x1, x2 = image.min(), image.max()
    y1, y2 = dynamic_range[0], dynamic_range[1]

    slope = (y2 - y1) / (x2 - x1)
    offset = (y1 * x2 - y2 * x1) / (x2 - x1)

    image = image * slope + offset

    # Fix precision issue.
    image = image.clip(y1, y2)
    return image


def get_dataloaders(
    config: AttributeHashmap
) -> Tuple[Tuple[torch.utils.data.DataLoader, ], AttributeHashmap]:
    if config.dataset == 'mnist':
        config.in_channels = 1
        config.num_classes = 10
        config.image_shape = (3, 28, 28)
        dataset_mean = (0.1307, )
        dataset_std = (0.3081, )
        torchvision_dataset = torchvision.datasets.MNIST

    elif config.dataset == 'cifar10':
        config.in_channels = 3
        config.num_classes = 10
        config.image_shape = (3, 32, 32)
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset = torchvision.datasets.CIFAR10

    elif config.dataset == 'cifar100':
        config.in_channels = 3
        config.num_classes = 100
        config.image_shape = (3, 32, 32)
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset = torchvision.datasets.CIFAR100

    elif config.dataset == 'stl10':
        config.in_channels = 3
        config.num_classes = 10
        config.image_shape = (3, 96, 96)
        dataset_mean = (0.4467, 0.4398, 0.4066)
        dataset_std = (0.2603, 0.2566, 0.2713)
        torchvision_dataset = torchvision.datasets.STL10

    else:
        raise ValueError(
            '`config.dataset` value not supported. Value provided: %s.' %
            config.dataset)

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    if config.dataset in ['mnist', 'cifar10', 'cifar100']:
        train_loader = torch.utils.data.DataLoader(
            torchvision_dataset(config.dataset_dir,
                                train=True,
                                download=True,
                                transform=transform_train),
            batch_size=config.batch_size,
            shuffle=True)
        val_loader = torch.utils.data.DataLoader(torchvision_dataset(
            config.dataset_dir,
            train=False,
            download=True,
            transform=transform_val),
                                                 batch_size=config.batch_size,
                                                 shuffle=False)
    elif config.dataset in ['stl10']:
        train_loader = torch.utils.data.DataLoader(
            torchvision_dataset(config.dataset_dir,
                                split='train',
                                download=True,
                                transform=transform_train),
            batch_size=config.batch_size,
            shuffle=True)
        val_loader = torch.utils.data.DataLoader(torchvision_dataset(
            config.dataset_dir,
            split='test',
            download=True,
            transform=transform_val),
                                                 batch_size=config.batch_size,
                                                 shuffle=False)

    return (train_loader, val_loader), config


def train(config):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    dataloaders, config = get_dataloaders(config=config)
    train_loader, val_loader = dataloaders

    if config.gan_name == 'GAN':
        model = GAN(learning_rate=config.learning_rate,
                    device=device,
                    batch_size=config.batch_size,
                    img_size=config.image_shape[1:],
                    z_dim=config.z_dim,
                    num_channel=config.image_size[0],
                    linearity_lambda=config.linearity_lambda)
    elif config.gan_name == 'WGAN':
        model = WGAN(learning_rate=config.learning_rate,
                     device=device,
                     batch_size=config.batch_size,
                    img_size=config.image_shape[1:],
                    z_dim=config.z_dim,
                    num_channel=config.image_size[0],
                     linearity_lambda=config.linearity_lambda)
    elif config.gan_name == 'WGANGP':
        model = WGANGP(learning_rate=config.learning_rate,
                       device=device,
                       batch_size=config.batch_size,
                    img_size=config.image_shape[1:],
                    z_dim=config.z_dim,
                    num_channel=config.image_size[0],
                       linearity_lambda=config.linearity_lambda)
    else:
        raise ValueError('`config.gan_name` not supported: %s.' %
                         args.gan_name)

    # Create folders.
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.plot_folder, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    log_path = '%s/%s.log' % (config.log_dir, config.config_file_name)

    # Log the config.
    config_str = 'Config: \n'
    for key in config.keys():
        config_str += '%s: %s\n' % (key, config[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=config.log_dir, to_console=False)

    # Our GAN Evaluator.
    evaluator = GAN_Evaluator(device=device,
                              num_images_real=len(train_loader.dataset),
                              num_images_fake=len(train_loader.dataset))

    # We can pre-load the real images in the format of a dataloader.
    # Of course you can do that in individual batches, but this way is neater.
    # Because in CIFAR10, each batch contains a (image, label) pair, we set `idx_in_loader` = 0.
    # If we only have images in the datalaoder, we can set `idx_in_loader` = None.
    evaluator.load_all_real_imgs(real_loader=train_loader, idx_in_loader=0)

    epoch_list, IS_list, FID_list = [], [], []
    for epoch_idx in range(config.max_epochs):

        model.train()
        for batch_idx, (x_real, _) in enumerate(tqdm(train_loader)):
            shall_plot = batch_idx % config.plot_interval == config.plot_interval - 1 or batch_idx == len(
                train_loader) - 1

            B = x_real.shape[0]
            x_real = normalize(x_real, dynamic_range=[-1, 1])
            num_visited += B

            # Train discriminator
            model.optimize_D(x_real=x_real)

            # Train generator
            model.optimize_G()

            # Here comes the IS and FID values.
            # These are the values evaluated with the data available so far.
            # `IS_std` is only meaningful if `EVALUATOR.IS_splits` > 1.
            if shall_plot:
                z = torch.randn((B, model.z_dim)).to(device)
                with torch.no_grad():
                    model.eval()
                    x_fake = model.forward_G(z)
                    model.train()
                IS_mean, _, FID = evaluator.fill_fake_img_batch(
                    fake_batch=x_fake)
                epoch_list.append(epoch_idx + batch_idx / len(train_loader))
                IS_list.append(IS_mean)
                FID_list.append(FID)
            else:
                evaluator.fill_fake_img_batch(fake_batch=x_fake,
                                              return_results=False)

            # Plot intermediate results.
            if shall_plot:
                num_samples = 5
                rng = torch.Generator(device=device)
                rng.manual_seed(config.random_seed)
                H, W = config.image_shape[1:]
                fig = plt.figure(figsize=(15, 6))
                for i in range(num_samples):
                    real_image = next(iter(train_loader))[0][0, ...][None, ...]
                    real_image = real_image.type(torch.FloatTensor).to(device)
                    with torch.no_grad():
                        model.eval()
                        y_pred_real = model.forward_D(real_image).view(-1)
                        model.train()
                    real_image = np.moveaxis(real_image.cpu().detach().numpy(),
                                             1, -1).reshape(H, W, 3)
                    real_image = normalize(real_image, dynamic_range=[0, 1])
                    ax = fig.add_subplot(2, num_samples, i + 1)
                    ax.imshow(real_image)
                    ax.set_axis_off()
                    ax.set_title('D(x): %.3f' % y_pred_real)
                    fix_z = torch.randn((1, model.z_dim, 1, 1),
                                        device=device,
                                        generator=rng)
                    with torch.no_grad():
                        model.eval()
                        generated_image = model.forward_G(fix_z)
                        y_pred_fake = model.forward_D(
                            generated_image.to(device)).view(-1)
                        model.train()
                    generated_image = normalize(generated_image,
                                                dynamic_range=[0, 1])
                    ax = fig.add_subplot(2, num_samples, num_samples + i + 1)
                    ax.imshow(
                        np.moveaxis(generated_image.cpu().detach().numpy(), 1,
                                    -1).reshape(H, W, 3))
                    ax.set_title('D(G(z)): %.3f' % y_pred_fake)
                    ax.set_axis_off()

                plt.tight_layout()
                plt.savefig('%s/epoch_%s_batch_%s_generated' %
                            (config.plot_folder, str(epoch_idx).zfill(4),
                             str(batch_idx).zfill(4)))
                plt.close(fig=fig)

                log('Train [E %s/%s, B %s/%s] IS: %.3f, FID: %.3f'
                    % (epoch_idx + 1, config.max_epochs, batch_idx + 1,
                       len(train_loader), IS_mean, FID),
                    filepath=config.log_dir,
                    to_console=False)

        # Update the IS and FID curves every epoch.
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(epoch_list, IS_list, color='firebrick')
        ax.plot(epoch_list, IS_list, color='firebrick')
        ax.set_ylabel('Inception Score (IS)')
        ax.set_xlabel('Epoch')
        ax.spines[['right', 'top']].set_visible(False)
        ax = fig.add_subplot(1, 2, 2)
        ax.scatter(epoch_list, FID_list, color='firebrick')
        ax.plot(epoch_list, FID_list, color='firebrick')
        ax.set_ylabel('Frechet Inception Distance (FID)')
        ax.set_xlabel('Epoch')
        ax.spines[['right', 'top']].set_visible(False)
        plt.tight_layout()
        plt.savefig('%s/IS_FID_curve' % config.plot_folder)
        plt.close(fig=fig)

        # Need to clear up the fake images every epoch.
        evaluator.clear_fake_imgs()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entry point to train student network.')
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--gpu-id',
                        help='Available GPU index.',
                        type=int,
                        default=0)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config = parse_setting(AttributeHashmap(config))

    seed_everything(config.random_seed)
    train(config=config)