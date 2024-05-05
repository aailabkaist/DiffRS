import os
import click
from tqdm.auto import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import tensorflow as tf
import io
from torchvision.utils import make_grid, save_image
import classifier_lib
import random
import time

#----------------------------------------------------------------------------
# Proposed DiffRS sampler.

def diffrs_sampler(
    boosting, time_min, time_max, vpsde, rej_percentile, discriminator,
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    backsteps=0, min_backsteps=0, max_backsteps=18, mode='default', outdir=None, adaptive_pickle=None, adaptive_pickle2=None,
    class_idx=None, batch_size=100, num_samples=50000, iter_warmup=10, max_iter=999999, no_zero=0,
):

    S_churn_vec = torch.tensor([S_churn] * latents.shape[0], device=latents.device)
    S_churn_max = torch.tensor([np.sqrt(2) - 1] * latents.shape[0], device=latents.device)
    S_noise_vec = torch.tensor([S_noise] * latents.shape[0], device=latents.device)
    gamma_vec = torch.minimum(S_churn_vec / num_steps, S_churn_max)

    def sampling_loop(x_next, lst_idx, log_ratio_prev, per_sample_nfe, labels, warmup=False):
        t_cur = t_steps[lst_idx]
        t_next = t_steps[lst_idx+1]

        x_cur = x_next

        bool_zero = lst_idx == 0
        if warmup:
            if bool_zero.sum() != 0:
                log_ratio_prev[bool_zero] = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_cur[bool_zero], t_steps[lst_idx][bool_zero], net.img_resolution, time_min, time_max, labels, log_only=True).detach().cpu()

                for i in range(len(log_ratio_prev[bool_zero])):
                    lst_adaptive[0].append(log_ratio_prev[bool_zero][i].cpu())
        else:
            if min_backsteps == 0:
                while bool_zero.sum() != 0:
                    x_check = x_cur[bool_zero]
                    labels_ = labels[bool_zero] if labels is not None else None
                    log_ratio_prev_check = log_ratio_prev[bool_zero]
                    log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_check, t_steps[lst_idx][bool_zero], net.img_resolution, time_min, time_max, labels_, log_only=True).detach().cpu()
                    bool_neg_log_ratio = log_ratio < adaptive[lst_idx][bool_zero] + torch.log(torch.rand_like(log_ratio) + 1e-7)
                    bool_reject = torch.arange(len(bool_zero), device=bool_zero.device)[bool_zero][bool_neg_log_ratio]
                    bool_accept = torch.arange(len(bool_zero), device=bool_zero.device)[bool_zero][~bool_neg_log_ratio]

                    if bool_neg_log_ratio.sum() != 0:
                        eps_rand = randn_like(x_check[bool_neg_log_ratio])
                        x_back = t_steps[0] * eps_rand
                        x_cur[bool_reject] = x_back

                    log_ratio_prev_check[~bool_neg_log_ratio] = log_ratio[~bool_neg_log_ratio]
                    log_ratio_prev[bool_zero] = log_ratio_prev_check
                    bool_zero[bool_accept] = False

        bool_gamma = (S_min <= t_cur) & (t_cur <= S_max)

        if bool_gamma.sum() != 0:
            t_hat_temp = net.round_sigma(t_cur + gamma_vec * t_cur)[bool_gamma]
            x_hat_temp = x_cur[bool_gamma] + (t_hat_temp ** 2 - t_cur[bool_gamma] ** 2).sqrt()[:, None, None, None] * S_noise_vec[bool_gamma, None, None,None] * randn_like(x_cur[bool_gamma])

            t_hat = t_cur
            x_hat = x_cur

            t_hat[bool_gamma] = t_hat_temp
            x_hat[bool_gamma] = x_hat_temp
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        denoised = net(x_hat, t_hat, labels).to(torch.float64)
        per_sample_nfe += 1
        if mode == 'debug':
            nonlocal total_nfe
            total_nfe += len(denoised)
        d_cur = (x_hat - denoised) / t_hat[:, None, None, None]
        x_next = x_hat + (t_next - t_hat)[:, None, None, None] * d_cur

        # Apply 2nd order correction.
        bool_2nd = lst_idx < num_steps - 1
        if bool_2nd.sum() != 0:
            labels_ = labels[bool_2nd] if labels is not None else None
            denoised = net(x_next[bool_2nd], t_next[bool_2nd], labels_).to(torch.float64)
            per_sample_nfe[bool_2nd] += 1
            if mode == 'debug':
                total_nfe += len(denoised)
            d_prime = (x_next[bool_2nd] - denoised) / t_next[bool_2nd][:, None, None, None]
            x_next[bool_2nd] = x_hat[bool_2nd] + (t_next - t_hat)[bool_2nd][:, None, None, None] * (0.5 * d_cur[bool_2nd] + 0.5 * d_prime)

        lst_idx = lst_idx + 1

        if warmup:
            assert adaptive_pickle == 'None'
            log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_next, t_steps[lst_idx], net.img_resolution, time_min, time_max, labels, log_only=True).detach().cpu()
            for i in range(len(log_ratio)):
                lst_adaptive[lst_idx[i]].append(log_ratio[i].cpu())
            for i in range(len(log_ratio)):
                lst_adaptive2[lst_idx[i]].append(log_ratio[i].cpu() - log_ratio_prev[i].cpu())
            log_ratio_prev = log_ratio[:]
            return x_next, lst_idx, log_ratio_prev, per_sample_nfe

        if backsteps != 0.:
            bool_check = (lst_idx > min_backsteps) & (lst_idx <= max_backsteps)
            if mode == 'debug':
                save_lst_idx = copy.deepcopy(lst_idx)
            count = 0
            while bool_check.sum() != 0:
                x_check = x_next[bool_check]
                labels_ = labels[bool_check] if labels is not None else None
                log_ratio_prev_check = log_ratio_prev[bool_check]
                log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_check, t_steps[lst_idx][bool_check], net.img_resolution, time_min, time_max, labels_, log_only=True).detach().cpu()
                if count == 0:
                    bool_neg_log_ratio = log_ratio < adaptive2[lst_idx][bool_check] + torch.log(torch.rand_like(log_ratio) + 1e-7) + log_ratio_prev_check
                else:
                    bool_neg_log_ratio = log_ratio < adaptive[lst_idx][bool_check] + torch.log(torch.rand_like(log_ratio) + 1e-7)
                bool_reject = torch.arange(len(bool_check), device=bool_check.device)[bool_check][bool_neg_log_ratio]
                bool_accept = torch.arange(len(bool_check), device=bool_check.device)[bool_check][~bool_neg_log_ratio]

                if bool_neg_log_ratio.sum() != 0:
                    eps_rand = randn_like(x_check[bool_neg_log_ratio])
                    x_back = x_check[bool_neg_log_ratio] + (t_steps[lst_idx - backsteps][bool_check] ** 2 - t_steps[lst_idx][bool_check] ** 2).sqrt()[bool_neg_log_ratio][:, None, None, None] * eps_rand
                    x_next[bool_reject] = x_back
                    lst_idx[bool_reject] = lst_idx[bool_reject] - backsteps

                if mode == 'debug':
                    for i in range(len(save_lst_idx[bool_check & (lst_idx <= min_backsteps)])):
                        from_num = save_lst_idx[bool_check & (lst_idx <= min_backsteps)][i]
                        to_num = lst_idx[bool_check & (lst_idx <= min_backsteps)][i]
                        dict_nfe['dict_nfe'][f'{from_num}_{to_num}'] = dict_nfe['dict_nfe'].get(f'{from_num}_{to_num}', 0) + 1
                    if count != 0:
                        for i in range(len(save_lst_idx[bool_check][~bool_neg_log_ratio])):
                            from_num = save_lst_idx[bool_check][~bool_neg_log_ratio][i]
                            to_num = lst_idx[bool_check][~bool_neg_log_ratio][i]
                            dict_nfe['dict_nfe'][f'{from_num}_{to_num}'] = dict_nfe['dict_nfe'].get(f'{from_num}_{to_num}', 0) + 1
                count += 1

                log_ratio_prev_check[~bool_neg_log_ratio] = log_ratio[~bool_neg_log_ratio]
                log_ratio_prev[bool_check] = log_ratio_prev_check
                bool_check[lst_idx <= min_backsteps] = False
                bool_check[bool_accept] = False

        bool_check2 = per_sample_nfe + (num_steps * 2 - 1 - lst_idx.cpu() * 2) > max_iter
        if bool_check2.sum() != 0:
            pbar.update(bool_check2.sum().item())
            eps_rand = randn_like(x_next[bool_check2])
            x_next[bool_check2] = t_steps[0] * eps_rand
            lst_idx[bool_check2] = 0
            per_sample_nfe[bool_check2] = 0

        return x_next, lst_idx, log_ratio_prev, per_sample_nfe

    def save_img(images, index, save_type="npz", batch_size=100):
        ## Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        if save_type == "png":
            count = 0
            for image_np in images_np:
                image_path = os.path.join(outdir, f'{index*batch_size+count:06d}.png')
                count += 1
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        elif save_type == "npz":
            # r = np.random.randint(1000000)
            with tf.io.gfile.GFile(os.path.join(outdir, f"samples_{index}.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                if class_labels == None:
                    np.savez_compressed(io_buffer, samples=images_np)
                else:
                    np.savez_compressed(io_buffer, samples=images_np, label=class_labels.cpu().numpy())
                fout.write(io_buffer.getvalue())

            nrow = int(np.sqrt(images_np.shape[0]))
            image_grid = make_grid(torch.tensor(images_np).permute(0, 3, 1, 2) / 255., nrow, padding=2)
            with tf.io.gfile.GFile(os.path.join(outdir, f"sample_{index}.png"), "wb") as fout:
                save_image(image_grid, fout)

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    if mode == 'debug':
        import copy
        total_nfe = 0
        nfe_path = os.path.join(outdir, f'nfe_analysis.pickle')
        with open(nfe_path, 'rb') as f:
            dict_nfe = pickle.load(f)

    if adaptive_pickle == 'None':
        # Warmup
        lst_adaptive = [[] for i in range(len(t_steps))]
        lst_adaptive2 = [[] for i in range(len(t_steps))]
        x_next = latents.to(torch.float64) * t_steps[0]
        lst_idx = torch.zeros((latents.shape[0],), device=latents.device).long()
        log_ratio_prev = torch.zeros((latents.shape[0],))
        per_sample_nfe = torch.zeros((latents.shape[0],)).long()
        num_warm = 0
        while num_warm < iter_warmup:
            x_next, lst_idx, log_ratio_prev, per_sample_nfe = sampling_loop(x_next, lst_idx, log_ratio_prev, per_sample_nfe, class_labels, warmup=True)
            bool_fin = lst_idx == num_steps
            if bool_fin.sum() > 0:
                x_next[bool_fin] = torch.randn_like(x_next[bool_fin]).to(torch.float64) * t_steps[0]
                lst_idx[bool_fin] = torch.zeros_like(lst_idx[bool_fin]).long()
                if (class_labels is not None) & (class_idx is None):
                    class_labels[bool_fin] = torch.eye(net.label_dim, device=class_labels.device)[torch.randint(net.label_dim, size=[bool_fin.sum()], device=class_labels.device)]
                num_warm += 1
        lst_adaptive = [torch.stack(lst_adaptive[i]) for i in range(0, len(t_steps))]
        lst_adaptive2 = [torch.zeros(len(x_next)*iter_warmup)] + [torch.stack(lst_adaptive2[i]) for i in range(1, len(t_steps))]
        adaptive_path = os.path.join(outdir, f'adaptive.pickle')
        adaptive2_path = os.path.join(outdir, f'adaptive2.pickle')
        with open(adaptive_path, 'wb') as f:
            pickle.dump(lst_adaptive, f)
        with open(adaptive2_path, 'wb') as f:
            pickle.dump(lst_adaptive2, f)
    else:
        with open(adaptive_pickle, 'rb') as f:
            lst_adaptive = pickle.load(f)
        with open(adaptive_pickle2, 'rb') as f:
            lst_adaptive2 = pickle.load(f)
    adaptive = torch.zeros_like(t_steps).cpu()
    for k in range(len(t_steps)):
        if no_zero:
            adaptive[k] = torch.quantile(lst_adaptive[k], rej_percentile).item()
        else:
            adaptive[k] = max(0., torch.quantile(lst_adaptive[k], rej_percentile).item())
    print(adaptive)
    adaptive2 = torch.zeros_like(t_steps).cpu()
    for k in range(len(t_steps)):
        if no_zero:
            adaptive2[k] = torch.quantile(lst_adaptive2[k], rej_percentile).item()
        else:
            adaptive2[k] = max(0., torch.quantile(lst_adaptive2[k], rej_percentile).item())
    print(adaptive2)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    lst_idx = torch.zeros((latents.shape[0],), device=latents.device).long()
    log_ratio_prev = torch.zeros((latents.shape[0],))
    per_sample_nfe = torch.zeros((latents.shape[0],)).long()

    pbar = tqdm(desc='Number of re-init. samples')

    x_fin = torch.zeros_like(x_next)
    tot_per_sample_nfe = []
    total_samples = 0
    index = 0
    current_time = time.time()
    while total_samples <= num_samples:
        x_next, lst_idx, log_ratio_prev, per_sample_nfe = sampling_loop(x_next, lst_idx, log_ratio_prev, per_sample_nfe, class_labels)
        bool_fin = lst_idx == num_steps
        if bool_fin.sum() > 0:
            if (batch_size - total_samples % batch_size) <= bool_fin.sum():
                x_fin[total_samples % batch_size:] = x_next[bool_fin][:batch_size - total_samples % batch_size]
                r = np.random.randint(1000000)
                save_img(x_fin, index=r)
                index += 1
                x_fin = torch.zeros_like(x_next)

                x_fin[:bool_fin.sum() - batch_size + total_samples % batch_size] = x_next[bool_fin][batch_size - total_samples % batch_size:]
                total_samples += bool_fin.sum()
            else:
                x_fin[total_samples % batch_size:total_samples % batch_size + bool_fin.sum()] = x_next[bool_fin]
                total_samples += bool_fin.sum()
            x_next[bool_fin] = torch.randn_like(x_next[bool_fin]).to(torch.float64) * t_steps[0]
            lst_idx[bool_fin] = torch.zeros_like(lst_idx[bool_fin]).long()
            log_ratio_prev[bool_fin] = torch.zeros_like(log_ratio_prev[bool_fin])

            tot_per_sample_nfe += per_sample_nfe[bool_fin].tolist()
            per_sample_nfe[bool_fin] = torch.zeros_like(per_sample_nfe[bool_fin]).long()

            if (class_labels is not None) & (class_idx is None):
                class_labels[bool_fin] = torch.eye(net.label_dim, device=class_labels.device)[torch.randint(net.label_dim, size=[bool_fin.sum()], device=class_labels.device)]

            if mode == 'debug':
                dict_nfe['total_nfe'] = total_nfe
                dict_nfe['total_samples'] = total_samples.item()
                dict_nfe['tot_per_sample_nfe'] = tot_per_sample_nfe
                with open(nfe_path, 'wb') as f:
                    pickle.dump(dict_nfe, f)
    print(time.time()-current_time)

    if mode == 'debug':
        dict_nfe['total_nfe'] = dict_nfe.get('total_nfe', 0) + total_nfe
        dict_nfe['total_samples'] = dict_nfe.get('total_samples', 0) + num_samples
        dict_nfe['tot_per_sample_nfe'] = tot_per_sample_nfe

        with open(nfe_path, 'wb') as f:
            pickle.dump(dict_nfe, f)
    # return x_next

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'batch_size',     help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=100, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

#---------------------------------------------------------------------------- Options for Discriminator-Guidance
## Sampling configureation
@click.option('--do_seed',                 help='Applying manual seed or not', metavar='INT',                       type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--seed',                    help='Seed number',                 metavar='INT',                       type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--num_samples',             help='Num samples',                 metavar='INT',                       type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--save_type',               help='png or npz',                  metavar='png|npz',                   type=click.Choice(['png', 'npz']), default='npz')
@click.option('--device',                  help='Device', metavar='STR',                                            type=str, default='cuda:0')

## DG configuration
@click.option('--time_min',                help='Minimum time[0,1] to apply DG', metavar='FLOAT',                   type=click.FloatRange(min=0., max=1.), default=0.01, show_default=True)
@click.option('--time_max',                help='Maximum time[0,1] to apply DG', metavar='FLOAT',                   type=click.FloatRange(min=0., max=1.), default=1.0, show_default=True)
@click.option('--boosting',                help='If true, dg scale up low log ratio samples', metavar='INT',        type=click.IntRange(min=0), default=0, show_default=True)

## Discriminator checkpoint
@click.option('--pretrained_classifier_ckpt',help='Path of ADM classifier(latent extractor)',  metavar='STR',       type=str, default='checkpoints/ADM_classifier/32x32_classifier.pt', show_default=True)
@click.option('--discriminator_ckpt',      help='Path of discriminator',  metavar='STR',                            type=str, default='checkpoints/discriminator/cifar_uncond/discriminator_60.pt', show_default=True)

## DiffRS configuration
@click.option('--rej_percentile',          help='Rejection percentile gamma',       metavar='FLOAT',                type=float, default=0., show_default=True)
@click.option('--cond',                    help='Is it conditional discriminator?', metavar='INT',                  type=click.IntRange(min=0, max=1), default=0, show_default=True)
@click.option('--backsteps',               help='backsteps', metavar='INT',                                         type=click.IntRange(min=0), default=1, show_default=True)
@click.option('--min_backsteps',           help='min_backsteps', metavar='INT',                                     type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--max_backsteps',           help='max_backsteps', metavar='INT',                                     type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--mode',                    help='Mode', metavar='STR',                                              type=str, default='default')
@click.option('--adaptive_pickle',         help='Path of adaptive',  metavar='STR',                                 type=str, default='None', show_default=True)
@click.option('--adaptive_pickle2',        help='Path of adaptive2',  metavar='STR',                                type=str, default='None', show_default=True)
@click.option('--iter_warmup',             help='iteration of warmup', metavar='INT',                               type=click.IntRange(min=0), default=10, show_default=True)
@click.option('--max_iter',                help='max_iter', metavar='INT',                                          type=click.IntRange(min=0), default=999999, show_default=True)
@click.option('--no_zero',                 help='Use zero minimum for M', metavar='INT',                            type=click.IntRange(min=0, max=1), default=0, show_default=True)

def main(boosting, time_min, time_max, rej_percentile, cond, pretrained_classifier_ckpt, discriminator_ckpt, save_type, batch_size, do_seed, seed, num_samples, network_pkl, outdir, class_idx, device, backsteps, min_backsteps, max_backsteps, mode, adaptive_pickle, adaptive_pickle2, iter_warmup, max_iter, no_zero, **sampler_kwargs):
    ## Load pretrained score network.
    print(f'Loading network from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)

    ## Load discriminator
    if 'ffhq' in network_pkl:
        depth = 4
    else:
        depth = 2
    discriminator = classifier_lib.get_discriminator(pretrained_classifier_ckpt, discriminator_ckpt,
                                                     net.label_dim and cond, net.img_resolution, device,
                                                     depth=depth, enable_grad=False)
    print(discriminator)
    vpsde = classifier_lib.vpsde()

    ## Loop over batches.
    print(f'Generating {num_samples} images to "{outdir}"...')
    os.makedirs(outdir, exist_ok=True)

    if mode == 'debug':
        dict_nfe = {'dict_nfe': {}, 'total_nfe': 0, 'total_samples': 0}
        nfe_path = os.path.join(outdir, f'nfe_analysis.pickle')
        with open(nfe_path, 'wb') as f:
            pickle.dump(dict_nfe, f)

    ## Set seed
    if do_seed:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    ## Pick latents and labels.
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
    if class_idx is not None:
        class_labels[:, :] = 0
        class_labels[:, class_idx] = 1

    ## Generate images.
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    diffrs_sampler(boosting, time_min, time_max, vpsde, rej_percentile, discriminator,
                   net, latents, class_labels, randn_like=torch.randn_like, backsteps=backsteps,
                   min_backsteps=min_backsteps, max_backsteps=max_backsteps, mode=mode, outdir=outdir,
                   adaptive_pickle=adaptive_pickle, adaptive_pickle2=adaptive_pickle2, class_idx=class_idx,
                   batch_size=batch_size, num_samples=num_samples, iter_warmup=iter_warmup, max_iter=max_iter,
                   no_zero=no_zero, **sampler_kwargs)

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
