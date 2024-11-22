
import os
import sys
import copy
import hydra
import pytorch_lightning as pl
import torch
import numpy as np
import scipy.io

from datasets.latent import MyDatasetInitial, MyDatasetInversion,MyDatasetResults
from models.callbacks import ImageWriter
from models.diffusion import DDPMv2, DDPMWrapper, SuperResModel
from models.vae import VAE
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import configure_device


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(version_base=None, config_path=r"your_path\main\configs")
def sample_cond(config):
    # Seed and setup
    config_ddpm = config.dataset.ddpm
    config_vae = config.dataset.vae
    seed_everything(config_ddpm.evaluation.seed, workers=True)

    batch_size = config_ddpm.evaluation.batch_size
    n_steps = config_ddpm.evaluation.n_steps
    n_samples = config_ddpm.evaluation.n_samples
    image_size = config_ddpm.data.image_size
    ddpm_latent_path = config_ddpm.data.ddpm_latent_path
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # Load pretrained VAE
    vae = VAE.load_from_checkpoint(
        config_vae.evaluation.chkpt_path,
        input_res=image_size,
    )
    vae.eval()

    # Load pretrained wrapper
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)  # "32,16,8"
    dim_mults = __parse_str(config_ddpm.model.dim_mults)  # "1,2,2,3,4"
    decoder = SuperResModel(
        in_channels=config_ddpm.data.n_channels,  # 1
        model_channels=config_ddpm.model.dim,  # 128
        out_channels=1,
        num_res_blocks=config_ddpm.model.n_residual,  # 2
        attention_resolutions=attn_resolutions,  # "32,16,8"
        channel_mult=dim_mults,  # "1,2,2,3,4"
        use_checkpoint=False,
        dropout=config_ddpm.model.dropout,  # 0.0
        num_heads=config_ddpm.model.n_heads,  # 1
        z_dim=config_ddpm.evaluation.z_dim,  # 100
        use_scale_shift_norm=config_ddpm.evaluation.z_cond,
        use_z=config_ddpm.evaluation.z_cond,
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    ddpm_cls = DDPMv2
    ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )

    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config_ddpm.evaluation.chkpt_path,
        network=ddpm,
        vae=vae,
        conditional=True,
        pred_steps=n_steps,
        eval_mode="sample",
        resample_strategy=config_ddpm.evaluation.resample_strategy,  # "spaced"
        skip_strategy=config_ddpm.evaluation.skip_strategy,  # "uniform"
        sample_method=config_ddpm.evaluation.sample_method,  # "ddpm"
        data_norm=config_ddpm.data.norm,  # False
        temp=config_ddpm.evaluation.temp,
        guidance_weight=config_ddpm.evaluation.guidance_weight,
        z_cond=config_ddpm.evaluation.z_cond,  # True
        strict=False,
        ddpm_latents=ddpm_latents,
    )

    if decision == 0:
        z_dataset = MyDatasetInitial(
            (n_samples, config_vae.model.z_dim, 1, 1),
            (n_samples, 1, image_size, image_size),
            share_ddpm_latent=True if ddpm_latent_path != "" else False,
        )
        idx_list = range(len(z_dataset))
        z_vae_column = torch.stack([z_dataset[idx][1] for idx in idx_list], dim=1)
        z_vae_column = z_vae_column.squeeze().cpu().numpy()
        z_vae_Ne = np.zeros((config_vae.model.z_dim, n_samples))

        for i in range(z_vae_column.shape[1]):
            z_vae_Ne[:, i] = z_vae_column[:, i]
        file_path1 = r'your_path\z_vae_100.mat'
        data1 = {"z_vae_100": z_vae_Ne}
        scipy.io.savemat(file_path1, data1)

        z_ddpm_array = z_dataset[0][0]
        z_ddpm_array = z_ddpm_array.squeeze().cpu().numpy()
        z_ddpm_array = z_ddpm_array.astype(np.double)
        file_path2 = r"your_path\z_ddpm_128_128.mat"
        data2 = {"z_ddpm": z_ddpm_array}
        scipy.io.savemat(file_path2, data2)

    elif decision == 1:
        data_dir = r"your_path"
        z_dataset = MyDatasetInversion(data_dir, n_samples)

    elif decision == 2:
        data_dir = r"your_path"
        z_dataset = MyDatasetResults(data_dir, config_ddpm.evaluation.iter)

        # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = config_ddpm.evaluation.device
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["gpus"] = devs

        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    val_loader = DataLoader(
        z_dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        num_workers=config_ddpm.evaluation.workers,
        **loader_kws,
    )

    # Predict trainer
    if decision == 0:
        save_path = os.path.join(config_ddpm.evaluation.save_path, "0")
    elif decision == 1:
        save_path = os.path.join(config_ddpm.evaluation.save_path, str(m))
    elif decision == 2:
        save_path = r"your_path\results"

    write_callback = ImageWriter(
        save_path,
        "batch",
        n_steps=n_steps,
        eval_mode="sample",
        conditional=True,
        sample_prefix=config_ddpm.evaluation.sample_prefix,
        save_vae=config_ddpm.evaluation.save_vae,
        is_norm=True,
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config_ddpm.evaluation.save_path
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, val_loader)


if __name__ == "__main__":
    import time
    import numpy as np
    import matlab.engine
    import scipy.io
    import os
    import datetime
    import time
    import torch

    import argparse
    import torch.nn.functional as F

    import torch
    import torch.nn as nn
    import sys
    import os
    # from torchsummary import summary

    import matplotlib.pyplot as plt
    import random
    import scipy.io
    import numpy as np
    from scipy.io import savemat
    eng = matlab.engine.start_matlab('MATLAB_R2021a')

    class _DenseLayer(nn.Sequential):

        def __init__(self, in_features, growth_rate, drop_rate=0, bn_size=4, bottleneck=False):
            super(_DenseLayer, self).__init__()
            if bottleneck and in_features > bn_size * growth_rate:
                self.add_module('norm1', nn.BatchNorm2d(in_features))
                self.add_module('relu1', nn.ReLU(inplace=True))
                self.add_module('conv1', nn.Conv2d(in_features, bn_size *
                                                   growth_rate, kernel_size=1, stride=1, bias=False))
                self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
                self.add_module('relu2', nn.ReLU(inplace=True))
                self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                                   kernel_size=3, stride=1, padding=1, bias=False))
            else:
                self.add_module('norm1', nn.BatchNorm2d(in_features))
                self.add_module('relu1', nn.ReLU(inplace=True))
                self.add_module('conv1',
                                nn.Conv2d(in_features, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
            self.drop_rate = drop_rate

        def forward(self, x):
            y = super(_DenseLayer, self).forward(x)
            if self.drop_rate > 0:
                y = F.dropout2d(y, p=self.drop_rate, training=self.training)
            z = torch.cat([x, y], 1)
            return z

    class _DenseBlock(nn.Sequential):
        def __init__(self, num_layers, in_features, growth_rate, drop_rate, bn_size=4, bottleneck=False):
            super(_DenseBlock, self).__init__()

            for i in range(num_layers):
                layer = _DenseLayer(in_features + i * growth_rate, growth_rate, drop_rate=drop_rate, bn_size=bn_size,
                                    bottleneck=bottleneck)
                self.add_module('denselayer%d' % (i + 1), layer)

    class _Transition(nn.Sequential):

        def __init__(self, in_features, out_features, encoding=True, drop_rate=0., last=False, out_channels=3,
                     outsize_even=True):
            super(_Transition, self).__init__()
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            if encoding:
                self.add_module('conv1',
                                nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                                   kernel_size=3, stride=2,
                                                   padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                # decoder设置
                if last:
                    ks = 6 if outsize_even else 3
                    out_convt = nn.ConvTranspose2d(out_features, out_channels, kernel_size=ks, stride=2, padding=1,
                                                   bias=False)
                else:
                    out_convt = nn.ConvTranspose2d(out_features, out_features, kernel_size=3, stride=2, padding=1,
                                                   output_padding=0, bias=False)

                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1',
                                nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False))

                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                self.add_module('convT2', out_convt)
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))


    class DenseED(nn.Module):
        def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
                     num_init_features=64, bn_size=4, drop_rate=0, outsize_even=True,
                     bottleneck=False):
            super(DenseED, self).__init__()

            if len(blocks) > 1 and len(blocks) % 2 == 0:
                ValueError('length of blocks must be an odd number, but got {}'.format(len(blocks)))

            enc_block_layers = blocks[: len(blocks) // 2]
            dec_block_layers = blocks[len(blocks) // 2:]
            self.features = nn.Sequential()
            self.features.add_module('in_conv',
                                     nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3,
                                               bias=False))
            num_features = num_init_features
            for i, num_layers in enumerate(enc_block_layers):
                block = _DenseBlock(num_layers=num_layers,
                                    in_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate,
                                    drop_rate=drop_rate, bottleneck=bottleneck)
                self.features.add_module('encblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate

                trans = _Transition(in_features=num_features,
                                    out_features=num_features // 2,
                                    encoding=True, drop_rate=drop_rate)
                self.features.add_module('down%d' % (i + 1), trans)
                num_features = num_features // 2

            for i, num_layers in enumerate(dec_block_layers):
                block = _DenseBlock(num_layers=num_layers,
                                    in_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate,
                                    drop_rate=drop_rate, bottleneck=bottleneck)
                self.features.add_module('decblock%d' % (i + 1), block)
                num_features += num_layers * growth_rate

                last_layer = True if i == len(dec_block_layers) - 1 else False

                trans = _Transition(in_features=num_features,
                                    out_features=num_features // 2,
                                    encoding=False, drop_rate=drop_rate,
                                    last=last_layer, out_channels=out_channels,
                                    outsize_even=outsize_even)
                self.features.add_module('up%d' % (i + 1), trans)
                num_features = num_features // 2

        def forward(self, x):
            y = self.features(x)
            y[:, 0] = F.softplus(y[:, 0].clone(), beta=1)

            return y

        def _num_parameters_convlayers(self):
            n_params, n_conv_layers = 0, 0
            for name, param in self.named_parameters():
                if 'conv' in name:
                    n_conv_layers += 1
                n_params += param.numel()
            return n_params, n_conv_layers

        def _count_parameters(self):
            n_params = 0
            for name, param in self.named_parameters():
                print(name)
                print(param.size())
                print(param.numel())
                n_params += param.numel()
                print('num of parameters so far: {}'.format(n_params))

        def reset_parameters(self, verbose=False):
            for module in self.modules():
                if isinstance(module, self.__class__):
                    continue
                if 'reset_parameters' in dir(module):
                    if callable(module.reset_parameters):
                        module.reset_parameters()
                        if verbose:
                            print("Reset parameters in {}".format(module))


    parser = argparse.ArgumentParser(description='Dnense Encoder-Decoder Convolutional Network')
    parser.add_argument('--exp-name', type=str, default='AR-Net', help='experiment name')
    parser.add_argument('--blocks', type=list, default=(5, 10, 5),
                        help='list of number of layers in each block in decoding net')
    parser.add_argument('--growth-rate', type=int, default=40, help='output of each conv')
    parser.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
    parser.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
    parser.add_argument('--bottleneck', action='store_true', default=False,
                        help='enable bottleneck in the dense blocks')
    parser.add_argument('--init-features', type=int, default=48,
                        help='# initial features after the first conv layer')
    args = parser.parse_args(args=[])

    model = DenseED(3, 2, blocks=args.blocks, growth_rate=args.growth_rate,
                    drop_rate=args.drop_rate, bn_size=args.bn_size,
                    num_init_features=args.init_features, bottleneck=args.bottleneck).cuda()

    model_dir = r""
    model.load_state_dict(torch.load(model_dir + '/model_epoch{}.pth'.format(200)))
    print('Loaded model')

#-------------------------------------------------------------------------------------------------------
    def get_simv(y_pred):
        scipy.io.savemat('/y_pred.mat', dict(y_pred=y_pred))
        eng.interp_matlab(nargout=0)
        y_sim = np.loadtxt("y_sim.dat")
        return y_sim

    def normalize(obj):
        B, C, H, W = obj.shape
        for i in range(1):
            channel_val = obj[:, i, :, :].view(B, -1)
            channel_val -= channel_val.min(1, keepdim=True)[0]
            channel_val /= (channel_val.max(1, keepdim=True)[0] - channel_val.min(1, keepdim=True)[0])
            channel_val = channel_val.view(B, H, W)
            obj[:, i, :, :] = channel_val
        return obj

    def Postprocess(file_path):
        img_data = torch.tensor(np.load(file_path))
        img_data = img_data.view(1, 1, 128, 128)
        img_data = normalize(img_data)
        b1 = torch.full(img_data[0][0].shape, 10.0)
        b2 = torch.full(img_data[0][0].shape, 0.4)
        img_data = torch.where(img_data >= 0.5, b1, b2)
        img_data = img_data.detach().cpu().numpy()
        img_data = np.squeeze(img_data, axis=0)
        return img_data

    def gene_ss_and_save(range_vals, N=1, file_path=''):
        if N is None:
            N = 1
        Npar = range_vals.shape[0]
        x = np.empty((Npar, N))
        for i in range(N):
            x[:, i] = range_vals[:, 0] + (range_vals[:, 1] - range_vals[:, 0]) * np.random.rand(Npar)
        savemat(file_path, {'ss_Ne': x})

#=============================================================================================================
    start_time1 = time.time()
    Ne = '???'
    ngx = '???'
    ngy = '???'
    Nt = '???'
    Nobs = '???'

    decision = 0
    sample_cond()

    mat_data = scipy.io.loadmat('/z_vae_100.mat')
    z_vae_100 = mat_data['z_vae_100']
    mat_data2 = scipy.io.loadmat('/ss_200.mat')
    ss_Ne = mat_data2['ss_Ne']
    x1 = np.vstack((z_vae_100, ss_Ne))
    scipy.io.savemat('/x1.mat', {'x1': x1})
    y1 = np.zeros((Nobs, Ne))

    model.eval()
    folder_path = r'\images'
    for i in range(0, Ne):
        k = i % 8
        batch_index = i//8
        file_name = f'???.npy'
        file_path = os.path.join(folder_path, file_name)
        output = Postprocess(file_path)
        output = np.squeeze(output)
        output = np.log(output)

        Sx_id = 11
        source = np.full((Nt, 128, 128), 0.0)
        for j in range(Nt):
            for p in range(33, 93):
                Sy_id = p
                source[j, Sy_id, Sx_id] = ss_Ne[j, i]

        x = np.full((1, 3, 128, 128), 0.0)
        y = np.full((Nt, 2, 128, 128), 0.0)
        y_i_1 = np.full((128, 128), 0.0)
        for q in range(Nt):
            x[0, 0, :, :] = output
            x[0, 1, :, :] = source[q]
            x[0, 2, :, :] = y_i_1
            x_tensor = (torch.FloatTensor(x)).cuda()
            with torch.no_grad():
                y_hat = model(x_tensor)
            y_hat = y_hat.data.cpu().numpy()
            y[q] = y_hat
            y_i_1 = y_hat[0, 0, :, :]

        y_pred = np.full((Nt + 1, 128, 128), 0.0)
        y[:, 0] *= 1000
        y_pred[:Nt] = y[:, 0]

        column_to_modify = y[0, 1]
        column_to_modify[column_to_modify > 11] = 11
        column_to_modify[column_to_modify < 10] = 10
        y[0, 1] = column_to_modify

        y_pred[Nt] = y[0, 1]

        y1[:, i] = get_simv(y_pred)

    scipy.io.savemat('/y1.mat', {'conc_head_Ne': y1})
    end_time1 = time.time()
    execution_time1 = end_time1 - start_time1
    print("Execution Time of Inversion:", execution_time1)
# -----------------------------------------------------------------------------------------------------
# 开始反演
    x_para = scipy.io.loadmat('/x1.mat')
    xf = x_para['x1']
    obs_model = scipy.io.loadmat('/y1.mat')
    yf = obs_model['conc_head_Ne']

    scipy.io.savemat('/xf.mat', {"xf": xf})
    scipy.io.savemat('/yf.mat', {"yf": yf})
    xall = xf
    yall = yf

    start_time2 = time.time()
    m = 0
    N_iter = '???'
    ngx = '???'
    ngy = '???'
    Nobs = '???'
    Nt = '???'
    Ne = '???'
    ya = np.zeros((Nobs, Ne))
    model.eval()

    eng.cd('', nargout=0)

    for i in range(0, N_iter):
        m += 1
        eng.ilues(nargout=0)
        xa = scipy.io.loadmat('/xa.mat')
        xa = xa['xa']

        print('iter=', i + 1)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_path = r"\timestamp.txt"
        with open(file_path, 'a') as file:
            content = f"iter {i+1} done: {timestamp}\n"
            file.write(content)

        z_a = xa[0:256, :]
        ss_a = xa[256:, :]

        file_path4 = r'\z_a.mat'
        data4 = {"z_a": z_a}
        scipy.io.savemat(file_path4, data4)

        file_path5 = r'\ss_a.mat'
        data5 = {"ss_a": ss_a}
        scipy.io.savemat(file_path5, data5)

        decision = 1
        sample_cond()

        folder_path = rf'\images'
        for j in range(0, Ne):
            k = j % 8
            batch_index = j // 8
            file_name = f'{batch_index}_{k}.npy'
            file_path = os.path.join(folder_path, file_name)
            output = Postprocess(file_path)
            output = np.squeeze(output)
            output = np.log(output)
            Sx_id = 11
            source = np.full((Nt, 128, 128), 0.0)
            for j in range(Nt):
                for p in range(33, 93):
                    Sy_id = p
                    source[j, Sy_id, Sx_id] = ss_a[j, i]

            x = np.full((1, 3, 128, 128),0.0)
            y = np.full((Nt, 2, 128, 128), 0.0)
            y_i_1 = np.full((128, 128), 0.0)
            for q in range(Nt):
                x[0, 0, :, :] = output  # hydraulic conductivity
                x[0, 1, :, :] = source[q]  # source rate
                x[0, 2, :, :] = y_i_1
                x_tensor = (torch.FloatTensor(x)).cuda()
                with torch.no_grad():
                    y_hat = model(x_tensor)
                y_hat = y_hat.data.cpu().numpy()
                y[q] = y_hat
                y_i_1 = y_hat[0, 0, :, :]  # the updated (i-1)^th predicted concentration field

            y_pred = np.full((Nt + 1, 128, 128), 0.0)
            y[:, 0] *= 1000
            y_pred[:Nt] = y[:, 0]   # the concentration fields at Nt time instances

            column_to_modify = y[0, 1]
            column_to_modify[column_to_modify > 11] = 11
            column_to_modify[column_to_modify < 10] = 10
            y[0, 1] = column_to_modify

            y_pred[Nt] = y[0, 1]  # the hydraulic head field

            ya[:, i] = get_simv(y_pred)

        scipy.io.savemat('/ya.mat', {"ya": ya})

        eng.update_samples(nargout=0)

        xa = scipy.io.loadmat('/xa.mat')  # The updated inputs
        ya = scipy.io.loadmat('/ya.mat')  # The updated outputs
        xa = xa['xa']
        ya = ya['ya']

        xall = np.concatenate((xall, xa), axis=1)
        yall = np.concatenate((yall, ya), axis=1)

    scipy.io.savemat('/results.mat', {"xall": xall, "yall": yall})  # save results

    end_time2 = time.time()
    execution_time2 = end_time2 - start_time2
    print("Execution Time of Inversion:", execution_time2)
