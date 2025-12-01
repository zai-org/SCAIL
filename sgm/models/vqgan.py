from packaging import version

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..modules.autoencoding.vqvae.vqvae_blocks import Encoder, Decoder
from ..modules.autoencoding.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ..modules.autoencoding.vqvae.movq_modules import MOVQDecoder

from ..util import instantiate_from_config
from ..modules.ema import LitEma

from einops import rearrange
import numpy as np


class MOVQ(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 learning_rate=None,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ema_decay=None
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = MOVQDecoder(zq_ch=embed_dim, **ddconfig)
        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ema_decay is not None:
            self.use_ema = True
            print('use_ema = True')
            self.ema_encoder = LitEma(self.encoder, ema_decay)
            self.ema_decoder = LitEma(self.decoder, ema_decay)
            self.ema_quantize = LitEma(self.quantize, ema_decay) 
            self.ema_quant_conv = LitEma(self.quant_conv, ema_decay) 
            self.ema_post_quant_conv = LitEma(self.post_quant_conv, ema_decay) 
            
        else:
            self.use_ema = False
        
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False
            
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if 'state_dict' in sd:
            sd = sd['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def encode_h(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, h

    def decode(self, quant):
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def decode_code(self, code_b):
        batch_size = code_b.shape[0]
        quant = self.quantize.embedding(code_b.flatten())
        grid_size = int((quant.shape[0] // batch_size)**0.5)
        quant = quant.view((1, 32, 32, 4))
        quant = rearrange(quant, 'b h w c -> b c h w').contiguous()
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
    
    def forward_ema(self, x):
        h = self.ema_encoder(x)
        h = self.ema_quant_conv(h)
        quant, emb_loss, info = self.ema_quantize(h)
        quant2 = self.ema_post_quant_conv(quant)
        dec = self.ema_decoder(quant2, quant)
        return dec, emb_loss
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_encoder(self.encoder)
            self.ema_decoder(self.decoder)
            self.ema_quantize(self.quantize)
            self.ema_quant_conv(self.quant_conv)
            self.ema_post_quant_conv(self.post_quant_conv)

    def training_step(self, batch, batch_idx):
        # rank = torch.distributed.get_rank()
        # with open(f'log_rank_{rank}', 'a') as f:
        #     f.write(f'On batch {batch_idx} batch mean {batch.mean()}\n')
        opts = self.optimizers()
        if not isinstance(opts, list):
            # Non-adversarial case
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        opt = opts[optimizer_idx]
        opt.zero_grad()

        with opt.toggle_model():
            x = batch
            xrec, qloss = self(x)

            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                # return aeloss
                loss = aeloss

            if optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                # return discloss
                loss = discloss
            self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm='value')
        opt.step()

    def validation_step(self, batch, batch_idx):
        x = batch
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        # x = self.get_input(batch, self.image_key)
        x = batch
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 learning_rate,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ema_decay=None
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ema_decay is not None:
            self.use_ema = True
            print('use_ema = True')
            self.ema_encoder = LitEma(self.encoder, ema_decay)
            self.ema_decoder = LitEma(self.decoder, ema_decay)
            self.ema_quantize = LitEma(self.quantize, ema_decay) 
            self.ema_quant_conv = LitEma(self.quant_conv, ema_decay) 
            self.ema_post_quant_conv = LitEma(self.post_quant_conv, ema_decay) 
            
        else:
            self.use_ema = False
            
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
    
    def forward_ema(self, x):
        h = self.ema_encoder(x)
        h = self.ema_quant_conv(h)
        quant, emb_loss, info = self.ema_quantize(h)
        quant2 = self.ema_post_quant_conv(quant)
        dec = self.ema_decoder(quant2, quant)
        return dec, emb_loss
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_encoder(self.encoder)
            self.ema_decoder(self.decoder)
            self.ema_quantize(self.quantize)
            self.ema_quant_conv(self.quant_conv)
            self.ema_post_quant_conv(self.post_quant_conv)
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = batch
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 temperature_scheduler_config,
                 n_embed=256,
                 embed_dim=8192,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        print(z_channels)
        super().__init__(learning_rate=0.0001,
                         ddconfig=ddconfig,
                         lossconfig=lossconfig,
                         n_embed=n_embed,
                         embed_dim=embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log
    

class MOVQFix(MOVQ):
    def __init__(self, train_encoder=False, train_decoder=False, **kwargs):
        super().__init__(**kwargs)
        self.train_encoder = train_encoder
        self.train_decoder = train_decoder
        if not train_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if not train_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False
        self.ref_movq = MOVQ(**kwargs)
        for p in self.ref_movq.parameters():
            p.requires_grad = False

    def encode_features(self, x):
        h, output_features = self.encoder.forward_with_features_output(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, h, output_features
    
    def decode_features(self, quant):
        quant2 = self.post_quant_conv(quant)
        dec, output_features = self.decoder.forward_with_features_output(quant2, quant)
        return dec, output_features

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.train_encoder:
            opt = torch.optim.Adam(list(self.encoder.parameters())+
                                   list(self.quant_conv.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        elif self.train_decoder:
            opt = torch.optim.Adam(list(self.decoder.parameters())+
                                   list(self.post_quant_conv.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        else:
            raise NotImplementedError
        
        return [opt], []
    
    def print_feat_stats(self, output_features):
        for key in output_features:
            print(key)
            print(f'Mean: {output_features[key].mean().item()}, Std: {output_features[key].std().item()}')
            print(f'Min: {output_features[key].max().item()}, Max: {output_features[key].min().item()}')

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        with opt.toggle_model():
            x = batch
            with torch.no_grad():
                ref_quant, ref_emb_loss, ref_info, ref_h = self.ref_movq.encode_h(x)
                ref_dec = self.ref_movq.decode(ref_quant)
            if self.train_encoder:
                quant, emb_loss, info, h, enc_out_features = self.encode_features(x)
                enc_loss = (h - ref_h).abs().mean()
                self.log("train/enc_loss", enc_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                feat_loss = 0
                for key in enc_out_features:
                    feat_loss += 1e-4 * F.relu(torch.abs(enc_out_features[key]) - 100).mean()
                self.log("train/feat_loss", feat_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict({}, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                loss = enc_loss + feat_loss
            elif self.train_decoder:
                dec, dec_out_features = self.decode_features(ref_quant)
                dec_loss = (dec - ref_dec).abs().mean()
                self.log("train/dec_loss", dec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                feat_loss = 0
                for key in dec_out_features:
                    feat_loss += 1e-4 * F.relu(torch.abs(dec_out_features[key]) - 100).mean()
                self.log("train/feat_loss", feat_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict({}, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                loss = dec_loss + feat_loss
            else:
                raise NotImplementedError
            self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm='value')
        opt.step()

class MOVQFixBoth(MOVQ):
    def __init__(self, train_encoder=False, train_decoder=False, **kwargs):
        super().__init__(**kwargs)
        self.ref_movq = MOVQ(**kwargs)
        for p in self.ref_movq.parameters():
            p.requires_grad = False

    def encode_features(self, x):
        h, output_features = self.encoder.forward_with_features_output(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, h, output_features
    
    def decode_features(self, quant):
        quant2 = self.post_quant_conv(quant)
        dec, output_features = self.decoder.forward_with_features_output(quant2, quant)
        return dec, output_features

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_enc = torch.optim.Adam(list(self.encoder.parameters())+
                                   list(self.quant_conv.parameters()),
                                   lr=lr, betas=(0.5, 0.9))
        opt_dec = torch.optim.Adam(list(self.decoder.parameters())+
                                   list(self.post_quant_conv.parameters()),
                                   lr=lr, betas=(0.5, 0.9))
        
        return [opt_enc, opt_dec], []
    
    def print_feat_stats(self, output_features):
        for key in output_features:
            print(key)
            print(f'Mean: {output_features[key].mean().item()}, Std: {output_features[key].std().item()}')
            print(f'Min: {output_features[key].max().item()}, Max: {output_features[key].min().item()}')

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        opt_enc, opt_dec = self.optimizers()
        opt_enc.zero_grad()
        opt_dec.zero_grad()

        x = batch
        with torch.no_grad():
            ref_quant, ref_emb_loss, ref_info, ref_h = self.ref_movq.encode_h(x)
            ref_dec = self.ref_movq.decode(ref_quant)

        with opt_enc.toggle_model():
            # encoder training
            quant, emb_loss, info, h, enc_out_features = self.encode_features(x)
            enc_recon_loss = (h - ref_h).abs().mean()
            self.log("train/enc_recon_loss", enc_recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            enc_feat_loss = 0
            for key in enc_out_features:
                enc_feat_loss += 1e-4 * F.relu(torch.abs(enc_out_features[key]) - 100).mean()
            self.log("train/enc_feat_loss", enc_feat_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            enc_loss = enc_recon_loss + enc_feat_loss
            self.manual_backward(enc_loss)
        self.clip_gradients(opt_enc, gradient_clip_val=1.0, gradient_clip_algorithm='value')
        opt_enc.step()

        with opt_dec.toggle_model():
            # decoder training
            dec, dec_out_features = self.decode_features(ref_quant)
            dec_recon_loss = (dec - ref_dec).abs().mean()
            self.log("train/dec_recon_loss", dec_recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            dec_feat_loss = 0
            for key in dec_out_features:
                dec_feat_loss += 2e-5 * F.relu(torch.abs(dec_out_features[key]) - 100).mean()
            self.log("train/dec_feat_loss", dec_feat_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            dec_loss = dec_recon_loss + dec_feat_loss
            self.manual_backward(dec_loss)
        self.clip_gradients(opt_dec, gradient_clip_val=1.0, gradient_clip_algorithm='value')
        opt_dec.step()

class MOVQInferenceWrapper(MOVQ):
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h
    
    def decode(self, z):
        quant, emb_loss, info = self.quantize(z)
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec
