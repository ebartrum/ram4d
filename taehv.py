#!/usr/bin/env python3
"""
Tiny AutoEncoder for Wan
(DNN for encoding / decoding videos to Wan Video's latent space)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import namedtuple
import os

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f*stride,n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f*stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)

def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N*T, C, H, W)
        # parallel over input timesteps, iterate over blocks
        for b in tqdm(model, disable=not show_progress_bar):
            if isinstance(b, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                mem = F.pad(_x, (0,0,0,0,0,0,1,0), value=0)[:,:T].reshape(x.shape) 
                x = b(x, mem)
            else:
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        # TODO(oboerbohan): at least on macos this still gradually uses more memory during decode...
        # need to fix :(
        out = []
        # iterate over input timesteps and also iterate over blocks.
        # because of the cursed TPool/TGrow blocks, this is not a nested loop,
        # it's actually a ***graph traversal*** problem! so let's make a queue
        work_queue = [TWorkItem(xt, 0) for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))]
        # in addition to manually managing our queue, we also need to manually manage our progressbar.
        # we'll update it for every source node that we consume.
        progress_bar = tqdm(range(T), disable=not show_progress_bar)
        # we'll also need a separate addressable memory per node as well
        mem = [None] * len(model)
        while work_queue:
            xt, i = work_queue.pop(0)
            if i == 0:
                # new source node consumed
                progress_bar.update(1)
            if i == len(model):
                # reached end of the graph, append result to output list
                out.append(xt)
            else:
                # fetch the block to process
                b = model[i]
                if isinstance(b, MemBlock):
                    # mem blocks are simple since we're visiting the graph in causal order
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt
                    else:
                        xt_new = b(xt, mem[i])
                        mem[i].copy_(xt) # inplace might reduce mysterious pytorch memory allocations? doesn't help though
                    # add successor to work queue
                    work_queue.insert(0, TWorkItem(xt_new, i+1))
                elif isinstance(b, TPool):
                    # pool blocks are miserable
                    if mem[i] is None:
                        mem[i] = [] # pool memory is itself a queue of inputs to pool
                    mem[i].append(xt)
                    if len(mem[i]) > b.stride:
                        # pool mem is in invalid state, we should have pooled before this
                        raise ValueError("???")
                    elif len(mem[i]) < b.stride:
                        # pool mem is not yet full, go back to processing the work queue
                        pass
                    else:
                        # pool mem is ready, run the pool block
                        N, C, H, W = xt.shape 
                        xt = b(torch.cat(mem[i], 1).view(N*b.stride, C, H, W))
                        # reset the pool mem
                        mem[i] = []
                        # add successor to work queue
                        work_queue.insert(0, TWorkItem(xt, i+1))
                elif isinstance(b, TGrow):
                    xt = b(xt)
                    NT, C, H, W = xt.shape
                    # each tgrow has multiple successor nodes
                    for xt_next in reversed(xt.view(N, b.stride*C, H, W).chunk(b.stride, 1)):
                        # add successor to work queue
                        work_queue.insert(0, TWorkItem(xt_next, i+1))
                else:
                    # normal block with no funny business
                    xt = b(xt)
                    # add successor to work queue
                    work_queue.insert(0, TWorkItem(xt, i+1))
        progress_bar.close()
        x = torch.stack(out, 1)
    return x

# Fix for PyTorch 2.6 weights_only issue
def safe_load_state_dict(checkpoint_path, map_location="cpu"):
    """Safely load state dict with fallback for weights_only parameter"""
    try:
        # Try with weights_only=True first
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as e:
        # If that fails, try with weights_only=False
        print(f"Warning: weights_only=True failed, falling back to weights_only=False: {e}")
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)

class BaseTAEHV(nn.Module):
    """Base class for TAEHV models with common functionality."""
    
    def __init__(self, checkpoint_path="taehv.pth", decoder_time_upscale=(True, True), 
                 decoder_space_upscale=(True, True, True), patch_size=1, latent_channels=16,
                 input_channels=3, output_channels=3):
        """Initialize base TAEHV model with common functionality.

        Args:
            checkpoint_path: path to weight file to load. taehv.pth for Hunyuan, taew2_1.pth for Wan 2.1.
            decoder_time_upscale: whether temporal upsampling is enabled for each block.
            decoder_space_upscale: whether spatial upsampling is enabled for each block.
            patch_size: input/output pixelshuffle patch-size for this model.
            latent_channels: number of latent channels (z dim) for this model.
            input_channels: number of input channels for the encoder.
            output_channels: number of output channels for the decoder.
        """
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.is_cogvideox = checkpoint_path is not None and "taecvx" in checkpoint_path
        if checkpoint_path is not None and "taew2_2" in checkpoint_path:
            self.patch_size, self.latent_channels = 2, 48
        self.encoder = nn.Sequential(
            conv(self.input_channels*self.patch_size**2, 64), nn.ReLU(inplace=True),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2**sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], self.output_channels*self.patch_size**2),
        )

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if needed.

        Args:
            sd: state dict to patch
        """
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd.get(key) is not None and sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def encode_video(self, x, parallel=True, show_progress_bar=True):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        if self.patch_size > 1: x = F.pixel_unshuffle(x, self.patch_size)
        if x.shape[1] % 4 != 0:
            # pad at end to multiple of 4
            n_pad = 4 - x.shape[1] % 4
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x, parallel=True, show_progress_bar=False):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent tensor with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW tensor with ~[0, 1] values.
        """
        skip_trim = self.is_cogvideox and x.shape[1] % 2 == 0
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = x.clamp_(0, 1)
        if self.patch_size > 1: x = F.pixel_shuffle(x, self.patch_size)
        if skip_trim:
            # skip trimming for cogvideox to make frame counts match.
            # this still doesn't have correct temporal alignment for certain frame counts
            # (cogvideox seems to pad at the start?), but for multiple-of-4 it's fine.
            return x
        return x[:, self.frames_to_trim:]


class TAEHV(BaseTAEHV):
    def __init__(self, checkpoint_path="taehv.pth", decoder_time_upscale=(True, True), 
                 decoder_space_upscale=(True, True, True), patch_size=1, latent_channels=16):
        """Initialize pretrained TAEHV from the given checkpoint.

        Arg:
            checkpoint_path: path to weight file to load. taehv.pth for Hunyuan, taew2_1.pth for Wan 2.1.
            decoder_time_upscale: whether temporal upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            decoder_space_upscale: whether spatial upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            patch_size: input/output pixelshuffle patch-size for this model.
            latent_channels: number of latent channels (z dim) for this model.
        """
        super().__init__(checkpoint_path=checkpoint_path, 
                         decoder_time_upscale=decoder_time_upscale,
                         decoder_space_upscale=decoder_space_upscale,
                         patch_size=patch_size,
                         latent_channels=latent_channels,
                         input_channels=3,
                         output_channels=3)
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_state_dict(self.patch_tgrow_layers(safe_load_state_dict(checkpoint_path, map_location="cpu")))


class RGBA_TAEHV(BaseTAEHV):
    """Extended TAEHV model with 4-channel output for RGBA.
    
    The encoder takes 3 channels (RGB) as input and the decoder outputs 4 channels (RGBA).
    """
    
    def __init__(self, checkpoint_path="checkpoints/taew2_1.pth", 
                 decoder_time_upscale=(True, True), 
                 decoder_space_upscale=(True, True, True), 
                 patch_size=1, 
                 latent_channels=16):
        """Initialize pretrained TAEHV from the given checkpoint and modify for RGBA output."""
        super().__init__(checkpoint_path=None,  # Don't load checkpoint in base class
                         decoder_time_upscale=decoder_time_upscale,
                         decoder_space_upscale=decoder_space_upscale,
                         patch_size=patch_size,
                         latent_channels=latent_channels,
                         input_channels=3,  # Encoder input channels (RGB)
                         output_channels=4)  # Decoder output channels (RGBA)
        
        # Load pretrained weights and modify for RGBA
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            original_state_dict = self.patch_tgrow_layers(
                safe_load_state_dict(checkpoint_path, map_location="cpu"))
            self._adapt_pretrained_weights(original_state_dict)
    
    def _adapt_pretrained_weights(self, original_state_dict):
        """Adapt pretrained weights from RGB to RGBA output."""
        current_state_dict = self.state_dict()
        modified_state_dict = {}
        
        # Copy all weights that match in shape
        for key, value in original_state_dict.items():
            if key in current_state_dict:
                if value.shape == current_state_dict[key].shape:
                    modified_state_dict[key] = value
                else:
                    # For the last layer, we need to handle the channel difference
                    if "decoder" in key and value.shape[0] == 3 * self.patch_size**2:  # RGB output layer
                        # Create new tensor with 4 channels (RGBA)
                        new_value = torch.zeros(current_state_dict[key].shape, dtype=value.dtype)
                        # Copy RGB channels
                        new_value[:3 * self.patch_size**2] = value
                        # Initialize alpha channel to zeros
                        modified_state_dict[key] = new_value
                    else:
                        # Keep the randomly initialized weights for other mismatched layers
                        modified_state_dict[key] = current_state_dict[key]
            else:
                # Keep the randomly initialized weights for keys not in original
                modified_state_dict[key] = current_state_dict[key]
        
        # For keys in current but not in original, keep current values
        for key in current_state_dict:
            if key not in modified_state_dict:
                modified_state_dict[key] = current_state_dict[key]
        
        self.load_state_dict(modified_state_dict)


class WanCompatibleTAEHV(TAEHV):
    """TAEHV wrapper that mimics WanVideoVAE interface."""
    def __init__(self, checkpoint_path="checkpoints/taew2_1.pth"):
        super().__init__(checkpoint_path=checkpoint_path)

    def encode(self, videos, device=None, **kwargs):
        """
        Encode list of videos to list of latents.
        videos: List[Tensor], each [C, T, H, W] in [-1, 1]
        Returns: List[Tensor], each [C, T_lat, H_lat, W_lat]
        """
        # Collect batch
        x = torch.stack(videos) # [B, C, T, H, W]
        # Allow device override or use model device
        if device is None:
             device = next(self.parameters()).device
        
        x = x.to(device)
        
        # [B, C, T, H, W] -> [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        # [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        
        latents = self.encode_video(x) # [B, T, C, H, W]
        
        # [B, T, C, H, W] -> [B, C, T, H, W]
        latents = latents.permute(0, 2, 1, 3, 4)
        
        return list(latents)

    def decode(self, latents, device=None, **kwargs):
        """
        Decode list of latents to list of videos.
        latents: List[Tensor], each [C, T_lat, H_lat, W_lat]
        Returns: List[Tensor], each [C, T, H, W] in [-1, 1]
        """
        z = torch.stack(latents) # [B, C, T, H, W]
        if device is None:
             device = next(self.parameters()).device
        z = z.to(device)
        
        # [B, C, T, H, W] -> [B, T, C, H, W]
        z = z.permute(0, 2, 1, 3, 4)
        
        video = self.decode_video(z)
        
        # [0, 1] -> [-1, 1]
        video = (video * 2.0) - 1.0
        
        # [B, T, C, H, W] -> [B, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)
        
        return list(video)

    def latent_image_condition(self, img_tensor, num_frames):
        # img_tensor: [B, C, H, W] (usually B=1)
        msk = torch.ones(1, num_frames, img_tensor.shape[2]//8, img_tensor.shape[3]//8, device=img_tensor.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4,
           img_tensor.shape[2]//8, img_tensor.shape[3]//8)
        msk = msk.transpose(1, 2)[0]
        
        # vae_input: [C, T, H, W]
        # img_tensor.transpose(0, 1) -> [C, B, H, W]. If B=1 -> [C, 1, H, W]
        vae_input = torch.concat([img_tensor.transpose(0, 1),
            torch.zeros(3, num_frames-1, img_tensor.shape[2],
                img_tensor.shape[3]).to(device=img_tensor.device, dtype=img_tensor.dtype)], dim=1)

        latent_img_condition = self.encode_latent_img_condition(vae_input, msk)
        return latent_img_condition

    def encode_latent_img_condition(self, vae_input, msk):
        # vae_input: [C, T, H, W]
        # self.encode expects List[Tensor] with shape [C, T, H, W]
        # returns List[Tensor]
        latent_img_condition = self.encode([vae_input], device=msk.device)[0] # [C, T, H, W]
        latent_img_condition = torch.concat([msk, latent_img_condition])
        latent_img_condition = latent_img_condition.unsqueeze(0)
        return latent_img_condition


@torch.no_grad()

def main():
    """Run TAEHV roundtrip reconstruction on the given video paths."""
    import os
    import sys
    import cv2 # no highly esteemed deed is commemorated here

    class VideoTensorReader:
        def __init__(self, video_file_path):
            self.cap = cv2.VideoCapture(video_file_path)
            assert self.cap.isOpened(), f"Could not load {video_file_path}"
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        def __iter__(self):
            return self
        def __next__(self):
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration  # End of video or error
            return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) # BGR HWC -> RGB CHW

    class VideoTensorWriter:
        def __init__(self, video_file_path, width_height, fps=30):
            self.writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, width_height)
            assert self.writer.isOpened(), f"Could not create writer for {video_file_path}"
        def write(self, frame_tensor):
            assert frame_tensor.ndim == 3 and frame_tensor.shape[0] == 3, f"{frame_tensor.shape}??"
            self.writer.write(cv2.cvtColor(frame_tensor.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)) # RGB CHW -> BGR HWC
        def __del__(self):
            if hasattr(self, 'writer'): self.writer.release()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    checkpoint_path = "checkpoints/taew2_1.pth"
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    taehv = TAEHV(checkpoint_path=checkpoint_path).to(dev, dtype)
    for video_path in sys.argv[1:]:
        print(f"Processing {video_path}...")
        video_in = VideoTensorReader(video_path)
        video = torch.stack(list(video_in), 0)[None]
        vid_dev = video.to(dev, dtype).div_(255.0)
        # convert to device tensor
        if video.numel() < 100_000_000:
            print(f"  {video_path} seems small enough, will process all frames in parallel")
            # convert to device tensor
            vid_enc = taehv.encode_video(vid_dev)
            print(f"  Encoded {video_path} -> {vid_enc.shape}. Decoding...")
            vid_dec = taehv.decode_video(vid_enc)
            print(f"  Decoded {video_path} -> {vid_dec.shape}")
        else:
            print(f"  {video_path} seems large, will process each frame sequentially")
            # convert to device tensor
            vid_enc = taehv.encode_video(vid_dev, parallel=False)
            print(f"  Encoded {video_path} -> {vid_enc.shape}. Decoding...")
            vid_dec = taehv.decode_video(vid_enc, parallel=False)
            print(f"  Decoded {video_path} -> {vid_dec.shape}")
        video_out_path = video_path.replace(".mp4", f"_recon_{checkpoint_name}.mp4")
        video_out = VideoTensorWriter(video_out_path, (vid_dec.shape[-1], vid_dec.shape[-2]), fps=int(round(video_in.fps)))
        for frame in vid_dec.clamp_(0, 1).mul_(255).round_().byte().cpu()[0]:
            video_out.write(frame)
        print(f"  Saved to {video_out_path}")

if __name__ == "__main__":
    main()
