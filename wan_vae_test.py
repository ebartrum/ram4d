import torch
import torchvision
import torchvision.io
from taehv import WanCompatibleTAEHV
import os
import sys
import torch.nn.functional as F
import argparse
import warnings
from huggingface_hub import snapshot_download

# Add Wan repo to path
sys.path.insert(0, os.path.abspath("official_wan_repo"))

try:
    import wan
    from wan.modules.vae import WanVAE
    from wan.configs import WAN_CONFIGS
except ImportError:
    print("Warning: Could not import wan package. Please ensure official_wan_repo is present.")
    WanVAE = None

def load_video(video_path, max_frames=81):
    try:
        # Suppress the specific deprecation warning from torchvision
        warnings.filterwarnings("ignore", category=UserWarning, message="The video decoding and encoding capabilities of torchvision are deprecated")
        
        # Using read_video. Default output is [T, H, W, C]
        video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit='sec')
        # If successful, output is [T, C, H, W]
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None

    # Handle Frame Count (1 mod 4)
    # Target should be <= max_frames
    T = video.shape[0]
    target_T = min(T, max_frames)
    # Ensure 1 mod 4
    # (target_T - 1) // 4 * 4 + 1
    target_T = (target_T - 1) // 4 * 4 + 1
    
    if target_T < 1: 
        target_T = 1 
            
    video = video[:target_T]
            
    # Handle Resolution
    # [T, C, H, W]
    _, _, H, W = video.shape
    new_H = round(H / 16) * 16
    new_W = round(W / 16) * 16
    
    if (new_H, new_W) != (H, W):
            video = F.interpolate(video, size=(new_H, new_W), mode='bilinear', align_corners=False, antialias=True)
            
    # Normalize to [-1, 1]
    if video.dtype == torch.uint8:
        video = video.float() / 255.0
        
    video = 2.0 * video - 1.0 # [-1, 1]
    
    return video.permute(1, 0, 2, 3) # [C, T, H, W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_type", type=str, default="wan_vae", choices=["wan_vae", "tiny_vae"], help="Type of VAE to use: wan_vae (default) or tiny_vae")
    args = parser.parse_args()

    video_path = "data/videos/davis_camel.mp4"
    
    if args.vae_type == "wan_vae":
        # Ensure model is downloaded
        print("Ensuring Wan2.1 checkpoint is downloaded...")
        try:
             checkpoint_dir = snapshot_download(repo_id="Wan-AI/Wan2.1-I2V-14B-720P")
             vae_path = os.path.join(checkpoint_dir, "Wan2.1_VAE.pth")
        except Exception as e:
             print(f"Failed to download/locate Wan model: {e}")
             # Fallback to local relative path if exists
             vae_path = "Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"
    else:
        vae_path = "checkpoints/taew2_1.pth"
    
    output_path = f"output/{args.vae_type}_comparison.mp4"
    os.makedirs("output", exist_ok=True)

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    if not os.path.exists(vae_path) and args.vae_type == "tiny_vae":
        print(f"VAE weights not found: {vae_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 # Use float32 for compatibility, or bf16 if supported

    print(f"Loading {args.vae_type} from {vae_path}...")
    if args.vae_type == "wan_vae":
        if WanVAE is None:
            print("WanVAE class not available. Exiting.")
            return
        # WanVAE init expects vae_pth
        vae = WanVAE(vae_pth=vae_path, device=device, dtype=dtype)
        # WanVAE loads weights in init
    else:
        vae = WanCompatibleTAEHV(checkpoint_path=vae_path).to(device).to(dtype)
        vae.eval()

    print(f"Loading video from {video_path}...")
    video_tensor = load_video(video_path) # [C, T, H, W]
    if video_tensor is None:
        return

    video_tensor = video_tensor.to(device).to(dtype)
    print(f"Video tensor shape: {video_tensor.shape}")

    print("Encoding video...")
    with torch.no_grad():
        # Encode
        # Input: List of [C, T, H, W]
        # video_tensor is [C, T, H, W]
        latents_list = vae.encode([video_tensor])
        latents = latents_list[0]
        print(f"Latents shape: {latents.shape}")

        # Decode
        print("Decoding latents...")
        # Input: List of [C, T_lat, H_lat, W_lat]
        recon_video_list = vae.decode([latents])
        recon_video = recon_video_list[0] # [C, T, H, W], [-1, 1]
        print(f"Reconstructed video shape: {recon_video.shape}")

    # Post-process reconstructed video
    recon_video = (recon_video + 1.0) / 2.0 # [0, 1]
    recon_video = recon_video.clamp(0, 1)
    recon_video = (recon_video * 255).to(torch.uint8) # [0, 255]

    # Post-process original video for comparison
    original_video = video_tensor
    original_video = (original_video + 1.0) / 2.0
    original_video = original_video.clamp(0, 1)
    original_video = (original_video * 255).to(torch.uint8)

    # Convert to [T, H, W, C] for saving
    recon_video = recon_video.permute(1, 2, 3, 0).cpu()
    original_video = original_video.permute(1, 2, 3, 0).cpu()

    # Dimensions check
    print(f"Original: {original_video.shape}, Recon: {recon_video.shape}")

    # Resize recon to match original if needed
    if original_video.shape != recon_video.shape:
        print("Resizing reconstructed video to match original dimensions...")
        # Resize spatial dims
        # recon_video is [T, H, W, C] -> [T, C, H, W] for interpolate
        recon_video_perm = recon_video.permute(0, 3, 1, 2).float()
        
        target_H, target_W = original_video.shape[1], original_video.shape[2]
        
        recon_video_resized = F.interpolate(
            recon_video_perm, 
            size=(target_H, target_W), 
            mode='bilinear', 
            align_corners=False, 
            antialias=True
        )
        
        recon_video = recon_video_resized.permute(0, 2, 3, 1).to(torch.uint8)
        
        # Handle temporal mismatch if any
        min_t = min(original_video.shape[0], recon_video.shape[0])
        original_video = original_video[:min_t]
        recon_video = recon_video[:min_t]

    # Concatenate side-by-side
    combined_video = torch.cat([original_video, recon_video], dim=2) # Concatenate along Width

    print(f"Saving comparison video to {output_path}...")
    torchvision.io.write_video(output_path, combined_video, fps=24) # Assuming 24 fps
    print("Done!")

if __name__ == "__main__":
    main()
