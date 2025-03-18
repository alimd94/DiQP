import torch
import torch.onnx
import argparse
from model import DiQP  # Model defined in model.py :contentReference[oaicite:0]{index=0}​:contentReference[oaicite:1]{index=1}

# Wrapper that adapts the model inputs:
# - The main input is replicated to 3 frames (F=3) to satisfy the model's inputFrames.
# - Guidance inputs (around, aheadCropped, aheadScaled) are replicated to 2 frames, which is expected during concatenation.
# - After passing through the model, the middle frame from the temporal dimension is taken.
class VideoRestorationONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super(VideoRestorationONNXWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # x input tensor in the shape [B, C, H, W]
        frames_main = 3       # For the main path, the model expects 3 frames.
        frames_guidance = 2   # Guidance inputs are expected with 2 frames, to achieve the correct dimension after concatenation.
        
        # Replication of the main input
        x_main = x.unsqueeze(2).repeat(1, 1, frames_main, 1, 1)   # [B, C, 3, H, W]
        # Guidance inputs
        guidance = x.unsqueeze(2).repeat(1, 1, frames_guidance, 1, 1)  # [B, C, 2, H, W]
        
        B = x.shape[0]
        # Dummy tensor for location data (6 elements per sample)
        loc = torch.zeros(6, B, 1, dtype=torch.long, device=x.device)
        # Dummy factor weightDecay – shape [B, 1, 1, 1, 1]
        weightDecay = torch.ones(B, 1, 1, 1, 1, device=x.device)
        
        # Pass through the original model: the main input goes as x_main, while the guidance inputs are passed for around, aheadCropped, and aheadScaled.
        output = self.model(x_main, guidance, guidance, guidance, loc, weightDecay)
        # The model output is a 5D tensor [B, C, F, H, W] – we take the middle frame (index 1 for F=3)
        mid = frames_main // 2  
        output = output[:, :, mid, :, :]
        return output

def main():
    parser = argparse.ArgumentParser(description='Conversion of PyTorch video restoration model to dynamic ONNX (opset 20)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the PyTorch checkpoint file')
    parser.add_argument('--output', type=str, default='model.onnx', help='Name of the output ONNX file')
    parser.add_argument('--img_size', type=int, default=512, help='Image size (height and width) for the dummy input')
    args = parser.parse_args()

    device = torch.device('cpu')
    
    # Instantiate the model – parameters can be adjusted according to the checkpoint.
    model = DiQP(
        img_size=args.img_size,
        embed_dim=15,
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        win_size=8,
        mlp_ratio=3.,
        token_projection='linear',
        token_mlp='steff',
        shift_flag=True
    )
    
    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Wrap the model with a wrapper that handles pre/post-processing of inputs/outputs
    wrapped_model = VideoRestorationONNXWrapper(model)
    wrapped_model.eval()

    # Create a dummy input – 4D tensor in NCHW format
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    # Define input/output names and dynamic axes
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"}
    }

    # Export to ONNX format with opset version 20
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"Model has been successfully converted to ONNX and saved as {args.output}")

if __name__ == '__main__':
    main()