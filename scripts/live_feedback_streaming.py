# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Streaming Live Fitness Coaching - Variable timing like original project."""

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.constants import FEEDBACK_BEGIN_TOKEN, FEEDBACK_END_TOKEN, VISION_TOKEN, INFERENCE_SPEED
from src.model_helpers import make_model
from src.vision_modules.vision_model import Hypermodel


class StreamingFeedbackCoach:
    """Streaming version of live feedback coach with variable timing.

    Uses the same asynchronous approach as the original project where the model
    decides when to generate feedback using <vision> and <answer> tokens.

    Key differences from lightweight version:
    - Model controls feedback timing (not fixed intervals)
    - Streaming generation loop (model decides with tokens)
    - Pre-extracts all features before generation (like original evaluation)
    - Uses blind frame mechanism to simulate real-time

    Note: "Streaming" refers to the generation approach (asynchronous token-based
    decisions), not real-time feature extraction. All frames are preprocessed and
    features extracted before the generation loop starts, matching the original
    evaluation approach.
    """

    def __init__(self, model, config, cnn_weights_path, max_buffer_frames=300):
        """Initialize streaming coach.

        Args:
            model: Stream-VLM model
            config: Configuration dictionary
            cnn_weights_path: Path to 3D CNN (EfficientNet) weights
            max_buffer_frames: Maximum preprocessed frames to buffer
        """
        self.model = model
        self.config = config
        self.sampling_kwargs = config["evaluator"]["sampling_kwargs"]
        self.feats_frequency = self.sampling_kwargs.get("feats_frequency", 6)

        # Load 3D CNN for feature extraction
        print("Loading 3D CNN for feature extraction...")
        self.cnn_model = Hypermodel(
            num_global_classes=23,
            num_frames_required=1,
            path_weights=cnn_weights_path,
            gpus=[0] if torch.cuda.is_available() else None,
            half_precision=False
        )
        self.cnn_model.initialize()
        print("3D CNN loaded successfully!")

        # Buffer for preprocessed frames (not CNN features yet)
        self.frame_buffer = deque(maxlen=max_buffer_frames)
        self.feedback_history = []

        # Special tokens
        self.special_tokens_dict = {
            VISION_TOKEN: self.model.tokenizer.encode(VISION_TOKEN)[-1],
            FEEDBACK_BEGIN_TOKEN: self.model.tokenizer.encode(FEEDBACK_BEGIN_TOKEN)[-1],
            FEEDBACK_END_TOKEN: self.model.tokenizer.encode(FEEDBACK_END_TOKEN)[-1],
        }

        # Tracking for streaming generation
        self.current_frame_idx = 0
        self.frames_seen_by_model = 0

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    def preprocess_frame(self, frame):
        """Preprocess a single frame using 3D CNN's transform.

        Args:
            frame: OpenCV BGR frame

        Returns:
            Preprocessed frame as numpy array [1, 3, H, W]
        """
        return self.cnn_model.transforms(frame)

    def extract_features_batch(self, frames_list):
        """Extract CNN features from a batch of preprocessed frames.

        Args:
            frames_list: List of preprocessed frames, each [1, 3, H, W]

        Returns:
            Dictionary with 'feats' [1, num_frames, 1, 1280] and 'spatial_res'
        """
        # Concatenate along batch dimension: [num_frames, 3, H, W]
        frames_batch = np.concatenate(frames_list, axis=0)

        # Convert to tensor
        frames_tensor = torch.from_numpy(frames_batch)
        if self.cnn_model.gpus is not None:
            frames_tensor = frames_tensor.cuda(self.cnn_model.gpus[0])

        with torch.no_grad():
            # Extract features from backbone
            cnn_features = self.cnn_model.features(frames_tensor)

            # Apply spatial pooling if needed
            if len(cnn_features.shape) == 4:  # [num_frames, 1280, H, W]
                cnn_features = cnn_features.mean(dim=-1).mean(dim=-1)  # [num_frames, 1280]

            # Move to model device
            cnn_features = cnn_features.to(self.model.device)

            # Reshape to [B, L, H*W, C] format
            # B=1 (batch), L=num_frames (temporal), H*W=1 (spatial), C=1280 (features)
            cnn_features = cnn_features.unsqueeze(0).unsqueeze(2)  # [1, num_frames, 1, 1280]

        # Clean up
        del frames_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'feats': cnn_features,
            'spatial_res': [1, 1]
        }

    def _get_vision_xattn_mask(self, input_ids):
        """Create vision cross-attention mask."""
        valid_video_indices = np.where(
            np.array(input_ids) == self.special_tokens_dict[VISION_TOKEN]
        )[0]
        vision_xattn_mask = np.array([0] * len(input_ids))
        vision_xattn_mask[valid_video_indices] = 1
        return vision_xattn_mask.tolist()

    @torch.no_grad()
    def generate_streaming(self, system_prompt, video_file=None):
        """Generate feedback with streaming approach - model decides timing.

        This mimics the original _generate_interactive method but for live video.
        We pre-extract all CNN features first, then run the generation loop.

        Args:
            system_prompt: System prompt describing the task
            video_file: Optional video file path (None for webcam)
        """
        min_frames = self.sampling_kwargs.get("min_frames_before_start", 12)

        print("Waiting for initial frames before starting generation...")

        # Wait for minimum frames
        while len(self.frame_buffer) < min_frames:
            time.sleep(0.1)

        print(f"Starting generation with {len(self.frame_buffer)} frames buffered...")

        # Pre-extract ALL CNN features from buffered frames
        # This matches the original approach where all features are available upfront
        all_frames = list(self.frame_buffer)
        print(f"Extracting CNN features from {len(all_frames)} frames...")
        encoded_video = self.extract_features_batch(all_frames)
        print(f"Features extracted: {encoded_video['feats'].shape}")

        # Prepare input prompt
        input_prompt = system_prompt + VISION_TOKEN
        input_ids = self.model.tokenizer.encode(input_prompt)
        vision_xattn_mask = self._get_vision_xattn_mask(input_ids)
        vision_xattn_mask = [2 if tok == 1 else 0 for tok in vision_xattn_mask]

        # Convert to tensors
        output_ids = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)
        vision_xattn_mask = torch.tensor(vision_xattn_mask).unsqueeze(0).to(self.model.device)

        # Generation state (follows original _generate_interactive)
        past_key_values = None
        feedback_mode = False
        current_feedback_tokens = []
        curr_response_len = 0
        input_vision_idx = 2  # Start with first 2 frames as in original
        skip_blind_frames = [False] * (input_vision_idx - 1)

        max_feedback_length = self.sampling_kwargs.get("max_feedback_length", 64)
        do_sample = self.sampling_kwargs.get("do_sample", False)
        temperature = self.sampling_kwargs.get("temperature", 0.0)

        # Continue generating until we've consumed all video frames
        while input_vision_idx < encoded_video["feats"].shape[1]:
            # Prepare video input (frames 1 to input_vision_idx, excluding blind frames)
            encoded_video_feats = encoded_video["feats"]
            encoded_video_in_range = {
                "feats": encoded_video_feats[:, 1:input_vision_idx][:, np.logical_not(skip_blind_frames)],
                "spatial_res": encoded_video["spatial_res"],
            }

            # Adapt video features
            multi_model_embedding = self.model.model.adapter(
                encoded_video_in_range, output_ids, vision_xattn_mask
            )

            # Generate next token
            lang_out = self.model.model.lang(
                inputs_embeds=multi_model_embedding,
                attention_mask=torch.ones_like(output_ids).to(self.model.device),
                use_cache=True,
                past_key_values=past_key_values,
            )

            # Update KV cache
            past_key_values = lang_out["past_key_values"]

            # Sample next token
            if not do_sample:
                next_token = torch.argmax(lang_out["logits"][:, -1], dim=-1)
            else:
                logits = lang_out["logits"][:, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Sanity checks for invalid states (from original)
            if feedback_mode:
                # In feedback mode, if invalid token or too long, force end
                if (next_token.item() == self.special_tokens_dict[VISION_TOKEN] or
                    next_token.item() == self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN] or
                    curr_response_len > max_feedback_length):
                    next_token = torch.tensor([self.special_tokens_dict[FEEDBACK_END_TOKEN]]).to(self.model.device)
            else:
                # Not in feedback mode - only allow <vision> or <answer> tokens
                if (next_token.item() != self.special_tokens_dict[VISION_TOKEN] and
                    next_token.item() != self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN]):
                    # Default to <vision> (continue watching)
                    next_token = torch.tensor([self.special_tokens_dict[VISION_TOKEN]]).to(self.model.device)

            # Add token to output
            output_ids = torch.cat([output_ids, next_token.unsqueeze(-1)], dim=1)

            # State changes based on output (from original)
            if next_token.item() == self.special_tokens_dict[VISION_TOKEN]:
                # Continue watching - consume next frame
                input_vision_idx += 1
                skip_blind_frames.append(False)

            elif next_token.item() == self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN]:
                # Start feedback mode
                feedback_mode = True
                curr_response_len = 0
                current_feedback_tokens = []
                print(f"\n[Frame {input_vision_idx}] Coach is speaking...", end="")

            elif next_token.item() == self.special_tokens_dict[FEEDBACK_END_TOKEN]:
                # End feedback mode
                feedback_mode = False

                # Decode feedback
                feedback_text = self.model.tokenizer.decode(current_feedback_tokens, skip_special_tokens=True)
                print(f" Done!")
                print(f"Coach: {feedback_text}")

                # Store feedback
                self.feedback_history.append((time.time(), feedback_text))

                # Skip frames that arrived during generation (simulate real-time)
                skip_forward = math.floor((curr_response_len / INFERENCE_SPEED) * self.feats_frequency)
                input_vision_idx += skip_forward
                skip_blind_frames += [True] * skip_forward
                curr_response_len = 0
                current_feedback_tokens = []

            else:
                # Regular text token in feedback mode
                if feedback_mode:
                    current_feedback_tokens.append(next_token.item())
                    curr_response_len += 1

            # Update vision cross-attention mask based on token type
            if next_token.item() == self.special_tokens_dict[VISION_TOKEN]:
                vision_xattn_mask = torch.cat([
                    vision_xattn_mask,
                    torch.ones(1, 1).to(vision_xattn_mask) * 2
                ], dim=1)
            else:
                vision_xattn_mask = torch.cat([
                    vision_xattn_mask,
                    torch.zeros(1, 1).to(vision_xattn_mask)
                ], dim=1)

    def run_live(self, exercise_type, video_file=None, headless=False):
        """Run live feedback with streaming generation.

        Args:
            exercise_type: Type of exercise (e.g., "squats")
            video_file: Optional video file path (None for webcam)
            headless: If True, don't display video window
        """
        # Open video source
        if video_file:
            cap = cv2.VideoCapture(video_file)
            print(f"Processing video: {video_file}")
        else:
            camera_id = self.config["evaluator"]["kwargs"].get("camera_id", 0)
            cap = cv2.VideoCapture(camera_id)
            print(f"Using camera: {camera_id}")

        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        print(f"\n=== FitCoach Streaming Live Feedback ===")
        print(f"Exercise: {exercise_type}")
        print(f"Feature rate: {self.feats_frequency} fps")
        print(f"Headless mode: {headless}")
        print(f"Max buffer: {self.frame_buffer.maxlen} frames")
        print(f"Press 'q' to quit\n")

        system_prompt = (
            f"You are an expert fitness coaching AI who coaches users as they exercise. "
            f"You assess their performance, count repetitions, and proactively provide feedback. "
            f"The user should be doing {exercise_type}."
        )

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS) if video_file else 30.0
        if video_fps == 0:
            video_fps = 30.0

        frame_delay = 1.0 / video_fps if video_file else 0
        preprocess_interval = 1.0 / self.sampling_kwargs.get("frame_preprocess_rate", 6)

        last_preprocess_time = time.time()
        frame_count = 0

        # First, preprocess ALL frames from the video
        print("Preprocessing all frames from video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()

            # Preprocess at specified rate
            if current_time - last_preprocess_time >= preprocess_interval:
                try:
                    preprocessed = self.preprocess_frame(frame)
                    self.frame_buffer.append(preprocessed)
                    last_preprocess_time = current_time
                except Exception as e:
                    print(f"Frame preprocessing error: {e}")

            # Delay for video playback timing
            if video_file and frame_delay > 0:
                time.sleep(frame_delay)

        print(f"Preprocessed {len(self.frame_buffer)} frames from {frame_count} total frames")

        # Now run streaming generation with all frames available
        try:
            self.generate_streaming(system_prompt, video_file)
        except KeyboardInterrupt:
            print("\n\nStopping...")
        except Exception as e:
            print(f"\n\nError during generation: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

        # Print summary
        print(f"\n=== Session Summary ===")
        print(f"Total frames: {frame_count}")
        print(f"Preprocessed frames: {len(self.frame_buffer)}")
        print(f"Total feedback: {len(self.feedback_history)}")

        if self.feedback_history:
            print(f"\nFeedback History:")
            for i, (timestamp, feedback) in enumerate(self.feedback_history, 1):
                print(f"{i}. {feedback}")


def auto_find_cnn_weights():
    """Try to automatically find CNN weights."""
    possible_paths = [
        "./ckpts_efficientnet/fitness_ally_hypermodel/efficientnet4Lite_1.8.3.checkpoint",
        "./ckpts_efficientnet/efficientnet4Lite_1.8.3.checkpoint",
        "./ckpts_efficientnet/efficientnet_3d_cnn.pth.tar",
        "./ckpts_efficientnet/ckpts/efficientnet_3d_cnn.pth.tar",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return path

    return None


def main():
    parser = argparse.ArgumentParser(description="FitCoach Streaming Live Feedback")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--video", type=str, default=None, help="Optional video file (default: webcam)")
    parser.add_argument("--exercise", type=str, default=None, help="Exercise type (overrides config)")
    parser.add_argument("--headless", action="store_true", help="Run without video display")
    parser.add_argument("--cnn_weights", type=str, default=None, help="Path to CNN weights (overrides config)")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override exercise if specified
    if args.exercise:
        config["evaluator"]["kwargs"]["exercise_type"] = args.exercise

    exercise_type = config["evaluator"]["kwargs"]["exercise_type"]

    # Find CNN weights
    cnn_weights_path = args.cnn_weights or config["evaluator"]["kwargs"].get("cnn_weights_path")
    if not cnn_weights_path or not Path(cnn_weights_path).exists():
        print("CNN weights not found in config, trying auto-detection...")
        cnn_weights_path = auto_find_cnn_weights()

    if not cnn_weights_path or not Path(cnn_weights_path).exists():
        print(f"Error: CNN weights not found at {cnn_weights_path}")
        print("Please specify with --cnn_weights or in config file")
        return

    print(f"Using CNN weights: {cnn_weights_path}")

    # Load model
    print("Loading Stream-VLM model...")
    llama2_7b_path = config["model"]["llama2_7b_path"]
    model_kwargs = config["model"]["kwargs"]
    model = make_model(llama2_7b_path, **model_kwargs)
    model.eval()
    print("Model loaded!")

    # Create coach
    coach = StreamingFeedbackCoach(
        model=model,
        config=config,
        cnn_weights_path=cnn_weights_path,
        max_buffer_frames=config["evaluator"]["kwargs"].get("max_buffer_frames", 300)
    )

    # Run live feedback
    coach.run_live(
        exercise_type=exercise_type,
        video_file=args.video,
        headless=args.headless
    )


if __name__ == "__main__":
    main()
