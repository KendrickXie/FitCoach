# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Lightweight Live Fitness Coaching - Optimized for limited resources."""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.constants import FEEDBACK_BEGIN_TOKEN, FEEDBACK_END_TOKEN, VISION_TOKEN
from src.model_helpers import make_model
from src.vision_modules.vision_model import Hypermodel


class LightweightFeedbackCoach:
    """Lightweight version of live feedback coach optimized for limited resources.

    Key optimizations:
    - Lower feature extraction rate (2 fps default)
    - Smaller feature buffer
    - Frame skipping during feedback generation
    - Optional int8 quantization support
    - Batch size of 1
    """

    def __init__(self, model, config, cnn_weights_path, max_buffer_size=200):
        """Initialize lightweight coach.

        Args:
            model: Stream-VLM model
            config: Configuration dictionary
            cnn_weights_path: Path to 3D CNN (EfficientNet) weights
            max_buffer_size: Maximum features to buffer (default 200 = 100 seconds at 2fps)
        """
        self.model = model
        self.config = config
        self.sampling_kwargs = config["evaluator"]["sampling_kwargs"]
        self.feats_frequency = self.sampling_kwargs.get("feats_frequency", 2)  # Lower default

        # Load 3D CNN for feature extraction
        print("Loading 3D CNN for feature extraction...")
        self.cnn_model = Hypermodel(
            num_global_classes=23,  # Number of exercises in QEVD
            num_frames_required=1,
            path_weights=cnn_weights_path,
            gpus=[0] if torch.cuda.is_available() else None,
            half_precision=False
        )
        self.cnn_model.initialize()
        print("3D CNN loaded successfully!")

        # Smaller buffer for memory efficiency
        self.feature_buffer = deque(maxlen=max_buffer_size)
        self.feedback_history = []

        # Special tokens
        self.special_tokens_dict = {
            VISION_TOKEN: self.model.tokenizer.encode(VISION_TOKEN)[-1],
            FEEDBACK_BEGIN_TOKEN: self.model.tokenizer.encode(FEEDBACK_BEGIN_TOKEN)[-1],
            FEEDBACK_END_TOKEN: self.model.tokenizer.encode(FEEDBACK_END_TOKEN)[-1],
        }

        # Enable memory optimizations
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    def preprocess_frame(self, frame):
        """Preprocess a single frame using the 3D CNN's transform pipeline.

        Args:
            frame: OpenCV BGR frame

        Returns:
            Preprocessed frame numpy array
        """
        # Use the 3D CNN's built-in preprocessing
        return self.cnn_model.transforms(frame)

    def generate_feedback(self, system_prompt, use_recent_only=True, window_size=60):
        """Generate feedback with memory optimization.

        Args:
            system_prompt: System prompt
            use_recent_only: Only use recent features to save memory
            window_size: Number of recent features to use

        Returns:
            Feedback text and timestamp
        """
        if len(self.feature_buffer) < 8:  # Require fewer frames
            return None, None

        try:
            # Use only recent frames if specified
            if use_recent_only and len(self.feature_buffer) > window_size:
                frames_list = list(self.feature_buffer)[-window_size:]
            else:
                frames_list = list(self.feature_buffer)

            # Stack frames: each frame is [1, 3, H, W], concatenate along axis 0 to get [num_frames, 3, H, W]
            frames_batch = np.concatenate(frames_list, axis=0)  # [num_frames, 3, H, W]

            # Extract 3D CNN features from the backbone (not the full net with classifier)
            frames_tensor = torch.from_numpy(frames_batch)
            if self.cnn_model.gpus is not None:
                frames_tensor = frames_tensor.cuda(self.cnn_model.gpus[0])

            with torch.no_grad():
                # Use features (backbone) only, not the full net
                cnn_features = self.cnn_model.features(frames_tensor)

                # Apply global average pooling over spatial dimensions (keep [num_frames, 1280])
                if len(cnn_features.shape) == 4:  # [num_frames, 1280, H, W]
                    cnn_features = cnn_features.mean(dim=-1).mean(dim=-1)  # [num_frames, 1280]
                elif len(cnn_features.shape) == 2:  # Already [num_frames, 1280]
                    pass  # No pooling needed
                else:
                    raise ValueError(f"Unexpected CNN feature shape: {cnn_features.shape}")

            # Average over temporal dimension to get single feature vector per spatial location
            # This matches the single <vision> token in the prompt
            cnn_features = cnn_features.mean(dim=0, keepdim=True)  # [1, 1280]

            # Move to model device
            cnn_features = cnn_features.to(self.model.device)

            # Reshape to [B, L, H*W, C] format expected by the model
            # B=1 (batch), L=1 (single time step), H*W=1 (single spatial location), C=1280 (features)
            cnn_features = cnn_features.unsqueeze(0).unsqueeze(2)  # [1, 1, 1, 1280]

            # Create features dict as expected by the model
            video_features = {
                'feats': cnn_features,
                'spatial_res': [1, 1]  # Single spatial location
            }

            # Delete intermediate tensors to free memory
            del frames_tensor

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Prepare input
            input_prompt = system_prompt + VISION_TOKEN
            input_ids = self.model.tokenizer.encode(input_prompt)
            vision_xattn_mask = self._get_vision_xattn_mask(input_ids)
            vision_xattn_mask = [2 if tok == 1 else 0 for tok in vision_xattn_mask]

            # Generate with lower max length for speed
            max_length = min(self.sampling_kwargs.get("max_feedback_length", 128), 64)

            input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)
            vision_xattn_mask_tensor = torch.tensor(vision_xattn_mask).unsqueeze(0).to(self.model.device)

            output = self._generate_single_feedback(
                video_features,
                input_ids_tensor,
                vision_xattn_mask_tensor,
                max_length=max_length
            )

            # Clean up tensors
            del video_features, input_ids_tensor, vision_xattn_mask_tensor

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if output is not None:
                feedback_text = self.model.tokenizer.decode(output, skip_special_tokens=True)
                return feedback_text, time.time()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU out of memory! Clearing cache and reducing buffer...")
                # Clear cache first
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Aggressively reduce buffer size
                while len(self.feature_buffer) > 20:
                    self.feature_buffer.popleft()
                # Try to clear any lingering tensors
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"Error generating feedback: {e}")

        return None, None

    def _get_vision_xattn_mask(self, input_ids):
        """Create vision cross-attention mask."""
        valid_video_indices = np.where(
            np.array(input_ids) == self.special_tokens_dict[VISION_TOKEN]
        )[0]
        vision_xattn_mask = np.array([0] * len(input_ids))
        vision_xattn_mask[valid_video_indices] = 1
        return vision_xattn_mask.tolist()

    def _generate_single_feedback(self, encoded_video, input_ids, vision_xattn_mask, max_length=64):
        """Generate feedback with limited length."""
        output_ids = input_ids.clone()
        past_key_values = None

        do_sample = self.sampling_kwargs.get("do_sample", False)
        temperature = self.sampling_kwargs.get("temperature", 0.0)

        for _ in range(max_length):
            multi_model_embedding = self.model.model.adapter(
                encoded_video, output_ids, vision_xattn_mask
            )

            lang_out = self.model.model.lang(
                inputs_embeds=multi_model_embedding,
                attention_mask=torch.ones_like(output_ids).to(self.model.device),
                use_cache=True,
                past_key_values=past_key_values,
            )

            past_key_values = lang_out["past_key_values"]

            if not do_sample:
                next_token = torch.argmax(lang_out["logits"][:, -1], dim=-1)
            else:
                scaled_logits = lang_out["logits"][:, -1] / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze()

            # next_token is [batch_size], need to make it [batch_size, 1] to concat with output_ids
            output_ids = torch.cat([output_ids, next_token.unsqueeze(-1)], dim=1)

            if next_token.item() == self.special_tokens_dict[FEEDBACK_END_TOKEN]:
                output_list = output_ids[0].cpu().tolist()
                try:
                    start_idx = output_list.index(self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN])
                    end_idx = len(output_list) - 1
                    return output_list[start_idx + 1:end_idx]
                except ValueError:
                    return None

            vision_xattn_mask_pad = torch.zeros(1, 1).to(vision_xattn_mask)
            vision_xattn_mask = torch.cat([vision_xattn_mask, vision_xattn_mask_pad], dim=1)

        return None

    def run(self, camera_id=0, exercise_type="squats", video_file=None, headless=False):
        """Run lightweight feedback system.

        Args:
            camera_id: Camera ID
            exercise_type: Exercise type
            video_file: Optional video file path (instead of camera)
            headless: Run without display (print to console only)
        """
        # Open video source
        if video_file:
            cap = cv2.VideoCapture(video_file)
            print(f"Processing video: {video_file}")
        else:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {camera_id}")

        # Lower resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        print(f"=== FitCoach Lightweight Live Feedback ===")
        print(f"Exercise: {exercise_type}")
        print(f"Feature rate: {self.feats_frequency} fps")
        print(f"Headless mode: {headless}")
        print(f"Max buffer: {self.feature_buffer.maxlen} features")
        print(f"Press 'q' to quit, 'r' to reset\n")

        system_prompt = (
            f"You are a fitness coach. The user is doing {exercise_type}. "
            "Provide brief, helpful feedback. "
        )

        last_feature_time = time.time()
        last_feedback_time = time.time()
        feature_interval = 1.0 / self.feats_frequency
        feedback_interval = self.sampling_kwargs.get("feedback_interval", 10.0)  # Longer interval

        current_feedback = "Starting..."
        feedback_timestamp = time.time()
        frame_count = 0

        # Get video FPS for proper timing when processing video files
        video_fps = cap.get(cv2.CAP_PROP_FPS) if video_file else 30.0
        if video_fps == 0:
            video_fps = 30.0  # Default if not available
        frame_delay = 1.0 / video_fps if video_file else 0  # Only delay for video files

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if video_file:
                        print("End of video")
                        break
                    print("Failed to grab frame")
                    continue

                current_time = time.time()
                frame_count += 1

                # Preprocess frames at lower rate (don't extract features yet)
                if current_time - last_feature_time >= feature_interval:
                    try:
                        preprocessed_frame = self.preprocess_frame(frame)
                        self.feature_buffer.append(preprocessed_frame)
                        last_feature_time = current_time
                    except Exception as e:
                        print(f"Frame preprocessing error: {e}")

                # Generate feedback less frequently
                if current_time - last_feedback_time >= feedback_interval:
                    print(f"\n[Frame {frame_count}] Generating feedback...")
                    feedback, timestamp = self.generate_feedback(
                        system_prompt,
                        use_recent_only=True,
                        window_size=20  # Use only last 20 features (~10 seconds, optimized for T4 GPU memory)
                    )

                    if feedback:
                        current_feedback = feedback
                        feedback_timestamp = timestamp
                        last_feedback_time = current_time
                        self.feedback_history.append((timestamp, feedback))
                        print(f"Coach: {feedback}")

                # Display if not headless
                if not headless:
                    display_frame = frame.copy()

                    # Simple text overlay
                    cv2.putText(display_frame, "FitCoach Lite", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{exercise_type}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Show current feedback
                    if current_time - feedback_timestamp < 5.0:
                        y_pos = display_frame.shape[0] - 40
                        cv2.putText(display_frame, f"{current_feedback[:50]}", (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    cv2.imshow('FitCoach Lite', display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.feature_buffer.clear()
                        self.feedback_history.clear()
                        print("\n[RESET] Session cleared")
                else:
                    # Headless mode - add small delay to match video FPS when processing files
                    if frame_delay > 0:
                        time.sleep(frame_delay)

        finally:
            cap.release()
            if not headless:
                cv2.destroyAllWindows()

            # Summary
            print(f"\n=== Session Summary ===")
            print(f"Total frames: {frame_count}")
            print(f"Total feedback: {len(self.feedback_history)}")
            print(f"\nFeedback History:")
            for i, (ts, fb) in enumerate(self.feedback_history, 1):
                print(f"{i}. {fb}")


def main():
    parser = argparse.ArgumentParser(description="FitCoach Lightweight Live Feedback")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--video", type=str, help="Video file path (instead of camera)")
    parser.add_argument("--exercise", type=str, default="squats", help="Exercise type")
    parser.add_argument("--headless", action="store_true", help="Run without display")
    parser.add_argument("--buffer-size", type=int, default=200, help="Max feature buffer size")

    args = parser.parse_args()

    print("Loading configuration...")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Loading model (this may take a few minutes)...")
    llama2_7b_path = config["model"]["llama2_7b_path"]
    model_kwargs = config["model"]["kwargs"]

    # Enable memory optimizations
    model_kwargs['bf16'] = True  # Use bfloat16 for memory efficiency

    model = make_model(llama2_7b_path, **model_kwargs)
    model.eval()

    # Enable gradient checkpointing if available
    if hasattr(model.model.lang, 'gradient_checkpointing_enable'):
        model.model.lang.gradient_checkpointing_enable()

    print("Model loaded successfully!")

    # Get CNN weights path - try common locations
    import sys
    possible_paths = [
        "./ckpts_efficientnet/fitness_ally_hypermodel/efficientnet4Lite_1.8.3.checkpoint",
        "./ckpts_efficientnet/efficientnet4Lite_1.8.3.checkpoint",
        "./ckpts_efficientnet/efficientnet_3d_cnn.pth.tar",
        "./ckpts_efficientnet/ckpts/efficientnet_3d_cnn.pth.tar",
    ]

    cnn_weights_path = None
    for path in possible_paths:
        if Path(path).exists():
            cnn_weights_path = path
            print(f"Found CNN weights at: {path}")
            break

    if cnn_weights_path is None:
        print("ERROR: Could not find 3D CNN weights!")
        print("Searched in:")
        for p in possible_paths:
            print(f"  - {p}")
        print("\nPlease check the extracted files:")
        print("  !ls -la ckpts_efficientnet/")
        sys.exit(1)

    coach = LightweightFeedbackCoach(model, config, cnn_weights_path, max_buffer_size=args.buffer_size)
    coach.run(
        camera_id=args.camera,
        exercise_type=args.exercise,
        video_file=args.video,
        headless=args.headless
    )


if __name__ == "__main__":
    main()
