# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Live Fitness Coaching Script - Real-time feedback from webcam input."""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.constants import FEEDBACK_BEGIN_TOKEN, FEEDBACK_END_TOKEN, VISION_TOKEN
from src.model_helpers import make_model


class LiveFeedbackCoach:
    """Real-time fitness coach that provides feedback from webcam input.

    This class captures video from a webcam, extracts features using the vision model,
    and generates live feedback using the Stream-VLM model.
    """

    def __init__(self, model, config):
        """Initialize the live feedback coach.

        Args:
            model: The Stream-VLM model wrapper
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.sampling_kwargs = config["evaluator"]["sampling_kwargs"]
        self.feats_frequency = self.sampling_kwargs.get("feats_frequency", 4)

        # Feature buffer to accumulate video features
        self.feature_buffer = deque(maxlen=1000)  # Store up to 1000 features (~4 minutes at 4fps)
        self.feedback_history = []

        # Special token IDs
        self.special_tokens_dict = {
            VISION_TOKEN: self.model.tokenizer.encode(VISION_TOKEN)[-1],
            FEEDBACK_BEGIN_TOKEN: self.model.tokenizer.encode(FEEDBACK_BEGIN_TOKEN)[-1],
            FEEDBACK_END_TOKEN: self.model.tokenizer.encode(FEEDBACK_END_TOKEN)[-1],
        }

    def extract_features_from_frame(self, frame):
        """Extract features from a single video frame.

        Args:
            frame: OpenCV BGR frame (H, W, 3)

        Returns:
            Feature tensor from the vision model
        """
        # Resize frame to model input size (assuming 224x224 like most vision models)
        frame_resized = cv2.resize(frame, (224, 224))

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to tensor
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0

        # Rearrange to (C, H, W) and add batch dimension
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

        # Move to device
        frame_tensor = frame_tensor.to(self.model.device)

        # Extract features using vision model
        with torch.no_grad():
            features = self.model.model.vision(frame_tensor)

        return features

    def generate_feedback(self, system_prompt):
        """Generate feedback based on accumulated features.

        Args:
            system_prompt: System prompt for the model

        Returns:
            Generated feedback text and timestamp
        """
        if len(self.feature_buffer) < 10:  # Need at least 10 frames (~2.5 seconds)
            return None, None

        # Stack features into a video tensor
        features_list = list(self.feature_buffer)

        # Features are already encoded, just stack them
        if isinstance(features_list[0], dict):
            # If features are dictionaries, extract the 'feats' key
            video_features = {
                'feats': torch.cat([f['feats'] for f in features_list], dim=1),
                'spatial_res': features_list[0].get('spatial_res', None)
            }
        else:
            video_features = torch.cat(features_list, dim=1)

        # Prepare input prompt
        input_prompt = system_prompt + VISION_TOKEN
        input_ids = self.model.tokenizer.encode(input_prompt)
        vision_xattn_mask = self._get_vision_xattn_mask(input_ids)
        vision_xattn_mask = [2 if tok == 1 else 0 for tok in vision_xattn_mask]

        # Generate feedback using interactive generation
        try:
            output = self._generate_single_feedback(
                video_features,
                torch.tensor(input_ids).unsqueeze(0).to(self.model.device),
                torch.tensor(vision_xattn_mask).unsqueeze(0).to(self.model.device),
            )

            if output is not None:
                # Extract feedback text
                feedback_text = self.model.tokenizer.decode(output, skip_special_tokens=True)
                current_time = time.time()
                return feedback_text, current_time
        except Exception as e:
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

    def _generate_single_feedback(self, encoded_video, input_ids, vision_xattn_mask):
        """Generate a single feedback using the model.

        This is a simplified version of the interactive generation that produces
        one feedback at a time rather than streaming through entire video.
        """
        output_ids = input_ids.clone()
        past_key_values = None

        max_tokens = self.sampling_kwargs.get("max_feedback_length", 128)
        do_sample = self.sampling_kwargs.get("do_sample", False)
        temperature = self.sampling_kwargs.get("temperature", 0.0)

        # Generate tokens until we get a feedback
        for _ in range(max_tokens):
            # Adapt video features
            multi_model_embedding = self.model.model.adapter(
                encoded_video, output_ids, vision_xattn_mask
            )

            # Generate next token
            lang_out = self.model.model.lang(
                inputs_embeds=multi_model_embedding,
                attention_mask=torch.ones_like(output_ids).to(self.model.device),
                use_cache=True,
                past_key_values=past_key_values,
            )

            past_key_values = lang_out["past_key_values"]

            # Sample next token
            if not do_sample:
                next_token = torch.argmax(lang_out["logits"][:, -1], dim=-1)
            else:
                scaled_logits = lang_out["logits"][:, -1] / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze()

            output_ids = torch.cat([output_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            # Check if we've completed a feedback
            if next_token.item() == self.special_tokens_dict[FEEDBACK_END_TOKEN]:
                # Extract the feedback between <answer> and <answer/>
                output_list = output_ids[0].cpu().tolist()
                try:
                    start_idx = output_list.index(self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN])
                    end_idx = len(output_list) - 1  # Current position
                    feedback_tokens = output_list[start_idx + 1:end_idx]
                    return feedback_tokens
                except ValueError:
                    return None

            # Update vision mask
            vision_xattn_mask_pad = torch.zeros(1, 1).to(vision_xattn_mask)
            vision_xattn_mask = torch.cat([vision_xattn_mask, vision_xattn_mask_pad], dim=1)

        return None

    def run(self, camera_id=0, exercise_type="squats"):
        """Run the live feedback system.

        Args:
            camera_id: Camera device ID (default 0 for built-in webcam)
            exercise_type: Type of exercise being performed
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"=== FitCoach Live Feedback System ===")
        print(f"Exercise: {exercise_type}")
        print(f"Camera: {camera_id}")
        print(f"Feature extraction rate: {self.feats_frequency} fps")
        print(f"Press 'q' to quit, 'r' to reset\n")

        # System prompt
        system_prompt = (
            "You are an expert fitness coaching AI who coaches users as they exercise. "
            f"The user is doing {exercise_type}. You assess their performance and "
            "proactively provide feedback. "
        )

        # Timing variables
        last_feature_time = time.time()
        last_feedback_time = time.time()
        feature_interval = 1.0 / self.feats_frequency
        feedback_interval = 5.0  # Generate feedback every 5 seconds

        current_feedback = "Starting session..."
        feedback_display_time = 3.0  # Display feedback for 3 seconds
        feedback_timestamp = time.time()

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                current_time = time.time()
                frame_count += 1

                # Extract features at specified frequency
                if current_time - last_feature_time >= feature_interval:
                    try:
                        features = self.extract_features_from_frame(frame)
                        self.feature_buffer.append(features)
                        last_feature_time = current_time
                    except Exception as e:
                        print(f"Error extracting features: {e}")

                # Generate feedback at specified interval
                if current_time - last_feedback_time >= feedback_interval:
                    print(f"\n[{frame_count}] Generating feedback...")
                    feedback, timestamp = self.generate_feedback(system_prompt)

                    if feedback:
                        current_feedback = feedback
                        feedback_timestamp = timestamp
                        last_feedback_time = current_time
                        self.feedback_history.append((timestamp, feedback))
                        print(f"Coach: {feedback}")

                # Display frame with feedback overlay
                display_frame = frame.copy()

                # Add info panel
                panel_height = 120
                panel = np.zeros((panel_height, display_frame.shape[1], 3), dtype=np.uint8)

                # Add text to panel
                cv2.putText(panel, "FitCoach Live", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(panel, f"Exercise: {exercise_type}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(panel, f"Frames: {frame_count} | Features: {len(self.feature_buffer)}",
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Display current feedback
                if current_time - feedback_timestamp < feedback_display_time:
                    # Wrap text if too long
                    max_chars = 60
                    if len(current_feedback) > max_chars:
                        feedback_line1 = current_feedback[:max_chars]
                        feedback_line2 = current_feedback[max_chars:max_chars*2]
                        cv2.putText(panel, f"Coach: {feedback_line1}", (10, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        if feedback_line2:
                            # Create additional panel space if needed
                            pass
                    else:
                        cv2.putText(panel, f"Coach: {current_feedback}", (10, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Combine panel and frame
                display_frame = np.vstack([panel, display_frame])

                # Show frame
                cv2.imshow('FitCoach Live Feedback', display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset session
                    self.feature_buffer.clear()
                    self.feedback_history.clear()
                    current_feedback = "Session reset..."
                    feedback_timestamp = time.time()
                    print("\n[RESET] Session cleared")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()

            # Print summary
            print(f"\n=== Session Summary ===")
            print(f"Total frames: {frame_count}")
            print(f"Total feedbacks generated: {len(self.feedback_history)}")
            print(f"\nFeedback History:")
            for i, (ts, feedback) in enumerate(self.feedback_history, 1):
                print(f"{i}. [{time.strftime('%H:%M:%S', time.localtime(ts))}] {feedback}")


def main():
    """Main entry point for live feedback script."""
    parser = argparse.ArgumentParser(description="FitCoach Live Feedback System")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the yaml config file"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default="squats",
        help="Type of exercise being performed (default: squats)"
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load model
    print("Loading model (this may take a few minutes)...")
    llama2_7b_path = config["model"]["llama2_7b_path"]
    model_kwargs = config["model"]["kwargs"]
    model = make_model(llama2_7b_path, **model_kwargs)
    model.eval()
    print("Model loaded successfully!")

    # Create live feedback coach
    coach = LiveFeedbackCoach(model, config)

    # Run live feedback
    coach.run(camera_id=args.camera, exercise_type=args.exercise)


if __name__ == "__main__":
    main()
