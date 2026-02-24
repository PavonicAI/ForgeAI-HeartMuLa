"""
ForgeAI HeartMuLa — ComfyUI Custom Nodes
All-in-one music generation + lyrics transcription for 16 GB GPUs.

Fixes included:
  - transformers 5.x compatibility (ignore_mismatched_sizes)
  - torchtune >= 0.5 RoPE init fix
  - OOM fix: model.cpu() offload before codec decode
  - soundfile instead of torchaudio (no torchcodec needed)
  - bitsandbytes 4-bit/8-bit quantization support
  - Pre-quantized 4-bit checkpoint auto-detection
  - Fade-out post-processing
"""

import os
import sys
import torch
import numpy as np
import folder_paths

# ---------------------------------------------------------------------------
# Model path setup
# ---------------------------------------------------------------------------
HEARTMULA_DIR = os.path.join(folder_paths.models_dir, "HeartMuLa")

# ---------------------------------------------------------------------------
# Scan for available HeartMuLa models
# ---------------------------------------------------------------------------
def _scan_model_folders():
    """Scan HeartMuLa directory for available model folders."""
    if not os.path.exists(HEARTMULA_DIR):
        return ["HeartMuLa-oss-3B"]

    folders = []
    skip = {"HeartCodec-oss", "heartlib", "HeartMuLa-oss-whisper"}
    for name in sorted(os.listdir(HEARTMULA_DIR)):
        path = os.path.join(HEARTMULA_DIR, name)
        if os.path.isdir(path) and name not in skip:
            # Check if it looks like a model folder (has config.json)
            if os.path.exists(os.path.join(path, "config.json")):
                folders.append(name)

    return folders if folders else ["HeartMuLa-oss-3B"]


# ---------------------------------------------------------------------------
# Patch heartlib on import: RoPE init fix for torchtune >= 0.5
# ---------------------------------------------------------------------------
def _apply_heartlib_patches():
    """Monkey-patch HeartMuLa.setup_caches to call rope_init() for torchtune >= 0.5"""
    try:
        from heartlib.heartmula.modeling_heartmula import HeartMuLa

        _original_setup_caches = HeartMuLa.setup_caches

        def _patched_setup_caches(self, batch_size, *args, **kwargs):
            _original_setup_caches(self, batch_size, *args, **kwargs)
            # RoPE init fix: torchtune >= 0.5 requires explicit rope_init()
            for m in self.modules():
                if hasattr(m, "rope_init"):
                    m.rope_init()
                    # Move RoPE buffers to same device as model
                    try:
                        p = next(self.parameters())
                        m.to(p.device)
                    except StopIteration:
                        pass

        HeartMuLa.setup_caches = _patched_setup_caches
        print("[ForgeAI] HeartMuLa RoPE patch applied")
    except ImportError:
        print("[ForgeAI] WARNING: heartlib not found, install with: pip install heartlib")


_apply_heartlib_patches()


# ---------------------------------------------------------------------------
# Pipeline cache (keep models loaded between runs)
# ---------------------------------------------------------------------------
_pipeline_cache = {}


def _get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_pipeline(model_folder, quantize="none"):
    """Load HeartMuLa generation pipeline with optional quantization.

    model_folder: name of the folder inside HeartMuLa/ (e.g. HeartMuLa-4bit-3B)
    """
    cache_key = f"{model_folder}_{quantize}"
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    from tokenizers import Tokenizer
    from heartlib.heartmula.modeling_heartmula import HeartMuLa
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec

    device = _get_device()
    dtype = torch.bfloat16

    # --- Load HeartCodec ---
    codec_path = os.path.join(HEARTMULA_DIR, "HeartCodec-oss")
    if not os.path.exists(codec_path):
        raise FileNotFoundError(
            f"HeartCodec not found at {codec_path}. "
            f"Download from https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B"
        )
    print(f"[ForgeAI] Loading HeartCodec from {codec_path}")
    codec = HeartCodec.from_pretrained(
        codec_path, device_map=device, ignore_mismatched_sizes=True
    )

    # --- Determine model path and quantization config ---
    model_path = os.path.join(HEARTMULA_DIR, model_folder)
    bnb_config = None
    device_map = None

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Download from https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B "
            f"or https://huggingface.co/PavonicAI/HeartMuLa-3B-4bit"
        )

    # Check if the model folder has a baked-in quantization config
    import json
    config_path = os.path.join(model_path, "config.json")
    has_baked_quant = False
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            model_cfg = json.load(f)
        if "quantization_config" in model_cfg:
            has_baked_quant = True
            print(f"[ForgeAI] Model has baked-in quantization config")

    if has_baked_quant:
        # Pre-quantized checkpoint — just load it, config.json has everything
        device_map = "cuda:0"
        print(f"[ForgeAI] Loading pre-quantized checkpoint: {model_folder}")
    elif quantize == "4bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        device_map = "cuda:0"
        print(f"[ForgeAI] Quantizing on-the-fly (4bit): {model_folder}")
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        device_map = "cuda:0"
        print(f"[ForgeAI] Quantizing on-the-fly (8bit): {model_folder}")
    else:
        print(f"[ForgeAI] Loading full precision: {model_folder}")

    print(f"[ForgeAI] Loading {model_folder} ({quantize})...")
    model = HeartMuLa.from_pretrained(
        model_path,
        dtype=dtype,
        quantization_config=bnb_config,
        device_map=device_map,
        ignore_mismatched_sizes=True,
    )

    # --- Load tokenizer + config ---
    tokenizer_path = os.path.join(HEARTMULA_DIR, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    gen_config_path = os.path.join(HEARTMULA_DIR, "gen_config.json")
    if not os.path.exists(gen_config_path):
        raise FileNotFoundError(f"gen_config.json not found at {gen_config_path}")

    with open(gen_config_path, encoding="utf-8") as f:
        gen_cfg = json.load(f)

    pipeline_data = {
        "model": model,
        "codec": codec,
        "tokenizer": tokenizer,
        "gen_config": gen_cfg,
        "device": device,
        "dtype": dtype,
        "num_quantizers": codec.config.num_quantizers,
        "muq_dim": model.config.muq_dim,
    }

    _pipeline_cache[cache_key] = pipeline_data
    print(f"[ForgeAI] Pipeline ready ({quantize})")
    return pipeline_data


# ---------------------------------------------------------------------------
# Generation logic (all fixes baked in)
# ---------------------------------------------------------------------------
def _generate_music(
    pipeline_data,
    tags: str,
    lyrics: str,
    max_duration_sec: int = 30,
    temperature: float = 1.0,
    top_k: int = 50,
    cfg_scale: float = 1.5,
):
    """Generate music from lyrics. Returns (waveform_tensor, sample_rate)."""
    from comfy.utils import ProgressBar

    model = pipeline_data["model"]
    codec = pipeline_data["codec"]
    tokenizer = pipeline_data["tokenizer"]
    gen_cfg = pipeline_data["gen_config"]
    device = pipeline_data["device"]
    dtype = pipeline_data["dtype"]
    parallel_number = pipeline_data["num_quantizers"] + 1
    muq_dim = pipeline_data["muq_dim"]

    text_bos_id = gen_cfg.get("text_bos_id", 128000)
    text_eos_id = gen_cfg.get("text_eos_id", 128001)
    audio_eos_id = gen_cfg.get("audio_eos_id", 8193)
    empty_id = gen_cfg.get("empty_id", 0)

    # --- Preprocess tags ---
    tags = tags.lower().strip()
    if not tags.startswith("<tag>"):
        tags = f"<tag>{tags}"
    if not tags.endswith("</tag>"):
        tags = f"{tags}</tag>"

    tags_ids = tokenizer.encode(tags).ids
    if tags_ids[0] != text_bos_id:
        tags_ids = [text_bos_id] + tags_ids
    if tags_ids[-1] != text_eos_id:
        tags_ids = tags_ids + [text_eos_id]

    # --- Preprocess lyrics ---
    lyrics = lyrics.lower().strip()
    lyrics_ids = tokenizer.encode(lyrics).ids
    if lyrics_ids[0] != text_bos_id:
        lyrics_ids = [text_bos_id] + lyrics_ids
    if lyrics_ids[-1] != text_eos_id:
        lyrics_ids = lyrics_ids + [text_eos_id]

    # --- Build prompt tokens ---
    muq_embed = torch.zeros([muq_dim], dtype=dtype)
    muq_idx = len(tags_ids)
    prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

    tokens = torch.zeros([prompt_len, parallel_number], dtype=torch.long)
    tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
    tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

    tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
    tokens_mask[:, -1] = True

    # CFG 1.0 causes batch_size mismatch in attention layers
    # Clamp to minimum 1.1 to always use batch_size=2
    if cfg_scale <= 1.0:
        print(f"[ForgeAI] CFG {cfg_scale} clamped to 1.1 (minimum for stable generation)")
        cfg_scale = 1.1
    bs_size = 2

    def _cfg_cat(tensor, cfg_s):
        tensor = tensor.unsqueeze(0)
        if cfg_s != 1.0:
            tensor = torch.cat([tensor, tensor], dim=0)
        return tensor

    prompt_tokens = _cfg_cat(tokens, cfg_scale).to(device)
    prompt_tokens_mask = _cfg_cat(tokens_mask, cfg_scale).to(device)
    continuous_segment = _cfg_cat(muq_embed, cfg_scale).to(device)
    starts = [muq_idx] * bs_size
    prompt_pos = _cfg_cat(
        torch.arange(prompt_len, dtype=torch.long), cfg_scale
    ).to(device)

    # --- Generate frames with ComfyUI progress bar ---
    max_audio_frames = (max_duration_sec * 1000) // 80
    frames = []

    model.setup_caches(bs_size)

    with torch.autocast(device_type=device.type, dtype=dtype):
        curr_token = model.generate_frame(
            tokens=prompt_tokens,
            tokens_mask=prompt_tokens_mask,
            input_pos=prompt_pos,
            temperature=temperature,
            topk=top_k,
            cfg_scale=cfg_scale,
            continuous_segments=continuous_segment,
            starts=starts,
        )
    frames.append(curr_token[0:1])

    print(f"[ForgeAI] Generating {max_duration_sec}s of audio...")
    pbar = ProgressBar(max_audio_frames)
    for i in range(max_audio_frames):
        padded_token = (
            torch.ones(
                (curr_token.shape[0], parallel_number),
                device=curr_token.device,
                dtype=torch.long,
            )
            * empty_id
        )
        padded_token[:, :-1] = curr_token
        padded_token = padded_token.unsqueeze(1)
        padded_token_mask = torch.ones_like(padded_token, dtype=torch.bool)
        padded_token_mask[..., -1] = False

        with torch.autocast(device_type=device.type, dtype=dtype):
            curr_token = model.generate_frame(
                tokens=padded_token,
                tokens_mask=padded_token_mask,
                input_pos=prompt_pos[..., -1:] + i + 1,
                temperature=temperature,
                topk=top_k,
                cfg_scale=cfg_scale,
                continuous_segments=None,
                starts=None,
            )
        if torch.any(curr_token[0:1, :] >= audio_eos_id):
            print(f"[ForgeAI] Audio EOS reached at frame {i}")
            break
        frames.append(curr_token[0:1])
        pbar.update(1)

    # --- Decode audio (with OOM fix: offload model first) ---
    frames_tensor = torch.stack(frames).permute(1, 2, 0).squeeze(0)
    model.reset_caches()
    model.cpu()
    torch.cuda.empty_cache()
    print("[ForgeAI] Model offloaded, decoding audio...")

    wav = codec.detokenize(frames_tensor)

    # Move model back to GPU for next run
    try:
        model.to(device)
    except Exception:
        pass  # quantized models may not support .to()

    return wav, 48000


# ---------------------------------------------------------------------------
# Post-processing: fade out
# ---------------------------------------------------------------------------
def _apply_fade_out(wav_np, sample_rate, fade_seconds=3.0):
    """Apply a smooth fade-out to the last N seconds of audio."""
    fade_samples = int(fade_seconds * sample_rate)
    if fade_samples >= len(wav_np):
        fade_samples = len(wav_np)

    # Create linear fade curve
    fade_curve = np.linspace(1.0, 0.0, fade_samples)

    if wav_np.ndim == 2:
        # Multi-channel: apply to all channels
        wav_np[-fade_samples:] *= fade_curve[:, np.newaxis]
    else:
        wav_np[-fade_samples:] *= fade_curve

    return wav_np


# ---------------------------------------------------------------------------
# Save audio to file (WAV or MP3) — uses PyAV for MP3 (no ffmpeg needed)
# ---------------------------------------------------------------------------
def _save_audio(wav_tensor, sample_rate, save_path, output_format="wav",
                fade_out_seconds=0.0):
    """Save audio tensor to file. Uses PyAV for MP3 encoding."""
    import soundfile as sf

    wav_np = wav_tensor.cpu().float().numpy()
    if wav_np.ndim == 2:
        wav_np = wav_np.T  # soundfile expects (samples, channels)

    # Apply fade-out if requested
    if fade_out_seconds > 0:
        wav_np = _apply_fade_out(wav_np, sample_rate, fade_out_seconds)
        print(f"[ForgeAI] Applied {fade_out_seconds}s fade-out")

    if output_format == "mp3":
        try:
            import av

            # Encode MP3 using PyAV (bundled with ComfyUI, no external ffmpeg needed)
            container = av.open(save_path, mode="w")
            channels = wav_np.shape[1] if wav_np.ndim == 2 else 1
            stream = container.add_stream("libmp3lame", rate=sample_rate)
            stream.bit_rate = 320000  # 320 kbps
            stream.layout = "stereo" if channels > 1 else "mono"

            # PyAV expects (samples, channels) float32
            if wav_np.ndim == 1:
                wav_np = wav_np.reshape(-1, 1)

            # Process in chunks to avoid memory issues
            chunk_size = sample_rate  # 1 second chunks
            for start in range(0, len(wav_np), chunk_size):
                chunk = wav_np[start : start + chunk_size]
                frame = av.AudioFrame.from_ndarray(
                    chunk.T,  # PyAV expects (channels, samples)
                    format="fltp",
                    layout=stream.layout.name,
                )
                frame.sample_rate = sample_rate
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush encoder
            for packet in stream.encode(None):
                container.mux(packet)
            container.close()
            print(f"[ForgeAI] Saved MP3: {save_path}")

        except Exception as e:
            # Fallback to WAV if MP3 encoding fails
            print(f"[ForgeAI] MP3 encoding failed ({e}), saving as WAV")
            save_path = save_path.rsplit(".", 1)[0] + ".wav"
            sf.write(save_path, wav_np, sample_rate)
            print(f"[ForgeAI] Saved WAV: {save_path}")
    else:
        sf.write(save_path, wav_np, sample_rate)
        print(f"[ForgeAI] Saved WAV: {save_path}")

    return save_path


# ===========================================================================
# ComfyUI Node: ForgeAI Music Generator
# ===========================================================================
class ForgeAI_HeartMuLa_Generate:
    """
    All-in-one HeartMuLa music generation node.
    Generates music from lyrics + genre tags. Supports 4-bit quantization
    for 16 GB GPUs. Saves output as WAV or MP3.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        model_folders = _scan_model_folders()
        return {
            "required": {
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[verse]\nHello world\nThis is a song\n\n[chorus]\nLa la la",
                    },
                ),
                "tags": (
                    "STRING",
                    {
                        "default": "warm, Piano, pop, acoustic guitar",
                    },
                ),
                "model": (model_folders,),
                "quantize": (["4bit", "8bit", "none"],),
                "duration_seconds": (
                    "INT",
                    {"default": 120, "min": 5, "max": 240, "step": 5},
                ),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05},
                ),
                "top_k": (
                    "INT",
                    {"default": 50, "min": 1, "max": 500, "step": 1},
                ),
                "fade_out": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5},
                ),
                "output_format": (["wav", "mp3"],),
                "filename_prefix": (
                    "STRING",
                    {"default": "ForgeAI_music"},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "generate"
    CATEGORY = "ForgeAI/Audio"
    OUTPUT_NODE = True

    def generate(
        self,
        lyrics,
        tags,
        model,
        quantize,
        duration_seconds,
        cfg_scale,
        temperature,
        top_k,
        fade_out,
        output_format,
        filename_prefix,
    ):
        # Load pipeline (model is now folder name directly)
        pipeline = _load_pipeline(model, quantize)

        # Generate audio
        wav, sample_rate = _generate_music(
            pipeline,
            tags=tags,
            lyrics=lyrics,
            max_duration_sec=duration_seconds,
            temperature=temperature,
            top_k=top_k,
            cfg_scale=cfg_scale,
        )

        # Build output path
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ext = output_format
        filename = f"{filename_prefix}_{timestamp}.{ext}"
        save_path = os.path.join(self.output_dir, filename)

        # Save file (with optional fade-out)
        actual_path = _save_audio(wav, sample_rate, save_path, output_format,
                                  fade_out_seconds=fade_out)
        actual_filename = os.path.basename(actual_path)

        # Build ComfyUI AUDIO output
        wav_np = wav.cpu().float().numpy()
        waveform = torch.from_numpy(wav_np)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T
        audio_output = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}

        # Return with UI dict so ComfyUI shows audio preview in assets
        return {
            "ui": {
                "audio": [
                    {
                        "filename": actual_filename,
                        "subfolder": "",
                        "type": "output",
                    }
                ]
            },
            "result": (audio_output, actual_path),
        }


# ===========================================================================
# ComfyUI Node: ForgeAI Lyrics Transcriber
# ===========================================================================
class ForgeAI_HeartMuLa_Transcribe:
    """
    Transcribe lyrics from audio using HeartMuLa's Whisper-based transcriber.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "transcribe"
    CATEGORY = "ForgeAI/Audio"

    def transcribe(self, audio):
        import soundfile as sf
        import tempfile

        # Extract waveform from ComfyUI AUDIO type
        waveform = audio["waveform"].squeeze(0)  # remove batch dim
        sample_rate = audio["sample_rate"]

        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            wav_np = waveform.cpu().float().numpy()
            if wav_np.ndim == 2:
                wav_np = wav_np.T
            sf.write(tmp_path, wav_np, sample_rate)

        try:
            # Load Whisper model
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            import librosa

            whisper_path = os.path.join(HEARTMULA_DIR, "HeartMuLa-oss-whisper")
            if not os.path.exists(whisper_path):
                # Fall back to standard whisper
                whisper_path = "openai/whisper-large-v3"
                print(f"[ForgeAI] HeartMuLa whisper not found, using {whisper_path}")

            processor = WhisperProcessor.from_pretrained(whisper_path)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(
                whisper_path,
                ignore_mismatched_sizes=True,
            ).to(_get_device())

            # Load and resample audio to 16kHz for Whisper
            audio_data, sr = librosa.load(tmp_path, sr=16000)
            input_features = processor(
                audio_data, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(_get_device())

            # Transcribe
            predicted_ids = whisper_model.generate(input_features)
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            # Clean up whisper model
            del whisper_model
            torch.cuda.empty_cache()

            print(f"[ForgeAI] Transcribed: {transcription[:100]}...")
            return (transcription,)

        finally:
            os.unlink(tmp_path)


# ===========================================================================
# ComfyUI Registration
# ===========================================================================
NODE_CLASS_MAPPINGS = {
    "ForgeAI_HeartMuLa_Generate": ForgeAI_HeartMuLa_Generate,
    "ForgeAI_HeartMuLa_Transcribe": ForgeAI_HeartMuLa_Transcribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForgeAI_HeartMuLa_Generate": "ForgeAI Music Generator (HeartMuLa)",
    "ForgeAI_HeartMuLa_Transcribe": "ForgeAI Lyrics Transcriber",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
