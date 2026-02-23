# ForgeAI HeartMuLa — ComfyUI Custom Nodes

All-in-one music generation node for ComfyUI, powered by [HeartMuLa](https://heartmula.github.io/).

**Works on 16 GB VRAM GPUs** (RTX 4060 Ti, RTX 5070 Ti, etc.) with built-in 4-bit quantization.

## Nodes

### ForgeAI Music Generator
Generate music from lyrics and genre tags in a single node.

- **Inputs**: Lyrics, Tags (genre/style), Duration, Quantization, Temperature, Top-K, CFG Scale, Output Format
- **Outputs**: Audio (ComfyUI AUDIO type) + File Path (saved WAV/MP3)
- **Quantization**: none / 4-bit NF4 / 8-bit (select in node)

### ForgeAI Lyrics Transcriber
Transcribe lyrics from audio using Whisper.

- **Input**: Audio (ComfyUI AUDIO type)
- **Output**: Transcribed lyrics text

## Installation

### 1. Clone into ComfyUI custom_nodes
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ForgeAI-HeartMuLa.git
```

### 2. Install dependencies
```bash
pip install -r ForgeAI-HeartMuLa/requirements.txt
```

### 3. Download HeartMuLa models
Download from [HuggingFace](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B) and place in:
```
ComfyUI/models/HeartMuLa/
  ├── HeartMuLa-oss-3B/
  ├── HeartCodec-oss/
  ├── tokenizer.json
  └── gen_config.json
```

Or use our pre-quantized 4-bit checkpoint: [PavonicAI/HeartMuLa-3B-4bit](https://huggingface.co/PavonicAI/HeartMuLa-3B-4bit)

## Compatibility Fixes Included

All fixes are applied automatically — no manual patching needed:

| Issue | Status |
|---|---|
| `ignore_mismatched_sizes` (transformers 5.x) | Fixed |
| `RoPE cache not built` (torchtune >= 0.5) | Fixed |
| OOM on 16 GB GPUs | Fixed (model CPU offload) |
| `torchcodec` missing (torchaudio >= 2.10) | Fixed (uses soundfile) |

## Hardware Tested

- NVIDIA RTX 5070 Ti (16 GB) with 4-bit quantization

## License

Apache-2.0

## Credits

- Original HeartMuLa model by [HeartMuLa Team](https://heartmula.github.io/)
- ComfyUI nodes & compatibility fixes by [ForgeAI](https://huggingface.co/PavonicAI)
