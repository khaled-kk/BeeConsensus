---
language: en
license: apache-2.0
tags:
  - minicpm
  - minicpm-o
  - multimodal
  - vision
  - audio
  - text-to-speech
  - tiny-model
  - random-weights
model-index:
  - name: MiniCPM-o-2_6
    results: []
---

# Tiny Random MiniCPM-o-2_6

This is a tiny random variant of the MiniCPM-o-2_6 model, specifically created for testing and validation purposes with Optimum-Intel.

## Model Details

- **Model Type**: Multimodal (Vision + Audio + Text-to-Speech)
- **Original Model**: [openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6)
- **Size**: ~4.4MB (significantly reduced from original 160MB)
- **Architecture**: MiniCPMO (multimodal architecture)
- **Purpose**: Testing and validation with Optimum-Intel

## Configuration Changes

This tiny model has the following modifications from the original:

- `hidden_size`: 16 (reduced from original)
- `intermediate_size`: 32 (reduced from original)
- `num_hidden_layers`: 1 (reduced from original)
- `vocab_size`: 32000 (maintained for compatibility)
- Vision and Audio components significantly reduced in size

## Usage

```python
from transformers import AutoModel, AutoTokenizer, AutoProcessor

model = AutoModel.from_pretrained("your-username/tiny-random-MiniCPM-o-2_6", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("your-username/tiny-random-MiniCPM-o-2_6", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("your-username/tiny-random-MiniCPM-o-2_6", trust_remote_code=True)
```

## Validation

This model has been tested and validated with Optimum-Intel tests:
- ✅ `test_compare_to_transformers_14_minicpmo`
- ✅ `test_generate_utils_14_minicpmo`  
- ✅ `test_model_can_be_loaded_after_saving_14_minicpmo`

## Limitations

⚠️ **Important**: This is a random weights model intended for testing purposes only. It will not produce meaningful outputs and should not be used for production applications.

## License

Apache 2.0