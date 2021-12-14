# from https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
from jiwer import wer


librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", output_layer_index=24).to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

librispeech_eval = librispeech_eval.map(map_to_array)

def map_to_pred(batch):
    inputs = processor(batch["speech"], return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = librispeech_eval.map(map_to_pred, batched=True, batch_size=20, remove_columns=["speech"])
print(result["transcription"])
print("WER:", wer(result["text"], result["transcription"]))
