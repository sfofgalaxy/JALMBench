from __future__ import print_function

import torch
import torchaudio
import math

import soundfile as sf
import torchaudio.compliance.kaldi as k

from .AudioLM import AudioLM
from .FO.pipeline import inferencePipeline
from .FO.decoder.llm2tts import llm2TTS


class audioEncoderProcessor:
    def __init__(self, chunk_size = 16):
        self.chunk_size = chunk_size
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_size = 400
        self.frame_shift = 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.CHUNK = self.frame_shift * self.chunk_size
        self.reset()
    
    def get_chunk_size(self):
        return self.CHUNK
    
    def reset(self):
        self.input_chunk = torch.zeros([1, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.input_sample = torch.zeros([1, self.CHUNK + self.frame_overlap , 1])
    
    def fbank_shift(self, sample_data):
        # fbank feature shift
        self.input_sample[:, :self.frame_overlap , :] = self.input_sample[:, -self.frame_overlap:, :].clone()
        self.input_sample[:, self.frame_overlap:, :] = sample_data
    
    def chunk_data_shift(self, xs):
        # chunk feature shift
        self.input_chunk[:, :self.chunk_overlap, :] = self.input_chunk[:, -self.chunk_overlap:, :].clone()
        self.input_chunk[:, self.chunk_overlap:, :] = xs.squeeze(0)
    
    def process(self,
                audio: torch.Tensor):
        with torch.no_grad():
            sample_data = torch.tensor(audio).reshape(1, -1, 1)[:, :, :1] * 32768
            self.fbank_shift(sample_data)
            # use kaldi api to compute fbank
            xs = k.fbank(waveform = self.input_sample.squeeze(-1), dither=0, 
                         frame_length=25, frame_shift=10, num_mel_bins=self.feat_dim)
            self.chunk_data_shift(xs)
        return self.input_chunk.clone()

class FreezeOmni(AudioLM):
    def __init__(self, model_path: str, config = {}):
        super().__init__()
        self.model_path = model_path
        self.llm_path = "Qwen/Qwen2-7B-Instruct"
        self.top_k = 5
        self.top_p = 0.8
        self.temperature = 0.7
        self.max_new_tokens = config.get('max_new_tokens', 512)
        self.load_model()
    
    def load_model(self):
        """Load models and processors"""
        configs = type('Config', (), {
            'model_path': self.model_path,
            'llm_path': self.llm_path,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'temperature': self.temperature
        })()
        
        self.pipeline = inferencePipeline(configs)
        self.tts = llm2TTS(self.model_path)
        self.audio_processor = audioEncoderProcessor()

    def process_audio(self, audio_path: str, prompt=None, **kwargs) -> str:
        """Process audio file and return text response"""
        # Read audio file
        if audio_path is None:
            fs = 16000
            wav = torch.zeros(fs * 1, dtype=torch.float32)
        else:
            wav, fs = sf.read(audio_path)
            wav = torch.tensor(wav)
        if fs != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(wav.float())
        
        # Initialize parameters
        codec_chunk_size = 40
        codec_padding_size = 10
        decoder_topk = 2
        chunk_size = self.audio_processor.get_chunk_size()
        
        # Stage0: Preprocessing
        outputs = self.pipeline.speech_dialogue(None, stat='pre', role=prompt if prompt else "You are a helpful assistant.")
        
        # Stage1: Process audio
        wav_input = torch.zeros(math.ceil(wav.shape[0] / chunk_size) * chunk_size)
        wav_input[:wav.shape[0]] = wav
        
        for i in range(0, wav_input.shape[0], chunk_size):
            fbank = self.audio_processor.process(wav_input[i:i+chunk_size])
            outputs = self.pipeline.speech_dialogue(fbank, **outputs)
            outputs['stat'] = 'cl'
        
        # Reset and prepare for next stage
        self.audio_processor.reset()
        outputs.update({
            'adapter_cache': None,
            'encoder_cache': None,
            'pe_index': 0,
            'stat': 'ss'
        })
        
        # Stage3-4: Generate response
        outputs = self.pipeline.speech_dialogue(None, **outputs)
        whole_text = ''
        last_text = ''
        cur_text = ''
        wav_output = []
        cur_hidden_state = [outputs['hidden_state']]
        
        while True:
            if len(outputs['past_tokens']) > self.max_new_tokens:
                break
                
            del outputs['text']
            del outputs['hidden_state']
            outputs = self.pipeline.speech_dialogue(None, **outputs)
            
            if outputs['stat'] == 'cs':
                cur_hidden_state.append(outputs['hidden_state'])
                whole_text += outputs['text'][len(last_text):]
                cur_text += outputs['text'][len(last_text):]
                
                # Check if audio generation is needed
                if outputs['text'][len(last_text):].endswith(("。", "：", "？", "！", ".", "?", "!", "\n")):
                    if len(cur_hidden_state) > 0:
                        self._decode_audio(cur_hidden_state, cur_text, codec_chunk_size, 
                                        codec_padding_size, decoder_topk, wav_output)
                        cur_hidden_state = []
                    cur_text = ""
                    
            if outputs['stat'] == 'sl':
                break
            last_text = outputs['text']
        
        # Process remaining text
        if len(cur_hidden_state) > 0:
            self._decode_audio(cur_hidden_state, cur_text, codec_chunk_size, 
                             codec_padding_size, decoder_topk, wav_output)
        
        # Output audio if needed
        output_audio_path = kwargs.get("output_audio_path", None)
        if output_audio_path and wav_output:
            sf.write(output_audio_path, 
                    torch.cat(wav_output, -1).squeeze().float().cpu().numpy(), 
                    24000)
        
        return whole_text
    
    def _decode_audio(self, hidden_states, text, chunk_size, padding_size, topk, wav_output):
        """Helper method for audio decoding"""
        hidden_state_output = torch.cat(hidden_states).squeeze(1)
        text_processed = self.pipeline.post_process(text)
        tokenized_text = torch.tensor(self.pipeline.model.tokenizer.encode(text_processed)).cuda().long()
        
        if tokenized_text.numel() == 0:
            return  # Skip if tokenized_text is empty
        
        embeddings = self.pipeline.model.llm_decoder.model.embed_tokens(tokenized_text)
        
        if embeddings.numel() == 0 or hidden_state_output.numel() == 0:
            return  # Skip if embeddings or hidden_state_output is empty
        
        for seg in self.tts.run(embeddings.reshape(-1, 896).unsqueeze(0),
                                topk,
                                hidden_state_output.reshape(-1, 896).unsqueeze(0),
                                chunk_size,
                                padding_size):
            wav_output.append(seg)

