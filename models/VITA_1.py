from typing import Optional
import torch
from .AudioLM import AudioLM
from .vita.model.builder import load_pretrained_model
from .vita.conversation import conv_templates
from .vita.constants import DEFAULT_AUDIO_TOKEN, IMAGE_TOKEN_INDEX
from .vita.util.mm_utils import KeywordsStoppingCriteria, tokenizer_image_audio_token, tokenizer_image_token
from .vita.util.utils import disable_torch_init

class VITA_1(AudioLM):
    def __init__(self, model_path: str, audio_encoder: str, vision_tower: str,
                 model_type: str = "mixtral-8x7b", conv_mode: str = "mixtral_two", 
                 config = {}):
        super().__init__()
        self.model_path = model_path
        self.audio_encoder = audio_encoder
        self.vision_tower = vision_tower
        self.model_type = model_type
        self.conv_mode = conv_mode
        self.do_sample = False
        self.use_cache = True
        
        self.max_new_tokens = config.get('max_new_tokens', 512)
        self.max_new_tokens = 256
        self.temperature = config.get('temperature', 0.5)
        self.top_p = config.get('top_p', 1.0)
        self.num_beams = config.get('num_beams', 1)
        
        self.load_model()
        
    def load_model(self):
        disable_torch_init()
        self.tokenizer, self.model, image_processor, context_len = load_pretrained_model(
            self.model_path, 
            self.model_type,
            self.audio_encoder,
            self.vision_tower
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.audio_encoder = self.model.get_audio_encoder()
        self.audio_encoder.to(device='cuda', dtype=torch.float16)
        self.processor = self.audio_encoder.audio_processor
        self.model.eval()
        
    def process_audio(self, audio_path: str, prompt: str = "", addtional_system_prompt: str = "", **kwargs) -> str:
        if not audio_path and not prompt:
            raise ValueError("Either audio_path or prompt must be provided")
        qs = prompt
        if audio_path is not None:
            audio, _ = self.processor.process(audio_path)
        else:
            # Create a 1-second silent audio with a sample rate of 16000
            sample_rate = 16000
            silent_audio = np.zeros(sample_rate * 1)
            os.makedirs('temp', exist_ok=True)
            audio_path = 'temp/none.wav'
            sf.write(audio_path, silent_audio, sample_rate)

        audio_length = audio.shape[0]
        # Ensure audio tensor size matches the expected dimensions (match the attention length (5000, which is 20000/4))
        expected_length = 20000  # Replace with the expected length
        if audio_length > expected_length:
            audio = audio[-expected_length-1:-1]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audios = {
            "audios": audio.half().cuda(),
            "lengths": audio_length.half().cuda()
        }
        
        image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=self.model.dtype, device="cuda")
        if audio_path:
            qs = (qs + DEFAULT_AUDIO_TOKEN) if qs else DEFAULT_AUDIO_TOKEN
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        sys_prompt = conv.get_prompt("lang")
        # note: this is the specific method of adding additional system prompt for the template of vita_1 with backbone LLM of mixtral-8x7b
        if addtional_system_prompt and addtional_system_prompt != "":
            sys_prompt_contents = sys_prompt.split("</s>\nuser:")
            sys_prompt_content = sys_prompt_contents[0]
            suffix = "</s>\nuser:" + sys_prompt_contents[1]
            sys_prompt = sys_prompt_content + " " + addtional_system_prompt + suffix

        if audio_path:
            input_ids = (
                tokenizer_image_audio_token(sys_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
        else:
            input_ids = (
                tokenizer_image_token(sys_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
        stop_str = conv.sep if conv.sep_style != "TWO" else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                audios=audios,
                do_sample=self.do_sample,
                temperature=self.temperature,
                num_beams=self.num_beams,
                top_p=self.top_p,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=self.max_new_tokens,
                use_cache=self.use_cache,
                stopping_criteria=[stopping_criteria]
            )

        output_ids = output_ids.sequences
        input_token_len = input_ids.shape[1]
        decode = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)
        outputs = decode[0]
        
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
            
        return outputs.strip() 