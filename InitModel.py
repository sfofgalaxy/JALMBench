import os
import dotenv
from models.Qwen import Qwen
from models.DiVA import DiVA
from models.LLaMAOmni import LLaMAOmni
from models.Gemini import Gemini
from models.GPT import GPT
from models.FreezeOmni import FreezeOmni
from models.GLM4Voice import GLM4Voice
from models.SpeechGPT import SpeechGPT
from models.VITA_1 import VITA_1
from models.VITA_1_5 import VITA_1_5
from models.SalmonnModel import SalmonnModel
from models.SpiritLM import SpiritLM

def init_common_config():
    temperature = float(os.getenv('TEMPERATURE', 0.5))
    top_p = float(os.getenv('TOP_P', 1.0))
    max_new_tokens = int(os.getenv('MAX_NEW_TOKENS', 512))
    num_beams = int(os.getenv('NUM_BEAMS', 1))
    return {
        'temperature': temperature,
        'top_p': top_p,
        'max_new_tokens': max_new_tokens,
        'num_beams': num_beams,
    }

def get_model(model_name):
    dotenv.load_dotenv()
    common_config = init_common_config()
    if model_name == 'qwen':
        model = Qwen(config=common_config)
    elif model_name == 'fo':
        model = FreezeOmni(os.getenv('FREEZE_OMNI_MODEL_PATH'), config=common_config)
    elif model_name == 'lo':
        model = LLaMAOmni(os.getenv('LLAMA_OMNI_MODEL_PATH'), os.getenv('LLAMA_OMNI_VOCODER_PATH'), config=common_config)
    elif model_name == 'diva':
        model = DiVA(hf_token=os.getenv('HUGGING_FACE_TOKEN'), config=common_config)
    elif model_name == 'vita':
        model = VITA_1(model_path=os.getenv('VITA_MODEL_PATH'),
                     audio_encoder=os.getenv('VITA_AUDIO_ENCODER_PATH'),
                     vision_tower=os.getenv('VITA_VISION_TOWER_PATH'),
                     model_type=os.getenv('VITA_MODEL_TYPE'),
                     conv_mode=os.getenv('VITA_CONV_MODE'),
                     config=common_config)
    elif model_name == 'vita_1_5':
        model = VITA_1_5(model_path=os.getenv('VITA_1_5_MODEL_PATH'),
                     audio_encoder=os.getenv('VITA_1_5_AUDIO_ENCODER_PATH'),
                     vision_tower=os.getenv('VITA_1_5_VISION_TOWER_PATH'),
                     model_type=os.getenv('VITA_1_5_MODEL_TYPE'),
                     conv_mode=os.getenv('VITA_1_5_CONV_MODE'),
                     config=common_config)
    elif model_name == 'glm':
        model = GLM4Voice(os.getenv('GLM_DECODER_PATH'), config=common_config)
    elif model_name == 'salmonn':
        model = SalmonnModel(os.getenv('SALMONN_LLAMA_PATH'),
                       os.getenv('SALMONN_WHISPER_PATH'),
                       os.getenv('SALMONN_BEATS_PATH'),
                       os.getenv('SALMONN_CKPT_PATH'),
                       config=common_config)
    elif model_name == 'speechgpt':
        model = SpeechGPT(os.getenv('SPEECHGPT_CM_MODEL_PATH'),
                          os.getenv('SPEECHGPT_COM_MODEL_PATH'),
                          os.getenv('SPEECHGPT_S2U_PATH'),
                          os.getenv('SPEECHGPT_VOCODER_PATH'),
                          config=common_config)
    elif model_name == 'spirit':
        model = SpiritLM("spirit-lm-base-7b",
                        model_path=os.getenv('SPIRIT_LM_MODEL_PATH'), 
                        speech_encoder_path=os.getenv('SPIRIT_LM_SPEECH_TOKENIZER_PATH'),
                        config=common_config)
    elif model_name == 'gemini':
        model = Gemini(os.getenv('GEMINI_API_KEY'), os.getenv('GEMINI_MODEL_NAME'), config=common_config)
    elif model_name == 'gpt':
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
        model = GPT(endpoint, api_key, deployment, config=common_config)
    else:
        raise ValueError(f'Invalid model: {model_name}')
    return model
