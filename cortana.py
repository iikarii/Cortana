print("Importing dependencies...")

import re
import os
import sounddevice
import soundfile
import torch
import torchaudio
import pyautogui
import ollama
from datetime import datetime
from pathlib import Path


WARNINGS = [


    "No LLM model loaded.\n"
    'Enter: -persona "model name" to load a langauge model.',
    "No XTTS model loaded.\n"
    'Enter: -voice "model name" to load a speech model.',


    "doesn't take any arguements",
    "doesn't take those arguements",
    "not in the function dictionary",

]

AUDIO_DEVICE = 'Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO), Windows DirectSound'

#LLM global variables 
model    = ''
messages = []
images   = []
stream   = True
raw      = False
options  = {

    "num_keep":24,
    #"seed": 42,
    "num_predict":200,
    "top_k":20,
    "top_p":0.40,
    "tfs_z":0.5,
    "typical_p":0.2,
    "repeat_last_n":1000,
    "temperature":0.1,
    "repeat_penalty":1.8,
    "presence_penalty":1.3,
    "frequency_penalty":1.3,
    #"mirostat": 1,
    #"mirostat_tau": 0.8,
    #"mirostat_eta": 0.6,
    #"penalize_newline": True,
    #"stop": [],
    #"numa": False,
    "num_ctx": 1000,
    #"num_batch": 2,
    #"num_gpu": 1,
    #"main_gpu": 0,
    #"low_vram": False,
    #"f16_kv": True,
    #"vocab_only": False,
    #"use_mmap": True,
    #"use_mlock": False,
    #"num_thread": 8

}



#Function Definitions
def print_globals():
    for name in globals(): print(name)

def print_messages():
    print(messages)

def clear_messages():
    messages.clear()

def print_images():
    print(images)

def clear_images():
    images.clear()

def append_image_by_name(image):
    if not os.path.isfile(image):
        print(f'Image "{image}" directory not found in {os.getcwd()}');return
    images.append(image)

def append_screenshot():
    pyautogui.screenshot().save('screenshot.png')
    images.append('screenshot.png')

def replay_xtts_audio():
        data,fs=soundfile.read('xtts.wav')
        sounddevice.play(data,fs,device=AUDIO_DEVICE)

def print_dict(dictionary):
    x=dictionary
    if 'x' in globals():
        if isinstance(x, dict):
            for k,v in x.items():print(f"{k}: {v}")
        else:print(f"{x} is not a dictionary.")
    else:print(f"{x} not a global variable.")

def load_xtts_model_by_name(xtts_model_name):
    x=xtts_model_name;print(f'Loading {x} config...')

    if not os.path.isdir(x):
        print(f'Model "{x}" directory not found in {os.getcwd()}');return

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    global XTTS_CONFIG, XTTS, GPT_LAT, SPEAKER_EMBED

    XTTS_CONFIG = XttsConfig();XTTS_CONFIG.load_json(f'{x}/config.json')
    XTTS = Xtts.init_from_config(XTTS_CONFIG)
    XTTS.load_checkpoint(XTTS_CONFIG,f"C:/Users/1/ai/{x}/");XTTS.cuda()

    print('Loading conditioning latents...')

    LATENT_PARAMS = {

        'audio_path':
        [f"{x}/voice_files/{x}-{i:02d}.wav" for i in range(1,100)],
        'max_ref_length':30,
        'gpt_cond_len':30,
        'gpt_cond_chunk_len':30,
        'librosa_trim_db':None,
        'sound_norm_refs':False,
        'load_sr':24000,

    }

    GPT_LAT,SPEAKER_EMBED,=XTTS.get_conditioning_latents(**LATENT_PARAMS)
    print(f'{x} loaded succesfully.')

def xtts_inference_and_audio(text):
    if 'XTTS' not in globals(): print(WARNINGS[1]); return
    if isinstance(text, list): text = " ".join(text[1:])
    print("Inferencing...")

    INFERENCE_PARAMS = {

        'text':text,
        'language':'en',
        'gpt_cond_latent':GPT_LAT,
        'speaker_embedding':SPEAKER_EMBED,
        'speed':1.00,
        'enable_text_splitting':True,

    }

    torch.cuda.empty_cache()
    out = XTTS.inference(**INFERENCE_PARAMS)
    torchaudio.save('xtts.wav', torch.tensor(out['wav']).unsqueeze(0), 24000)
    data,fs = soundfile.read('xtts.wav')
    sounddevice.play(data,fs,device=AUDIO_DEVICE)

def load_ollama_model_by_name(ollama_model_name):
    global model

    x=ollama_model_name
    LIST_OLLAMA_MODELS=[model['name'] for model in ollama.list()['models']]

    if f'{x}:latest' not in LIST_OLLAMA_MODELS:
        print(f"'{x}:latest' model not found in {LIST_OLLAMA_MODELS}");return
    
    model=x;print(f'"{x}" loaded into OLLAMA.')

def ollama_chat_response():
    if model == '':
        print("No LLM model loaded.");return

    OLLAMA_CHAT_PARARMS = {

    'model':model,
    'messages':messages,
    'stream':stream,
    #'format':'json',
    'options':options,
    #'keep_alive':'60m'

    }

    text=''; response=ollama.chat(**OLLAMA_CHAT_PARARMS)

    for chunk in response:
        word = chunk['message']['content']; text += word
        print(word,end='',flush=True)

    messages.append({'role':'assistant','content':f'{text}'})

    if 'XTTS' in globals():
        text=trim_unfinished_sentence(text)
        text=remove_emotes(text);xtts_inference_and_audio(text)

def llava_generate_response(user_input):
    OLLAMA_GENERATE_PARAMS = {

        'model': 'llava',
        'prompt': user_input,
        #'system': """ system promtp info """,
        #'template': """ template info """,
        #'context': 
        'stream': stream,
        'raw': raw,
        #'format': 'json'
        'images': images,
        'options': options,
        #'keep_alive':'60m',

        }
    
    text='';response=ollama.generate(**OLLAMA_GENERATE_PARAMS)
    for chunk in response:word=chunk['response']; text += word
    print(word,end='',flush=True)
    messages.append({'role':'assistant','content':text});print('\n')

def add_time():
    prompt_path='C:/Users/1/ai/prompts/daily/'
    date,time,day=datetime.now().strftime('%Y-%m-%d %I:%M:%p %A').split()

    message=f"""
    
    Today is {day}, {date}. 
    The current time is {time}. 
    {Path(prompt_path + day + '.txt').read_text()}

    """

    if messages:del messages[0];messages.insert(0,{'role':'user','content':message})

def remove_emotes(text):
    text=re.sub(r'\*([^*]+)\*|\([^)]*\)','',text);return text

def trim_unfinished_sentence(text):
    end=max(text.rfind("."),text.rfind("!"),text.rfind("?"));return text[:end+1]
 
def twitch_irc_listen():
    print("Connecting to irc.chat.twitch.tv...\n")
    import socket
    SOCKET = socket.socket()
    SOCKET.settimeout(36000.0)
    SOCKET.connect(('irc.chat.twitch.tv', 6667))
    SOCKET.send('CAP REQ :twitch.tv/tags\r\n'.encode())
    SOCKET.send('PASS SCHMOOPIIE\r\n'.encode()) 
    SOCKET.send('NICK justinfan67420\r\n'.encode())
    SOCKET.send('JOIN #auskaai\r\n'.encode())

    while True:
            raw  = SOCKET.recv(4096).decode(); print(raw)

            if('PING :tmi.twitch.tv' in raw): 
                SOCKET.send(('PONG :tmi.twitch.tv'+'\r\n').encode())

            elif('PRIVMSG' in raw):
                bisection = raw.split('#auskaai :'); data = {}
                data['message'] = bisection[1]
                pairs = bisection[0].split(';')  

                for pair in pairs:
                    key, value = pair.split("="); data[key] = value
                
                if data['display-name'] == 'AuskaAI':
                    data['display-name'] = 'Ikari'
                
                message = f"{data['display-name']}:{data['message']}"
                message = message.replace('\r\n','');
                messages.append({'role':'user','content':message})
                ollama_chat_response()
            
            else: continue



def call_function_map(call):
    x = call.split()

    if x[0] in FUNCTION_DICT:
        if len(x) == 1: 
            try: FUNCTION_DICT[x[0]]()
            except TypeError: print(f"{x[0]} {WARNINGS[1]}.")
        else:
            try: FUNCTION_DICT[x[0]](*x[1:])
            except TypeError: print(f"{x[0]} {WARNINGS[2]}.")

    else: print(f"{x[0]} {WARNINGS[3]}.")



FUNCTION_DICT = {

    '-globals': print_globals,
    '-show': print_dict,
    '-messages': print_messages,
    '-images': print_images,
    '-clear': clear_messages,
    '-clearim': clear_images,
    '-persona': load_ollama_model_by_name,
    '-voice': load_xtts_model_by_name,
    #'-tts': xtts_inference_and_audio,
    '-twitch': twitch_irc_listen,

    }

def main():
    while True:
        x = input('>:')
        if x[:1] == '-': call_function_map(x)
        elif model == '': print(WARNINGS[0])
        else: 
            messages.append({'role':'user','content':f'{x}'})
            print();add_time();ollama_chat_response()
 

print("Dependencies imported.")
main()
