import os
if 'XDG_CACHE_HOME' not in os.environ:
	os.environ['XDG_CACHE_HOME'] = os.path.realpath(os.path.join(os.getcwd(), './models/'))

if 'TORTOISE_MODELS_DIR' not in os.environ:
	os.environ['TORTOISE_MODELS_DIR'] = os.path.realpath(os.path.join(os.getcwd(), './models/tortoise/'))

if 'TRANSFORMERS_CACHE' not in os.environ:
	os.environ['TRANSFORMERS_CACHE'] = os.path.realpath(os.path.join(os.getcwd(), './models/transformers/'))

import argparse
import time
import math
import json
import base64
import re
import urllib.request
import signal
import gc
import subprocess
import psutil
import yaml
import hashlib
import string
import random
import shutil

from tqdm import tqdm
import torch
import torchaudio
import music_tag
import gradio as gr
import gradio.utils
import pandas as pd
import numpy as np

from glob import glob
from datetime import datetime
from datetime import timedelta

from tortoise.api import TextToSpeech as TorToise_TTS, MODELS, get_model_path, pad_or_truncate
from tortoise.api_fast import TextToSpeech as Toroise_TTS_Hifi
from tortoise.utils.audio import load_audio, load_voice, load_voices, get_voice_dir, get_voices
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.device import get_device_name, set_device_name, get_device_count, get_device_vram, get_device_batch_size, do_gc
# TODO: The below import blocks any CLI parameters.
#       Try running with --low-vram
from rvc_pipe.rvc_infer import rvc_convert

MODELS['dvae.pth'] = "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/3704aea61678e7e468a06d8eea121dba368a798e/.models/dvae.pth"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]
WHISPER_SPECIALIZED_MODELS = ["tiny.en", "base.en", "small.en", "medium.en"]
WHISPER_BACKENDS = ["openai/whisper", "lightmare/whispercpp", "m-bain/whisperx"]
VOCODERS = ['univnet', 'bigvgan_base_24khz_100band', 'bigvgan_24khz_100band']
TTSES = ['tortoise']

INFERENCING = False
GENERATE_SETTINGS_ARGS = None

LEARNING_RATE_SCHEMES = {"Multistep": "MultiStepLR", "Cos. Annealing": "CosineAnnealingLR_Restart"}
LEARNING_RATE_SCHEDULE = [ 2, 4, 9, 18, 25, 33, 50 ]

RESAMPLERS = {}

MIN_TRAINING_DURATION = 1.6 # Original value was 0.6
MAX_TRAINING_DURATION = 11.6097505669
MAX_TRAINING_CHAR_LENGTH = 200

VALLE_ENABLED = False
BARK_ENABLED = False

VERBOSE_DEBUG = True

KKS = None
PYKAKASI_ENABLED = False

import traceback

try:
	import pykakasi
	KKS = pykakasi.kakasi()
	PYKAKASI_ENABLED = True
except Exception as e:
	#if VERBOSE_DEBUG:
	#	print(traceback.format_exc())
	pass

try:
	from whisper.normalizers.english import EnglishTextNormalizer
	from whisper.normalizers.basic import BasicTextNormalizer
	from whisper.tokenizer import LANGUAGES 

	print("Whisper detected")
except Exception as e:
	if VERBOSE_DEBUG:
		print(traceback.format_exc())
	pass
'''
try:
	from vall_e.emb.qnt import encode as valle_quantize
	from vall_e.emb.g2p import encode as valle_phonemize

	from vall_e.inference import TTS as VALLE_TTS

	import soundfile

	print("VALL-E detected")
	VALLE_ENABLED = True
except Exception as e:
	if VERBOSE_DEBUG:
		print(traceback.format_exc())
	pass
'''
if VALLE_ENABLED:
	TTSES.append('vall-e')

# torchaudio.set_audio_backend('soundfile')

'''
try:
	import bark
	from bark import text_to_semantic
	from bark.generation import SAMPLE_RATE as BARK_SAMPLE_RATE, ALLOWED_PROMPTS, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic, load_codec_model
	from bark.api import generate_audio as bark_generate_audio
	from encodec.utils import convert_audio

	from scipy.io.wavfile import write as write_wav

	print("Bark detected")
	BARK_ENABLED = True
except Exception as e:
	if VERBOSE_DEBUG:
		print(traceback.format_exc())
	pass
'''
if BARK_ENABLED:
	TTSES.append('bark')

	def semantic_to_audio_tokens(
	    semantic_tokens,
	    history_prompt = None,
	    temp = 0.7,
	    silent = False,
	    output_full = False,
	):
	    coarse_tokens = generate_coarse(
	        semantic_tokens, history_prompt=history_prompt, temp=temp, silent=silent, use_kv_caching=True
	    )
	    fine_tokens = generate_fine(coarse_tokens, history_prompt=history_prompt, temp=0.5)

	    if output_full:
	        full_generation = {
	            "semantic_prompt": semantic_tokens,
	            "coarse_prompt": coarse_tokens,
	            "fine_prompt": fine_tokens,
	        }
	        return full_generation
	    return fine_tokens

	class Bark_TTS():
		def __init__(self, small=False):
			self.input_sample_rate = BARK_SAMPLE_RATE
			self.output_sample_rate = BARK_SAMPLE_RATE # args.output_sample_rate

			preload_models(
				text_use_gpu=True,
				coarse_use_gpu=True,
				fine_use_gpu=True,
				codec_use_gpu=True,

				text_use_small=small,
				coarse_use_small=small,
				fine_use_small=small,
				
				force_reload=False
			)

			self.device = get_device_name()

			try:
				from vocos import Vocos
				self.vocos_enabled = True
				print("Vocos detected")
			except Exception as e:
				if VERBOSE_DEBUG:
					print(traceback.format_exc())
				self.vocos_enabled = False

			try:
				from hubert.hubert_manager import HuBERTManager

				hubert_manager = HuBERTManager()
				hubert_manager.make_sure_hubert_installed()
				hubert_manager.make_sure_tokenizer_installed()

				self.hubert_enabled = True
				print("HuBERT detected")
			except Exception as e:
				if VERBOSE_DEBUG:
					print(traceback.format_exc())
				self.hubert_enabled = False

			if self.vocos_enabled:
				self.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(self.device)

		def create_voice( self, voice ):
			transcription_json = f'./training/{voice}/whisper.json'
			if not os.path.exists(transcription_json):
				raise f"Transcription for voice not found: {voice}"
			
			transcriptions = json.load(open(transcription_json, 'r', encoding="utf-8"))
			candidates = []
			for file in transcriptions:
				result = transcriptions[file]
				added = 0

				for segment in result['segments']:
					path = file.replace(".wav", f"_{pad(segment['id'], 4)}.wav")
					# check if the slice actually exists
					if not os.path.exists(f'./training/{voice}/audio/{path}'):
						continue

					entry = (
						path,
						segment['end'] - segment['start'],
						segment['text']
					)
					candidates.append(entry)
					added = added + 1

				# if nothing got added (assuming because nothign was sliced), use the master file
				if added == 0: # added < len(result['segments']):
					start = 0
					end = 0
					for segment in result['segments']:
						start = max( start, segment['start'] )
						end = max( end, segment['end'] )

					entry = (
						file,
						end - start,
						result['text']
					)
					candidates.append(entry)

			candidates.sort(key=lambda x: x[1])
			candidate = random.choice(candidates)
			audio_filepath = f'./training/{voice}/audio/{candidate[0]}'
			text = candidate[-1]

			print("Using as reference:", audio_filepath, text)

			# Load and pre-process the audio waveform
			model = load_codec_model(use_gpu=True)
			wav, sr = torchaudio.load(audio_filepath)
			wav = convert_audio(wav, sr, model.sample_rate, model.channels)

			# generate semantic tokens

			if self.hubert_enabled:
				from hubert.pre_kmeans_hubert import CustomHubert
				from hubert.customtokenizer import CustomTokenizer
				
				wav = wav.to(self.device)

				# Extract discrete codes from EnCodec
				with torch.no_grad():
					encoded_frames = model.encode(wav.unsqueeze(0))
				codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

				# get seconds of audio
				seconds = wav.shape[-1] / model.sample_rate

				# Load the HuBERT model
				hubert_model = CustomHubert(checkpoint_path='./data/models/hubert/hubert.pt').to(self.device)

				# Load the CustomTokenizer model
				tokenizer = CustomTokenizer.load_from_checkpoint('./data/models/hubert/tokenizer.pth').to(self.device)

				semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
				semantic_tokens = tokenizer.get_token(semantic_vectors)

				# move codes to cpu
				codes = codes.cpu().numpy()
				# move semantic tokens to cpu
				semantic_tokens = semantic_tokens.cpu().numpy()
			else:
				wav = wav.unsqueeze(0).to(self.device)

				# Extract discrete codes from EnCodec
				with torch.no_grad():
					encoded_frames = model.encode(wav)
				codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze().cpu().numpy()  # [n_q, T]

				# get seconds of audio
				seconds = wav.shape[-1] / model.sample_rate

				# generate semantic tokens
				semantic_tokens = generate_text_semantic(text, max_gen_duration_s=seconds, top_k=50, top_p=.95, temp=0.7)

			# print(bark.__file__)
			bark_location = os.path.dirname(os.path.relpath(bark.__file__)) # './modules/bark/bark/'
			output_path = f'./{bark_location}/assets/prompts/' + voice.replace("/", "_") + '.npz'
			np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

		def inference( self, text, voice, text_temp=0.7, waveform_temp=0.7 ):
			if voice == "random":
				voice = None
			else:
				if not os.path.exists('./modules/bark/bark/assets/prompts/' + voice + '.npz'):
					self.create_voice( voice )
				voice = voice.replace("/", "_")
				if voice not in ALLOWED_PROMPTS:
					ALLOWED_PROMPTS.add( voice )

			semantic_tokens = text_to_semantic(text, history_prompt=voice, temp=text_temp, silent=False)
			audio_tokens = semantic_to_audio_tokens( semantic_tokens, history_prompt=voice, temp=waveform_temp, silent=False, output_full=False )

			if self.vocos_enabled:
				audio_tokens_torch = torch.from_numpy(audio_tokens).to(self.device)
				features = self.vocos.codes_to_features(audio_tokens_torch)
				wav = self.vocos.decode(features, bandwidth_id=torch.tensor([2], device=self.device))
			else:
				wav = codec_decode( audio_tokens )

			return ( wav, BARK_SAMPLE_RATE )
			# return (bark_generate_audio(text, history_prompt=voice, text_temp=text_temp, waveform_temp=waveform_temp), BARK_SAMPLE_RATE)

args = None
tts = None
tts_loading = False
webui = None
voicefixer = None

whisper_model = None
whisper_align_model = None

training_state = None

current_voice = None

def cleanup_voice_name( name ):
	return name.split("/")[-1]

def resample( waveform, input_rate, output_rate=44100 ):
	# mono-ize
	waveform = torch.mean(waveform, dim=0, keepdim=True)

	if input_rate == output_rate:
		return waveform, output_rate

	key = f'{input_rate}:{output_rate}'
	if not key in RESAMPLERS:
		RESAMPLERS[key] = torchaudio.transforms.Resample(
			input_rate,
			output_rate,
			lowpass_filter_width=16,
			rolloff=0.85,
			resampling_method="kaiser_window",
			beta=8.555504641634386,
		)

	return RESAMPLERS[key]( waveform ), output_rate

def generate(**kwargs):
	if args.tts_backend == "tortoise":
		return generate_tortoise(**kwargs)
	if args.tts_backend == "vall-e":
		return generate_valle(**kwargs)
	if args.tts_backend == "bark":
		return generate_bark(**kwargs)

def generate_bark(**kwargs):
	parameters = {}
	parameters.update(kwargs)

	voice = parameters['voice']
	progress = parameters['progress'] if 'progress' in parameters else None
	if parameters['seed'] == 0:
		parameters['seed'] = None

	usedSeed = parameters['seed']

	global args
	global tts

	unload_whisper()
	unload_voicefixer()

	if not tts:
		# should check if it's loading or unloaded, and load it if it's unloaded
		if tts_loading:
			raise Exception("TTS is still initializing...")
		if progress is not None:
			notify_progress("Initializing TTS...", progress=progress)
		load_tts()
	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	do_gc()

	voice_samples = None
	conditioning_latents = None
	sample_voice = None

	voice_cache = {}

	def get_settings( override=None ):
		settings = {
			'voice': parameters['voice'],
			'text_temp': float(parameters['temperature']),
			'waveform_temp': float(parameters['temperature']),
		}

		# could be better to just do a ternary on everything above, but i am not a professional
		selected_voice = voice
		if override is not None:
			if 'voice' in override:
				selected_voice = override['voice']

			for k in override:
				if k not in settings:
					continue
				settings[k] = override[k]

		return settings

	if not parameters['delimiter']:
		parameters['delimiter'] = "\n"
	elif parameters['delimiter'] == "\\n":
		parameters['delimiter'] = "\n"

	if parameters['delimiter'] and parameters['delimiter'] != "" and parameters['delimiter'] in parameters['text']:
		texts = parameters['text'].split(parameters['delimiter'])
	else:
		texts = split_and_recombine_text(parameters['text'])
 
	full_start_time = time.time()
 
	outdir = f"{args.results_folder}/{voice}/"
	os.makedirs(outdir, exist_ok=True)

	audio_cache = {}

	volume_adjust = torchaudio.transforms.Vol(gain=args.output_volume, gain_type="amplitude") if args.output_volume != 1 else None

	idx = 0
	idx_cache = {}
	for i, file in enumerate(os.listdir(outdir)):
		filename = os.path.basename(file)
		extension = os.path.splitext(filename)[-1][1:]
		if extension != "json" and extension != "wav":
			continue
		match = re.findall(rf"^{cleanup_voice_name(voice)}_(\d+)(?:.+?)?{extension}$", filename)
		if match and len(match) > 0:
			key = int(match[0])
			idx_cache[key] = True

	if len(idx_cache) > 0:
		keys = sorted(list(idx_cache.keys()))
		idx = keys[-1] + 1

	idx = pad(idx, 4)

	def get_name(line=0, candidate=0, combined=False):
		name = f"{idx}"
		if combined:
			name = f"{name}_combined"
		elif len(texts) > 1:
			name = f"{name}_{line}"
		if parameters['candidates'] > 1:
			name = f"{name}_{candidate}"
		return name

	def get_info( voice, settings = None, latents = True ):
		info = {}
		info.update(parameters)

		info['time'] = time.time()-full_start_time
		info['datetime'] = datetime.now().isoformat()

		info['progress'] = None
		del info['progress']

		if info['delimiter'] == "\n":
			info['delimiter'] = "\\n"

		if settings is not None:
			for k in settings:
				if k in info:
					info[k] = settings[k]
		return info

	INFERENCING = True
	for line, cut_text in enumerate(texts):	
		tqdm_prefix = f'[{str(line+1)}/{str(len(texts))}]'
		print(f"{tqdm_prefix} Generating line: {cut_text}")
		start_time = time.time()

		# do setting editing
		match = re.findall(r'^(\{.+\}) (.+?)$', cut_text) 
		override = None
		if match and len(match) > 0:
			match = match[0]
			try:
				override = json.loads(match[0])
				cut_text = match[1].strip()
			except Exception as e:
				raise Exception("Prompt settings editing requested, but received invalid JSON")

		settings = get_settings( override=override )

		gen = tts.inference(cut_text, **settings )

		run_time = time.time()-start_time
		print(f"Generating line took {run_time} seconds")

		if not isinstance(gen, list):
			gen = [gen]

		for j, g in enumerate(gen):
			wav, sr = g
			name = get_name(line=line, candidate=j)

			settings['text'] = cut_text
			settings['time'] = run_time
			settings['datetime'] = datetime.now().isoformat()

			# save here in case some error happens mid-batch
			if tts.vocos_enabled:
				torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', wav.cpu(), sr)
			else:
				write_wav(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', sr, wav)
			wav, sr = torchaudio.load(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')

			audio_cache[name] = {
				'audio': wav,
				'settings': get_info(voice=override['voice'] if override and 'voice' in override else voice, settings=settings)
			}

	del gen
	do_gc()
	INFERENCING = False

	for k in audio_cache:
		audio = audio_cache[k]['audio']

		audio, _ = resample(audio, tts.output_sample_rate, args.output_sample_rate)
		if volume_adjust is not None:
			audio = volume_adjust(audio)

		audio_cache[k]['audio'] = audio
		torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{k}.wav', audio, args.output_sample_rate)

	output_voices = []
	for candidate in range(parameters['candidates']):
		if len(texts) > 1:
			audio_clips = []
			for line in range(len(texts)):
				name = get_name(line=line, candidate=candidate)
				audio = audio_cache[name]['audio']
				audio_clips.append(audio)
			
			name = get_name(candidate=candidate, combined=True)
			audio = torch.cat(audio_clips, dim=-1)
			torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', audio, args.output_sample_rate)

			audio = audio.squeeze(0).cpu()
			audio_cache[name] = {
				'audio': audio,
				'settings': get_info(voice=voice),
				'output': True
			}
		else:
			try:
				name = get_name(candidate=candidate)
				audio_cache[name]['output'] = True
			except Exception as e:
				for name in audio_cache:
					audio_cache[name]['output'] = True


	if args.voice_fixer:
		if not voicefixer:
			notify_progress("Loading voicefix...", progress=progress)
			load_voicefixer()

		try:
			fixed_cache = {}
			for name in tqdm(audio_cache, desc="Running voicefix..."):
				del audio_cache[name]['audio']
				if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
					continue

				path = f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav'
				fixed = f'{outdir}/{cleanup_voice_name(voice)}_{name}_fixed.wav'
				voicefixer.restore(
					input=path,
					output=fixed,
					cuda=get_device_name() == "cuda" and args.voice_fixer_use_cuda,
					#mode=mode,
				)
				
				fixed_cache[f'{name}_fixed'] = {
					'settings': audio_cache[name]['settings'],
					'output': True
				}
				audio_cache[name]['output'] = False
			
			for name in fixed_cache:
				audio_cache[name] = fixed_cache[name]
		except Exception as e:
			print(e)
			print("\nFailed to run Voicefixer")

	for name in audio_cache:
		if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
			if args.prune_nonfinal_outputs:
				audio_cache[name]['pruned'] = True
				os.remove(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')
			continue

		output_voices.append(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')

		if not args.embed_output_metadata:
			with open(f'{outdir}/{cleanup_voice_name(voice)}_{name}.json', 'w', encoding="utf-8") as f:
				f.write(json.dumps(audio_cache[name]['settings'], indent='\t') )

	if args.embed_output_metadata:
		for name in tqdm(audio_cache, desc="Embedding metadata..."):
			if 'pruned' in audio_cache[name] and audio_cache[name]['pruned']:
				continue

			metadata = music_tag.load_file(f"{outdir}/{cleanup_voice_name(voice)}_{name}.wav")
			metadata['lyrics'] = json.dumps(audio_cache[name]['settings'])
			metadata.save()
 
	if sample_voice is not None:
		sample_voice = (tts.input_sample_rate, sample_voice.numpy())

	info = get_info(voice=voice, latents=False)
	print(f"Generation took {info['time']} seconds, saved to '{output_voices[0]}'\n")

	info['seed'] = usedSeed
	if 'latents' in info:
		del info['latents']

	os.makedirs('./config/', exist_ok=True)
	with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(info, indent='\t') )

	stats = [
		[ parameters['seed'], "{:.3f}".format(info['time']) ]
	]

	return (
		sample_voice,
		output_voices,
		stats,
	)

def generate_valle(**kwargs):
	parameters = {}
	parameters.update(kwargs)

	voice = parameters['voice']
	progress = parameters['progress'] if 'progress' in parameters else None
	if parameters['seed'] == 0:
		parameters['seed'] = None

	usedSeed = parameters['seed']

	global args
	global tts

	unload_whisper()
	unload_voicefixer()

	if not tts:
		# should check if it's loading or unloaded, and load it if it's unloaded
		if tts_loading:
			raise Exception("TTS is still initializing...")
		if progress is not None:
			notify_progress("Initializing TTS...", progress=progress)
		load_tts()
	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	do_gc()

	voice_samples = None
	conditioning_latents = None
	sample_voice = None

	voice_cache = {}
	def fetch_voice( voice ):
		if voice in voice_cache:
			return voice_cache[voice]

		"""
		voice_dir = f'./training/{voice}/audio/'

		if not os.path.isdir(voice_dir) or len(os.listdir(voice_dir)) == 0:
			voice_dir = f'./voices/{voice}/'

		files = [ f'{voice_dir}/{d}' for d in os.listdir(voice_dir) if d[-4:] == ".wav" ]
		"""

		if os.path.isdir(f'./training/{voice}/audio/'):
			files = get_voice(name="audio", dir=f"./training/{voice}/", load_latents=False)
		else:
			files = get_voice(name=voice, load_latents=False)

		# return files
		voice_cache[voice] = random.sample(files, k=min(3, len(files)))
		return voice_cache[voice]

	def get_settings( override=None ):
		settings = {
			'ar_temp': float(parameters['temperature']),
			'nar_temp': float(parameters['temperature']),
			'max_ar_steps': parameters['num_autoregressive_samples'],
		}

		# could be better to just do a ternary on everything above, but i am not a professional
		selected_voice = voice
		if override is not None:
			if 'voice' in override:
				selected_voice = override['voice']

			for k in override:
				if k not in settings:
					continue
				settings[k] = override[k]

		settings['references'] = fetch_voice(voice=selected_voice) # [ fetch_voice(voice=selected_voice) for _ in range(3) ]
		return settings

	if not parameters['delimiter']:
		parameters['delimiter'] = "\n"
	elif parameters['delimiter'] == "\\n":
		parameters['delimiter'] = "\n"

	if parameters['delimiter'] and parameters['delimiter'] != "" and parameters['delimiter'] in parameters['text']:
		texts = parameters['text'].split(parameters['delimiter'])
	else:
		texts = split_and_recombine_text(parameters['text'])
 
	full_start_time = time.time()
 
	outdir = f"{args.results_folder}/{voice}/"
	os.makedirs(outdir, exist_ok=True)

	audio_cache = {}

	volume_adjust = torchaudio.transforms.Vol(gain=args.output_volume, gain_type="amplitude") if args.output_volume != 1 else None

	idx = 0
	idx_cache = {}
	for i, file in enumerate(os.listdir(outdir)):
		filename = os.path.basename(file)
		extension = os.path.splitext(filename)[-1][1:]
		if extension != "json" and extension != "wav":
			continue
		match = re.findall(rf"^{voice}_(\d+)(?:.+?)?{extension}$", filename)
		if match and len(match) > 0:
			key = int(match[0])
			idx_cache[key] = True

	if len(idx_cache) > 0:
		keys = sorted(list(idx_cache.keys()))
		idx = keys[-1] + 1

	idx = pad(idx, 4)

	def get_name(line=0, candidate=0, combined=False):
		name = f"{idx}"
		if combined:
			name = f"{name}_combined"
		elif len(texts) > 1:
			name = f"{name}_{line}"
		if parameters['candidates'] > 1:
			name = f"{name}_{candidate}"
		return name

	def get_info( voice, settings = None, latents = True ):
		info = {}
		info.update(parameters)

		info['time'] = time.time()-full_start_time
		info['datetime'] = datetime.now().isoformat()

		info['progress'] = None
		del info['progress']

		if info['delimiter'] == "\n":
			info['delimiter'] = "\\n"

		if settings is not None:
			for k in settings:
				if k in info:
					info[k] = settings[k]
		return info

	INFERENCING = True
	for line, cut_text in enumerate(texts):	
		tqdm_prefix = f'[{str(line+1)}/{str(len(texts))}]'
		print(f"{tqdm_prefix} Generating line: {cut_text}")
		start_time = time.time()

		# do setting editing
		match = re.findall(r'^(\{.+\}) (.+?)$', cut_text) 
		override = None
		if match and len(match) > 0:
			match = match[0]
			try:
				override = json.loads(match[0])
				cut_text = match[1].strip()
			except Exception as e:
				raise Exception("Prompt settings editing requested, but received invalid JSON")

		name = get_name(line=line, candidate=0)

		settings = get_settings( override=override )
		references = settings['references']
		settings.pop("references")
		settings['out_path'] = f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav'

		gen = tts.inference(cut_text, references, **settings )

		run_time = time.time()-start_time
		print(f"Generating line took {run_time} seconds")

		if not isinstance(gen, list):
			gen = [gen]

		for j, g in enumerate(gen):
			wav, sr = g
			name = get_name(line=line, candidate=j)

			settings['text'] = cut_text
			settings['time'] = run_time
			settings['datetime'] = datetime.now().isoformat()

			# save here in case some error happens mid-batch
			#torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', wav.cpu(), sr)
			#soundfile.write(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', wav.cpu()[0,0], sr)
			wav, sr = torchaudio.load(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')

			audio_cache[name] = {
				'audio': wav,
				'settings': get_info(voice=override['voice'] if override and 'voice' in override else voice, settings=settings)
			}

	del gen
	do_gc()
	INFERENCING = False

	for k in audio_cache:
		audio = audio_cache[k]['audio']

		audio, _ = resample(audio, tts.output_sample_rate, args.output_sample_rate)
		if volume_adjust is not None:
			audio = volume_adjust(audio)

		audio_cache[k]['audio'] = audio
		torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{k}.wav', audio, args.output_sample_rate)

	output_voices = []
	for candidate in range(parameters['candidates']):
		if len(texts) > 1:
			audio_clips = []
			for line in range(len(texts)):
				name = get_name(line=line, candidate=candidate)
				audio = audio_cache[name]['audio']
				audio_clips.append(audio)
			
			name = get_name(candidate=candidate, combined=True)
			audio = torch.cat(audio_clips, dim=-1)
			torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', audio, args.output_sample_rate)

			audio = audio.squeeze(0).cpu()
			audio_cache[name] = {
				'audio': audio,
				'settings': get_info(voice=voice),
				'output': True
			}
		else:
			name = get_name(candidate=candidate)
			audio_cache[name]['output'] = True


	if args.voice_fixer:
		if not voicefixer:
			notify_progress("Loading voicefix...", progress=progress)
			load_voicefixer()

		try:
			fixed_cache = {}
			for name in tqdm(audio_cache, desc="Running voicefix..."):
				del audio_cache[name]['audio']
				if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
					continue

				path = f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav'
				fixed = f'{outdir}/{cleanup_voice_name(voice)}_{name}_fixed.wav'
				voicefixer.restore(
					input=path,
					output=fixed,
					cuda=get_device_name() == "cuda" and args.voice_fixer_use_cuda,
					#mode=mode,
				)
				
				fixed_cache[f'{name}_fixed'] = {
					'settings': audio_cache[name]['settings'],
					'output': True
				}
				audio_cache[name]['output'] = False
			
			for name in fixed_cache:
				audio_cache[name] = fixed_cache[name]
		except Exception as e:
			print(e)
			print("\nFailed to run Voicefixer")

	for name in audio_cache:
		if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
			if args.prune_nonfinal_outputs:
				audio_cache[name]['pruned'] = True
				os.remove(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')
			continue

		output_voices.append(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')

		if not args.embed_output_metadata:
			with open(f'{outdir}/{cleanup_voice_name(voice)}_{name}.json', 'w', encoding="utf-8") as f:
				f.write(json.dumps(audio_cache[name]['settings'], indent='\t') )

	if args.embed_output_metadata:
		for name in tqdm(audio_cache, desc="Embedding metadata..."):
			if 'pruned' in audio_cache[name] and audio_cache[name]['pruned']:
				continue

			metadata = music_tag.load_file(f"{outdir}/{cleanup_voice_name(voice)}_{name}.wav")
			metadata['lyrics'] = json.dumps(audio_cache[name]['settings'])
			metadata.save()
 
	if sample_voice is not None:
		sample_voice = (tts.input_sample_rate, sample_voice.numpy())

	info = get_info(voice=voice, latents=False)
	print(f"Generation took {info['time']} seconds, saved to '{output_voices[0]}'\n")

	info['seed'] = usedSeed
	if 'latents' in info:
		del info['latents']

	os.makedirs('./config/', exist_ok=True)
	with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(info, indent='\t') )

	stats = [
		[ parameters['seed'], "{:.3f}".format(info['time']) ]
	]

	return (
		sample_voice,
		output_voices,
		stats,
	)

def generate_tortoise(**kwargs):
	parameters = {}
	parameters.update(kwargs)

	voice = parameters['voice']
	progress = parameters['progress'] if 'progress' in parameters else None
	if parameters['seed'] == 0:
		parameters['seed'] = None

	usedSeed = parameters['seed']

	global args
	global tts

	unload_whisper()
	unload_voicefixer()

	if not tts:
		# should check if it's loading or unloaded, and load it if it's unloaded
		if tts_loading:
			raise Exception("TTS is still initializing...")
		load_tts()
	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	do_gc()

	voice_samples = None
	conditioning_latents = None
	sample_voice = None

	voice_cache = {}
	def fetch_voice( voice ):
		cache_key = f'{voice}:{tts.autoregressive_model_hash[:8]}'
		if cache_key in voice_cache:
			return voice_cache[cache_key]

		print(f"Loading voice: {voice} with model {tts.autoregressive_model_hash[:8]}")
		sample_voice = None
		if voice == "microphone":
			if parameters['mic_audio'] is None:
				raise Exception("Please provide audio from mic when choosing `microphone` as a voice input")
			voice_samples, conditioning_latents = [load_audio(parameters['mic_audio'], tts.input_sample_rate)], None
		elif voice == "random":
			voice_samples, conditioning_latents = None, tts.get_random_conditioning_latents()
		else:
			if progress is not None:
				notify_progress(f"Loading voice: {voice}", progress=progress)

			voice_samples, conditioning_latents = load_voice(voice, model_hash=tts.autoregressive_model_hash)
			
		if voice_samples and len(voice_samples) > 0:
			if conditioning_latents is None:
				conditioning_latents = compute_latents(voice=voice, voice_samples=voice_samples, voice_latents_chunks=parameters['voice_latents_chunks'])
				
			sample_voice = torch.cat(voice_samples, dim=-1).squeeze().cpu()
			voice_samples = None

		voice_cache[cache_key] = (voice_samples, conditioning_latents, sample_voice)
		return voice_cache[cache_key]

	def get_settings( override=None ):
		settings = {
			'temperature': float(parameters['temperature']),

			'top_p': float(parameters['top_p']),
			'diffusion_temperature': float(parameters['diffusion_temperature']),
			'length_penalty': float(parameters['length_penalty']),
			'repetition_penalty': float(parameters['repetition_penalty']),
			'cond_free_k': float(parameters['cond_free_k']),

			'num_autoregressive_samples': parameters['num_autoregressive_samples'],
			'sample_batch_size': args.sample_batch_size,
			'diffusion_iterations': parameters['diffusion_iterations'],

			'voice_samples': None,
			'conditioning_latents': None,

			'use_deterministic_seed': parameters['seed'],
			'return_deterministic_state': True,
			'k': parameters['candidates'],
			'diffusion_sampler': parameters['diffusion_sampler'],
			'breathing_room': parameters['breathing_room'],
			'half_p': "Half Precision" in parameters['experimentals'],
			'cond_free': "Conditioning-Free" in parameters['experimentals'],
			'cvvp_amount': parameters['cvvp_weight'],
			
			'autoregressive_model': args.autoregressive_model,
			'diffusion_model': args.diffusion_model,
			'tokenizer_json': args.tokenizer_json,
		}

		# could be better to just do a ternary on everything above, but i am not a professional
		selected_voice = voice
		if override is not None:
			if 'voice' in override:
				selected_voice = override['voice']

			for k in override:
				if k not in settings:
					continue
				settings[k] = override[k]

		if settings['autoregressive_model'] is not None:
			if settings['autoregressive_model'] == "auto":
				settings['autoregressive_model'] = deduce_autoregressive_model(selected_voice)
			tts.load_autoregressive_model(settings['autoregressive_model'])

		if not args.use_hifigan:
			if settings['diffusion_model'] is not None:
				if settings['diffusion_model'] == "auto":
					settings['diffusion_model'] = deduce_diffusion_model(selected_voice)
				tts.load_diffusion_model(settings['diffusion_model'])
		
		if settings['tokenizer_json'] is not None:
			tts.load_tokenizer_json(settings['tokenizer_json'])

		settings['voice_samples'], settings['conditioning_latents'], _ = fetch_voice(voice=selected_voice)

		# clamp it down for the insane users who want this
		# it would be wiser to enforce the sample size to the batch size, but this is what the user wants
		settings['sample_batch_size'] = args.sample_batch_size
		if not settings['sample_batch_size']:
			settings['sample_batch_size'] = tts.autoregressive_batch_size
		if settings['num_autoregressive_samples'] < settings['sample_batch_size']:
			settings['sample_batch_size'] = settings['num_autoregressive_samples']

		if settings['conditioning_latents'] is not None and len(settings['conditioning_latents']) == 2 and settings['cvvp_amount'] > 0:
			print("Requesting weighing against CVVP weight, but voice latents are missing some extra data. Please regenerate your voice latents with 'Slimmer voice latents' unchecked.")
			settings['cvvp_amount'] = 0
			
		return settings

	if not parameters['delimiter']:
		parameters['delimiter'] = "\n"
	elif parameters['delimiter'] == "\\n":
		parameters['delimiter'] = "\n"

	if parameters['delimiter'] and parameters['delimiter'] != "" and parameters['delimiter'] in parameters['text']:
		texts = parameters['text'].split(parameters['delimiter'])
	else:
		texts = split_and_recombine_text(parameters['text'])
 
	full_start_time = time.time()
 
	outdir = f"{args.results_folder}/{voice}/"
	os.makedirs(outdir, exist_ok=True)

	audio_cache = {}

	volume_adjust = torchaudio.transforms.Vol(gain=args.output_volume, gain_type="amplitude") if args.output_volume != 1 else None

	idx = 0
	idx_cache = {}
	for i, file in enumerate(os.listdir(outdir)):
		filename = os.path.basename(file)
		extension = os.path.splitext(filename)[-1][1:]
		if extension != "json" and extension != "wav":
			continue
		match = re.findall(rf"^{voice}_(\d+)(?:.+?)?{extension}$", filename)
		if match and len(match) > 0:
			key = int(match[0])
			idx_cache[key] = True

	if len(idx_cache) > 0:
		keys = sorted(list(idx_cache.keys()))
		idx = keys[-1] + 1

	idx = pad(idx, 4)

	def get_name(line=0, candidate=0, combined=False):
		name = f"{idx}"
		if combined:
			name = f"{name}_combined"
		elif len(texts) > 1:
			name = f"{name}_{line}"
		if parameters['candidates'] > 1:
			name = f"{name}_{candidate}"
		return name

	def get_info( voice, settings = None, latents = True ):
		info = {}
		info.update(parameters)

		info['time'] = time.time()-full_start_time
		info['datetime'] = datetime.now().isoformat()

		info['model'] = tts.autoregressive_model_path
		info['model_hash'] = tts.autoregressive_model_hash 

		info['progress'] = None
		del info['progress']

		if info['delimiter'] == "\n":
			info['delimiter'] = "\\n"

		if settings is not None:
			for k in settings:
				if k in info:
					info[k] = settings[k]

			if 'half_p' in settings and 'cond_free' in settings:
				info['experimentals'] = []
				if settings['half_p']:
					info['experimentals'].append("Half Precision")
				if settings['cond_free']:
					info['experimentals'].append("Conditioning-Free")

		if latents and "latents" not in info:
			voice = info['voice']
			model_hash = settings["model_hash"][:8] if settings is not None and "model_hash" in settings else tts.autoregressive_model_hash[:8]

			dir = f'{get_voice_dir()}/{voice}/'
			# TODO: Use of model_hash here causes issues in development as new hashes are added to the repo.
			latents_path = f'{dir}/cond_latents_{model_hash}.pth'

			if voice == "random" or voice == "microphone":
				if args.use_hifigan:
					if latents and settings is not None and torch.any(settings['conditioning_latents']):
						os.makedirs(dir, exist_ok=True)
						torch.save(conditioning_latents, latents_path)
				else: 
					if latents and settings is not None and settings['conditioning_latents']:
						os.makedirs(dir, exist_ok=True)
						torch.save(conditioning_latents, latents_path)

			if latents_path and os.path.exists(latents_path):
				try:
					with open(latents_path, 'rb') as f:
						info['latents'] = base64.b64encode(f.read()).decode("ascii")
				except Exception as e:
					pass

		return info

	INFERENCING = True
	for line, cut_text in enumerate(texts):
		if should_phonemize():
			cut_text = phonemizer( cut_text )

		if parameters['emotion'] == "Custom":
			if parameters['prompt'] and parameters['prompt'].strip() != "":
				cut_text = f"[{parameters['prompt']},] {cut_text}"
		elif parameters['emotion'] != "None" and parameters['emotion']:
			cut_text = f"[I am really {parameters['emotion'].lower()},] {cut_text}"
		
		tqdm_prefix = f'[{str(line+1)}/{str(len(texts))}]'
		print(f"{tqdm_prefix} Generating line: {cut_text}")
		start_time = time.time()

		# do setting editing
		match = re.findall(r'^(\{.+\}) (.+?)$', cut_text) 
		override = None
		if match and len(match) > 0:
			match = match[0]
			try:
				override = json.loads(match[0])
				cut_text = match[1].strip()
			except Exception as e:
				raise Exception("Prompt settings editing requested, but received invalid JSON")

		settings = get_settings( override=override )
		print(settings)
		try:
			if args.use_hifigan:
				# Removing unused arguments when running hifigan, hf transformers doesn't like all these unused args
				# This only happens when loading with hifigan as it doesn't load the diffusion model... bandage for now
				unused_args = ['diffusion_temperature', 'cond_free_k', 'sample_batch_size', 'diffusion_iterations',
							'return_deterministic_state', 'diffusion_sampler', 'breathing_room', 'half_p', 'cond_free',
							'autoregressive_model', 'diffusion_model', 'tokenizer_json']
				filtered_settings = {k: v for k, v in settings.items() if k not in unused_args}
				
				gen = tts.tts(cut_text, **filtered_settings)
			else:
				gen, additionals = tts.tts(cut_text, **settings )
				parameters['seed'] = additionals[0]
		except Exception as e:
			raise RuntimeError(f'Possible latent mismatch: click the "(Re)Compute Voice Latents" button and then try again. Error: {e}')
		# print(type(gen))
		# print(gen)
		run_time = time.time()-start_time
		print(f"Generating line took {run_time} seconds")

		if not isinstance(gen, list):
			gen = [gen]

		for j, g in enumerate(gen):
			audio = g.squeeze(0).cpu()
			name = get_name(line=line, candidate=j)

			settings['text'] = cut_text
			settings['time'] = run_time
			settings['datetime'] = datetime.now().isoformat()
			if args.tts_backend == "tortoise":
				settings['model'] = tts.autoregressive_model_path
				settings['model_hash'] = tts.autoregressive_model_hash

			audio_cache[name] = {
				'audio': audio,
				'settings': get_info(voice=override['voice'] if override and 'voice' in override else voice, settings=settings)
			}
			# save here in case some error happens mid-batch
			torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', audio, tts.output_sample_rate)

	del gen
	do_gc()
	INFERENCING = False

	for k in audio_cache:
		audio = audio_cache[k]['audio']

		audio, _ = resample(audio, tts.output_sample_rate, args.output_sample_rate)
		if volume_adjust is not None:
			audio = volume_adjust(audio)

		audio_cache[k]['audio'] = audio
		torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{k}.wav', audio, args.output_sample_rate)

	output_voices = []
	for candidate in range(parameters['candidates']):
		if len(texts) > 1:
			audio_clips = []
			for line in range(len(texts)):
				name = get_name(line=line, candidate=candidate)
				audio = audio_cache[name]['audio']
				audio_clips.append(audio)
			
			name = get_name(candidate=candidate, combined=True)
			audio = torch.cat(audio_clips, dim=-1)
			torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', audio, args.output_sample_rate)

			audio = audio.squeeze(0).cpu()
			audio_cache[name] = {
				'audio': audio,
				'settings': get_info(voice=voice),
				'output': True
			}
		else:
			name = get_name(candidate=candidate)
			audio_cache[name]['output'] = True


	if args.voice_fixer:
		if not voicefixer:
			notify_progress("Loading voicefix...", progress=progress)
			load_voicefixer()

		try:
			fixed_cache = {}
			for name in tqdm(audio_cache, desc="Running voicefix..."):
				del audio_cache[name]['audio']
				if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
					continue

				path = f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav'
				fixed = f'{outdir}/{cleanup_voice_name(voice)}_{name}_fixed.wav'
				voicefixer.restore(
					input=path,
					output=fixed,
					cuda=get_device_name() == "cuda" and args.voice_fixer_use_cuda,
					#mode=mode,
				)
				
				fixed_cache[f'{name}_fixed'] = {
					'settings': audio_cache[name]['settings'],
					'output': True
				}
				audio_cache[name]['output'] = False
			
			for name in fixed_cache:
				audio_cache[name] = fixed_cache[name]
		except Exception as e:
			print(e)
			print("\nFailed to run Voicefixer")

	for name in audio_cache:
		if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
			if args.prune_nonfinal_outputs:
				audio_cache[name]['pruned'] = True
				os.remove(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')
			continue

		output_voices.append(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')

		if not args.embed_output_metadata:
			with open(f'{outdir}/{cleanup_voice_name(voice)}_{name}.json', 'w', encoding="utf-8") as f:
				f.write(json.dumps(audio_cache[name]['settings'], indent='\t') )

	if args.embed_output_metadata:
		for name in tqdm(audio_cache, desc="Embedding metadata..."):
			if 'pruned' in audio_cache[name] and audio_cache[name]['pruned']:
				continue

			metadata = music_tag.load_file(f"{outdir}/{cleanup_voice_name(voice)}_{name}.wav")
			metadata['lyrics'] = json.dumps(audio_cache[name]['settings'])
			metadata.save()
 
	if sample_voice is not None:
		sample_voice = (tts.input_sample_rate, sample_voice.numpy())

	info = get_info(voice=voice, latents=False)

	#insert rvc stuff
	if args.use_rvc:
		rvc_settings = load_rvc_settings()
		rvc_model_path = os.path.join("models", "rvc_models", rvc_settings['rvc_model'])
		rvc_index_path = os.path.join("models", "rvc_models", rvc_settings['file_index'])
		print (rvc_model_path)

		for i, output_voice in enumerate(output_voices):
			rvc_out_path = rvc_convert(model_path=rvc_model_path, 
										input_path=output_voice,
										f0_up_key=rvc_settings['f0_up_key'],
										file_index=rvc_index_path,
										index_rate=rvc_settings['index_rate'],
										filter_radius=rvc_settings['filter_radius'],
										resample_sr=rvc_settings['resample_sr'],
										rms_mix_rate=rvc_settings['rms_mix_rate'],
										protect=rvc_settings['protect'])
			
			# Read the contents from rvc_out_path
			with open(rvc_out_path, 'rb') as file:
				content = file.read()

			# Write the contents to output_voices[0], effectively replacing its contents
			with open(output_voice, 'wb') as file:
				file.write(content)


	print(f"Generation took {info['time']} seconds, saved to '{output_voices[0]}'\n")

	info['seed'] = usedSeed
	if 'latents' in info:
		del info['latents']

	os.makedirs('./config/', exist_ok=True)
	with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(info, indent='\t') )

	stats = [
		[ parameters['seed'], "{:.3f}".format(info['time']) ]
	]

	return (
		sample_voice,
		output_voices,
		stats,
	)

def cancel_generate():
	if not INFERENCING:
		return
		
	import tortoise.api

	tortoise.api.STOP_SIGNAL = True

def hash_file(path, algo="md5", buffer_size=0):
	hash = None
	if algo == "md5":
		hash = hashlib.md5()
	elif algo == "sha1":
		hash = hashlib.sha1()
	else:
		raise Exception(f'Unknown hash algorithm specified: {algo}')

	if not os.path.exists(path):
		raise Exception(f'Path not found: {path}')

	with open(path, 'rb') as f:
		if buffer_size > 0:
			while True:
				data = f.read(buffer_size)
				if not data:
					break
				hash.update(data)
		else:
			hash.update(f.read())

	return "{0}".format(hash.hexdigest())

def update_baseline_for_latents_chunks( voice ):
	global current_voice
	current_voice = voice

	path = f'{get_voice_dir()}/{voice}/'
	if not os.path.isdir(path):
		return 1

	dataset_file = f'./training/{voice}/train.txt'
	if os.path.exists(dataset_file):
		return 0 # 0 will leverage using the LJspeech dataset for computing latents

	files = os.listdir(path)
	
	total = 0
	total_duration = 0

	for file in files:
		if file[-4:] != ".wav":
			continue

		metadata = torchaudio.info(f'{path}/{file}')
		duration = metadata.num_frames / metadata.sample_rate
		total_duration += duration
		total = total + 1


	# brain too fried to figure out a better way
	if args.autocalculate_voice_chunk_duration_size == 0:
		return int(total_duration / total) if total > 0 else 1
	return int(total_duration / args.autocalculate_voice_chunk_duration_size) if total_duration > 0 else 1

def compute_latents(voice=None, voice_samples=None, voice_latents_chunks=0, original_ar=False, original_diffusion=False):
	global tts
	global args
	
	unload_whisper()
	unload_voicefixer()

	if not tts:
		if tts_loading:
			raise Exception("TTS is still initializing...")
		load_tts()

	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	if args.tts_backend == "bark":
		tts.create_voice( voice )
		return

	if args.autoregressive_model == "auto":
		tts.load_autoregressive_model(deduce_autoregressive_model(voice))

	if voice:
		load_from_dataset = voice_latents_chunks == 0

		if load_from_dataset:
			dataset_path = f'./training/{voice}/train.txt'
			if not os.path.exists(dataset_path):
				load_from_dataset = False
			else:
				with open(dataset_path, 'r', encoding="utf-8") as f:
					lines = f.readlines()

				print("Leveraging dataset for computing latents")

				voice_samples = []
				max_length = 0
				for line in lines:
					filename = f'./training/{voice}/{line.split("|")[0]}'
					
					waveform = load_audio(filename, 22050)
					max_length = max(max_length, waveform.shape[-1])
					voice_samples.append(waveform)

				for i in range(len(voice_samples)):
					voice_samples[i] = pad_or_truncate(voice_samples[i], max_length)

				voice_latents_chunks = len(voice_samples)
				if voice_latents_chunks == 0:
					print("Dataset is empty!")
					load_from_dataset = True
		if not load_from_dataset:
			voice_samples, _ = load_voice(voice, load_latents=False)

	if voice_samples is None:
		return

	conditioning_latents = tts.get_conditioning_latents(voice_samples, return_mels=not args.latents_lean_and_mean, slices=voice_latents_chunks, force_cpu=args.force_cpu_for_conditioning_latents, original_ar=original_ar, original_diffusion=original_diffusion)

	if len(conditioning_latents) == 4:
		conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)
	
	outfile = f'{get_voice_dir()}/{voice}/cond_latents_{tts.autoregressive_model_hash[:8]}.pth'
	torch.save(conditioning_latents, outfile)
	print(f'Saved voice latents: {outfile}')

	return conditioning_latents

# superfluous, but it cleans up some things
class TrainingState():
	def __init__(self, config_path, keep_x_past_checkpoints=0, start=True):
		self.killed = False
		
		self.training_dir = os.path.dirname(config_path)
		with open(config_path, 'r') as file:
			self.yaml_config = yaml.safe_load(file)

		self.json_config = json.load(open(f"{self.training_dir}/train.json", 'r', encoding="utf-8"))
		self.dataset_path = f"{self.training_dir}/train.txt"
		with open(self.dataset_path, 'r', encoding="utf-8") as f:
			self.dataset_size = len(f.readlines())

		self.batch_size = self.json_config["batch_size"]
		self.save_rate = self.json_config["save_rate"]

		self.epoch = 0
		self.epochs = self.json_config["epochs"]
		self.it = 0
		self.its = calc_iterations( self.epochs, self.dataset_size, self.batch_size )
		self.step = 0
		self.steps = int(self.its / self.dataset_size)
		self.checkpoint = 0
		self.checkpoints = int((self.its - self.it) / self.save_rate)

		self.gpus = self.json_config['gpus']

		self.buffer = []

		self.open_state = False
		self.training_started = False

		self.info = {}		
		
		self.it_rate = ""
		self.it_rates = 0
		
		self.epoch_rate = ""

		self.eta = "?"
		self.eta_hhmmss = "?"

		self.nan_detected = False

		self.last_info_check_at = 0
		self.statistics = {
			'loss': [],
			'lr': [],
			'grad_norm': [],
		}
		self.losses = []
		self.metrics = {
			'step': "",
			'rate': "",
			'loss': "",
		}

		self.loss_milestones = [ 1.0, 0.15, 0.05 ]

		if args.tts_backend=="vall-e":
			self.valle_last_it = 0
			self.valle_steps = 0

		if keep_x_past_checkpoints > 0:
			self.cleanup_old(keep=keep_x_past_checkpoints)
		if start:
			self.spawn_process(config_path=config_path, gpus=self.gpus)

	def spawn_process(self, config_path, gpus=1):
		if args.tts_backend == "vall-e":
			self.cmd = ['deepspeed', f'--num_gpus={gpus}', '--module', 'vall_e.train', f'yaml="{config_path}"']
		else:
			self.cmd = ['train.bat', config_path] if os.name == "nt" else ['./train.sh', config_path]

		print("Spawning process: ", " ".join(self.cmd))
		self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

	def parse_metrics(self, data):
		if isinstance(data, str):
			if line.find('Training Metrics:') >= 0:
				data = json.loads(line.split("Training Metrics:")[-1])
				data['mode'] = "training"
			elif line.find('Validation Metrics:') >= 0:
				data = json.loads(line.split("Validation Metrics:")[-1])
				data['mode'] = "validation"
			else:
				return

		self.info = data
		if 'epoch' in self.info:
			self.epoch = int(self.info['epoch'])
		if 'it' in self.info:
			self.it = int(self.info['it'])
		if 'step' in self.info:
			self.step = int(self.info['step'])
		if 'steps' in self.info:
			self.steps = int(self.info['steps'])

		if 'elapsed_time' in self.info:
			self.info['iteration_rate'] = self.info['elapsed_time']
			del self.info['elapsed_time']

		if 'iteration_rate' in self.info:
			it_rate = self.info['iteration_rate']
			self.it_rate = f'{"{:.3f}".format(1/it_rate)}it/s' if 0 < it_rate and it_rate < 1 else f'{"{:.3f}".format(it_rate)}s/it'
			self.it_rates += it_rate

			if self.it_rates > 0 and self.it * self.steps > 0:
				epoch_rate = self.it_rates / self.it * self.steps
				self.epoch_rate = f'{"{:.3f}".format(1/epoch_rate)}epoch/s' if 0 < epoch_rate and epoch_rate < 1 else f'{"{:.3f}".format(epoch_rate)}s/epoch'

			try:
				self.eta = (self.its - self.it) * (self.it_rates / self.it)
				eta = str(timedelta(seconds=int(self.eta)))
				self.eta_hhmmss = eta
			except Exception as e:
				self.eta_hhmmss = "?"
				pass

		self.metrics['step'] = [f"{self.epoch}/{self.epochs}"]
		if self.epochs != self.its:
			self.metrics['step'].append(f"{self.it}/{self.its}")
		if self.steps > 1:
			self.metrics['step'].append(f"{self.step}/{self.steps}")
		self.metrics['step'] = ", ".join(self.metrics['step'])

		if args.tts_backend == "tortoise":
			epoch = self.epoch + (self.step / self.steps)
		else:
			epoch = self.info['epoch'] if 'epoch' in self.info else self.it

		if self.it > 0:
			# probably can double for-loop but whatever
			keys = {
				'lrs': ['lr'],
				'losses': ['loss_text_ce', 'loss_mel_ce'],
				'accuracies': [],
				'precisions': [],
				'grad_norms': [],
			}
			if args.tts_backend == "vall-e":
				keys['lrs'] = [
					'ar.lr', 'nar.lr',
				]
				keys['losses'] = [
				#	'ar.loss', 'nar.loss', 'ar+nar.loss',
					'ar.loss.nll', 'nar.loss.nll',
				]

				keys['accuracies'] = [
					'ar.loss.acc', 'nar.loss.acc',
					'ar.stats.acc', 'nar.loss.acc',
				]
				keys['precisions'] = [ 'ar.loss.precision', 'nar.loss.precision', ]
				keys['grad_norms'] = ['ar.grad_norm', 'nar.grad_norm']

			for k in keys['lrs']:
				if k not in self.info:
					continue

				self.statistics['lr'].append({'epoch': epoch, 'it': self.it, 'value': self.info[k], 'type': k})

			for k in keys['accuracies']:
				if k not in self.info:
					continue

				self.statistics['loss'].append({'epoch': epoch, 'it': self.it, 'value': self.info[k], 'type': k})

			for k in keys['precisions']:
				if k not in self.info:
					continue

				self.statistics['loss'].append({'epoch': epoch, 'it': self.it, 'value': self.info[k], 'type': k})
			
			for k in keys['losses']:
				if k not in self.info:
					continue

				prefix = ""

				if "mode" in self.info and self.info["mode"] == "validation":
					prefix = f'{self.info["name"] if "name" in self.info else "val"}_'

				self.statistics['loss'].append({'epoch': epoch, 'it': self.it, 'value': self.info[k], 'type': f'{prefix}{k}' })

			self.losses.append( self.statistics['loss'][-1] )

			for k in keys['grad_norms']:
				if k not in self.info:
					continue
				self.statistics['grad_norm'].append({'epoch': epoch, 'it': self.it, 'value': self.info[k], 'type': k})

		return data

	def get_status(self):
		message = None

		self.metrics['rate'] = []
		if self.epoch_rate:
			self.metrics['rate'].append(self.epoch_rate)
		if self.it_rate and self.epoch_rate[:-7] != self.it_rate[:-4]:
			self.metrics['rate'].append(self.it_rate)
		self.metrics['rate'] = ", ".join(self.metrics['rate'])

		eta_hhmmss = self.eta_hhmmss if self.eta_hhmmss else "?"

		self.metrics['loss'] = []
		if 'lr' in self.info:
			self.metrics['loss'].append(f'LR: {"{:.3e}".format(self.info["lr"])}')

		if len(self.losses) > 0:
			self.metrics['loss'].append(f'Loss: {"{:.3f}".format(self.losses[-1]["value"])}')

		if False and len(self.losses) >= 2:
			deriv = 0
			accum_length = len(self.losses)//2 # i *guess* this is fine when you think about it
			loss_value = self.losses[-1]["value"]

			for i in range(accum_length):
				d1_loss = self.losses[accum_length-i-1]["value"]
				d2_loss = self.losses[accum_length-i-2]["value"]
				dloss = (d2_loss - d1_loss)

				d1_step = self.losses[accum_length-i-1]["it"]
				d2_step = self.losses[accum_length-i-2]["it"]
				dstep = (d2_step - d1_step)

				if dstep == 0:
					continue
		
				inst_deriv = dloss / dstep
				deriv += inst_deriv

			deriv = deriv / accum_length

			print("Deriv: ", deriv)

			if deriv != 0: # dloss < 0:
				next_milestone = None
				for milestone in self.loss_milestones:
					if loss_value > milestone:
						next_milestone = milestone
						break

				print(f"Loss value: {loss_value} | Next milestone: {next_milestone} | Distance: {loss_value - next_milestone}")
						
				if next_milestone:
					# tfw can do simple calculus but not basic algebra in my head
					est_its = (next_milestone - loss_value) / deriv * 100
					print(f"Estimated: {est_its}")
					if est_its >= 0:
						self.metrics['loss'].append(f'Est. milestone {next_milestone} in: {int(est_its)}its')
				else:
					est_loss = inst_deriv * (self.its - self.it) + loss_value
					if est_loss >= 0:
						self.metrics['loss'].append(f'Est. final loss: {"{:.3f}".format(est_loss)}')

		self.metrics['loss'] = ", ".join(self.metrics['loss'])

		message = f"[{self.metrics['step']}] [{self.metrics['rate']}] [ETA: {eta_hhmmss}] [{self.metrics['loss']}]"
		if self.nan_detected:
			message = f"[!NaN DETECTED! {self.nan_detected}] {message}"

		return message

	def load_statistics(self, update=False):
		if not os.path.isdir(self.training_dir):
			return

		if args.tts_backend == "tortoise":
			logs = sorted([f'{self.training_dir}/finetune/{d}' for d in os.listdir(f'{self.training_dir}/finetune/') if d[-4:] == ".log" ])
		else:
			log_dir = "logs"
			logs = sorted([f'{self.training_dir}/{log_dir}/{d}/log.txt' for d in os.listdir(f'{self.training_dir}/{log_dir}/') ])

		if update:
			logs = [logs[-1]]

		infos = {}
		highest_step = self.last_info_check_at

		if not update:
			self.statistics['loss'] = []
			self.statistics['lr'] = []
			self.statistics['grad_norm'] = []
			self.it_rates = 0

		unq = {}
		averager = None
		prev_state = 0

		for log in logs:
			with open(log, 'r', encoding="utf-8") as f:
				lines = f.readlines()

			for line in lines:
				line = line.strip()
				if not line:
					continue
					
				if line[-1] == ".":
					line = line[:-1]

				if line.find('Training Metrics:') >= 0:
					split = line.split("Training Metrics:")[-1]
					data = json.loads(split)
					
					name = "train"
					mode = "training"
					prev_state = 0
				elif line.find('Validation Metrics:') >= 0:
					data = json.loads(line.split("Validation Metrics:")[-1])
					if "it" not in data:
						data['it'] = it
					if "epoch" not in data:
						data['epoch'] = epoch

					# name = data['name'] if 'name' in data else "val"
					mode = "validation"

					if prev_state == 0:
						name = "subtrain"
					else:
						name = "val"

					prev_state += 1
				else:
					continue

				if "it" not in data:
					continue
				
				it = data['it']
				epoch = data['epoch']
				
				if args.tts_backend == "vall-e":
					if not averager or averager['key'] != f'{it}_{name}' or averager['mode'] != mode:
						averager = {
							'key': f'{it}_{name}',
							'name': name,
							'mode': mode,
							"metrics": {}
						}
						for k in data:
							if data[k] is None:
								continue
							averager['metrics'][k] = [ data[k] ]
					else:
						for k in data:
							if data[k] is None:
								continue
							if k not in averager['metrics']:
								averager['metrics'][k] = [ data[k] ]
							else:
								averager['metrics'][k].append( data[k] )

					unq[f'{it}_{mode}_{name}'] = averager
				else:
					unq[f'{it}_{mode}_{name}'] = data

				if update and it <= self.last_info_check_at:
					continue
		
		blacklist = [ "batch", "eval" ]
		for it in unq:
			if args.tts_backend == "vall-e":
				stats = unq[it]
				data = {k: sum(v) / len(v) for k, v in stats['metrics'].items() if k not in blacklist }
				#data = {k: min(v) for k, v in stats['metrics'].items() if k not in blacklist }
				#data = {k: max(v) for k, v in stats['metrics'].items() if k not in blacklist }
				data['name'] = stats['name']
				data['mode'] = stats['mode']
				data['steps'] = len(stats['metrics']['it'])
			else:
				data = unq[it]
			self.parse_metrics(data)

		self.last_info_check_at = highest_step

	def cleanup_old(self, keep=2):
		if keep <= 0:
			return

		if args.tts_backend == "vall-e":
			return

		if not os.path.isdir(f'{self.training_dir}/finetune/'):
			return
			
		models = sorted([ int(d[:-8]) for d in os.listdir(f'{self.training_dir}/finetune/models/') if d[-8:] == "_gpt.pth" ])
		states = sorted([ int(d[:-6]) for d in os.listdir(f'{self.training_dir}/finetune/training_state/') if d[-6:] == ".state" ])
		remove_models = models[:-keep]
		remove_states = states[:-keep]

		for d in remove_models:
			path = f'{self.training_dir}/finetune/models/{d}_gpt.pth'
			print("Removing", path)
			os.remove(path)
		for d in remove_states:
			path = f'{self.training_dir}/finetune/training_state/{d}.state'
			print("Removing", path)
			os.remove(path)

	def parse(self, line, verbose=False, keep_x_past_checkpoints=0, buffer_size=8, progress=None ):
		self.buffer.append(f'{line}')

		data = None
		percent = 0
		message = None
		should_return = False

		MESSAGE_START = 'Start training from epoch'
		MESSAGE_FINSIHED = 'Finished training'
		MESSAGE_SAVING = 'Saving models and training states.'

		MESSAGE_METRICS_TRAINING = 'Training Metrics:'
		MESSAGE_METRICS_VALIDATION = 'Validation Metrics:'

		if line.find(MESSAGE_FINSIHED) >= 0:
			self.killed = True
		# rip out iteration info
		elif not self.training_started:
			if line.find(MESSAGE_START) >= 0:
				self.training_started = True # could just leverage the above variable, but this is python, and there's no point in these aggressive microoptimizations

				match = re.findall(r'epoch: ([\d,]+)', line)
				if match and len(match) > 0:
					self.epoch = int(match[0].replace(",", ""))
				match = re.findall(r'iter: ([\d,]+)', line)
				if match and len(match) > 0:
					self.it = int(match[0].replace(",", ""))

				self.checkpoints = int((self.its - self.it) / self.save_rate)

				self.load_statistics()

				should_return = True
		else:
			if line.find(MESSAGE_SAVING) >= 0:
				self.checkpoint += 1
				message = f"[{self.checkpoint}/{self.checkpoints}] Saving checkpoint..."
				percent = self.checkpoint / self.checkpoints

				self.cleanup_old(keep=keep_x_past_checkpoints)
			elif line.find(MESSAGE_METRICS_TRAINING) >= 0:
				data = json.loads(line.split(MESSAGE_METRICS_TRAINING)[-1])
				data['mode'] = "training"
			elif line.find(MESSAGE_METRICS_VALIDATION) >= 0:
				data = json.loads(line.split(MESSAGE_METRICS_VALIDATION)[-1])
				data['mode'] = "validation"

		if data is not None:
			if ': nan' in line and not self.nan_detected:
				self.nan_detected = self.it
			
			self.parse_metrics( data )
			message = self.get_status()
			
			if message:
				percent = self.it / float(self.its) # self.epoch / float(self.epochs)
				if progress is not None:
					progress(percent, message)

				self.buffer.append(f'[{"{:.3f}".format(percent*100)}%] {message}')
				should_return = True

		if verbose and not self.training_started:
			should_return = True

		self.buffer = self.buffer[-buffer_size:]
		
		result = None
		if should_return:
			result = "".join(self.buffer) if not self.training_started else message

		return (
			result,
			percent,
			message,
		)

try:
	import altair as alt
	alt.data_transformers.enable('default', max_rows=None)
except Exception as e:
	print(e)
	pass

def run_training(config_path, verbose=False, keep_x_past_checkpoints=0, progress=gr.Progress(track_tqdm=True)):
	global training_state
	if training_state and training_state.process:
		return "Training already in progress"


	# ensure we have the dvae.pth
	if args.tts_backend == "tortoise":
		get_model_path('dvae.pth')
	
	# I don't know if this is still necessary, as it was bitching at me for not doing this, despite it being in a separate process
	torch.multiprocessing.freeze_support()

	unload_tts()
	unload_whisper()
	unload_voicefixer()

	training_state = TrainingState(config_path=config_path, keep_x_past_checkpoints=keep_x_past_checkpoints)

	for line in iter(training_state.process.stdout.readline, ""):
		if training_state is None or training_state.killed:
			return

		result, percent, message = training_state.parse( line=line, verbose=verbose, keep_x_past_checkpoints=keep_x_past_checkpoints, progress=progress )
		print(f"[Training] [{datetime.now().isoformat()}] {line[:-1]}")
		if result:
			yield result

			if progress is not None and message:
				progress(percent, message)

	if training_state:
		training_state.process.stdout.close()
		return_code = training_state.process.wait()
		training_state = None

def update_training_dataplot(x_min=None, x_max=None, y_min=None, y_max=None, config_path=None):
	global training_state
	losses = None
	lrs = None
	grad_norms = None

	x_lim = [ x_min, x_max ]
	y_lim = [ y_min, y_max ]

	if not training_state:
		if config_path:
			training_state = TrainingState(config_path=config_path, start=False)
			training_state.load_statistics()
			message = training_state.get_status()
	
	if training_state:
		if not x_lim[-1]:
			x_lim[-1] = training_state.epochs

		if not y_lim[-1]:
			y_lim = None

		if len(training_state.statistics['loss']) > 0:
			losses = gr.LinePlot(
				value = pd.DataFrame(training_state.statistics['loss']),
				x_lim=x_lim, y_lim=y_lim,
				x="epoch", y="value", # x="it",
				title="Loss Metrics", color="type", tooltip=['epoch', 'it', 'value', 'type'],
				width=500, height=350
			)
		if len(training_state.statistics['lr']) > 0:
			lrs = gr.LinePlot(
				value = pd.DataFrame(training_state.statistics['lr']),
				x_lim=x_lim,
				x="epoch", y="value", # x="it",
				title="Learning Rate", color="type", tooltip=['epoch', 'it', 'value', 'type'],
				width=500, height=350
			)
		if len(training_state.statistics['grad_norm']) > 0:
			grad_norms = gr.LinePlot(
				value = pd.DataFrame(training_state.statistics['grad_norm']),
				x_lim=x_lim,
				x="epoch", y="value", # x="it",
				title="Gradient Normals", color="type", tooltip=['epoch', 'it', 'value', 'type'],
				width=500, height=350
			)
	
	if config_path:
		del training_state
		training_state = None

	return (losses, lrs, grad_norms)

def reconnect_training(verbose=False, progress=gr.Progress(track_tqdm=True)):
	global training_state
	if not training_state or not training_state.process:
		return "Training not in progress"

	for line in iter(training_state.process.stdout.readline, ""):
		result, percent, message = training_state.parse( line=line, verbose=verbose, progress=progress )
		print(f"[Training] [{datetime.now().isoformat()}] {line[:-1]}")
		if result:
			yield result

			if progress is not None and message:
				progress(percent, message)

def stop_training():
	global training_state
	if training_state is None:
		return "No training in progress"
	print("Killing training process...")
	training_state.killed = True

	children = []
	if args.tts_backend == "tortoise":
		# wrapped in a try/catch in case for some reason this fails outside of Linux
		try:
			children = [p.info for p in psutil.process_iter(attrs=['pid', 'name', 'cmdline']) if './src/train.py' in p.info['cmdline']]
		except Exception as e:
			pass

		training_state.process.stdout.close()
		training_state.process.terminate()
		training_state.process.kill()
	elif args.tts_backend == "vall-e":
		print(training_state.process.communicate(input='quit')[0])

	return_code = training_state.process.wait()

	for p in children:
		os.kill( p['pid'], signal.SIGKILL )

	training_state = None
	print("Killed training process.")
	return f"Training cancelled: {return_code}"

def get_halfp_model_path():
	autoregressive_model_path = get_model_path('autoregressive.pth')
	return autoregressive_model_path.replace(".pth", "_half.pth")

def convert_to_halfp():
	autoregressive_model_path = get_model_path('autoregressive.pth')
	print(f'Converting model to half precision: {autoregressive_model_path}')
	model = torch.load(autoregressive_model_path)
	for k in model:
		model[k] = model[k].half()

	outfile = get_halfp_model_path()
	torch.save(model, outfile)
	print(f'Converted model to half precision: {outfile}')


# collapses short segments into the previous segment
def whisper_sanitize( results ):
	sanitized = json.loads(json.dumps(results))
	sanitized['segments'] = []

	for segment in results['segments']:
		length = segment['end'] - segment['start']
		if length >= MIN_TRAINING_DURATION or len(sanitized['segments']) == 0:
			sanitized['segments'].append(segment)
			continue

		last_segment = sanitized['segments'][-1]
		# segment already asimilitated it, somehow
		if last_segment['end'] >= segment['end']:
			continue
		"""
		# segment already asimilitated it, somehow
		if last_segment['text'].endswith(segment['text']):
			continue
		"""
		last_segment['text'] += segment['text']
		last_segment['end'] = segment['end']

	for i in range(len(sanitized['segments'])):
		sanitized['segments'][i]['id'] = i

	return sanitized

def whisper_transcribe( file, language=None ):
	# shouldn't happen, but it's for safety
	global whisper_model
	global whisper_align_model

	if not whisper_model:
		load_whisper_model(language=language)

	if args.whisper_backend == "openai/whisper":
		if not language:
			language = None

		return whisper_model.transcribe(file, language=language)

	if args.whisper_backend == "lightmare/whispercpp":
		res = whisper_model.transcribe(file)
		segments = whisper_model.extract_text_and_timestamps( res )

		result = {
			'text': [],
			'segments': []
		}
		for segment in segments:
			reparsed = {
				'start': segment[0] / 100.0,
				'end': segment[1] / 100.0,
				'text': segment[2],
				'id': len(result['segments'])
			}
			result['text'].append( segment[2] )
			result['segments'].append(reparsed)

		result['text'] = " ".join(result['text'])
		return result

	if args.whisper_backend == "m-bain/whisperx":
		import whisperx

		device = "cuda" if get_device_name() == "cuda" else "cpu"
		if language == "ja":
			# This is to prevent whisperx from segmenting audio that is too long for tortoise to handle.  Check changelog 1/30/2024
			chunk_size = 8
		else:
			chunk_size = 30
		result = whisper_model.transcribe(file, batch_size=args.whisper_batchsize, language=language, chunk_size=chunk_size)
			
		align_model, metadata = whisper_align_model
		result_aligned = whisperx.align(result["segments"], align_model, metadata, file, device, return_char_alignments=False)

		result['segments'] = result_aligned['segments']
		result['text'] = []
		for segment in result['segments']:
			segment['id'] = len(result['text'])
			result['text'].append(segment['text'].strip())
		result['text'] = " ".join(result['text'])

		return result

def validate_waveform( waveform, sample_rate, min_only=False ):
	if not torch.any(waveform < 0):
		return "Waveform is empty"

	num_channels, num_frames = waveform.shape
	duration = num_frames / sample_rate
	
	if duration < MIN_TRAINING_DURATION:
		return "Duration too short ({:.3f}s < {:.3f}s)".format(duration, MIN_TRAINING_DURATION)

	if not min_only:
		if duration > MAX_TRAINING_DURATION:
			return "Duration too long ({:.3f}s < {:.3f}s)".format(MAX_TRAINING_DURATION, duration)

	return

def transcribe_dataset( voice, language=None, skip_existings=False, progress=None ):
	unload_tts()

	global whisper_model
	if whisper_model is None:
		load_whisper_model(language=language)

	results = {}

	files = get_voice(voice, load_latents=False)
	indir = f'./training/{voice}/'
	infile = f'{indir}/whisper.json'

	quantize_in_memory = args.tts_backend == "vall-e"
	
	os.makedirs(f'{indir}/audio/', exist_ok=True)
	
	TARGET_SAMPLE_RATE = 22050
	if args.tts_backend != "tortoise":
		TARGET_SAMPLE_RATE = 24000
	if tts:
		TARGET_SAMPLE_RATE = tts.input_sample_rate

	if os.path.exists(infile):
		results = json.load(open(infile, 'r', encoding="utf-8"))

	for file in tqdm(files, desc="Iterating through voice files"):
		basename = os.path.basename(file)

		if basename in results and skip_existings:
			print(f"Skipping already parsed file: {basename}")
			continue

		try:
			result = whisper_transcribe(file, language=language)
		except Exception as e:
			print("Failed to transcribe:", file, e)
			continue

		results[basename] = result

		if not quantize_in_memory:
			waveform, sample_rate = torchaudio.load(file)
			# resample to the input rate, since it'll get resampled for training anyways
			# this should also "help" increase throughput a bit when filling the dataloaders
			waveform, sample_rate = resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
			if waveform.shape[0] == 2:
				waveform = waveform[:1]
			
			try:
				kwargs = {}
				if basename[-4:] == ".wav":
					kwargs['encoding'] = "PCM_S"
					kwargs['bits_per_sample'] = 16

				torchaudio.save(f"{indir}/audio/{basename}", waveform, sample_rate, **kwargs)
			except Exception as e:
				print(e)

		with open(infile, 'w', encoding="utf-8") as f:
			f.write(json.dumps(results, indent='\t'))

		do_gc()

	modified = False
	for basename in results:
		try:
			sanitized = whisper_sanitize(results[basename])
			if len(sanitized['segments']) > 0 and len(sanitized['segments']) != len(results[basename]['segments']):
				results[basename] = sanitized
				modified = True
				print("Segments sanizited: ", basename)
		except Exception as e:
			print("Failed to sanitize:", basename, e)
			pass

	if modified:
		os.rename(infile, infile.replace(".json", ".unsanitized.json"))
		with open(infile, 'w', encoding="utf-8") as f:
			f.write(json.dumps(results, indent='\t'))

	return f"Processed dataset to: {indir}"

def slice_waveform( waveform, sample_rate, start, end, trim ):
	start = int(start * sample_rate)
	end = int(end * sample_rate)

	if start < 0:
		start = 0
	if end >= waveform.shape[-1]:
		end = waveform.shape[-1] - 1

	sliced = waveform[:, start:end]

	error = validate_waveform( sliced, sample_rate, min_only=True )
	if trim and not error:
		sliced = torchaudio.functional.vad( sliced, sample_rate )

	return sliced, error

def slice_dataset( voice, trim_silence=True, start_offset=0, end_offset=0, results=None, progress=gr.Progress() ):
	indir = f'./training/{voice}/'
	infile = f'{indir}/whisper.json'
	messages = []

	if not os.path.exists(infile):
		message = f"Missing dataset: {infile}"
		print(message)
		return message

	if results is None:
		results = json.load(open(infile, 'r', encoding="utf-8"))

	TARGET_SAMPLE_RATE = 22050
	if args.tts_backend != "tortoise":
		TARGET_SAMPLE_RATE = 24000
	if tts:
		TARGET_SAMPLE_RATE = tts.input_sample_rate

	files = 0
	segments = 0
	for filename in results:
		path = f'./voices/{voice}/{filename}'
		extension = os.path.splitext(filename)[-1][1:]
		out_extension = extension # "wav"

		if not os.path.exists(path):
			path = f'./training/{voice}/{filename}'

		if not os.path.exists(path):
			message = f"Missing source audio: {filename}"
			print(message)
			messages.append(message)
			continue

		files += 1
		result = results[filename]
		waveform, sample_rate = torchaudio.load(path)
		num_channels, num_frames = waveform.shape
		duration = num_frames / sample_rate

		for segment in result['segments']: 
			file = filename.replace(f".{extension}", f"_{pad(segment['id'], 4)}.{out_extension}")
			
			sliced, error = slice_waveform( waveform, sample_rate, segment['start'] + start_offset, segment['end'] + end_offset, trim_silence )
			if error:
				message = f"{error}, skipping... {file}"
				print(message)
				messages.append(message)
				continue
		
			sliced, _ = resample( sliced, sample_rate, TARGET_SAMPLE_RATE )

			if waveform.shape[0] == 2:
				waveform = waveform[:1]
				
			kwargs = {}
			if file[-4:] == ".wav":
				kwargs['encoding'] = "PCM_S"
				kwargs['bits_per_sample'] = 16

			torchaudio.save(f"{indir}/audio/{file}", sliced, TARGET_SAMPLE_RATE, **kwargs)
			
			segments +=1

	messages.append(f"Sliced segments: {files} => {segments}.")
	return "\n".join(messages)

# takes an LJSpeech-dataset-formatted .txt file and phonemize it
def phonemize_txt_file( path ):
	with open(path, 'r', encoding='utf-8') as f:
		lines = f.readlines()

	reparsed = []
	with open(path.replace(".txt", ".phn.txt"), 'a', encoding='utf-8') as f:
		for line in tqdm(lines, desc='Phonemizing...'):
			split = line.split("|")
			audio = split[0]
			text = split[2]

			phonemes = phonemizer( text )
			reparsed.append(f'{audio}|{phonemes}')
			f.write(f'\n{audio}|{phonemes}')
	

	joined = "\n".join(reparsed)
	with open(path.replace(".txt", ".phn.txt"), 'w', encoding='utf-8') as f:
		f.write(joined)

	return joined

# takes an LJSpeech-dataset-formatted .txt (and phonemized .phn.txt from the above) and creates a JSON that should slot in as whisper.json
def create_dataset_json( path ):
	with open(path, 'r', encoding='utf-8') as f:
		lines = f.readlines()

	phonemes = None
	phn_path = path.replace(".txt", ".phn.txt")
	if os.path.exists(phn_path):
		with open(phn_path, 'r', encoding='utf-8') as f:
			phonemes = f.readlines()

	data = {}

	for line in lines:
		split = line.split("|")
		audio = split[0]
		text = split[1]

		data[audio] = {
			'text': text.strip()
		}

	for line in phonemes:
		split = line.split("|")
		audio = split[0]
		text = split[1]

		data[audio]['phonemes'] = text.strip()

	with open(path.replace(".txt", ".json"), 'w', encoding='utf-8') as f:
		f.write(json.dumps(data, indent="\t"))


cached_backends = {}

def phonemizer( text, language="en-us" ):
	from phonemizer import phonemize
	from phonemizer.backend import BACKENDS

	def _get_backend( language="en-us", backend="espeak" ):
		key = f'{language}_{backend}'
		if key in cached_backends:
			return cached_backends[key]

		if backend == 'espeak':
			phonemizer = BACKENDS[backend]( language, preserve_punctuation=True, with_stress=True)
		elif backend == 'espeak-mbrola':
			phonemizer = BACKENDS[backend]( language )
		else: 
			phonemizer = BACKENDS[backend]( language, preserve_punctuation=True )

		cached_backends[key] = phonemizer
		return phonemizer
	if language == "en":
		language = "en-us"

	backend = _get_backend(language=language, backend=args.phonemizer_backend)
	if backend is not None:
		tokens = backend.phonemize( [text], strip=True )
	else:
		tokens = phonemize( [text], language=language, strip=True, preserve_punctuation=True, with_stress=True )

	return tokens[0] if len(tokens) == 0 else tokens
	tokenized = " ".join( tokens )

def should_phonemize():
	if args.tts_backend == "vall-e":
		return False
		
	should = args.tokenizer_json is not None and args.tokenizer_json[-8:] == "ipa.json"
	if should:
		try:
			from phonemizer import phonemize
		except Exception as e:
			return False
	return should

def prepare_dataset( voice, use_segments=False, text_length=0, audio_length=0, progress=gr.Progress() ):
	indir = f'./training/{voice}/'
	infile = f'{indir}/whisper.json'
	if not os.path.exists(infile):
		message = f"Missing dataset: {infile}"
		print(message)
		return message

	results = json.load(open(infile, 'r', encoding="utf-8"))

	errored = 0
	messages = []
	normalize = False # True
	phonemize = should_phonemize()
	lines = { 'training': [], 'validation': [] }
	segments = {}

	quantize_in_memory = args.tts_backend == "vall-e"

	if args.tts_backend != "tortoise":
		text_length = 0
		audio_length = 0

	start_offset = -0.1
	end_offset = 0.1
	trim_silence = False

	TARGET_SAMPLE_RATE = 22050
	if args.tts_backend != "tortoise":
		TARGET_SAMPLE_RATE = 24000
	if tts:
		TARGET_SAMPLE_RATE = tts.input_sample_rate

	for filename in tqdm(results, desc="Parsing results"):
		use_segment = use_segments

		extension = os.path.splitext(filename)[-1][1:]
		out_extension = extension # "wav"
		result = results[filename]
		lang = result['language']
		language = LANGUAGES[lang] if lang in LANGUAGES else lang
		normalizer = EnglishTextNormalizer() if language and language == "english" else BasicTextNormalizer()

		# check if unsegmented text exceeds 200 characters
		if not use_segment:
			if len(result['text']) > MAX_TRAINING_CHAR_LENGTH:
				message = f"Text length too long ({MAX_TRAINING_CHAR_LENGTH} < {len(result['text'])}), using segments: {filename}"
				print(message)
				messages.append(message)
				use_segment = True

		# check if unsegmented audio exceeds 11.6s
		if not use_segment:
			path = f'{indir}/audio/{filename}'
			if not quantize_in_memory and not os.path.exists(path):
				messages.append(f"Missing source audio: {filename}")
				errored += 1
				continue

			duration = 0
			for segment in result['segments']:
				duration = max(duration, segment['end'])

			if duration >= MAX_TRAINING_DURATION:
				message = f"Audio too large, using segments: {filename}"
				print(message)
				messages.append(message)
				use_segment = True

		# implicitly segment
		if use_segment and not use_segments:
			exists = True
			for segment in result['segments']:
				duration = segment['end'] - segment['start']
				if duration <= MIN_TRAINING_DURATION or MAX_TRAINING_DURATION <= duration:
					continue

				path = f'{indir}/audio/' + filename.replace(f".{extension}", f"_{pad(segment['id'], 4)}.{out_extension}")
				if os.path.exists(path):
					continue
				exists = False
				break

			if not quantize_in_memory and not exists:
				tmp = {}
				tmp[filename] = result
				print(f"Audio not segmented, segmenting: {filename}")
				message = slice_dataset( voice, results=tmp )
				print(message)
				messages = messages + message.split("\n")
		
		waveform = None
		

		if quantize_in_memory:
			path = f'{indir}/audio/{filename}'
			if not os.path.exists(path):
				path = f'./voices/{voice}/{filename}'

			if not os.path.exists(path):
				message = f"Audio not found: {path}"
				print(message)
				messages.append(message)
				#continue
			else:
				waveform = torchaudio.load(path)
				waveform = resample(waveform[0], waveform[1], TARGET_SAMPLE_RATE)

		if not use_segment:
			segments[filename] = {
				'text': result['text'],
				'lang': lang,
				'language': language,
				'normalizer': normalizer,
				'phonemes': result['phonemes'] if 'phonemes' in result else None
			}

			if waveform:
				segments[filename]['waveform'] = waveform
		else:
			for segment in result['segments']:
				duration = segment['end'] - segment['start']
				if duration <= MIN_TRAINING_DURATION or MAX_TRAINING_DURATION <= duration:
					continue

				file = filename.replace(f".{extension}", f"_{pad(segment['id'], 4)}.{out_extension}")

				segments[file] = {
					'text': segment['text'],
					'lang': lang,
					'language': language,
					'normalizer': normalizer,
					'phonemes': segment['phonemes'] if 'phonemes' in segment else None
				}

				if waveform:
					sliced, error = slice_waveform( waveform[0], waveform[1], segment['start'] + start_offset, segment['end'] + end_offset, trim_silence )
					if error:
						message = f"{error}, skipping... {file}"
						print(message)
						messages.append(message)
						segments[file]['error'] = error
						#continue
					else:
						segments[file]['waveform'] = (sliced, waveform[1])

	jobs = {
		'quantize':  [[], []],
		'phonemize': [[], []],
	}

	for file in tqdm(segments, desc="Parsing segments"):
		extension = os.path.splitext(file)[-1][1:]
		result = segments[file]
		path = f'{indir}/audio/{file}'

		text = result['text']
		lang = result['lang']
		language = result['language']
		normalizer = result['normalizer']
		phonemes = result['phonemes']
		if phonemize and phonemes is None:
			phonemes = phonemizer( text, language=lang )
		
		normalized = normalizer(text) if normalize else text

		if len(text) > MAX_TRAINING_CHAR_LENGTH:
			message = f"Text length too long ({MAX_TRAINING_CHAR_LENGTH} < {len(text)}), skipping... {file}"
			print(message)
			messages.append(message)
			errored += 1
			continue

		# num_channels, num_frames = waveform.shape
		#duration = num_frames / sample_rate


		culled = len(text) < text_length
		if not culled and audio_length > 0:
			culled = duration < audio_length

		line = f'audio/{file}|{phonemes if phonemize and phonemes else text}'

		lines['training' if not culled else 'validation'].append(line) 

		if culled or args.tts_backend != "vall-e":
			continue
		
		os.makedirs(f'{indir}/valle/', exist_ok=True)
		#os.makedirs(f'./training/valle/data/{voice}/', exist_ok=True)

		phn_file = f'{indir}/valle/{file.replace(f".{extension}",".phn.txt")}'
		#phn_file = f'./training/valle/data/{voice}/{file.replace(f".{extension}",".phn.txt")}'
		if not os.path.exists(phn_file):
			jobs['phonemize'][0].append(phn_file)
			jobs['phonemize'][1].append(normalized)
			"""
			phonemized = valle_phonemize( normalized )
			open(f'{indir}/valle/{file.replace(".wav",".phn.txt")}', 'w', encoding='utf-8').write(" ".join(phonemized))
			print("Phonemized:", file, normalized, text)
			"""

		qnt_file = f'{indir}/valle/{file.replace(f".{extension}",".qnt.pt")}'
		#qnt_file = f'./training/valle/data/{voice}/{file.replace(f".{extension}",".qnt.pt")}'
		if 'error' not in result:
			if not quantize_in_memory and not os.path.exists(path):
				message = f"Missing segment, skipping... {file}"
				print(message)
				messages.append(message)
				errored += 1
				continue

		if not os.path.exists(qnt_file):
			waveform = None
			if 'waveform' in result:
				waveform, sample_rate = result['waveform']
			elif os.path.exists(path):
				waveform, sample_rate = torchaudio.load(path)
				error = validate_waveform( waveform, sample_rate )
				if error:
					message = f"{error}, skipping... {file}"
					print(message)
					messages.append(message)
					errored += 1
					continue

			if waveform is not None:
				jobs['quantize'][0].append(qnt_file)
				jobs['quantize'][1].append((waveform, sample_rate))
				"""
				quantized = valle_quantize( waveform, sample_rate ).cpu()
				torch.save(quantized, f'{indir}/valle/{file.replace(".wav",".qnt.pt")}')
				print("Quantized:", file)
				"""

	for i in tqdm(range(len(jobs['quantize'][0])), desc="Quantizing"):
		qnt_file = jobs['quantize'][0][i]
		waveform, sample_rate = jobs['quantize'][1][i]

		quantized = valle_quantize( waveform, sample_rate ).cpu()
		torch.save(quantized, qnt_file)
		#print("Quantized:", qnt_file)

	for i in tqdm(range(len(jobs['phonemize'][0])), desc="Phonemizing"):
		phn_file = jobs['phonemize'][0][i]
		normalized = jobs['phonemize'][1][i]

		if language == "japanese":
			language = "ja"

		if language == "ja" and PYKAKASI_ENABLED and KKS is not None:
			normalized = KKS.convert(normalized)
			normalized = [ n["hira"] for n in normalized ]
			normalized = "".join(normalized)

		try:
			phonemized = valle_phonemize( normalized )
			open(phn_file, 'w', encoding='utf-8').write(" ".join(phonemized))
			#print("Phonemized:", phn_file)
		except Exception as e:
			message = f"Failed to phonemize: {phn_file}: {normalized}"
			messages.append(message)
			print(message)


	training_joined = "\n".join(lines['training'])
	validation_joined = "\n".join(lines['validation'])

	with open(f'{indir}/train.txt', 'w', encoding="utf-8") as f:
		f.write(training_joined)

	with open(f'{indir}/validation.txt', 'w', encoding="utf-8") as f:
		f.write(validation_joined)

	messages.append(f"Prepared {len(lines['training'])} lines (validation: {len(lines['validation'])}, culled: {errored}).\n{training_joined}\n\n{validation_joined}")
	return "\n".join(messages)

def calc_iterations( epochs, lines, batch_size ):
	return int(math.ceil(epochs * math.ceil(lines / batch_size)))

def schedule_learning_rate( iterations, schedule=LEARNING_RATE_SCHEDULE ):
	return [int(iterations * d) for d in schedule]

def optimize_training_settings( **kwargs ):
	messages = []
	settings = {}
	settings.update(kwargs)

	dataset_path = f"./training/{settings['voice']}/train.txt"
	with open(dataset_path, 'r', encoding="utf-8") as f:
		lines = len(f.readlines())

	if lines == 0:
		raise Exception("Empty dataset.")

	if settings['batch_size'] > lines:
		settings['batch_size'] = lines
		messages.append(f"Batch size is larger than your dataset, clamping batch size to: {settings['batch_size']}")	

	"""
	if lines % settings['batch_size'] != 0:
		settings['batch_size'] = int(lines / settings['batch_size'])
		if settings['batch_size'] == 0:
			settings['batch_size'] = 1
		messages.append(f"Batch size not neatly divisible by dataset size, adjusting batch size to: {settings['batch_size']}")
	"""
	if settings['gradient_accumulation_size'] == 0:
		settings['gradient_accumulation_size'] = 1
	
	if settings['batch_size'] / settings['gradient_accumulation_size'] < 2:
		settings['gradient_accumulation_size'] = int(settings['batch_size'] / 2)
		if settings['gradient_accumulation_size'] == 0:
			settings['gradient_accumulation_size'] = 1

		messages.append(f"Gradient accumulation size is too large for a given batch size, clamping gradient accumulation size to: {settings['gradient_accumulation_size']}")
	elif settings['batch_size'] % settings['gradient_accumulation_size'] != 0:
		settings['gradient_accumulation_size'] -= settings['batch_size'] % settings['gradient_accumulation_size']
		if settings['gradient_accumulation_size'] == 0:
			settings['gradient_accumulation_size'] = 1

		messages.append(f"Batch size is not evenly divisible by the gradient accumulation size, adjusting gradient accumulation size to: {settings['gradient_accumulation_size']}")

	if settings['batch_size'] % settings['gpus'] != 0:
		settings['batch_size'] -= settings['batch_size'] % settings['gpus']
		if settings['batch_size'] == 0:
			settings['batch_size'] = 1
		messages.append(f"Batch size not neatly divisible by GPU count, adjusting batch size to: {settings['batch_size']}")


	def get_device_batch_size( vram ):
		DEVICE_BATCH_SIZE_MAP = [
			(70, 128), # based on an A100-80G, I can safely get a ratio of 4096:32 = 128
			(32, 64), # based on my two 6800XTs, I can only really safely get a ratio of 128:2 = 64
			(16, 8), # based on an A4000, I can do a ratio of 512:64 = 8:1
			(8, 4), # interpolated
			(6, 2), # based on my 2060, it only really lets me have a batch ratio of 2:1
		]
		for k, v in DEVICE_BATCH_SIZE_MAP:
			if vram > (k-1):
				return v
		return 1

	if settings['gpus'] > get_device_count():
		settings['gpus'] = get_device_count()
		messages.append(f"GPU count exceeds defacto GPU count, clamping to: {settings['gpus']}")

	if settings['gpus'] <= 1:
		settings['gpus'] = 1
	else:
		messages.append(f"! EXPERIMENTAL ! Multi-GPU training is extremely particular, expect issues.")

	# assuming you have equal GPUs
	vram = get_device_vram() * settings['gpus']
	batch_ratio = int(settings['batch_size'] / settings['gradient_accumulation_size'])
	batch_cap = get_device_batch_size(vram)

	if batch_ratio > batch_cap:
		settings['gradient_accumulation_size'] = int(settings['batch_size'] / batch_cap)
		messages.append(f"Batch ratio ({batch_ratio}) is expected to exceed your VRAM capacity ({'{:.3f}'.format(vram)}GB, suggested {batch_cap} batch size cap), adjusting gradient accumulation size to: {settings['gradient_accumulation_size']}")

	iterations = calc_iterations(epochs=settings['epochs'], lines=lines, batch_size=settings['batch_size'])

	if settings['epochs'] < settings['save_rate']:
		settings['save_rate'] = settings['epochs']
		messages.append(f"Save rate is too small for the given iteration step, clamping save rate to: {settings['save_rate']}")

	if settings['epochs'] < settings['validation_rate']:
		settings['validation_rate'] = settings['epochs']
		messages.append(f"Validation rate is too small for the given iteration step, clamping validation rate to: {settings['validation_rate']}")

	if settings['resume_state'] and not os.path.exists(settings['resume_state']):
		settings['resume_state'] = None
		messages.append("Resume path specified, but does not exist. Disabling...")

	if settings['bitsandbytes']:
		messages.append("! EXPERIMENTAL ! BitsAndBytes requested.")

	if settings['half_p']:
		if settings['bitsandbytes']:
			settings['half_p'] = False
			messages.append("Half Precision requested, but BitsAndBytes is also requested. Due to redundancies, disabling half precision...")
		else:
			messages.append("! EXPERIMENTAL ! Half Precision requested.")
			if not os.path.exists(get_halfp_model_path()):
				convert_to_halfp()	

	steps = int(iterations / settings['epochs'])

	messages.append(f"For {settings['epochs']} epochs with {lines} lines in batches of {settings['batch_size']}, iterating for {iterations} steps ({steps}) steps per epoch)")

	return settings, messages

def save_training_settings( **kwargs ):
	messages = []
	settings = {}
	settings.update(kwargs)
	

	outjson = f'./training/{settings["voice"]}/train.json'
	with open(outjson, 'w', encoding="utf-8") as f:
		f.write(json.dumps(settings, indent='\t') )

	settings['dataset_path'] = f"./training/{settings['voice']}/train.txt"
	settings['validation_path'] = f"./training/{settings['voice']}/validation.txt"

	with open(settings['dataset_path'], 'r', encoding="utf-8") as f:
		lines = len(f.readlines())

	settings['iterations'] = calc_iterations(epochs=settings['epochs'], lines=lines, batch_size=settings['batch_size'])

	if not settings['source_model'] or settings['source_model'] == "auto":
		settings['source_model'] = f"./models/tortoise/autoregressive{'_half' if settings['half_p'] else ''}.pth"

	if settings['half_p']:
		if not os.path.exists(get_halfp_model_path()):
			convert_to_halfp()

	messages.append(f"For {settings['epochs']} epochs with {lines} lines, iterating for {settings['iterations']} steps")

	iterations_per_epoch = settings['iterations'] / settings['epochs']

	settings['save_rate'] = int(settings['save_rate'] * iterations_per_epoch)
	settings['validation_rate'] = int(settings['validation_rate'] * iterations_per_epoch)

	iterations_per_epoch = int(iterations_per_epoch)
	
	if settings['save_rate'] < 1:
		settings['save_rate'] = 1
	"""
	if settings['validation_rate'] < 1:
		settings['validation_rate'] = 1
	"""
	"""
	if settings['iterations'] % settings['save_rate'] != 0:
		adjustment = int(settings['iterations'] / settings['save_rate']) * settings['save_rate']
		messages.append(f"Iteration rate is not evenly divisible by save rate, adjusting: {settings['iterations']} => {adjustment}")
		settings['iterations'] = adjustment
	"""

	settings['validation_batch_size'] = int(settings['batch_size'] / settings['gradient_accumulation_size'])
	if not os.path.exists(settings['validation_path']):
		settings['validation_enabled'] = False
		messages.append("Validation not found, disabling validation...")
	elif settings['validation_batch_size'] == 0:
		settings['validation_enabled'] = False
		messages.append("Validation batch size == 0, disabling validation...")
	else:
		with open(settings['validation_path'], 'r', encoding="utf-8") as f:
			validation_lines = len(f.readlines())

		if validation_lines < settings['validation_batch_size']:
			settings['validation_batch_size'] = validation_lines
			messages.append(f"Batch size exceeds validation dataset size, clamping validation batch size to {validation_lines}")

	settings['tokenizer_json'] = args.tokenizer_json if args.tokenizer_json else get_tokenizer_jsons()[0]

	if settings['gpus'] > get_device_count():
		settings['gpus'] = get_device_count()

	# what an utter mistake this was
	settings['optimizer'] = 'adamw' # if settings['gpus'] == 1 else 'adamw_zero'

	if 'learning_rate_scheme' not in settings or settings['learning_rate_scheme'] not in LEARNING_RATE_SCHEMES:
		settings['learning_rate_scheme'] = "Multistep"

	settings['learning_rate_scheme'] = LEARNING_RATE_SCHEMES[settings['learning_rate_scheme']]

	learning_rate_schema = [f"default_lr_scheme: {settings['learning_rate_scheme']}"]
	if settings['learning_rate_scheme'] == "MultiStepLR":
		if not settings['learning_rate_schedule']:
			settings['learning_rate_schedule'] = LEARNING_RATE_SCHEDULE
		elif isinstance(settings['learning_rate_schedule'],str):
			settings['learning_rate_schedule'] = json.loads(settings['learning_rate_schedule'])

		settings['learning_rate_schedule'] = schedule_learning_rate( iterations_per_epoch, settings['learning_rate_schedule'] )

		learning_rate_schema.append(f"  gen_lr_steps: {settings['learning_rate_schedule']}")
		learning_rate_schema.append(f"  lr_gamma: 0.5")
	elif settings['learning_rate_scheme'] == "CosineAnnealingLR_Restart":
		epochs = settings['epochs']
		restarts = settings['learning_rate_restarts']
		restart_period = int(epochs / restarts)

		if 'learning_rate_warmup' not in settings:
			settings['learning_rate_warmup'] = 0
		if 'learning_rate_min' not in settings:
			settings['learning_rate_min'] = 1e-08

		if 'learning_rate_period' not in settings:
			settings['learning_rate_period'] = [ iterations_per_epoch * restart_period for x in range(epochs) ]

		settings['learning_rate_restarts'] = [ iterations_per_epoch * (x+1) * restart_period for x in range(restarts) ] # [52, 104, 156, 208]

		if 'learning_rate_restart_weights' not in settings:
			settings['learning_rate_restart_weights'] = [ ( restarts - x - 1 ) / restarts for x in range(restarts) ] # [.75, .5, .25, .125]
			settings['learning_rate_restart_weights'][-1] = settings['learning_rate_restart_weights'][-2] * 0.5

		learning_rate_schema.append(f"  T_period: {settings['learning_rate_period']}")
		learning_rate_schema.append(f"  warmup: {settings['learning_rate_warmup']}")
		learning_rate_schema.append(f"  eta_min: !!float {settings['learning_rate_min']}")
		learning_rate_schema.append(f"  restarts: {settings['learning_rate_restarts']}")
		learning_rate_schema.append(f"  restart_weights: {settings['learning_rate_restart_weights']}")
	settings['learning_rate_scheme'] = "\n".join(learning_rate_schema)

	if settings['resume_state']:
		settings['source_model'] = f"# pretrain_model_gpt: '{settings['source_model']}'"
		settings['resume_state'] = f"resume_state: '{settings['resume_state']}'"
	else:
		settings['source_model'] = f"pretrain_model_gpt: '{settings['source_model']}'"
		settings['resume_state'] = f"# resume_state: '{settings['resume_state']}'"

	def use_template(template, out):
		with open(template, 'r', encoding="utf-8") as f:
			yaml = f.read()

		# i could just load and edit the YAML directly, but this is easier, as I don't need to bother with path traversals
		for k in settings:
			if settings[k] is None:
				continue
			yaml = yaml.replace(f"${{{k}}}", str(settings[k]))

		with open(out, 'w', encoding="utf-8") as f:
			f.write(yaml)
	
	if args.tts_backend == "tortoise":
		use_template(f'./models/.template.dlas.yaml', f'./training/{settings["voice"]}/train.yaml')
	elif args.tts_backend == "vall-e":
		settings['model_name'] = "[ 'ar-quarter', 'nar-quarter' ]"
		use_template(f'./models/.template.valle.yaml', f'./training/{settings["voice"]}/config.yaml')

	messages.append(f"Saved training output")
	return settings, messages

def import_voices(files, saveAs=None, progress=None):
	global args

	if not isinstance(files, list):
		files = [files]

	for file in tqdm(files, desc="Importing voice files"):
		j, latents = read_generate_settings(file, read_latents=True)
		
		if j is not None and saveAs is None:
			saveAs = j['voice']
		if saveAs is None or saveAs == "":
			raise Exception("Specify a voice name")

		outdir = f'{get_voice_dir()}/{saveAs}/'
		os.makedirs(outdir, exist_ok=True)

		if latents:
			print(f"Importing latents to {latents}")
			with open(f'{outdir}/cond_latents.pth', 'wb') as f:
				f.write(latents)
			latents = f'{outdir}/cond_latents.pth'
			print(f"Imported latents to {latents}")
		else:
			filename = file.name
			if filename[-4:] != ".wav":
				raise Exception("Please convert to a WAV first")

			path = f"{outdir}/{os.path.basename(filename)}"
			print(f"Importing voice to {path}")

			waveform, sample_rate = torchaudio.load(filename)

			if args.voice_fixer:
				if not voicefixer:
					load_voicefixer()

				waveform, sample_rate = resample(waveform, sample_rate, 44100)
				torchaudio.save(path, waveform, sample_rate)

				print(f"Running 'voicefixer' on voice sample: {path}")
				voicefixer.restore(
					input = path,
					output = path,
					cuda=get_device_name() == "cuda" and args.voice_fixer_use_cuda,
					#mode=mode,
				)
			else:
				torchaudio.save(path, waveform, sample_rate)

			print(f"Imported voice to {path}")

def relative_paths( dirs ):
	return [ './' + os.path.relpath( d ).replace("\\", "/") for d in dirs ]

def get_voice( name, dir=get_voice_dir(), load_latents=True, extensions=["wav", "mp3", "flac"] ):
	subj = f'{dir}/{name}/'
	if not os.path.isdir(subj):
		return
	files = os.listdir(subj)
	
	if load_latents:
		extensions.append("pth")

	voice = []
	for file in files:
		ext = os.path.splitext(file)[-1][1:]
		if ext not in extensions:
			continue

		voice.append(f'{subj}/{file}') 

	return sorted( voice )

def get_voice_list(dir=get_voice_dir(), append_defaults=False, extensions=["wav", "mp3", "flac", "pth", "opus", "m4a", "webm", "mp4"]):
	defaults = [ "random", "microphone" ]
	os.makedirs(dir, exist_ok=True)
	#res = sorted([d for d in os.listdir(dir) if d not in defaults and os.path.isdir(os.path.join(dir, d)) and len(os.listdir(os.path.join(dir, d))) > 0 ])

	res = []
	for name in os.listdir(dir):
		if name in defaults:
			continue
		if not os.path.isdir(f'{dir}/{name}'):
			continue
		if len(os.listdir(os.path.join(dir, name))) == 0:
			continue
		files = get_voice( name, dir=dir, extensions=extensions )

		if len(files) > 0:
			res.append(name)
		else:
			for subdir in os.listdir(f'{dir}/{name}'):
				if not os.path.isdir(f'{dir}/{name}/{subdir}'):
					continue
				files = get_voice( f'{name}/{subdir}', dir=dir, extensions=extensions )
				if len(files) == 0:
					continue
				res.append(f'{name}/{subdir}')

	res = sorted(res)
	
	if append_defaults:
		res = res + defaults
	
	return res


    
    

def get_valle_models(dir="./training/"):
	return [ f'{dir}/{d}/config.yaml' for d in os.listdir(dir) if os.path.exists(f'{dir}/{d}/config.yaml') ]

def get_autoregressive_models(dir="./models/finetunes/", prefixed=False, auto=False):
	os.makedirs(dir, exist_ok=True)
	base = [get_model_path('autoregressive.pth')]
	halfp = get_halfp_model_path()
	if os.path.exists(halfp):
		base.append(halfp)

	additionals = sorted([f'{dir}/{d}' for d in os.listdir(dir) if d[-4:] == ".pth" ])
	found = []
	for training in os.listdir(f'./training/'):
		if not os.path.isdir(f'./training/{training}/') or not os.path.isdir(f'./training/{training}/finetune/') or not os.path.isdir(f'./training/{training}/finetune/models/'):
			continue
		models = sorted([ int(d[:-8]) for d in os.listdir(f'./training/{training}/finetune/models/') if d[-8:] == "_gpt.pth" ])
		found = found + [ f'./training/{training}/finetune/models/{d}_gpt.pth' for d in models ]

	res = base + additionals + found
	
	if prefixed:
		for i in range(len(res)):
			path = res[i]
			hash = hash_file(path)
			shorthash = hash[:8]

			res[i] = f'[{shorthash}] {path}'

	paths = relative_paths(res)
	if auto:
		paths = ["auto"] + paths 

	return paths

def get_diffusion_models(dir="./models/finetunes/", prefixed=False):
	return relative_paths([ get_model_path('diffusion_decoder.pth') ])

def get_tokenizer_jsons( dir="./models/tokenizers/" ):
	additionals = sorted([ f'{dir}/{d}' for d in os.listdir(dir) if d[-5:] == ".json" ]) if os.path.isdir(dir) else []
	return relative_paths([ "./modules/tortoise-tts/tortoise/data/tokenizer.json" ] + additionals)

def tokenize_text( text, config=None, stringed=True, skip_specials=False ):
	from tortoise.utils.tokenizer import VoiceBpeTokenizer

	if not config:
		config = args.tokenizer_json if args.tokenizer_json else get_tokenizer_jsons()[0]

	if not tts:
		tokenizer = VoiceBpeTokenizer(config)
	else:
		tokenizer = tts.tokenizer

	encoded = tokenizer.encode(text)
	decoded = tokenizer.tokenizer.decode(encoded, skip_special_tokens=skip_specials).split(" ")

	if stringed:
		return "\n".join([ str(encoded), str(decoded) ])

	return decoded

def get_dataset_list(dir="./training/"):
	return sorted([d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and "train.txt" in os.listdir(os.path.join(dir, d)) ])

def get_training_list(dir="./training/"):
	if args.tts_backend == "tortoise":
		return sorted([f'./training/{d}/train.yaml' for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and "train.yaml" in os.listdir(os.path.join(dir, d)) ])
	else:
		return sorted([f'./training/{d}/config.yaml' for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and "config.yaml" in os.listdir(os.path.join(dir, d)) ])

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

def curl(url):
	try:
		req = urllib.request.Request(url, headers={'User-Agent': 'Python'})
		conn = urllib.request.urlopen(req)
		data = conn.read()
		data = data.decode()
		data = json.loads(data)
		conn.close()
		return data
	except Exception as e:
		print(e)
		return None

def check_for_updates( dir = None ):
	if dir is None:
		check_for_updates("./.git/")
		check_for_updates("./.git/modules/dlas/")
		check_for_updates("./.git/modules/tortoise-tts/")
		return

	git_dir = dir
	if not os.path.isfile(f'{git_dir}/FETCH_HEAD'):
		print(f"Cannot check for updates for {dir}: not from a git repo")
		return False

	with open(f'{git_dir}/FETCH_HEAD', 'r', encoding="utf-8") as f:
		head = f.read()
	
	match = re.findall(r"^([a-f0-9]+).+?https:\/\/(.+?)\/(.+?)\/(.+?)\n", head)
	if match is None or len(match) == 0:
		print(f"Cannot check for updates for {dir}: cannot parse FETCH_HEAD")
		return False

	match = match[0]

	local = match[0]
	host = match[1]
	owner = match[2]
	repo = match[3]

	res = curl(f"https://{host}/api/v1/repos/{owner}/{repo}/branches/") #this only works for gitea instances

	if res is None or len(res) == 0:
		print(f"Cannot check for updates for {dir}: cannot fetch from remote")
		return False

	remote = res[0]["commit"]["id"]

	if remote != local:
		print(f"New version found for {dir}: {local[:8]} => {remote[:8]}")
		return True

	return False

def notify_progress(message, progress=None, verbose=True):
	if verbose:
		print(message)

	if progress is None:
		tqdm.write(message)
	else:
		progress(0, desc=message)

def get_args():
	global args
	return args

def setup_args(cli=False):
	global args

	default_arguments = {
		'share': False,
		'listen': None,
		'check-for-updates': False,
		'models-from-local-only': False,
		'low-vram': False,
		'sample-batch-size': None,
		'unsqueeze-sample-batches': False,
		'embed-output-metadata': True,
		'latents-lean-and-mean': True,
		'voice-fixer': False, # getting tired of long initialization times in a Colab for downloading a large dataset for it
		'use-deepspeed': False,
		#stuff that jarod has added
		'use-hifigan': False,
		'use-rvc' : False,
		'rvc-model' : None,

		'voice-fixer-use-cuda': True,

		
		'force-cpu-for-conditioning-latents': False,
		'defer-tts-load': False,
		'device-override': None,
		'prune-nonfinal-outputs': True,
		'concurrency-count': 2,
		'autocalculate-voice-chunk-duration-size': 10,

		'output-sample-rate': 44100,
		'output-volume': 1,
		'results-folder': "./results/",
		
		'hf-token': None,
		'tts-backend': TTSES[0],
		
		'autoregressive-model': None,
		'diffusion-model': None,
		'vocoder-model': VOCODERS[-1],
		'tokenizer-json': None,

		'phonemizer-backend': 'espeak',
		
		'valle-model': None,

		'whisper-backend': 'openai/whisper',
		'whisper-model': "base",
		'whisper-batchsize': 1,

		'training-default-halfp': False,
		'training-default-bnb': True,

		'websocket-listen-address': "127.0.0.1",
		'websocket-listen-port': 8069,
		'websocket-enabled': False
	}

	if os.path.isfile('./config/exec.json'):
		with open(f'./config/exec.json', 'r', encoding="utf-8") as f:
			try:
				overrides = json.load(f)
				for k in overrides:
					default_arguments[k] = overrides[k]
			except Exception as e:
				print(e)
				pass

	parser = argparse.ArgumentParser(allow_abbrev=not cli)
	parser.add_argument("--share", action='store_true', default=default_arguments['share'], help="Lets Gradio return a public URL to use anywhere")
	parser.add_argument("--listen", default=default_arguments['listen'], help="Path for Gradio to listen on")
	parser.add_argument("--check-for-updates", action='store_true', default=default_arguments['check-for-updates'], help="Checks for update on startup")
	parser.add_argument("--models-from-local-only", action='store_true', default=default_arguments['models-from-local-only'], help="Only loads models from disk, does not check for updates for models")
	parser.add_argument("--low-vram", action='store_true', default=default_arguments['low-vram'], help="Disables some optimizations that increases VRAM usage")
	parser.add_argument("--no-embed-output-metadata", action='store_false', default=not default_arguments['embed-output-metadata'], help="Disables embedding output metadata into resulting WAV files for easily fetching its settings used with the web UI (data is stored in the lyrics metadata tag)")
	parser.add_argument("--latents-lean-and-mean", action='store_true', default=default_arguments['latents-lean-and-mean'], help="Exports the bare essentials for latents.")
	parser.add_argument("--voice-fixer", action='store_true', default=default_arguments['voice-fixer'], help="Uses python module 'voicefixer' to improve audio quality, if available.")
	parser.add_argument("--voice-fixer-use-cuda", action='store_true', default=default_arguments['voice-fixer-use-cuda'], help="Hints to voicefixer to use CUDA, if available.")
	parser.add_argument("--use-deepspeed", action='store_true', default=default_arguments['use-deepspeed'], help="Use deepspeed for speed bump.")
	parser.add_argument("--use-hifigan", action='store_true', default=default_arguments['use-hifigan'], help="Use Hifigan instead of Diffusion")
	parser.add_argument("--use-rvc", action='store_true', default=default_arguments['use-rvc'], help="Run the outputted audio thorugh RVC")
	parser.add_argument("--rvc-model", action='store_true', default=default_arguments['rvc-model'], help="Specifies RVC model to use")

	# parser.add_argument("--f0_up_key", action='store_true', default=default_arguments['f0_up_key'], help="transpose of the audio file, changes pitch (positive makes voice higher pitch)")
	# parser.add_argument("--f0method", action='store_true', default=default_arguments['f0method'], help="picks which f0 method to use: dio, harvest, crepe, rmvpe (requires rmvpe.pt)")
	# parser.add_argument("--file_index", action='store_true', default=default_arguments['file_index'], help="path to file_index, defaults to None")
	# parser.add_argument("--index_rate", action='store_true', default=default_arguments['index_rate'], help="strength of the index file if provided")
	# parser.add_argument("--filter_radius", action='store_true', default=default_arguments['filter_radius'], help="if >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.")
	# parser.add_argument("--resample_sr", action='store_true', default=default_arguments['resample_sr'], help="quality at which to resample audio to, defaults to no resample")
	# parser.add_argument("--rms_mix_rate", action='store_true', default=default_arguments['rms_mix_rate'], help="adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume")
	# parser.add_argument("--protect", action='store_true', default=default_arguments['protect'], help="protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy")
	

	parser.add_argument("--force-cpu-for-conditioning-latents", default=default_arguments['force-cpu-for-conditioning-latents'], action='store_true', help="Forces computing conditional latents to be done on the CPU (if you constantyl OOM on low chunk counts)")
	parser.add_argument("--defer-tts-load", default=default_arguments['defer-tts-load'], action='store_true', help="Defers loading TTS model")
	parser.add_argument("--prune-nonfinal-outputs", default=default_arguments['prune-nonfinal-outputs'], action='store_true', help="Deletes non-final output files on completing a generation")
	parser.add_argument("--device-override", default=default_arguments['device-override'], help="A device string to override pass through Torch")
	parser.add_argument("--sample-batch-size", default=default_arguments['sample-batch-size'], type=int, help="Sets how many batches to use during the autoregressive samples pass")
	parser.add_argument("--unsqueeze-sample-batches", default=default_arguments['unsqueeze-sample-batches'], action='store_true', help="Unsqueezes sample batches to process one by one after sampling")
	parser.add_argument("--concurrency-count", type=int, default=default_arguments['concurrency-count'], help="How many Gradio events to process at once")
	parser.add_argument("--autocalculate-voice-chunk-duration-size", type=float, default=default_arguments['autocalculate-voice-chunk-duration-size'], help="Number of seconds to suggest voice chunk size for (for example, 100 seconds of audio at 10 seconds per chunk will suggest 10 chunks)")
	parser.add_argument("--output-sample-rate", type=int, default=default_arguments['output-sample-rate'], help="Sample rate to resample the output to (from 24KHz)")
	parser.add_argument("--output-volume", type=float, default=default_arguments['output-volume'], help="Adjusts volume of output")
	parser.add_argument("--results-folder", type=str, default=default_arguments['results-folder'], help="Sets output directory")
	
	parser.add_argument("--hf-token", type=str, default=default_arguments['hf-token'], help="HuggingFace Token")
	parser.add_argument("--tts-backend", default=default_arguments['tts-backend'], help="Specifies which TTS backend to use.")

	parser.add_argument("--autoregressive-model", default=default_arguments['autoregressive-model'], help="Specifies which autoregressive model to use for sampling.")
	parser.add_argument("--diffusion-model", default=default_arguments['diffusion-model'], help="Specifies which diffusion model to use for sampling.")
	parser.add_argument("--vocoder-model", default=default_arguments['vocoder-model'], action='store_true', help="Specifies with vocoder to use")
	parser.add_argument("--tokenizer-json", default=default_arguments['tokenizer-json'], help="Specifies which tokenizer json to use for tokenizing.")

	parser.add_argument("--phonemizer-backend", default=default_arguments['phonemizer-backend'], help="Specifies which phonemizer backend to use.")
	
	parser.add_argument("--valle-model", default=default_arguments['valle-model'], help="Specifies which VALL-E model to use for sampling.")
	
	parser.add_argument("--whisper-backend", default=default_arguments['whisper-backend'], action='store_true', help="Picks which whisper backend to use (openai/whisper, lightmare/whispercpp)")
	parser.add_argument("--whisper-model", default=default_arguments['whisper-model'], help="Specifies which whisper model to use for transcription.")
	parser.add_argument("--whisper-batchsize", type=int, default=default_arguments['whisper-batchsize'], help="Specifies batch size for WhisperX")
	
	parser.add_argument("--training-default-halfp", action='store_true', default=default_arguments['training-default-halfp'], help="Training default: halfp")
	parser.add_argument("--training-default-bnb", action='store_true', default=default_arguments['training-default-bnb'], help="Training default: bnb")
	
	parser.add_argument("--websocket-listen-port", type=int, default=default_arguments['websocket-listen-port'], help="Websocket server listen port, default: 8069")
	parser.add_argument("--websocket-listen-address", default=default_arguments['websocket-listen-address'], help="Websocket server listen address, default: 127.0.0.1")
	parser.add_argument("--websocket-enabled", action='store_true', default=default_arguments['websocket-enabled'], help="Websocket API server enabled, default: false")

	if cli:
		args, unknown = parser.parse_known_args()
	else:
		args = parser.parse_args()

	args.embed_output_metadata = not args.no_embed_output_metadata

	if not args.device_override:
		set_device_name(args.device_override)

	if args.sample_batch_size == 0 and get_device_batch_size() == 1:
		print("!WARNING! Automatically deduced sample batch size returned 1.")

	args.listen_host = None
	args.listen_port = None
	args.listen_path = None
	if args.listen:
		try:
			match = re.findall(r"^(?:(.+?):(\d+))?(\/.*?)?$", args.listen)[0]

			args.listen_host = match[0] if match[0] != "" else "0.0.0.0"
			args.listen_port = match[1] if match[1] != "" else None
			args.listen_path = match[2] if match[2] != "" else "/"
		except Exception as e:
			pass

	if args.listen_port is not None:
		args.listen_port = int(args.listen_port)
		if args.listen_port == 0:
			args.listen_port = None
	
	return args

def get_default_settings( hypenated=True ):
	settings = {
		'listen': None if not args.listen else args.listen,
		'share': args.share,
		'low-vram':args.low_vram,
		'check-for-updates':args.check_for_updates,
		'models-from-local-only':args.models_from_local_only,
		'force-cpu-for-conditioning-latents': args.force_cpu_for_conditioning_latents,
		'defer-tts-load': args.defer_tts_load,
		'prune-nonfinal-outputs': args.prune_nonfinal_outputs,
		'device-override': args.device_override,
		'sample-batch-size': args.sample_batch_size,
		'unsqueeze-sample-batches': args.unsqueeze_sample_batches,
		'embed-output-metadata': args.embed_output_metadata,
		'latents-lean-and-mean': args.latents_lean_and_mean,
		'voice-fixer': args.voice_fixer,
		'use-deepspeed': args.use_deepspeed,
		'use-hifigan': args.use_hifigan,
		'use-rvc': args.use_rvc,
		'rvc-model' : args.rvc_model,
		'voice-fixer-use-cuda': args.voice_fixer_use_cuda,
		'concurrency-count': args.concurrency_count,
		'output-sample-rate': args.output_sample_rate,
		'autocalculate-voice-chunk-duration-size': args.autocalculate_voice_chunk_duration_size,
		'output-volume': args.output_volume,
		'results-folder': args.results_folder,
		
		'hf-token': args.hf_token,
		'tts-backend': args.tts_backend,

		'autoregressive-model': args.autoregressive_model,
		'diffusion-model': args.diffusion_model,
		'vocoder-model': args.vocoder_model,
		'tokenizer-json': args.tokenizer_json,

		'phonemizer-backend': args.phonemizer_backend,
		
		'valle-model': args.valle_model,

		'whisper-backend': args.whisper_backend,
		'whisper-model': args.whisper_model,
		'whisper-batchsize': args.whisper_batchsize,

		'training-default-halfp': args.training_default_halfp,
		'training-default-bnb': args.training_default_bnb,
	}

	res = {}
	for k in settings:
		res[k.replace("-", "_") if not hypenated else k] = settings[k]
	return res

def update_args( **kwargs ):
	global args

	settings = get_default_settings(hypenated=False)
	settings.update(kwargs)

	args.listen = settings['listen']
	args.share = settings['share']
	args.check_for_updates = settings['check_for_updates']
	args.models_from_local_only = settings['models_from_local_only']
	args.low_vram = settings['low_vram']
	args.force_cpu_for_conditioning_latents = settings['force_cpu_for_conditioning_latents']
	args.defer_tts_load = settings['defer_tts_load']
	args.prune_nonfinal_outputs = settings['prune_nonfinal_outputs']
	args.device_override = settings['device_override']
	args.sample_batch_size = settings['sample_batch_size']
	args.unsqueeze_sample_batches = settings['unsqueeze_sample_batches']
	args.embed_output_metadata = settings['embed_output_metadata']
	args.latents_lean_and_mean = settings['latents_lean_and_mean']
	args.voice_fixer = settings['voice_fixer']
	args.voice_fixer_use_cuda = settings['voice_fixer_use_cuda']
	args.use_deepspeed = settings['use_deepspeed']
	args.use_hifigan = settings['use_hifigan']
	args.use_rvc = settings['use_rvc']
	args.rvc_model = settings['rvc_model']
	args.concurrency_count = settings['concurrency_count']
	args.output_sample_rate = 44000
	args.autocalculate_voice_chunk_duration_size = settings['autocalculate_voice_chunk_duration_size']
	args.output_volume = settings['output_volume']
	args.results_folder = settings['results_folder']
	
	args.hf_token = settings['hf_token']
	args.tts_backend = settings['tts_backend']
	
	args.autoregressive_model = settings['autoregressive_model']
	args.diffusion_model = settings['diffusion_model']
	args.vocoder_model = settings['vocoder_model']
	args.tokenizer_json = settings['tokenizer_json']

	args.phonemizer_backend = settings['phonemizer_backend']
	
	args.valle_model = settings['valle_model']

	args.whisper_backend = settings['whisper_backend']
	args.whisper_model = settings['whisper_model']
	args.whisper_batchsize = settings['whisper_batchsize']

	args.training_default_halfp = settings['training_default_halfp']
	args.training_default_bnb = settings['training_default_bnb']

	save_args_settings()

def save_args_settings():
	global args
	settings = get_default_settings()

	os.makedirs('./config/', exist_ok=True)
	with open(f'./config/exec.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(settings, indent='\t') )

# super kludgy )`;
def import_generate_settings(file = None):
	if not file:
		file = "./config/generate.json"

	res = {
		'text': None,
		'delimiter': None,
		'emotion': None,
		'prompt': None,
		'voice': "random",
		'mic_audio': None,
		'voice_latents_chunks': None,
		'candidates': None,
		'seed': None,
		'num_autoregressive_samples': 16,
		'diffusion_iterations': 30,
		'temperature': 0.8,
		'diffusion_sampler': "DDIM",
		'breathing_room': 8  ,
		'cvvp_weight': 0.0,
		'top_p': 0.8,
		'diffusion_temperature': 1.0,
		'length_penalty': 1.0,
		'repetition_penalty': 2.0,
		'cond_free_k': 2.0,
		'experimentals': None,
	}

	settings, _ = read_generate_settings(file, read_latents=False)

	if settings is not None:
		res.update(settings)
	
	return res

def reset_generate_settings():
	with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps({}, indent='\t') )
	return import_generate_settings()

def read_generate_settings(file, read_latents=True):
	j = None
	latents = None

	if isinstance(file, list) and len(file) == 1:
		file = file[0]

	try:
		if file is not None:
			if hasattr(file, 'name'):
				file = file.name

			if file[-4:] == ".wav":
					metadata = music_tag.load_file(file)
					if 'lyrics' in metadata:
						j = json.loads(str(metadata['lyrics']))
			elif file[-5:] == ".json":
				with open(file, 'r') as f:
					j = json.load(f)
	except Exception as e:
		pass

	if j is not None:
		if 'latents' in j:
			if read_latents:
				latents = base64.b64decode(j['latents'])
			del j['latents']
		

		if "time" in j:
			j["time"] = "{:.3f}".format(j["time"])



	return (
		j,
		latents,
	)

def version_check_tts( min_version ):
	global tts
	if not tts:
		raise Exception("TTS is not initialized")

	if not hasattr(tts, 'version'):
		return False

	if min_version[0] > tts.version[0]:
		return True
	if min_version[1] > tts.version[1]:
		return True
	if min_version[2] >= tts.version[2]:
		return True
	return False

def load_tts( restart=False, 
	# TorToiSe configs
	autoregressive_model=None, diffusion_model=None, vocoder_model=None, tokenizer_json=None,
	# VALL-E configs
	valle_model=None,
):
	global args
	global tts

	if restart:
		unload_tts()

	tts_loading = True
	if args.tts_backend == "tortoise":
		if autoregressive_model:
			args.autoregressive_model = autoregressive_model
		else:
			autoregressive_model = args.autoregressive_model

		if autoregressive_model == "auto":
			autoregressive_model = deduce_autoregressive_model()

		if diffusion_model:
			args.diffusion_model = diffusion_model
		else:
			diffusion_model = args.diffusion_model

		if vocoder_model:
			args.vocoder_model = vocoder_model
		else:
			vocoder_model = args.vocoder_model

		if tokenizer_json:
			args.tokenizer_json = tokenizer_json
		else:
			tokenizer_json = args.tokenizer_json

		if get_device_name() == "cpu":
			print("!!!! WARNING !!!! No GPU available in PyTorch. You may need to reinstall PyTorch.")

		
		if args.use_hifigan:
			print("Loading Tortoise with Hifigan")
			tts = Toroise_TTS_Hifi(autoregressive_model_path=autoregressive_model,  
							tokenizer_json=tokenizer_json, 
							use_deepspeed=args.use_deepspeed)
		else:
			print(f"Loading TorToiSe... (AR: {autoregressive_model}, diffusion: {diffusion_model}, vocoder: {vocoder_model})")
			tts = TorToise_TTS(minor_optimizations=not args.low_vram, 
							autoregressive_model_path=autoregressive_model, 
							diffusion_model_path=diffusion_model, 
							vocoder_model=vocoder_model, 
							tokenizer_json=tokenizer_json, 
							unsqueeze_sample_batches=args.unsqueeze_sample_batches, 
							use_deepspeed=args.use_deepspeed)
			
	elif args.tts_backend == "vall-e":
		if valle_model:
			args.valle_model = valle_model
		else:
			valle_model = args.valle_model

		print(f"Loading VALL-E... (Config: {valle_model})")
		tts = VALLE_TTS(config=args.valle_model)
	elif args.tts_backend == "bark":

		print(f"Loading Bark...")
		tts = Bark_TTS(small=args.low_vram)

	print("Loaded TTS, ready for generation.")
	tts_loading = False
	return tts

def unload_tts():
	global tts

	if tts:
		del tts
		tts = None
		print("Unloaded TTS")
	do_gc()

def reload_tts():
	in_docker = os.environ.get("IN_DOCKER", "false")
	if in_docker == "false":
		subprocess.Popen(["start.bat"])
	with open("reload_flag.txt", "w") as f:
		f.write("reload")
	os.kill(os.getpid(), signal.SIGTERM)  # Or signal.SIGKILL for an even harder kill
	# unload_tts()
	# load_tts()

def get_current_voice():
	global current_voice
	if current_voice:
		return current_voice

	settings, _ = read_generate_settings("./config/generate.json", read_latents=False)
	
	if settings and "voice" in settings['voice']:
		return settings["voice"]
	
	return None

def deduce_autoregressive_model(voice=None):
	if not voice:
		voice = get_current_voice()

	if voice:
		if os.path.exists(f'./models/finetunes/{voice}.pth'):
			return f'./models/finetunes/{voice}.pth'
		
		dir = f'./training/{voice}/finetune/models/'
		if os.path.isdir(dir):
			counts = sorted([ int(d[:-8]) for d in os.listdir(dir) if d[-8:] == "_gpt.pth" ])
			names = [ f'{dir}/{d}_gpt.pth' for d in counts ]
			if len(names) > 0:
				return names[-1]

	if args.autoregressive_model != "auto":
		return args.autoregressive_model

	return get_model_path('autoregressive.pth')

def update_autoregressive_model(autoregressive_model_path):
	if args.tts_backend != "tortoise":
		raise f"Unsupported backend: {args.tts_backend}"

	if autoregressive_model_path == "auto":
		autoregressive_model_path = deduce_autoregressive_model()
	else:
		match = re.findall(r'^\[[a-fA-F0-9]{8}\] (.+?)$', autoregressive_model_path)
		if match:
			autoregressive_model_path = match[0]

	if not autoregressive_model_path or not os.path.exists(autoregressive_model_path):
		print(f"Invalid model: {autoregressive_model_path}")
		return

	args.autoregressive_model = autoregressive_model_path
	save_args_settings()
	print(f'Stored autoregressive model to settings: {autoregressive_model_path}')

	global tts
	if not tts:
		if tts_loading:
			raise Exception("TTS is still initializing...")
		return
	
	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")


	if autoregressive_model_path == tts.autoregressive_model_path:
		return

	tts.load_autoregressive_model(autoregressive_model_path)

	do_gc()
	
	return autoregressive_model_path

def update_diffusion_model(diffusion_model_path):
	if args.tts_backend != "tortoise":
		raise f"Unsupported backend: {args.tts_backend}"

	match = re.findall(r'^\[[a-fA-F0-9]{8}\] (.+?)$', diffusion_model_path)
	if match:
		diffusion_model_path = match[0]

	if not diffusion_model_path or not os.path.exists(diffusion_model_path):
		print(f"Invalid model: {diffusion_model_path}")
		return

	args.diffusion_model = diffusion_model_path
	save_args_settings()
	print(f'Stored diffusion model to settings: {diffusion_model_path}')

	global tts
	if not tts:
		if tts_loading:
			raise Exception("TTS is still initializing...")
		return
	
	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	if diffusion_model_path == "auto":
		diffusion_model_path = deduce_diffusion_model()

	if diffusion_model_path == tts.diffusion_model_path:
		return

	tts.load_diffusion_model(diffusion_model_path)

	do_gc()
	
	return diffusion_model_path

def update_vocoder_model(vocoder_model):
	if args.tts_backend != "tortoise":
		raise f"Unsupported backend: {args.tts_backend}"

	args.vocoder_model = vocoder_model
	save_args_settings()
	print(f'Stored vocoder model to settings: {vocoder_model}')

	global tts
	if not tts:
		if tts_loading:
			raise Exception("TTS is still initializing...")
		return

	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	print(f"Loading model: {vocoder_model}")
	tts.load_vocoder_model(vocoder_model)
	print(f"Loaded model: {tts.vocoder_model}")

	do_gc()
	
	return vocoder_model

def update_tokenizer(tokenizer_json):
	if args.tts_backend != "tortoise":
		raise f"Unsupported backend: {args.tts_backend}"

	args.tokenizer_json = tokenizer_json
	save_args_settings()
	print(f'Stored tokenizer to settings: {tokenizer_json}')

	global tts
	if not tts:
		if tts_loading:
			raise Exception("TTS is still initializing...")
		return

	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	print(f"Loading tokenizer vocab: {tokenizer_json}")
	tts.load_tokenizer_json(tokenizer_json)
	print(f"Loaded tokenizer vocab: {tts.tokenizer_json}")

	do_gc()
	
	return vocoder_model

def load_voicefixer(restart=False):
	global voicefixer

	if restart:
		unload_voicefixer()

	try:
		print("Loading Voicefixer")
		from voicefixer import VoiceFixer
		voicefixer = VoiceFixer()
		print("Loaded Voicefixer")
	except Exception as e:
		print(f"Error occurred while tring to initialize voicefixer: {e}")
		if voicefixer:
			del voicefixer
		voicefixer = None

def unload_voicefixer():
	global voicefixer

	if voicefixer:
		del voicefixer
		voicefixer = None
		print("Unloaded Voicefixer")

	do_gc()

def load_whisper_model(language=None, model_name=None, progress=None):
	global whisper_model
	global whisper_align_model

	if args.whisper_backend not in WHISPER_BACKENDS:
		raise Exception(f"unavailable backend: {args.whisper_backend}")

	if not model_name:
		model_name = args.whisper_model
	else:
		args.whisper_model = model_name
		save_args_settings()

	if language and f'{model_name}.{language}' in WHISPER_SPECIALIZED_MODELS:
		model_name = f'{model_name}.{language}'
		print(f"Loading specialized model for language: {language}")

	notify_progress(f"Loading Whisper model: {model_name}", progress=progress)

	if args.whisper_backend == "openai/whisper":
		import whisper
		try:
			#is it possible for model to fit on vram but go oom later on while executing on data?
			whisper_model = whisper.load_model(model_name)
		except:
			print("Out of VRAM memory. falling back to loading Whisper on CPU.")
			whisper_model = whisper.load_model(model_name, device="cpu")
	elif args.whisper_backend == "lightmare/whispercpp":
		from whispercpp import Whisper
		if not language:
			language = 'auto'

		b_lang = language.encode('ascii')
		whisper_model = Whisper(model_name, models_dir='./models/', language=b_lang)
	elif args.whisper_backend == "m-bain/whisperx":
		import whisper, whisperx
		device = "cuda" if get_device_name() == "cuda" else "cpu"
		whisper_model = whisperx.load_model(model_name, device)
		whisper_align_model = whisperx.load_align_model(model_name="WAV2VEC2_ASR_LARGE_LV60K_960H" if language=="en" else None, language_code=language, device=device)

	print("Loaded Whisper model")

def unload_whisper():
	global whisper_model
	global whisper_align_model

	if whisper_align_model:
		del whisper_align_model
		whisper_align_model = None

	if whisper_model:
		del whisper_model
		whisper_model = None
		print("Unloaded Whisper")

	do_gc()	

# shamelessly borrowed from Voldy's Web UI: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/extras.py#L74
def merge_models( primary_model_name, secondary_model_name, alpha, progress=gr.Progress() ):
	key_blacklist = []

	def weighted_sum(theta0, theta1, alpha):
		return ((1 - alpha) * theta0) + (alpha * theta1)

	def read_model( filename ):
		print(f"Loading {filename}")
		return torch.load(filename)

	theta_func = weighted_sum

	theta_0 = read_model(primary_model_name)
	theta_1 = read_model(secondary_model_name)

	for key in tqdm(theta_0.keys(), desc="Merging..."):
		if key in key_blacklist:
			print("Skipping ignored key:", key)
			continue
		
		a = theta_0[key]
		b = theta_1[key]

		if a.dtype != torch.float32 and a.dtype != torch.float16:
			print("Skipping key:", key, a.dtype)
			continue

		if b.dtype != torch.float32 and b.dtype != torch.float16:
			print("Skipping key:", key, b.dtype)
			continue

		theta_0[key] = theta_func(a, b, alpha)

	del theta_1

	primary_basename = os.path.splitext(os.path.basename(primary_model_name))[0]
	secondary_basename = os.path.splitext(os.path.basename(secondary_model_name))[0]
	suffix = "{:.3f}".format(alpha)
	output_path = f'./models/finetunes/{primary_basename}_{secondary_basename}_{suffix}_merge.pth'

	torch.save(theta_0, output_path)
	message = f"Saved to {output_path}"
	print(message)
	return message

#Stuff added by Jarod
def get_rvc_models():
	folder_path = 'models/rvc_models'
	return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pth')]
def get_rvc_indexes():
	folder_path = 'models/rvc_models'
	return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.index')]

def load_rvc_settings():
    rvc_settings_path = './config/rvc.json'
    if os.path.exists(rvc_settings_path):
        with open(rvc_settings_path, 'r') as file:
            return json.load(file)
    else:
        return {}  # Return an empty dict if the file doesn't exist
    
def get_training_folder(voice) -> str:
    '''
    voice(str) : voice to retrieve training folder from
    '''
    return f"./training/{voice}"

def archive_dataset(voice):
    training_folder = get_training_folder(voice)
    archive_root = os.path.join(training_folder,"archived_data")
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_folder = os.path.join(archive_root,current_datetime)
    
    items_to_move = ["train.txt", "validation.txt", "audio"]
    training_folder_contents = os.listdir(training_folder)

    if not any(item in training_folder_contents for item in items_to_move):
        raise gr.Error("No files to move")
    
    for item in items_to_move:
        os.makedirs(archive_folder, exist_ok=True)
        move_item_path = os.path.join(training_folder, item)
        if os.path.exists(move_item_path):
            try:
                shutil.move(move_item_path, archive_folder)
            except:
                raise gr.Error(f'Close out of any windows using where "{item} is located!')
    
    gr.Info('Finished archiving files to "archived_data" folder')