import os
import argparse
import time
import json
import base64
import re
import inspect
import urllib.request

import torch
import torchaudio
import music_tag
import gradio as gr
import gradio.utils

from datetime import datetime

import tortoise.api
from tortoise.utils.audio import get_voice_dir, get_voices
from tortoise.utils.device import get_device_count

from utils import *

args = setup_args()

GENERATE_SETTINGS = {}
TRANSCRIBE_SETTINGS = {}
EXEC_SETTINGS = {}
TRAINING_SETTINGS = {}
MERGER_SETTINGS = {}
GENERATE_SETTINGS_ARGS = []

PRESETS = {
	'Ultra Fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
	'Fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
	'Standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
	'High Quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
}

HISTORY_HEADERS = {
	"Name": "",
	"Samples": "num_autoregressive_samples",
	"Iterations": "diffusion_iterations",
	"Temp.": "temperature",
	"Sampler": "diffusion_sampler",
	"CVVP": "cvvp_weight",
	"Top P": "top_p",
	"Diff. Temp.": "diffusion_temperature",
	"Len Pen": "length_penalty",
	"Rep Pen": "repetition_penalty",
	"Cond-Free K": "cond_free_k",
	"Time": "time",
	"Datetime": "datetime",
	"Model": "model",
	"Model Hash": "model_hash",
}

# can't use *args OR **kwargs if I want to retain the ability to use progress
def generate_proxy(
	text,
	delimiter,
	emotion,
	prompt,
	voice,
	mic_audio,
	voice_latents_chunks,
	candidates,
	seed,
	num_autoregressive_samples,
	diffusion_iterations,
	temperature,
	diffusion_sampler,
	breathing_room,
	cvvp_weight,
	top_p,
	diffusion_temperature,
	length_penalty,
	repetition_penalty,
	cond_free_k,
	experimentals,
	voice_latents_original_ar,
	voice_latents_original_diffusion,
	progress=gr.Progress(track_tqdm=True)
):
	kwargs = locals()

	try:
		sample, outputs, stats = generate(**kwargs)
	except Exception as e:
		message = str(e)
		if message == "Kill signal detected":
			unload_tts()

		raise e
	
	return (
		outputs[0],
		gr.update(value=sample, visible=sample is not None),
		gr.update(choices=outputs, value=outputs[0], visible=len(outputs) > 1, interactive=True),
		gr.update(value=stats, visible=True),
	)


def update_presets(value):
	if value in PRESETS:
		preset = PRESETS[value]
		return (gr.update(value=preset['num_autoregressive_samples']), gr.update(value=preset['diffusion_iterations']))
	else:
		return (gr.update(), gr.update())

def get_training_configs():
	configs = []
	for i, file in enumerate(sorted(os.listdir(f"./training/"))):
		if file[-5:] != ".yaml" or file[0] == ".":
			continue
		configs.append(f"./training/{file}")

	return configs

def update_training_configs():
	return gr.update(choices=get_training_list())

def history_view_results( voice ):
	results = []
	files = []
	outdir = f"{args.results_folder}/{voice}/"
	for i, file in enumerate(sorted(os.listdir(outdir))):
		if file[-4:] != ".wav":
			continue

		metadata, _ = read_generate_settings(f"{outdir}/{file}", read_latents=False)
		if metadata is None:
			continue
			
		values = []
		for k in HISTORY_HEADERS:
			v = file
			if k != "Name":
				v = metadata[HISTORY_HEADERS[k]] if HISTORY_HEADERS[k] in metadata else '?'
			values.append(v)


		files.append(file)
		results.append(values)

	return (
		results,
		gr.Dropdown.update(choices=sorted(files))
	)

def import_generate_settings_proxy( file=None ):
	global GENERATE_SETTINGS_ARGS
	settings = import_generate_settings( file )

	res = []
	for k in GENERATE_SETTINGS_ARGS:
		res.append(settings[k] if k in settings else None)

	return tuple(res)

def reset_generate_settings_proxy():
	global GENERATE_SETTINGS_ARGS
	settings = reset_generate_settings()

	res = []
	for k in GENERATE_SETTINGS_ARGS:
		res.append(settings[k] if k in settings else None)

	return tuple(res)

def compute_latents_proxy(voice, voice_latents_chunks, original_ar, original_diffusion, progress=gr.Progress(track_tqdm=True)):
	compute_latents( voice=voice, voice_latents_chunks=voice_latents_chunks, original_ar=original_ar, original_diffusion=original_diffusion )
	return voice


def import_voices_proxy(files, name, progress=gr.Progress(track_tqdm=True)):
	import_voices(files, name, progress)
	return gr.update()

def read_generate_settings_proxy(file, saveAs='.temp'):
	j, latents = read_generate_settings(file)

	if latents:
		outdir = f'{get_voice_dir()}/{saveAs}/'
		os.makedirs(outdir, exist_ok=True)
		with open(f'{outdir}/cond_latents.pth', 'wb') as f:
			f.write(latents)
		
		latents = f'{outdir}/cond_latents.pth'

	return (
		gr.update(value=j, visible=j is not None),
		gr.update(value=latents, visible=latents is not None),
		None if j is None else j['voice'],
		gr.update(visible=j is not None),
	)

def slice_dataset_proxy( voice, trim_silence, start_offset, end_offset, progress=gr.Progress(track_tqdm=True) ):
	return slice_dataset( voice, trim_silence=trim_silence, start_offset=start_offset, end_offset=end_offset, results=None, progress=progress )

def diarize_dataset( voice, progress=gr.Progress(track_tqdm=True) ):
	from pyannote.audio import Pipeline
	pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=args.hf_token)

	messages = []
	files = get_voice(voice, load_latents=False)
	for file in enumerate_progress(files, desc="Iterating through voice files", progress=progress):
		diarization = pipeline(file)
		for turn, _, speaker in diarization.itertracks(yield_label=True):
			message = f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}"
			print(message)
			messages.append(message)

	return "\n".join(messages)

def prepare_all_datasets( language, validation_text_length, validation_audio_length, skip_existings, slice_audio, trim_silence, slice_start_offset, slice_end_offset, progress=gr.Progress(track_tqdm=True) ):
	kwargs = locals()

	messages = []
	voices = get_voice_list()

	for voice in voices:
		print("Processing:", voice)
		message = transcribe_dataset( voice=voice, language=language, skip_existings=skip_existings, progress=progress )
		messages.append(message)

	if slice_audio:
		for voice in voices:
			print("Processing:", voice)
			message = slice_dataset( voice, trim_silence=trim_silence, start_offset=slice_start_offset, end_offset=slice_end_offset, results=None, progress=progress )
			messages.append(message)
			
	for voice in voices:
		print("Processing:", voice)
		message = prepare_dataset( voice, use_segments=slice_audio, text_length=validation_text_length, audio_length=validation_audio_length, progress=progress )
		messages.append(message)

	return "\n".join(messages)

def prepare_dataset_proxy( voice, language, validation_text_length, validation_audio_length, skip_existings, slice_audio, trim_silence, slice_start_offset, slice_end_offset, progress=gr.Progress(track_tqdm=True) ):
	messages = []
	
	message = transcribe_dataset( voice=voice, language=language, skip_existings=skip_existings, progress=progress )
	messages.append(message)

	if slice_audio:
		message = slice_dataset( voice, trim_silence=trim_silence, start_offset=slice_start_offset, end_offset=slice_end_offset, results=None, progress=progress )
		messages.append(message)

	message = prepare_dataset( voice, use_segments=slice_audio, text_length=validation_text_length, audio_length=validation_audio_length, progress=progress )
	messages.append(message)

	return "\n".join(messages)

def update_args_proxy( *args ):
	kwargs = {}
	keys = list(EXEC_SETTINGS.keys())
	for i in range(len(args)):
		k = keys[i]
		v = args[i]
		kwargs[k] = v

	update_args(**kwargs)
def optimize_training_settings_proxy( *args ):
	kwargs = {}
	keys = list(TRAINING_SETTINGS.keys())
	for i in range(len(args)):
		k = keys[i]
		v = args[i]
		kwargs[k] = v

	settings, messages = optimize_training_settings(**kwargs)
	output = list(settings.values())
	return output[:-1] + ["\n".join(messages)]

def import_training_settings_proxy( voice ):
	messages = []
	injson = f'./training/{voice}/train.json'
	statedir = f'./training/{voice}/finetune/training_state/'
	output = {}

	try:
		with open(injson, 'r', encoding="utf-8") as f:
			settings = json.loads(f.read())
	except:
		messages.append(f"Error import /{voice}/train.json")

		for k in TRAINING_SETTINGS:
			output[k] = TRAINING_SETTINGS[k].value

		output = list(output.values())
		return output[:-1] + ["\n".join(messages)]

	if os.path.isdir(statedir):
		resumes = sorted([int(d[:-6]) for d in os.listdir(statedir) if d[-6:] == ".state" ])

		if len(resumes) > 0:
			settings['resume_state'] = f'{statedir}/{resumes[-1]}.state'
			messages.append(f"Found most recent training state: {settings['resume_state']}")

	output = {}
	for k in TRAINING_SETTINGS:
		if k not in settings:
			output[k] = gr.update()
		else:
			output[k] = gr.update(value=settings[k])

	output = list(output.values())

	messages.append(f"Imported training settings: {injson}")

	return output[:-1] + ["\n".join(messages)]

def save_training_settings_proxy( *args ):
	kwargs = {}
	keys = list(TRAINING_SETTINGS.keys())
	for i in range(len(args)):
		k = keys[i]
		v = args[i]
		kwargs[k] = v

	settings, messages = save_training_settings(**kwargs)
	return "\n".join(messages)

def update_voices():
	return (
		gr.Dropdown.update(choices=get_voice_list(append_defaults=True)),
		gr.Dropdown.update(choices=get_voice_list()),
		gr.Dropdown.update(choices=get_voice_list(args.results_folder)),
	)

def history_copy_settings( voice, file ):
	return import_generate_settings( f"{args.results_folder}/{voice}/{file}" )

def setup_gradio():
	global args
	global ui

	if not args.share:
		def noop(function, return_value=None):
			def wrapped(*args, **kwargs):
				return return_value
			return wrapped
		gradio.utils.version_check = noop(gradio.utils.version_check)
		gradio.utils.initiated_analytics = noop(gradio.utils.initiated_analytics)
		gradio.utils.launch_analytics = noop(gradio.utils.launch_analytics)
		gradio.utils.integration_analytics = noop(gradio.utils.integration_analytics)
		gradio.utils.error_analytics = noop(gradio.utils.error_analytics)
		gradio.utils.log_feature_analytics = noop(gradio.utils.log_feature_analytics)
		#gradio.utils.get_local_ip_address = noop(gradio.utils.get_local_ip_address, 'localhost')

	if args.models_from_local_only:
		os.environ['TRANSFORMERS_OFFLINE']='1'

	voice_list_with_defaults = get_voice_list(append_defaults=True)
	voice_list = get_voice_list()
	result_voices = get_voice_list(args.results_folder)
	
	valle_models = get_valle_models()

	autoregressive_models = get_autoregressive_models()
	diffusion_models = get_diffusion_models()
	tokenizer_jsons = get_tokenizer_jsons()

	dataset_list = get_dataset_list()
	training_list = get_training_list()

	global GENERATE_SETTINGS_ARGS
	GENERATE_SETTINGS_ARGS = list(inspect.signature(generate_proxy).parameters.keys())[:-1]
	for i in range(len(GENERATE_SETTINGS_ARGS)):
		arg = GENERATE_SETTINGS_ARGS[i]
		GENERATE_SETTINGS[arg] = None

	with gr.Blocks() as ui:
		with gr.Tab("Generate"):
			with gr.Row():
				with gr.Column():
					GENERATE_SETTINGS["text"] = gr.Textbox(lines=4, value="Your prompt here.", label="Input Prompt")
			with gr.Row():
				with gr.Column():
					GENERATE_SETTINGS["delimiter"] = gr.Textbox(lines=1, label="Line Delimiter", placeholder="\\n")

					GENERATE_SETTINGS["emotion"] = gr.Radio( ["Happy", "Sad", "Angry", "Disgusted", "Arrogant", "Custom", "None"], value="None", label="Emotion", type="value", interactive=True, visible=args.tts_backend=="tortoise" )
					GENERATE_SETTINGS["prompt"] = gr.Textbox(lines=1, label="Custom Emotion", visible=False)
					GENERATE_SETTINGS["voice"] = gr.Dropdown(choices=voice_list_with_defaults, label="Voice", type="value", value=voice_list_with_defaults[0]) # it'd be very cash money if gradio was able to default to the first value in the list without this shit
					GENERATE_SETTINGS["mic_audio"] = gr.Audio( label="Microphone Source", source="microphone", type="filepath", visible=False )
					GENERATE_SETTINGS["voice_latents_chunks"] = gr.Number(label="Voice Chunks", precision=0, value=0, visible=args.tts_backend=="tortoise")
					GENERATE_SETTINGS["voice_latents_original_ar"] = gr.Checkbox(label="Use Original Latents Method (AR)", visible=args.tts_backend=="tortoise")
					GENERATE_SETTINGS["voice_latents_original_diffusion"] = gr.Checkbox(label="Use Original Latents Method (Diffusion)", visible=args.tts_backend=="tortoise")
					with gr.Row():
						refresh_voices = gr.Button(value="Refresh Voice List")
						recompute_voice_latents = gr.Button(value="(Re)Compute Voice Latents")

					GENERATE_SETTINGS["voice"].change(
						fn=update_baseline_for_latents_chunks,
						inputs=GENERATE_SETTINGS["voice"],
						outputs=GENERATE_SETTINGS["voice_latents_chunks"]
					)
					GENERATE_SETTINGS["voice"].change(
						fn=lambda value: gr.update(visible=value == "microphone"),
						inputs=GENERATE_SETTINGS["voice"],
						outputs=GENERATE_SETTINGS["mic_audio"],
					)
				with gr.Column():
					preset = None						
					GENERATE_SETTINGS["candidates"] = gr.Slider(value=1, minimum=1, maximum=6, step=1, label="Candidates", visible=args.tts_backend=="tortoise")
					GENERATE_SETTINGS["seed"] = gr.Number(value=0, precision=0, label="Seed", visible=args.tts_backend=="tortoise")

					preset = gr.Radio( ["Ultra Fast", "Fast", "Standard", "High Quality"], label="Preset", type="value", value="Ultra Fast", visible=args.tts_backend=="tortoise" )

					GENERATE_SETTINGS["num_autoregressive_samples"] = gr.Slider(value=16, minimum=2, maximum=2048 if args.tts_backend=="vall-e" else 512, step=1, label="Samples", visible=args.tts_backend!="bark")
					GENERATE_SETTINGS["diffusion_iterations"] = gr.Slider(value=30, minimum=0, maximum=512, step=1, label="Iterations", visible=args.tts_backend=="tortoise")

					GENERATE_SETTINGS["temperature"] = gr.Slider(value=0.95 if args.tts_backend=="vall-e" else 0.2, minimum=0, maximum=1, step=0.05, label="Temperature")
					
					show_experimental_settings = gr.Checkbox(label="Show Experimental Settings", visible=args.tts_backend=="tortoise")
					reset_generate_settings_button = gr.Button(value="Reset to Default")
				with gr.Column(visible=False) as col:
					experimental_column = col

					GENERATE_SETTINGS["experimentals"] = gr.CheckboxGroup(["Half Precision", "Conditioning-Free"], value=["Conditioning-Free"], label="Experimental Flags")
					GENERATE_SETTINGS["breathing_room"] = gr.Slider(value=8, minimum=1, maximum=32, step=1, label="Pause Size")
					GENERATE_SETTINGS["diffusion_sampler"] = gr.Radio(
						["P", "DDIM"], # + ["K_Euler_A", "DPM++2M"],
						value="DDIM", label="Diffusion Samplers", type="value"
					)
					GENERATE_SETTINGS["cvvp_weight"] = gr.Slider(value=0, minimum=0, maximum=1, label="CVVP Weight")
					GENERATE_SETTINGS["top_p"] = gr.Slider(value=0.8, minimum=0, maximum=1, label="Top P")
					GENERATE_SETTINGS["diffusion_temperature"] = gr.Slider(value=1.0, minimum=0, maximum=1, label="Diffusion Temperature")
					GENERATE_SETTINGS["length_penalty"] = gr.Slider(value=1.0, minimum=0, maximum=8, label="Length Penalty")
					GENERATE_SETTINGS["repetition_penalty"] = gr.Slider(value=2.0, minimum=0, maximum=8, label="Repetition Penalty")
					GENERATE_SETTINGS["cond_free_k"] = gr.Slider(value=2.0, minimum=0, maximum=4, label="Conditioning-Free K")
				with gr.Column():
					with gr.Row():
						submit = gr.Button(value="Generate")
						stop = gr.Button(value="Stop")

					generation_results = gr.Dataframe(label="Results", headers=["Seed", "Time"], visible=False)
					source_sample = gr.Audio(label="Source Sample", visible=False)
					output_audio = gr.Audio(label="Output")
					candidates_list = gr.Dropdown(label="Candidates", type="value", visible=False, choices=[""], value="")

					def change_candidate( val ):
						if not val:
							return
						return val

					candidates_list.change(
						fn=change_candidate,
						inputs=candidates_list,
						outputs=output_audio,
					)
		with gr.Tab("History"):
			with gr.Row():
				with gr.Column():
					history_info = gr.Dataframe(label="Results", headers=list(HISTORY_HEADERS.keys()))
			with gr.Row():
				with gr.Column():
					history_voices = gr.Dropdown(choices=result_voices, label="Voice", type="value", value=result_voices[0] if len(result_voices) > 0 else "")
				with gr.Column():
					history_results_list = gr.Dropdown(label="Results",type="value", interactive=True, value="")
				with gr.Column():
					history_audio = gr.Audio()
					history_copy_settings_button = gr.Button(value="Copy Settings")
		with gr.Tab("Utilities"):
			with gr.Tab("Import / Analyze"):
				with gr.Row():
					with gr.Column():
						audio_in = gr.Files(type="file", label="Audio Input", file_types=["audio"])
						import_voice_name = gr.Textbox(label="Voice Name")
						import_voice_button = gr.Button(value="Import Voice")
					with gr.Column(visible=False) as col:
						utilities_metadata_column = col

						metadata_out = gr.JSON(label="Audio Metadata")
						copy_button = gr.Button(value="Copy Settings")
						latents_out = gr.File(type="binary", label="Voice Latents")
			with gr.Tab("Tokenizer"):
				with gr.Row():
					text_tokenizier_input = gr.TextArea(label="Text", max_lines=4)
					text_tokenizier_output = gr.TextArea(label="Tokenized Text", max_lines=4)

				with gr.Row():
					text_tokenizier_button = gr.Button(value="Tokenize Text")
			with gr.Tab("Model Merger"):
				with gr.Column():
					with gr.Row():
						MERGER_SETTINGS["model_a"] = gr.Dropdown( choices=autoregressive_models, label="Model A", type="value", value=autoregressive_models[0] )
						MERGER_SETTINGS["model_b"] = gr.Dropdown( choices=autoregressive_models, label="Model B", type="value", value=autoregressive_models[0] )
					with gr.Row():
						MERGER_SETTINGS["weight_slider"] = gr.Slider(label="Weight (from A to B)", value=0.5, minimum=0, maximum=1)
					with gr.Row():
						merger_button = gr.Button(value="Run Merger")
				with gr.Column():
					merger_output = gr.TextArea(label="Console Output", max_lines=8)
		with gr.Tab("Training"):
			with gr.Tab("Prepare Dataset"):
				with gr.Row():
					with gr.Column():
						DATASET_SETTINGS = {}
						DATASET_SETTINGS['voice'] = gr.Dropdown( choices=voice_list, label="Dataset Source", type="value", value=voice_list[0] if len(voice_list) > 0 else "" )
						with gr.Row():
							DATASET_SETTINGS['language'] = gr.Textbox(label="Language", value="en")
							DATASET_SETTINGS['validation_text_length'] = gr.Number(label="Validation Text Length Threshold", value=12, precision=0, visible=args.tts_backend=="tortoise")
							DATASET_SETTINGS['validation_audio_length'] = gr.Number(label="Validation Audio Length Threshold", value=1, visible=args.tts_backend=="tortoise" )
						with gr.Row():
							DATASET_SETTINGS['skip'] = gr.Checkbox(label="Skip Existing", value=False)
							DATASET_SETTINGS['slice'] = gr.Checkbox(label="Slice Segments", value=False)
							DATASET_SETTINGS['trim_silence'] = gr.Checkbox(label="Trim Silence", value=False)
						with gr.Row():
							DATASET_SETTINGS['slice_start_offset'] = gr.Number(label="Slice Start Offset", value=0)
							DATASET_SETTINGS['slice_end_offset'] = gr.Number(label="Slice End Offset", value=0)

						transcribe_button = gr.Button(value="Transcribe and Process")
						transcribe_all_button = gr.Button(value="Transcribe All")
						diarize_button = gr.Button(value="Diarize", visible=False)
						
						with gr.Row():
							slice_dataset_button = gr.Button(value="(Re)Slice Audio")
							prepare_dataset_button = gr.Button(value="(Re)Create Dataset")

						with gr.Row():
							EXEC_SETTINGS['whisper_backend'] = gr.Dropdown(WHISPER_BACKENDS, label="Whisper Backends", value=args.whisper_backend)
							EXEC_SETTINGS['whisper_model'] = gr.Dropdown(WHISPER_MODELS, label="Whisper Model", value=args.whisper_model)

						dataset_settings = list(DATASET_SETTINGS.values())
					with gr.Column():
						prepare_dataset_output = gr.TextArea(label="Console Output", interactive=False, max_lines=8)
			with gr.Tab("Generate Configuration", visible=args.tts_backend != "bark"):
				with gr.Row():
					with gr.Column():
						TRAINING_SETTINGS["epochs"] = gr.Number(label="Epochs", value=500, precision=0)
						with gr.Row(visible=args.tts_backend=="tortoise"):
							TRAINING_SETTINGS["learning_rate"] = gr.Slider(label="Learning Rate", value=1e-5, minimum=0, maximum=1e-4, step=1e-6)
							TRAINING_SETTINGS["mel_lr_weight"] = gr.Slider(label="Mel LR Ratio", value=1.00, minimum=0, maximum=1)
							TRAINING_SETTINGS["text_lr_weight"] = gr.Slider(label="Text LR Ratio", value=0.01, minimum=0, maximum=1)
							
						with gr.Row(visible=args.tts_backend=="tortoise"):
							lr_schemes = list(LEARNING_RATE_SCHEMES.keys())
							TRAINING_SETTINGS["learning_rate_scheme"] = gr.Radio(lr_schemes, label="Learning Rate Scheme", value=lr_schemes[0], type="value")
							TRAINING_SETTINGS["learning_rate_schedule"] = gr.Textbox(label="Learning Rate Schedule", placeholder=str(LEARNING_RATE_SCHEDULE), visible=True)
							TRAINING_SETTINGS["learning_rate_restarts"] = gr.Number(label="Learning Rate Restarts", value=4, precision=0, visible=False)

							TRAINING_SETTINGS["learning_rate_scheme"].change(
								fn=lambda x: ( gr.update(visible=x == lr_schemes[0]), gr.update(visible=x == lr_schemes[1]) ),
								inputs=TRAINING_SETTINGS["learning_rate_scheme"],
								outputs=[
									TRAINING_SETTINGS["learning_rate_schedule"],
									TRAINING_SETTINGS["learning_rate_restarts"],
								]
							)
						with gr.Row():
							TRAINING_SETTINGS["batch_size"] = gr.Number(label="Batch Size", value=128, precision=0)
							TRAINING_SETTINGS["gradient_accumulation_size"] = gr.Number(label="Gradient Accumulation Size", value=4, precision=0)
						with gr.Row():
							TRAINING_SETTINGS["save_rate"] = gr.Number(label="Save Frequency (in epochs)", value=5, precision=0)
							TRAINING_SETTINGS["validation_rate"] = gr.Number(label="Validation Frequency (in epochs)", value=5, precision=0)

						with gr.Row():
							TRAINING_SETTINGS["half_p"] = gr.Checkbox(label="Half Precision", value=args.training_default_halfp, visible=args.tts_backend=="tortoise")
							TRAINING_SETTINGS["bitsandbytes"] = gr.Checkbox(label="BitsAndBytes", value=args.training_default_bnb, visible=args.tts_backend=="tortoise")
							TRAINING_SETTINGS["validation_enabled"] = gr.Checkbox(label="Validation Enabled", value=False)

						with gr.Row():
							TRAINING_SETTINGS["workers"] = gr.Number(label="Worker Processes", value=2, precision=0, visible=args.tts_backend=="tortoise")
							TRAINING_SETTINGS["gpus"] = gr.Number(label="GPUs", value=get_device_count(), precision=0)

						TRAINING_SETTINGS["source_model"] = gr.Dropdown( choices=autoregressive_models, label="Source Model", type="value", value=autoregressive_models[0], visible=args.tts_backend=="tortoise" )
						TRAINING_SETTINGS["resume_state"] = gr.Textbox(label="Resume State Path", placeholder="./training/${voice}/finetune/training_state/${last_state}.state", visible=args.tts_backend=="tortoise")
						
						TRAINING_SETTINGS["voice"] = gr.Dropdown( choices=dataset_list, label="Dataset", type="value", value=dataset_list[0] if len(dataset_list) else ""  )

						with gr.Row():
							training_refresh_dataset = gr.Button(value="Refresh Dataset List")
							training_import_settings = gr.Button(value="Reuse/Import Dataset")
					with gr.Column():
						training_configuration_output = gr.TextArea(label="Console Output", interactive=False, max_lines=8)
						with gr.Row():
							training_optimize_configuration = gr.Button(value="Validate Training Configuration")
							training_save_configuration = gr.Button(value="Save Training Configuration")
			with gr.Tab("Run Training", visible=args.tts_backend != "bark"):
				with gr.Row():
					with gr.Column():
						training_configs = gr.Dropdown(label="Training Configuration", choices=training_list, value=training_list[0] if len(training_list) else "")
						refresh_configs = gr.Button(value="Refresh Configurations")
						training_output = gr.TextArea(label="Console Output", interactive=False, max_lines=8)
						verbose_training = gr.Checkbox(label="Verbose Console Output", value=True)
						
						keep_x_past_checkpoints = gr.Slider(label="Keep X Previous States", minimum=0, maximum=8, value=0, step=1)
						
						with gr.Row():
							training_graph_x_min = gr.Number(label="X Min", precision=0, value=0)
							training_graph_x_max = gr.Number(label="X Max", precision=0, value=0)
							training_graph_y_min = gr.Number(label="Y Min", precision=0, value=0)
							training_graph_y_max = gr.Number(label="Y Max", precision=0, value=0)

						with gr.Row():
							start_training_button = gr.Button(value="Train")
							stop_training_button = gr.Button(value="Stop")
							reconnect_training_button = gr.Button(value="Reconnect")
						
						
					with gr.Column():
						training_loss_graph = gr.LinePlot(label="Training Metrics",
							x="it", # x="epoch",
							y="value",
							title="Loss Metrics",
							color="type",
							tooltip=['epoch', 'it', 'value', 'type'],
							width=500,
							height=350,
						)
						training_lr_graph = gr.LinePlot(label="Training Metrics",
							x="it", # x="epoch",
							y="value",
							title="Learning Rate",
							color="type",
							tooltip=['epoch', 'it', 'value', 'type'],
							width=500,
							height=350,
						)
						training_grad_norm_graph = gr.LinePlot(label="Training Metrics",
							x="it", # x="epoch",
							y="value",
							title="Gradient Normals",
							color="type",
							tooltip=['epoch', 'it', 'value', 'type'],
							width=500,
							height=350,
							visible=False, # args.tts_backend=="vall-e"
						)
						view_losses = gr.Button(value="View Losses")

		with gr.Tab("Settings"):
			with gr.Row():
				exec_inputs = []
				with gr.Column():
					EXEC_SETTINGS['listen'] = gr.Textbox(label="Listen", value=args.listen, placeholder="127.0.0.1:7860/")
					EXEC_SETTINGS['share'] = gr.Checkbox(label="Public Share Gradio", value=args.share)
					EXEC_SETTINGS['check_for_updates'] = gr.Checkbox(label="Check For Updates", value=args.check_for_updates)
					EXEC_SETTINGS['models_from_local_only'] = gr.Checkbox(label="Only Load Models Locally", value=args.models_from_local_only)
					EXEC_SETTINGS['low_vram'] = gr.Checkbox(label="Low VRAM", value=args.low_vram)
					EXEC_SETTINGS['embed_output_metadata'] = gr.Checkbox(label="Embed Output Metadata", value=args.embed_output_metadata)
					EXEC_SETTINGS['latents_lean_and_mean'] = gr.Checkbox(label="Slimmer Computed Latents", value=args.latents_lean_and_mean)
					EXEC_SETTINGS['voice_fixer'] = gr.Checkbox(label="Use Voice Fixer on Generated Output", value=args.voice_fixer)
					EXEC_SETTINGS['use_deepspeed'] = gr.Checkbox(label="Use DeepSpeed for Speed Bump.", value=args.use_deepspeed)
					EXEC_SETTINGS['use_hifigan'] = gr.Checkbox(label="Use Hifigan instead of Diffusion.", value=args.use_hifigan)
					EXEC_SETTINGS['voice_fixer_use_cuda'] = gr.Checkbox(label="Use CUDA for Voice Fixer", value=args.voice_fixer_use_cuda)
					EXEC_SETTINGS['force_cpu_for_conditioning_latents'] = gr.Checkbox(label="Force CPU for Conditioning Latents", value=args.force_cpu_for_conditioning_latents)
					EXEC_SETTINGS['defer_tts_load'] = gr.Checkbox(label="Do Not Load TTS On Startup", value=args.defer_tts_load)
					EXEC_SETTINGS['prune_nonfinal_outputs'] = gr.Checkbox(label="Delete Non-Final Output", value=args.prune_nonfinal_outputs)
				with gr.Column():
					EXEC_SETTINGS['sample_batch_size'] = gr.Number(label="Sample Batch Size", precision=0, value=args.sample_batch_size)
					EXEC_SETTINGS['unsqueeze_sample_batches'] = gr.Checkbox(label="Unsqueeze Sample Batches", value=args.unsqueeze_sample_batches)
					EXEC_SETTINGS['concurrency_count'] = gr.Number(label="Gradio Concurrency Count", precision=0, value=args.concurrency_count)
					EXEC_SETTINGS['autocalculate_voice_chunk_duration_size'] = gr.Number(label="Auto-Calculate Voice Chunk Duration (in seconds)", precision=0, value=args.autocalculate_voice_chunk_duration_size)
					EXEC_SETTINGS['output_volume'] = gr.Slider(label="Output Volume", minimum=0, maximum=2, value=args.output_volume)
					EXEC_SETTINGS['device_override'] = gr.Textbox(label="Device Override", value=args.device_override)

					EXEC_SETTINGS['results_folder'] = gr.Textbox(label="Results Folder", value=args.results_folder)
					# EXEC_SETTINGS['tts_backend'] = gr.Dropdown(TTSES, label="TTS Backend", value=args.tts_backend if args.tts_backend else TTSES[0])
					
				if args.tts_backend=="vall-e":
					with gr.Column():
						EXEC_SETTINGS['valle_model'] = gr.Dropdown(choices=valle_models, label="VALL-E Model Config", value=args.valle_model if args.valle_model else valle_models[0])

				with gr.Column(visible=args.tts_backend=="tortoise"):
					EXEC_SETTINGS['autoregressive_model'] = gr.Dropdown(choices=["auto"] + autoregressive_models, label="Autoregressive Model", value=args.autoregressive_model if args.autoregressive_model else "auto")
					EXEC_SETTINGS['diffusion_model'] = gr.Dropdown(choices=diffusion_models, label="Diffusion Model", value=args.diffusion_model if args.diffusion_model else diffusion_models[0])
					EXEC_SETTINGS['vocoder_model'] = gr.Dropdown(VOCODERS, label="Vocoder", value=args.vocoder_model if args.vocoder_model else VOCODERS[-1])
					EXEC_SETTINGS['tokenizer_json'] = gr.Dropdown(tokenizer_jsons, label="Tokenizer JSON Path", value=args.tokenizer_json if args.tokenizer_json else tokenizer_jsons[0])
					
					EXEC_SETTINGS['training_default_halfp'] = TRAINING_SETTINGS['half_p']
					EXEC_SETTINGS['training_default_bnb'] = TRAINING_SETTINGS['bitsandbytes']

					with gr.Row():
						autoregressive_models_update_button = gr.Button(value="Refresh Model List")
						gr.Button(value="Check for Updates").click(check_for_updates)
						gr.Button(value="(Re)Load TTS").click(
							reload_tts,
							inputs=None,
							outputs=None
						)
						# kill_button = gr.Button(value="Close UI")

					def update_model_list_proxy( autoregressive, diffusion, tokenizer ):
						autoregressive_models = get_autoregressive_models()
						if autoregressive not in autoregressive_models:
							autoregressive = autoregressive_models[0]

						diffusion_models = get_diffusion_models()
						if diffusion not in diffusion_models:
							diffusion = diffusion_models[0]

						tokenizer_jsons = get_tokenizer_jsons()
						if tokenizer not in tokenizer_jsons:
							tokenizer = tokenizer_jsons[0]

						return (
							gr.update( choices=autoregressive_models, value=autoregressive ),
							gr.update( choices=diffusion_models, value=diffusion ),
							gr.update( choices=tokenizer_jsons, value=tokenizer ),
						)

					autoregressive_models_update_button.click(
						update_model_list_proxy,
						inputs=[
							EXEC_SETTINGS['autoregressive_model'],
							EXEC_SETTINGS['diffusion_model'],
							EXEC_SETTINGS['tokenizer_json'],
						],
						outputs=[
							EXEC_SETTINGS['autoregressive_model'],
							EXEC_SETTINGS['diffusion_model'],
							EXEC_SETTINGS['tokenizer_json'],
						],
					)

				exec_inputs = list(EXEC_SETTINGS.values())
				for k in EXEC_SETTINGS:
					EXEC_SETTINGS[k].change( fn=update_args_proxy, inputs=exec_inputs )
				
				EXEC_SETTINGS['autoregressive_model'].change(
					fn=update_autoregressive_model,
					inputs=EXEC_SETTINGS['autoregressive_model'],
					outputs=None,
					api_name="set_autoregressive_model"
				)

				EXEC_SETTINGS['vocoder_model'].change(
					fn=update_vocoder_model,
					inputs=EXEC_SETTINGS['vocoder_model'],
					outputs=None
				)

		history_voices.change(
			fn=history_view_results,
			inputs=history_voices,
			outputs=[
				history_info,
				history_results_list,
			]
		)
		history_results_list.change(
			fn=lambda voice, file: f"{args.results_folder}/{voice}/{file}",
			inputs=[
				history_voices,
				history_results_list,
			],
			outputs=history_audio
		)
		audio_in.upload(
			fn=read_generate_settings_proxy,
			inputs=audio_in,
			outputs=[
				metadata_out,
				latents_out,
				import_voice_name,
				utilities_metadata_column,
			]
		)

		import_voice_button.click(
			fn=import_voices_proxy,
			inputs=[
				audio_in,
				import_voice_name,
			],
			outputs=import_voice_name #console_output
		)
		show_experimental_settings.change(
			fn=lambda x: gr.update(visible=x),
			inputs=show_experimental_settings,
			outputs=experimental_column
		)
		if preset:
			preset.change(fn=update_presets,
				inputs=preset,
				outputs=[
					GENERATE_SETTINGS['num_autoregressive_samples'],
					GENERATE_SETTINGS['diffusion_iterations'],
				],
			)

		recompute_voice_latents.click(compute_latents_proxy,
			inputs=[
				GENERATE_SETTINGS['voice'],
				GENERATE_SETTINGS['voice_latents_chunks'],
				GENERATE_SETTINGS['voice_latents_original_ar'],
				GENERATE_SETTINGS['voice_latents_original_diffusion'],
			],
			outputs=GENERATE_SETTINGS['voice'],
		)
		
		GENERATE_SETTINGS['emotion'].change(
			fn=lambda value: gr.update(visible=value == "Custom"),
			inputs=GENERATE_SETTINGS['emotion'],
			outputs=GENERATE_SETTINGS['prompt']
		)
		GENERATE_SETTINGS['mic_audio'].change(fn=lambda value: gr.update(value="microphone"),
			inputs=GENERATE_SETTINGS['mic_audio'],
			outputs=GENERATE_SETTINGS['voice']
		)

		refresh_voices.click(update_voices,
			inputs=None,
			outputs=[
				GENERATE_SETTINGS['voice'],
				DATASET_SETTINGS['voice'],
				history_voices
			]
		)

		generate_settings = list(GENERATE_SETTINGS.values())
		submit.click(
			lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
			outputs=[source_sample, candidates_list, generation_results],
		)

		submit_event = submit.click(generate_proxy,
			inputs=generate_settings,
			outputs=[output_audio, source_sample, candidates_list, generation_results],
			api_name="generate",
		)


		copy_button.click(import_generate_settings_proxy,
			inputs=audio_in, # JSON elements cannot be used as inputs
			outputs=generate_settings
		)

		reset_generate_settings_button.click(
			fn=reset_generate_settings_proxy,
			inputs=None,
			outputs=generate_settings
		)

		history_copy_settings_button.click(history_copy_settings,
			inputs=[
				history_voices,
				history_results_list,
			],
			outputs=generate_settings
		)

		text_tokenizier_button.click(tokenize_text,
			inputs=text_tokenizier_input,
			outputs=text_tokenizier_output
		)

		merger_button.click(merge_models,
			inputs=list(MERGER_SETTINGS.values()),
			outputs=merger_output
		)

		refresh_configs.click(
			lambda: gr.update(choices=get_training_list()),
			inputs=None,
			outputs=training_configs
		)
		start_training_button.click(run_training,
			inputs=[
				training_configs,
				verbose_training,
				keep_x_past_checkpoints,
			],
			outputs=[
				training_output,
			],
		)
		training_output.change(
			fn=update_training_dataplot,
			inputs=[
				training_graph_x_min,
				training_graph_x_max,
				training_graph_y_min,
				training_graph_y_max,
			],
			outputs=[
				training_loss_graph,
				training_lr_graph,
				training_grad_norm_graph,
			],
			show_progress=False,
		)

		view_losses.click(
			fn=update_training_dataplot,
			inputs=[
				training_graph_x_min,
				training_graph_x_max,
				training_graph_y_min,
				training_graph_y_max,
				training_configs,
			],
			outputs=[
				training_loss_graph,
				training_lr_graph,
				training_grad_norm_graph,
			],
		)

		stop_training_button.click(stop_training,
			inputs=None,
			outputs=training_output #console_output
		)
		reconnect_training_button.click(reconnect_training,
			inputs=[
				verbose_training,
			],
			outputs=training_output #console_output
		)
		transcribe_button.click(
			prepare_dataset_proxy,
			inputs=dataset_settings,
			outputs=prepare_dataset_output #console_output
		)
		transcribe_all_button.click(
			prepare_all_datasets,
			inputs=dataset_settings[1:],
			outputs=prepare_dataset_output #console_output
		)
		diarize_button.click(
			diarize_dataset,
			inputs=dataset_settings[0],
			outputs=prepare_dataset_output #console_output
		)
		prepare_dataset_button.click(
			prepare_dataset,
			inputs=[
				DATASET_SETTINGS['voice'],
				DATASET_SETTINGS['slice'],
				DATASET_SETTINGS['validation_text_length'],
				DATASET_SETTINGS['validation_audio_length'],
			],
			outputs=prepare_dataset_output #console_output
		)
		slice_dataset_button.click(
			slice_dataset_proxy,
			inputs=[
				DATASET_SETTINGS['voice'],
				DATASET_SETTINGS['trim_silence'],
				DATASET_SETTINGS['slice_start_offset'],
				DATASET_SETTINGS['slice_end_offset'],
			],
			outputs=prepare_dataset_output
		)
		
		training_refresh_dataset.click(
			lambda: gr.update(choices=get_dataset_list()),
			inputs=None,
			outputs=TRAINING_SETTINGS["voice"],
		)
		training_settings = list(TRAINING_SETTINGS.values())
		training_optimize_configuration.click(optimize_training_settings_proxy,
			inputs=training_settings,
			outputs=training_settings[:-1] + [training_configuration_output] #console_output
		)
		training_import_settings.click(import_training_settings_proxy,
			inputs=TRAINING_SETTINGS['voice'],
			outputs=training_settings[:-1] + [training_configuration_output] #console_output
		)
		training_save_configuration.click(save_training_settings_proxy,
			inputs=training_settings,
			outputs=training_configuration_output #console_output
		)

		if os.path.isfile('./config/generate.json'):
			ui.load(import_generate_settings_proxy, inputs=None, outputs=generate_settings)
		
		if args.check_for_updates:
			ui.load(check_for_updates)

		stop.click(fn=cancel_generate, inputs=None, outputs=None)


	ui.queue(concurrency_count=args.concurrency_count)
	webui = ui
	return webui