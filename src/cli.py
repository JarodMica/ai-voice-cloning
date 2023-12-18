import os
import argparse

if 'TORTOISE_MODELS_DIR' not in os.environ:
	os.environ['TORTOISE_MODELS_DIR'] = os.path.realpath(os.path.join(os.getcwd(), './models/tortoise/'))

if 'TRANSFORMERS_CACHE' not in os.environ:
	os.environ['TRANSFORMERS_CACHE'] = os.path.realpath(os.path.join(os.getcwd(), './models/transformers/'))

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from utils import *

if __name__ == "__main__":
	args = setup_args(cli=True)

	default_arguments = import_generate_settings()
	parser = argparse.ArgumentParser(allow_abbrev=False)
	parser.add_argument("--text", default=default_arguments['text'])
	parser.add_argument("--delimiter", default=default_arguments['delimiter'])
	parser.add_argument("--emotion", default=default_arguments['emotion'])
	parser.add_argument("--prompt", default=default_arguments['prompt'])
	parser.add_argument("--voice", default=default_arguments['voice'])
	parser.add_argument("--mic_audio", default=default_arguments['mic_audio'])
	parser.add_argument("--voice_latents_chunks", default=default_arguments['voice_latents_chunks'])
	parser.add_argument("--candidates", default=default_arguments['candidates'])
	parser.add_argument("--seed", default=default_arguments['seed'])
	parser.add_argument("--num_autoregressive_samples", default=default_arguments['num_autoregressive_samples'])
	parser.add_argument("--diffusion_iterations", default=default_arguments['diffusion_iterations'])
	parser.add_argument("--temperature", default=default_arguments['temperature'])
	parser.add_argument("--diffusion_sampler", default=default_arguments['diffusion_sampler'])
	parser.add_argument("--breathing_room", default=default_arguments['breathing_room'])
	parser.add_argument("--cvvp_weight", default=default_arguments['cvvp_weight'])
	parser.add_argument("--top_p", default=default_arguments['top_p'])
	parser.add_argument("--diffusion_temperature", default=default_arguments['diffusion_temperature'])
	parser.add_argument("--length_penalty", default=default_arguments['length_penalty'])
	parser.add_argument("--repetition_penalty", default=default_arguments['repetition_penalty'])
	parser.add_argument("--cond_free_k", default=default_arguments['cond_free_k'])

	args, unknown = parser.parse_known_args()
	kwargs = {
		'text': args.text,
		'delimiter': args.delimiter,
		'emotion': args.emotion,
		'prompt': args.prompt,
		'voice': args.voice,
		'mic_audio': args.mic_audio,
		'voice_latents_chunks': args.voice_latents_chunks,
		'candidates': args.candidates,
		'seed': args.seed,
		'num_autoregressive_samples': args.num_autoregressive_samples,
		'diffusion_iterations': args.diffusion_iterations,
		'temperature': args.temperature,
		'diffusion_sampler': args.diffusion_sampler,
		'breathing_room': args.breathing_room,
		'cvvp_weight': args.cvvp_weight,
		'top_p': args.top_p,
		'diffusion_temperature': args.diffusion_temperature,
		'length_penalty': args.length_penalty,
		'repetition_penalty': args.repetition_penalty,
		'cond_free_k': args.cond_free_k,
		'experimentals': default_arguments['experimentals'],
	}

	tts = load_tts()
	generate(**kwargs)