# Moved all of the imports into __name__ == "__main__" due to how multiprocessing spawns instances, makes multiprocessing faster as it reduces import overhead

# Need to check hz of dataset prep

if __name__ == "__main__":
	import os
	import sys

	if os.path.exists("runtime"):
		# Get the directory where the script is located
		script_dir = os.path.dirname(os.path.abspath(__file__))

		# Add this directory to sys.path
		if script_dir not in sys.path:
			sys.path.insert(0, script_dir)

	if 'TORTOISE_MODELS_DIR' not in os.environ:
		os.environ['TORTOISE_MODELS_DIR'] = os.path.realpath(os.path.join(os.getcwd(), './models/tortoise/'))

	if 'TRANSFORMERS_CACHE' not in os.environ:
		os.environ['TRANSFORMERS_CACHE'] = os.path.realpath(os.path.join(os.getcwd(), './models/transformers/'))

	os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

	from utils import *
	from webui import *

	from api.websocket_server import start_websocket_server
	args = setup_args()

	if args.listen_path is not None and args.listen_path != "/":
		import uvicorn
		uvicorn.run("main:app", host=args.listen_host, port=args.listen_port if not None else 8000)
	else:
		webui = setup_gradio()
		webui.launch(share=args.share, prevent_thread_lock=True, show_error=True, server_name=args.listen_host, server_port=args.listen_port)
		if not args.defer_tts_load:
			tts = load_tts()

		if args.websocket_enabled:
			start_websocket_server(args.websocket_listen_address, args.websocket_listen_port)

		webui.block_thread()
elif __name__ == "main":
	from fastapi import FastAPI
	import gradio as gr

	import sys
	sys.argv = [sys.argv[0]]

	app = FastAPI()
	args = setup_args()
	webui = setup_gradio()
	app = gr.mount_gradio_app(app, webui, path=args.listen_path)

	if not args.defer_tts_load:
		tts = load_tts()

