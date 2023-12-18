import asyncio
import json
from threading import Thread

from websockets.server import serve

from utils import generate, get_autoregressive_models, get_voice_list, args, update_autoregressive_model, update_diffusion_model, update_tokenizer

# this is a not so nice workaround to set values to None if their string value is "None"
def replaceNoneStringWithNone(message):
    ignore_fields = ['text']  # list of fields which CAN have "None" as literal String value

    for member in message:
        if message[member] == 'None' and member not in ignore_fields:
            message[member] = None

    return message


async def _handle_generate(websocket, message):
    # update args parameters which control the model settings
    if message.get('autoregressive_model'):
        update_autoregressive_model(message['autoregressive_model'])

    if message.get('diffusion_model'):
        update_diffusion_model(message['diffusion_model'])

    if message.get('tokenizer_json'):
        update_tokenizer(message['tokenizer_json'])

    if message.get('sample_batch_size'):
        global args
        args.sample_batch_size = message['sample_batch_size']

    message['result'] = generate(**message)
    await websocket.send(json.dumps(replaceNoneStringWithNone(message)))


async def _handle_get_autoregressive_models(websocket, message):
    message['result'] = get_autoregressive_models()
    await websocket.send(json.dumps(replaceNoneStringWithNone(message)))


async def _handle_get_voice_list(websocket, message):
    message['result'] = get_voice_list()
    await websocket.send(json.dumps(replaceNoneStringWithNone(message)))


async def _handle_message(websocket, message):
    message = replaceNoneStringWithNone(message)

    if message.get('action') and message['action'] == 'generate':
        await _handle_generate(websocket, message)
    elif message.get('action') and message['action'] == 'get_voices':
        await _handle_get_voice_list(websocket, message)
    elif message.get('action') and message['action'] == 'get_autoregressive_models':
        await _handle_get_autoregressive_models(websocket, message)
    else:
        print("websocket: undhandled message: " + message)


async def _handle_connection(websocket, path):
    print("websocket: client connected")

    async for message in websocket:
        try:
            await _handle_message(websocket, json.loads(message))
        except ValueError:
            print("websocket: malformed json received")


async def _run(host: str, port: int):
    print(f"websocket: server started on ws://{host}:{port}")

    async with serve(_handle_connection, host, port, ping_interval=None):
        await asyncio.Future()  # run forever


def _run_server(listen_address: str, port: int):
    asyncio.run(_run(host=listen_address, port=port))


def start_websocket_server(listen_address: str, port: int):
    Thread(target=_run_server, args=[listen_address, port], daemon=True).start()
