import os
import json
import asyncio
import threading
from flask import Flask, render_template
from flask_sock import Sock
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

app = Flask(__name__)
sock = Sock(app)

@app.route("/")
def index():
    return render_template("index.html", sample_rate=24000)

@sock.route("/ws")
def websocket_handler(ws):

    async def realtime_session():

        client = AsyncOpenAI(
            api_key=AZURE_API_KEY,
            base_url=f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}"
        )

        async with client.beta.realtime.connect(
            model=AZURE_DEPLOYMENT
        ) as connection:

            await connection.session.update(
                session={
                    "modalities": ["audio", "text"],
                    "turn_detection": {"type": "none"},
                }
            )

            ws.send(json.dumps({"type": "status", "message": "Connected to Azure"}))

            while True:
                message = ws.receive()

                if message is None:
                    break

                # Binary audio from browser
                if isinstance(message, (bytes, bytearray)):
                    await connection.input_audio_buffer.append(
                        audio=message
                    )

                else:
                    data = json.loads(message)

                    if data["type"] == "start":
                        await connection.input_audio_buffer.clear()

                    if data["type"] == "stop":
                        await connection.input_audio_buffer.commit()
                        await connection.response.create()

                        async for event in connection:
                            if event.type == "response.output_text.delta":
                                ws.send(json.dumps({
                                    "type": "text_delta",
                                    "delta": event.delta
                                }))

                            if event.type == "response.output_audio.delta":
                                ws.send(event.delta)

                            if event.type == "response.completed":
                                ws.send(json.dumps({"type": "done"}))
                                break

    # Run async loop inside thread
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_until_complete, args=(realtime_session(),)).start()


if __name__ == "__main__":
    app.run(port=5000, debug=True)
