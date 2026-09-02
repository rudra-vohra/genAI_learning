import os
import wave
import base64
import asyncio
import speech_recognition as sr

from dotenv import load_dotenv
from openai import OpenAI
from google import genai
import winsound

load_dotenv()


client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

tts_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def save_audio(filename, audio_data):
    """Save Gemini PCM audio as WAV."""

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data)

async def text_to_speech(text):
    # generating speech
    interaction = await tts_client.aio.interactions.create(
        model="gemini-3.1-flash-tts-preview",

        input=f"""
        Say naturally and cheerfully.:

        {text}
        """,

        response_format={
            "type": "audio"
        },

        generation_config={
            "speech_config": [
                {
                    "voice": "Orus"
                }
            ]
        }
    )

    # Gemini returns Base64 encoded audio
    audio_data = base64.b64decode(
        interaction.output_audio.data
    )

    save_audio(
        "out.wav",
        audio_data
    )

    print("Speech generated → out.wav")
    winsound.PlaySound(
        "out.wav",
        winsound.SND_FILENAME
    )

async def main():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 4

        system_prompt = """
            You're an expert voice agent.You're given a transcript of what user has said using voice/
            You need to output as if you're a voice agent and whatever you speak will be converted back to audio using AI and played back to the user. 
        """
        messages = [
             {"role":"system","content":system_prompt},
        ]

        while True:
            print("Speak something")
            audio = r.listen(source)

            print("Processing audio..")
            stt = r.recognize_google(audio)

            messages.append({"role":"user","content":stt})
            print("you said..",stt)

            response = client.chat.completions.create(
                model="gemini-3.5-flash-lite",
                messages=messages
            )

            print("AI response:- ",response.choices[0].message.content)
            await text_to_speech(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
