"""Synthesizes speech from the input string of text or ssml.
Make sure to be working in a virtual environment.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""

from google.cloud import texttospeech
import os

class GoogleTTS():
    def __init__(self, **kwargs):
        self.client_options = {"api_key": kwargs.get('api_key', os.getenv('GOOLE_TTS_API_KEY'))}

    def text_to_speech(self, text: str, output_path: str, target_lang="en-US") -> None:
        # Instantiates a client
        client = texttospeech.TextToSpeechClient(client_options=self.client_options)

        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code=target_lang,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

        # Select the type of audio file
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input, 
            voice=voice, 
            audio_config=audio_config
        )

        # Write the response to the output file
        with open(output_path, "wb") as out:
            out.write(response.audio_content)