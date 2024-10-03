#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

import sys
import os
import glob
import ffmpeg
import openai
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 videoTodocumentation.py videoFile.mp4")
        sys.exit(1)

    input_video = sys.argv[1]

    if not os.path.isfile(input_video):
        print(f"Error: File '{input_video}' does not exist.")
        sys.exit(1)

    # Set up API keys
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    # Paths
    base_filename = os.path.splitext(os.path.basename(input_video))[0]
    output_audio = f'{base_filename}_audio.wav'
    output_folder = f'{base_filename}_screenshots'

    # Step 1: Extract audio
    print("Extracting audio from the video...")
    (
        ffmpeg
        .input(input_video)
        .output(output_audio, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
        .run(overwrite_output=True)
    )

    # Step 2: Transcribe audio
    print("Transcribing audio to text using OpenAI Whisper...")
    with open(output_audio, 'rb') as audio_file:
        transcript = openai.Audio.transcribe('whisper-1', audio_file)
    transcribed_text = transcript['text']

    # Step 3: Capture screenshots
    print("Capturing screenshots from the video...")
    os.makedirs(output_folder, exist_ok=True)
    (
        ffmpeg
        .input(input_video)
        .filter('fps', fps=0.1)
        .output(f'{output_folder}/out%d.png')
        .run(overwrite_output=True)
    )

    # Step 4: Describe screenshots
    print("Generating descriptions for the screenshots...")
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

    def describe_image(image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors='pt')
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        return description

    screenshot_descriptions = []
    image_files = sorted(glob.glob(f'{output_folder}/*.png'))

    for image_file in image_files:
        description = describe_image(image_file)
        screenshot_descriptions.append(f"{os.path.basename(image_file)}: {description}")

    visual_descriptions = "\n".join(screenshot_descriptions)

    # Step 5: Set up LangChain
    print("Generating detailed explanation using LangChain and OpenAI GPT...")
    llm = LangchainOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    prompt = PromptTemplate(
        input_variables=['transcription', 'visuals'],
        template="""
You are provided with a video transcription and descriptions of its screenshots.

**Transcription:**
{transcription}

**Visual Descriptions:**
{visuals}

Please generate a detailed, step-by-step explanation of what is happening in the video. Make sure to reference the visuals when appropriate.
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # Step 6: Run the chain
    inputs = {
        'transcription': transcribed_text,
        'visuals': visual_descriptions
    }
    detailed_explanation = chain.run(inputs)

    # Output the result
    output_text_file = f'{base_filename}_documentation.txt'
    with open(output_text_file, 'w') as f:
        f.write(detailed_explanation)

    print(f"Detailed explanation has been saved to '{output_text_file}'.")

if __name__ == '__main__':
    main()
