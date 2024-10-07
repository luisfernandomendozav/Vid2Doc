#!/usr/bin/env python3

import sys
import os
import glob
from dotenv import load_dotenv
load_dotenv()

import ffmpeg
import openai
import re
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 videoToDocumentation.py videoFile.mp4")
        sys.exit(1)

    input_video = sys.argv[1]

    if not os.path.isfile(input_video):
        print(f"Error: File '{input_video}' does not exist.")
        sys.exit(1)

    # Set up API keys
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    openai.api_key = openai_api_key

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
        .filter('fps', fps=0.1)  # Adjust fps as needed
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

    # Create a list to store tuples of (image_path, description)
    screenshots_with_descriptions = []
    image_files = sorted(glob.glob(f'{output_folder}/*.png'))

    for image_file in image_files:
        description = describe_image(image_file)
        screenshots_with_descriptions.append((image_file, description))

    # Prepare visual descriptions for the prompt
    visual_descriptions = ""
    for idx, (_, description) in enumerate(screenshots_with_descriptions):
        visual_descriptions += f"[IMAGE_{idx+1}]: {description}\n"

    # Step 5: Set up LangChain
    print("Generating detailed explanation using LangChain and OpenAI GPT...")
    llm = LangchainOpenAI(model_name='gpt-3.5-turbo', temperature=0)
   
    # Adjusted PromptTemplate to ensure placeholders are included
    prompt = PromptTemplate(
        input_variables=['transcription', 'visuals'],
        template="""
You are provided with a video transcription and descriptions of its screenshots.

**Transcription:**
{transcription}

**Visual Descriptions:**
{visuals}

Please generate a detailed, step-by-step explanation of what is happening in the video.

**Important Instructions**:
- **Include the image placeholders exactly as they appear, such as [IMAGE_1], [IMAGE_2], etc.**
- **Each placeholder should be on its own line and correspond to the image number provided in the visual descriptions.**

**Example**:

"The video starts with an overview of the application.

[IMAGE_1]

Next, the user logs into the system.

[IMAGE_2]"

Make sure the explanation flows naturally, and that images are placed at relevant points.
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # Step 6: Run the chain
    inputs = {
        'transcription': transcribed_text,
        'visuals': visual_descriptions
    }
    detailed_explanation = chain.run(inputs)

    # Debugging: Check if placeholders are present
    print("Generated Detailed Explanation:")
    print(detailed_explanation)

    # Step 6.5: Generate Summary
    print("Generating summary using OpenAI GPT...")
    summary_prompt = PromptTemplate(
        input_variables=['detailed_explanation'],
        template="""
Please provide a concise summary of the following detailed explanation in 3-5 sentences:

{detailed_explanation}
"""
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary = summary_chain.run({'detailed_explanation': detailed_explanation})

    # Output the result
    output_text_file = f'{base_filename}_documentation.txt'
    with open(output_text_file, 'w') as f:
        f.write(detailed_explanation)

    print(f"Detailed explanation has been saved to '{output_text_file}'.")

    # Step 7: Create PDF with images embedded in the text
    create_pdf(base_filename, summary, detailed_explanation, screenshots_with_descriptions)

def create_pdf(base_filename, summary, detailed_explanation, screenshots_with_descriptions):
    print("Creating PDF with images embedded in the text...")
    pdf_filename = f'{base_filename}_documentation.pdf'

    # Create a SimpleDocTemplate
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
   
    # Initialize the Story list
    Story = []

    # Modify existing styles
    styles['Title'].fontSize = 24
    styles['Title'].spaceAfter = 12
   
    styles['Heading1'].fontSize = 18
    styles['Heading1'].spaceAfter = 12
   
    styles['BodyText'].fontSize = 12
    styles['BodyText'].leading = 14

    # Title Page
    Story.append(Paragraph(f"Documentation for {base_filename}", styles['Title']))
    Story.append(Spacer(1, 12))
   
    # Add Summary
    Story.append(Paragraph("Summary", styles['Heading1']))
    Story.append(Paragraph(summary.strip(), styles['BodyText']))
    Story.append(Spacer(1, 12))
    Story.append(PageBreak())

    # Add Detailed Explanation
    Story.append(Paragraph("Detailed Explanation", styles['Heading1']))
    Story.append(Spacer(1, 12))

    # Placeholder pattern allowing optional whitespace after image number
    placeholder_pattern = re.compile(r'\[IMAGE_(\d+)\s*\]')

    # Map placeholders to images
    image_map = {f'[IMAGE_{i+1}]': image_path for i, (image_path, _) in enumerate(screenshots_with_descriptions)}

    # Debugging: Print the image map
    print("Image Map:", image_map)

    # Split the text by placeholders
    parts = placeholder_pattern.split(detailed_explanation)

    # Debugging: Print the parts after splitting
    print("Parts after splitting:", parts)

    i = 0
    while i < len(parts):
        text = parts[i].strip()
        if text:
            paragraphs = text.split('\n')
            for para_text in paragraphs:
                if para_text.strip():
                    Story.append(Paragraph(para_text.strip(), styles['BodyText']))
                    Story.append(Spacer(1, 12))
        i += 1
        if i < len(parts):
            image_num = parts[i]
            placeholder = f'[IMAGE_{image_num}]'
            image_path = image_map.get(placeholder)
            if image_path:
                img = RLImage(image_path, width=400, height=225)
                Story.append(img)
                Story.append(Spacer(1, 12))
            else:
                print(f"Warning: No image found for placeholder {placeholder}")
            i += 1

    # Build the PDF
    doc.build(Story)
    print(f"PDF documentation has been saved to '{pdf_filename}'.")

if __name__ == '__main__':
    main()
