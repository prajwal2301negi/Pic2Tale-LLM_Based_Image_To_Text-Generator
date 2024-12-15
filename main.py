from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
import requests
import streamlit as st
import os
import tempfile

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not HUGGINGFACEHUB_API_TOKEN or not OPENAI_API_KEY:
    raise ValueError("API keys are missing. Please check your .env file.")

# Image to text function
def img2txt(image_path):
    try:
        image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        text = image_to_text(image_path)[0]['generated_text']
        return text
    except Exception as e:
        print(f"Error in img2txt: {e}")
        return "Error generating text from image"

# LLM for story generation
# def generate_story(scenario):
#     try:
#         template = """ 
#         You are a storyteller;
#         You can generate a short story based on a simple narrative, the story should be no more than 200 words.
        
#         CONTEXT: {scenario}
#         STORY:
#         """
#         prompt = PromptTemplate(template=template, input_variables=['scenario'])

#         story_llm = LLMChain(
#             llm=OpenAI(model_name='gpt-3.5-turbo', temperature=0.8),
#             prompt=prompt,
#             verbose=False
#         )
#         story = story_llm.run(scenario=scenario)
#         return story
#     except Exception as e:
#         print(f"Error in generate_story: {e}")
#         return "Error generating story"

# Text to speech function
# def text2speech(messages):
#     try:
#         API_URL = 'https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits'
#         headers = {'Authorization': f'Bearer {HUGGINGFACEHUB_API_TOKEN}'}
#         response = requests.post(API_URL, headers=headers, json={'inputs': messages})

#         if response.status_code == 200:
#             with open('audio.flac', 'wb') as file:
#                 file.write(response.content)
#             return 'audio.flac'
#         else:
#             print(f"Error in text2speech: {response.status_code} - {response.text}")
#             return None
#     except Exception as e:
#         print(f"Error in text2speech: {e}")
#         return None

# Main Streamlit app
def main():
    st.set_page_config(page_title='Image to Audio Story', page_icon='')
    st.header("Turn an Image into an Audio Story")

    uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_filename = temp_file.name

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Processing image to text..."):
            scenario = img2txt(temp_filename)

        # with st.spinner("Generating story..."):
        #     story = generate_story(scenario)

        # with st.spinner("Converting story to audio..."):
        #     audio_file = text2speech(story)

        with st.expander("Scenario from Image"):
            st.write(scenario)

        # with st.expander("Generated Story"):
        #     st.write(story)

        # if audio_file:
        #     st.audio(audio_file, format="audio/flac")
        # else:
        #     st.error("Audio generation failed. Please try again later.")

if __name__ == '__main__':
    main()
