import requests
from PIL import Image
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import json
from groq import Groq
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import textwrap
from gtts import gTTS
from scipy.io import wavfile
from moviepy import *
import base64


############# API KEYS
news_api='63be046bac5d43019c8285e2804e8dbf'   ###news api
gr_api='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'   ### groq api
hf_key='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'   ### HF api
### embed creation
client = InferenceClient(
    "sentence-transformers/all-MiniLM-L6-v2",
    token=hf_key,
    )

def news_extract(query):
    ########## NEWS API call
    language = 'en'
    api_key = news_api
    url = f'https://newsapi.org/v2/everything?q={query}&language={language}&apiKey={api_key}&sortBy=relevancy'

    response = requests.get(url)
    data = response.json()
    import urllib.request

    desc = []
    image = [];sites=[]
    for i, each in enumerate(data['articles']):
        desc.append(each['description'])
        sites.append(each['url'])
        if each['urlToImage'] is not None:
            image.append(each['urlToImage'])
        else:
            image.append('')
    return desc,sites

def embed(desc, image):
    query_embeddings = []
    for x in desc:
        query_embed = client.feature_extraction(x).tolist()
        query_embeddings.append(query_embed)

    q_embed = client.feature_extraction(query).tolist()
    similarity_scores = cosine_similarity([q_embed], query_embeddings)
    scores = []
    for i in range(len(similarity_scores[0])):
        scores.append(similarity_scores[0][i])

    df = pd.DataFrame(columns=['content', 'img_url', 'simil'])
    df['content'] = desc
    df['img_url'] = image
    df['simil'] = scores
    # df[df['simil']>0.70]
    df1 = df.sort_values('simil', ascending=False)
    df2 = df1.head(5)
    text = ', '.join(df2['content'])
    img_url = df2['img_url'].tolist()[0]

    return text

def summariser(text,emotion, platform):
    funct = f'''Tasks:
    1. Summarize the given content {text} to one generic sentences with 20 words according to the tone: {emotion} for platform: {platform}.
    2. Create concise single liner meme text with 10 words using the given content {text} according to the tone: {emotion}.
    3. Generate a concise single liner description text for generating an image without any dialogue for the given content {text} with tone {emotion}. 
    
    Please provide the outputs in this format:
    {{
      "summary": "<summary>",
      "meme": "<meme>",
      "image": "<image>"
    }}'''
    client = Groq(
        # This is the default and can be omitted
        api_key=gr_api,
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant who help to summarize the text."
            },
            {
                "role": "user",
                "content": f"Create text mentioned in the function {funct} using the text given. and return the response in json format.  ",
            }
        ],
        model="llama3-8b-8192",
        response_format={"type": "json_object"},
        max_tokens=4096
    )
    sum_cont = chat_completion.choices[0].message.content
    response=json.loads(sum_cont)
    return response

def resentence(text):
        client = Groq(
            # This is the default and can be omitted
            api_key=gr_api,
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "you are a helpful assistant who help to restructure the text."
                },
                {
                    "role": "user",
                    "content": f" Rewrite the given text {text} into a concise single liner for generating an image. ",
                }
            ],
            model="llama3-8b-8192"
        )
        sum_cont = chat_completion.choices[0].message.content
        return sum_cont

def image_generation(sum_cont):
    client = InferenceClient("stabilityai/stable-diffusion-3-medium-diffusers", token=hf_key)

    # output is a PIL.Image object
    image = client.text_to_image(sum_cont)
    return image

def create_speech_from_text(text, output_audio="output_audio.wav"):
    """
    Converts given text to speech and saves it as a WAV audio file.
    """
    tts = gTTS(text, lang="en")
    tts.save("output_audio.mp3")

    # Convert MP3 to WAV using ffmpeg (you need to install ffmpeg)
    os.system(f"ffmpeg -i output_audio.mp3 {output_audio}")
    print(f"Audio saved as {output_audio}")
    return output_audio

def create_video_with_audio_from_images(image_files, audio_file, output_video="output_with_audio.mp4", fps=24):
    """
    Combines image frames with an audio file to create a video with sound.

    Args:
        image_files (list): List of image file paths.
        audio_file (str): Path to the audio file (e.g., MP3).
        output_video (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    # Step 1: Ensure the images are sorted in correct order
    image_files = sorted(image_files)

    # Step 2: Load the audio to calculate duration
    audio = AudioFileClip(audio_file)
    audio_duration = audio.duration  # Duration in seconds

    # Step 3: Calculate total frames required to match audio duration
    total_frames = int(audio_duration * fps)

    # Step 4: Repeat image frames to match the total frame count
    repeated_frames = []
    num_images = len(image_files)
    frames_per_image = total_frames // num_images  # How many frames each image should last

    for image_file in image_files:
        repeated_frames.extend([image_file] * frames_per_image)

    # If there's a remainder, add extra frames to match the total frame count
    remainder = total_frames - len(repeated_frames)
    if remainder > 0:
        repeated_frames.extend(image_files[:remainder])

    # Step 5: Create a video from frames
    clip = ImageSequenceClip(repeated_frames, fps=fps)

    # Step 6: Set audio to the video
    clip = clip.with_audio(audio)

    # Step 7: Write the final video file
    clip.write_videofile(output_video, codec="libx264", audio_codec="aac")
    print(f"Video saved as {output_video}")

def frame_desc_generation(text):
    funct = f'''
    Please provide the outputs in this format:
    {{
      "frame1": "<frame1>",
      "frame2": "<frame2>",
      "frame3": "<frame3>"
    }}'''
    client = Groq(
        # This is the default and can be omitted
        api_key=gr_api,
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant who help to create the text."
            },
            {
                "role": "user",
                "content": f"Create single line text prompt with description for creating 3 sequential image frames for a video from the text given {text} and return each frame response in json format as mentioned in {funct}.  ",
            }
        ],
        model="llama3-8b-8192",
        response_format={"type": "json_object"},
        max_tokens=4096
    )
    sum_cont = chat_completion.choices[0].message.content
    response=json.loads(sum_cont)
    return response

def frame_generation(out):
    client = InferenceClient("stabilityai/stable-diffusion-3-medium-diffusers", token=hf_key)

    # output is a PIL.Image object
    for k,each in out.items():
      image = client.text_to_image(each)
      image.save('./images/'+k+'.png')

def video_generation(text_content):
    # Step 1: Convert text to speech
    audio_file = create_speech_from_text(text_content)
    frame_description=frame_desc_generation(text)
    frame_generation(frame_description)
    # Step 2: Create video with images and the generated speech
    image_folder = "./images"  # Replace with your folder containing images
    audio_file = "output_audio.mp3"  # Replace with your audio file path
    output_video = "output_with_audio.mp4"

    # Get all image file paths from the folder
    image_files = sorted([
        os.path.join(image_folder, file)
        for file in os.listdir(image_folder)
        if file.endswith((".png", ".jpg", ".jpeg"))
    ])

    # Create video with audio
    create_video_with_audio_from_images(image_files, audio_file, output_video, fps=24)

def create_meme(image_path, text, output_path="meme_image.png"):
    """
    Creates a meme by adding top and bottom text to an image.

    Args:
        image_path (str): Path to the input image.
        top_text (str): Text to display at the top of the meme.
        bottom_text (str): Text to display at the bottom of the meme.
        output_path (str): Path to save the generated meme image.
    """
    image = Image.open(image_path)
    image = image.convert("RGB")
    width, height = image.size

    # Create white space at the top
    white_space_height = int(height * 0.2)  # White space is 20% of the image height
    new_height = height + white_space_height
    new_image = Image.new("RGB", (width, new_height), "white")  # Create a blank white image
    new_image.paste(image, (0, white_space_height))  # Paste original image below the white space

    # Prepare drawing context
    draw = ImageDraw.Draw(new_image)

    # Load a font (use default font if no custom font is available)
    font_size = int(white_space_height * 0.3)  # Font size relative to white space height
    try:
        font = ImageFont.truetype("OleoScriptSwashCaps-Regular.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Wrap text to fit the width of the image
    max_chars_per_line = width // (font_size // 2)  # Approximation for characters per line
    wrapped_text = textwrap.fill(text, width=max_chars_per_line)

    # Calculate text position
    try:
        # Use `textbbox` for accurate bounding box calculation
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    except AttributeError:
        # Fallback to textsize for older Pillow versions
        text_width, text_height = draw.textsize(wrapped_text, font=font)

    # Center the text in the white space
    text_x = (width - text_width) / 2
    text_y = (white_space_height - text_height) / 2

    # Draw the text
    draw.multiline_text((text_x, text_y), wrapped_text, font=font, fill="black", align="center")

    # Save the resulting image
    new_image.save(output_path)

def sidebar_bg(side_bg):

   main_bg_ext = 'jpg'

   st.markdown(
       f"""
            <style>
            .stApp {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
                background-size: cover
            }},
            </style>
            """,
       unsafe_allow_html=True
   )


def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    #text = link.split('=')[1]
    print(link)
    return f'<a target="_blank" href="{link}"</a>'

if __name__ == "__main__":
  side_bg = 'bckg.jpg'
  sidebar_bg(side_bg)
  st.title('NewsFlare')
  st.sidebar.image('logo.png',width=400)
  query=st.text_input('User query')
  option = st.selectbox(
    "Tone",
    ("Humorous", "Casual","Professional"),
  )
  option1 = st.selectbox(
      "Platform",
      ("LinkedIn", "Twitter", "Instagram"),
  )
  cont=st.radio(label='Format',options=['Text','Image','Video','Meme'],horizontal=True)
  col1,col2,col3,col4,col5=st.columns(5)
  with col1:
    button1 = st.button("Generate", type="primary")
  #with col2:
  #  button2 = st.button("Regenerate")
  if button1:
      desc, sites=news_extract(query)
      site_df=pd.DataFrame(sites,columns=['URL'])
      text=','.join(desc[:10])
      #text=embed(desc, img)
      responses=summariser(text,option, option1)

      #st.write(responses)
      if cont=='Text':
          st.text_area("Text Response:",f'{responses['summary']}')
      if cont=='Image':
          with st.spinner('In progress...'):
              resp_img = image_generation(responses['image'])
              resp_img.save('img_out.png')
          st.write("Image Response:")
          st.image("img_out.png")
      if cont=='Meme':
          with st.spinner('In progress...'):
              re_text=resentence(responses['image'])
              resp_img = image_generation(re_text)
              resp_img.save('meme_out.png')
              create_meme('meme_out.png',responses['meme'])
          st.write("Meme Response:")
          st.image("meme_image.png")
      if cont=='Video':
          with st.spinner('In progress...'):
            video_generation(responses['summary'])
          st.write("Video Response:")
          st.video("output_with_audio.mp4")
      df = pd.DataFrame()
      df['References'] = site_df['URL'].head(5)
      st.write(df.to_html(escape=False, index=False,render_links=True), unsafe_allow_html=True)
