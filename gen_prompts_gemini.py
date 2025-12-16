import os
from google import genai
from google.genai import types
import time

def generate_video_summary(client, video_bytes):
    response = client.models.generate_content(
        model='models/gemini-2.5-pro',
        contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(
                        data=video_bytes,
                        mime_type='video/mp4'),
                    video_metadata=types.VideoMetadata(fps=6)
                ),
                types.Part(text='Please summarize this video. No line breaks.' \
                                'You should focus on the movements of the main character or characters. You should avoid describing the lighting and the detailed background.' \
                                'You should not describe the camera movements.' \
                                'Depict the body movements in a concise manner rather than inferring what the character or characters might be doing or being too detailed. '\
                                'Example 1: "A young woman dances on an escalator. She wears a gray long-sleeved top and blue skinny jeans, paired with thick-soled sneakers. Her long hair cascades down her shoulders as she sways to the rhythm, her body moving freely in sync with the music. The entire video exudes energy and youthful vibes, capturing the carefree and joyful mood of the dancer."' \
                                'Example 2: "A woman is dancing in a bright room with large windows. She is wearing a black short-sleeved top with a pink pattern, a pink pleated skirt, black boots, and a pink hat with ear flaps. She is also wearing black-framed glasses. She is performing a dance routine, moving her arms and Legs in various ways, including spreading her arms, crossing her arms, raising her Hands, and placing her Hands on her Head. She is also lifting her Legs and performing other dance moves. The room is decorated with several Potted Plants."'
                            )
            ]
        )
    )
    return response.text

def generate_final_prompt(client, action_description, image_bytes):
    sys_prefix = """
    You are part of a team of bots that creates videos. 
    You work with an assistant bot that will draw anything you say.
    You will be prompted by people looking to create detailed, amazing videos. 
    The way to accomplish this is to combine the object and the scene in the image to make a new description.
    Generally, remove the main object and the scene in the provided caption, replace them with the object and the scene in the image, keep the action in the provided caption.
    For example, if the provided caption shows a guy wearing a orange outfit is waving his hands in a room, and the image shows a girl in a red dress in a bar, you should rewrite the caption to be the description "a girl in a red dress is waving her hands in a bar".
    You should use the image to describe the scene and the main object, the provided caption to describe the action, then make a detailed and descriptive description of the video.
    There are a few rules to follow :
    Avoid describing the lights and the camera. Avoid mentioning the color. 
    Especially, if the character is in anime style, you should mention the anime style. Begin with sentences like An Anime Character..., A humanoid figure..., A Disney Princess..., etc.
    You will only ever output a single video description per user request. 
    """
    
    prompt = f"Provided Caption: {action_description}"

    response = client.models.generate_content(
        model='models/gemini-2.5-pro',
        contents=types.Content(
            parts=[
                types.Part(text=sys_prefix),
                types.Part(
                    inline_data=types.Blob(
                        data=image_bytes,
                        mime_type='image/png')
                ),
                types.Part(text=prompt)
            ]
        )
    )
    return response.text

if __name__ == '__main__':
    
    client = genai.Client()
    video_folder_path = "your_video_folder_path"
            
    video_file_path = os.path.join(video_folder_path, "driving.mp4")
    image_file_path = os.path.join(video_folder_path, "ref.png")
    final_prompt_file_path = os.path.join(video_folder_path, "prompt.txt")
    
    if not os.path.exists(video_file_path):
        print(f"Video not found: {video_file_path}")
        
    if not os.path.exists(image_file_path):
        print(f"Image not found: {image_file_path}")

    # Step 1: Get action description from video
    video_bytes = open(video_file_path, 'rb').read()
    action_summary = None
    while not action_summary:
        try:
            action_summary = generate_video_summary(client, video_bytes)
            print(f"  Action Summary: {action_summary}")
        except Exception as e:
            print(f"  Step 1 API call failed with error: {e}. Retrying...")
            time.sleep(10)
    
    # Step 2: Generate final prompt using action description and reference image
    image_bytes = open(image_file_path, 'rb').read()
    final_prompt = None
    while not final_prompt:
        try:
            final_prompt = generate_final_prompt(client, action_summary, image_bytes)
            print(f"  Final Prompt: {final_prompt}")
        except Exception as e:
            print(f"  Step 2 API call failed with error: {e}. Retrying...")
            time.sleep(10)

    with open(final_prompt_file_path, "w") as f:
        f.write(final_prompt)
