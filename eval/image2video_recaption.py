from openai import OpenAI

prefix ='''
**Objective**: **Give a highly descriptive video caption based on input image. **. As an expert, delve deep into the image with a discerning eye, leveraging rich creativity, meticulous thought. When describing the details of an image, include appropriate dynamic information to ensure that the video caption contains reasonable actions and plots. 

**Note**: The input image is the first frame of the video, and the output video caption should describe the motion starting from the current image.

**Answering Style**:
Answers should be comprehensive, conversational, and use complete sentences. Provide context where necessary and maintain a certain tone.  Begin directly without introductory phrases like "The image/video showcases" "The photo captures" and more. For example, say "A woman is on a beach", instead of "A woman is depicted in the image".

**Output Format**: "[highly descriptive image caption here]"

'''
import base64
from mimetypes import guess_type
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def get_answer(path):
    client = OpenAI(api_key="sk-YB69LUO4RBhdjTXaF60fBb3e9eA541138cD30a03C2C2A6C6", base_url="https://one-api.glm.ai/v1")
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prefix},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": local_image_to_data_url(path),
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
    )
    answer = response.choices[0].message.content
    return answer

if __name__ == "__main__":
    files = "/workspace/ckpt/yzy/sora/eval/image2video.txt"
    output_files = "/workspace/ckpt/yzy/sora/eval/image2video_recaption.txt"
    with open(files, "r") as f:
        lines = f.readlines()
    output = []
    for line in lines:
        path = line.strip().split("@@")[-1]
        if len(path) < 5:
            break
        answer = get_answer(path)
        output.append([answer, path])
    with open(output_files, "w") as f:
        for line in output:
            f.write(line[0] + "@@" + line[1] + "\n")