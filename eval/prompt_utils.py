import re

from openai import OpenAI
import traceback

def clean_string(s):
    s = s.replace("\n", " ")
    # 去除开头和结尾的空格
    s = s.strip()
    # 替换内部超过两个以上的空格为一个空格
    s = re.sub(r'\s{2,}', ' ', s)
    return s


def upsample_prompt_glm(prompt: str) -> str:

    url = "http://172.19.128.150:9090/v1"
    client = OpenAI(api_key="sk-",
                    base_url=url)
    prefix = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.
For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow :
You will only ever output a single video description per user request.
When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.
Video descriptions must have the same num of words as examples below. Extra words will be ignored.
{" role ": " user " , " content ": " Create an imaginative image descriptive caption or modify an earlier caption for the user input : " 一个技工在推着装有汽车发动机的手推车 "} ,
{" role ": " assistant " , " content ": " A mechanic in a blue uniform is seen pushing a red trolley with an engine component towards a service area in a well-equipped workshop, under bright fluorescent lighting. Over time, the mechanic, wearing blue coveralls and white shoes, continues this task, with a white SUV lifted on a hydraulic lift in the background, indicating maintenance work. The shop is clean, organized, and filled with various tools and parts. Another individual is partially visible, suggesting a busy work environment. The mechanic's consistent efforts in moving the trolley under the SUV highlight ongoing mechanical tasks in a professional, busy setting."} ,
{" role ": " user " , " content ": " Create an imaginative image descriptive caption or modify an earlier caption for the user input : " 人把书放到盒子里"} ,
{" role ": " assistant " , " content ": "A cardboard box is open on a wooden floor, with a person partially visible, suggesting a moment of organization or relocation. The person is seen packing books into the box, carefully placing them, including a white-spined book among others with dark covers, indicating a personal move or decluttering. The individual gives a thumbs up, signaling approval of the packing process. The focus is on placing a dark blue hardcover book among other books, hinting at a meticulous packing process in a personal setting. The scene captures a sense of concentration and transition as the person organizes a diverse and personal library."}""".replace(
        "\n", "").strip()

    text = prompt.strip()
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{prefix}"
                },
                {
                    "role": "assistant",
                    "content": "OK! I will create input-based imaginative and descriptive caption in ENGLISH."
                },
                {
                    "role": "user",
                    "content": f"Create an imaginative image descriptive caption or modify an earlier caption in ENGLISH for the user input: \" {text} \""
                }
            ],
            model="",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=300
        )
        prompt = response.choices[0].message.content
        if prompt:
            prompt = clean_string(prompt)
    except Exception as e:
        traceback.print_exc()
    return prompt

from tqdm import tqdm
if __name__ == '__main__':
    with open("video_input.txt", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            prompt = line.strip()
            prompt = upsample_prompt_glm(prompt)
            print(prompt, flush=True)
    # prompt = "一个小男孩在奔跑"
    # prompt = upsample_prompt_glm(prompt)
    # print(prompt)