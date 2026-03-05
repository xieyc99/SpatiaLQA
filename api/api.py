from openai import OpenAI

def call_api(model_name, prompt, image_url):
    client = OpenAI(
        api_key="xxx",
        base_url='xxx',
    )

    if isinstance(image_url, str):
        content_list = [
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            },
            {
                "type": "text", 
                "text": prompt  # zero-shot
            }
        ]
    elif isinstance(image_url, list):
        content_list = []
        for url in image_url:
            content_list.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url}
                }
            )
        content_list.append(
            {
                "type": "text", 
                "text": prompt
            }
        )
    
    response = client.chat.completions.create(
        model=model_name, # ModleScope Model-Id
        messages = [
            {
                "role": "user",
                "content": content_list,
            }
        ],
        stream=True,
        temperature=0,
    )

    s = ''
    for chunk in response:
        # print(chunk)
        if chunk.choices != []:
            content = chunk.choices[0].delta.content
            if content != None:
                s += content
            else:
                break

    return s
