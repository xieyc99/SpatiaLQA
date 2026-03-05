import json
from api.api import call_api
from tqdm import tqdm
import re
import os
import openai
from utils import *
import time
import multiprocessing
import time
import os
import ast

os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

def main():
    batch = 'batch_all'
    model_name = 'gpt-4o'

    samples_path = f'annotation/{batch}/annotation_all.json'
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    exp_path = 'prompt/example.json'
    with open(exp_path, 'r', encoding='utf-8') as f:
        example = json.load(f)
    example = example[0]
    
    prompt_template_path = 'prompt/zero_shot.txt'
   
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    save_path = os.path.join('result', batch, model_name.split('/')[-1])
    os.makedirs(save_path, exist_ok=True)
    
    save_frq = 2
    
    start_id = 0
    end_id = len(samples)

    if end_id == len(samples):
        samples = samples[start_id:]
        output_path = os.path.join(save_path, f'{start_id}_end.json')
        id_output_path = os.path.join(save_path, f'id_output_{start_id}_end.json')
    else:
        samples = samples[start_id:end_id]
        output_path = os.path.join(save_path, f'{start_id}_{end_id}.json')
        id_output_path = os.path.join(save_path, f'id_output_{start_id}_{end_id}.json')
    print('output_path:', output_path)
        
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            pre_answers = json.load(f)
        restart_id = len(pre_answers)
        samples = samples[restart_id:]
        with open(id_output_path, 'r', encoding='utf-8') as f:
            d_id_output = json.load(f)
    else:
        pre_answers = []
        d_id_output = {}

    iter = 0
    for sample in tqdm(samples, desc=model_name):
        question = sample.get("question", "")
        image = sample.get("image", "")
        id = sample.get("id", "")

        prompt = template.format(question=question, example=example)

        try:
            output_text = call_api(model_name, prompt, image)
        except openai.BadRequestError as e:
            print(e)  # 打印详细报错信息
            output_text = None
        
        d_id_output[id] = output_text

        if output_text != None:
            if model_name == 'Qwen/Qwen2.5-VL-7B-Instruct' or model_name == 'qwen-vl-plus':
                try:
                    pre_answer = parse_qwen7b_answer_from_str(output_text)
                except:
                    pre_answer = None
            elif model_name == 'Qwen/Qwen2.5-VL-32B-Instruct' or 'gemma' in model_name:
                try:
                    pre_answer = parse_qwen32b_answer_from_str(output_text)
                except:
                    pre_answer = None
            else:
                match = re.search(r'<ans>(.*?)</ans>', output_text, re.DOTALL)
                if match:
                    pre_answer = match.group(1).strip()  # 去除前后空格和换行
                    try:
                        pre_answer = ast.literal_eval(pre_answer)
                    except:
                        pre_answer = None
                else:
                    pre_answer = None
        else:
            pre_answer = None
        
        if pre_answer != None and isinstance(pre_answer, dict) and 'answer' in pre_answer.keys():
            pre_answer = pre_answer['answer']
        
        if isinstance(pre_answer, dict):
            for k,v in pre_answer.items():
                if isinstance(v, dict) and 'precondition' in v.keys():
                    var = v['precondition']
                    if isinstance(var, set):
                        v['precondition'] = list(var)
                    elif isinstance(var, list):
                        for sub_var in var:
                            if isinstance(sub_var, set):
                                sub_var = list(sub_var)

        sample['answer'] = pre_answer   
        pre_answers.append(sample)
        iter += 1
    
        if (iter % save_frq) == 0 or iter == len(samples):
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(pre_answers, f_out, ensure_ascii=False, indent=4)

            with open(id_output_path, 'w', encoding='utf-8') as f_out:
                json.dump(d_id_output, f_out, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
        

