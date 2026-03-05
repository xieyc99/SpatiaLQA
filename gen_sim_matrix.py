import json
# from api.hf_offline import load_vlm, single_answer
from api.api import call_api
from tqdm import tqdm
import re
import os
import openai
from utils import *
import time
import multiprocessing
import time
import ast

def main():
    model_name = 'gpt-4o'
    api_model_name = 'gpt-4o'

    samples_path = r'annotation\batch1\annotation_all.json'
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    save_path = os.path.join('result', 'batch_all', model_name.split('/')[-1])
    pre_samples_path = os.path.join(save_path, '0_end.json')
    with open(pre_samples_path, 'r', encoding='utf-8') as f:
        pre_samples = json.load(f)
    
    prompt_template_path = r'prompt\gen_sim_matrix.txt'
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    save_frq = 2
    
    start_id = 0
    end_id = len(samples)

    if end_id == len(samples):
        samples = samples[start_id:]
        output_path = os.path.join(save_path, f'{start_id}_end_matrix.json')
    else:
        samples = samples[start_id:end_id]
        output_path = os.path.join(save_path, f'{start_id}_{end_id}_matrix.json')
        
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            d_id_output = json.load(f)
        restart_id = len(d_id_output)
        pre_samples = pre_samples[restart_id:]
    else:
        d_id_output = {}

    iter = 0

    output_path = os.path.join(save_path, f'{start_id}_end_matrix.json')
    d_gt_steps = {}
    for sample in samples:
        answer = sample.get("answer", {})
        id = sample.get("id", "")
        steps = [answer[k]['content'] for k in answer.keys()]
        d_gt_steps[id] = steps

    # pre_samples = pre_samples[:5]
    for sample in tqdm(pre_samples, desc=model_name):
        image = sample.get("image", "")
        pre_answer = sample.get("answer", {})
        pre_id = sample.get("id", "")

        if pre_answer == {} or pre_answer == None:
            d_id_output[pre_id] = None
        else:
            pre_steps = [pre_answer[k]['content'] for k in pre_answer.keys()]
            gt_steps = d_gt_steps[pre_id]
            
            prompt = template.format(ground_truth_steps=gt_steps, predicted_steps=pre_steps)

            try:
                output_text = call_api(api_model_name, prompt, image)
            except openai.BadRequestError as e:
                print(e)  # 打印详细报错信息
                output_text = None
            
            if output_text != None:
                match = re.search(r'<ans>(.*?)</ans>', output_text, re.DOTALL)
                if match:
                    str_mat = match.group(1).strip()  # 去除前后空格和换行
                    try:
                        list_mat = ast.literal_eval(str_mat)
                    except:
                        list_mat = None
                    d_id_output[pre_id] = list_mat
                else:
                    d_id_output[pre_id] = None
            else:
                d_id_output[pre_id] = None

        iter += 1
    
        if (iter % save_frq) == 0 or iter == len(pre_samples):
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(d_id_output, f_out, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
        