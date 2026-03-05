import re
import json
from datetime import datetime
import os
import multiprocessing
import time
from scipy.optimize import linear_sum_assignment
import numpy as np
import ast
import torch
import random
from typing import Dict, Any, List

def fix_json_quotes(text):
    # 替换字段名两端的单引号为双引号
    fields = ['question', 'answer', 'step\\d+', 'content', 'precondition']
    for field in fields:
        # 匹配类似 'question': 或 'step1':
        pattern = rf"'({field})'\s*:"
        text = re.sub(pattern, r'"\1":', text)
    return text

def string_to_dict(json_str):
    try:
        # 去除首尾空格并解析为字典
        return json.loads(json_str.strip())
    except json.JSONDecodeError as e:
        print("JSON 解析错误:", e)
        return None
    
def clean_keys(d):
    """
    递归处理：删除字典中所有键的空格
    """
    if isinstance(d, dict):
        return {k.replace(" ", ""): clean_keys(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_keys(item) for item in d]
    else:
        return d
    
def remove_all_postaction(sample):
    """
    删除字典中的"postaction"(针对qwen7b)
    """
    if "answer" in sample:
        keys_to_delete = []
        for key, value in sample["answer"].items():
            # 情况1：postaction是step内部的字段
            if isinstance(value, dict) and "postaction" in value:
                del value["postaction"]
            # 情况2：postaction本身是一个step级别的键（例如"postaction": {}）
            if key == "postaction":
                keys_to_delete.append(key)

        # 延后删除，避免迭代时修改字典
        for key in keys_to_delete:
            del sample["answer"][key]
    return sample

    
def parse_qwen7b_answer_from_str(input_str):
    # re.sub(r"(\])(\s*\n)(\s*```)", r"\1\2}\n\3", input_str)
    match = re.search(r"\]\s*\n?\s*```", input_str, re.DOTALL)
    if match:
        input_str = re.sub(r"\]\s*\n?\s*```", r"]\n }\n ```", input_str)
        # print('111')

    match = re.search(r"\}\,\s*\n?\s*\}\s*\n?\s*\]\s*\n?\s*\}\s*\n?```", input_str, re.DOTALL)
    if match:
        input_str = re.sub(r"\}\,\s*\n?\s*\}\s*\n?\s*\]\s*\n?\s*\}\s*\n?```", r"}\n }\n ]\n }\n ```", input_str)
        # print('111')
    # print(input_str)

    # 去除 markdown 符号和 <ans> 包裹部分
    match = re.search(r'```json\s*(\{.*?\})\s*```', input_str, re.DOTALL)
    if not match:
        print("未找到JSON代码块")
        return None
    input_str = match.group(1)
    
    if input_str.strip().lower().startswith("json"):
        input_str = input_str.strip()[4:].lstrip()

    if '<ans>' in input_str:
        input_str = input_str.split('<ans>')[0]
    # input_str = input_str.replace("<ans>", "").replace("</ans>", "")
    input_str = input_str.strip()
    # print(input_str)

    # 尝试解析 JSON
    try:
        parsed = json.loads(input_str)
    except json.JSONDecodeError as e:
        print("解析失败:", e)
        return None

    # 提取 answer 列表
    # print(parsed)
    answer_list = parsed.get("answer", [])
    result = {}

    for block in answer_list:
        for step_name, step_content in block.items():
            # 去掉键里的空格  
            step_name = step_name.replace(" ", "")
            step_content = clean_keys(step_content)
            # print(step_content)

            # 处理 precondition 字段
            if isinstance(step_content, dict):
                precond = step_content.get("precondition", [])
                if not precond or (len(precond) == 1 and precond[0] == ""):
                    precond = []
                step_content["precondition"] = precond
                result[step_name] = step_content

    return result

def fix_qwen7b_answer_format(samples_path, id_output_path, output_path):
    # 示例
    # id_output_path = r'result\batch1\Qwen2.5-VL-7B-Instruct\id_output_0_900.json'
    with open(id_output_path, 'r', encoding='utf-8') as f:
        id_output_samples = json.load(f)
    # s = samples['742ce5c9-5bf5-44b9-8e63-86f83d0e2b94']

    # samples_path = r'result\batch1\Qwen2.5-VL-7B-Instruct\0_900.json'
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    for sample in samples:
        # answer = sample.get("answer", {})
        id = sample.get("id", "")
        # print(id)
        if sample['answer'] == None:
            raw_str = id_output_samples[id]
            answer_json = parse_qwen7b_answer_from_str(raw_str)
            sample['answer'] = answer_json
        elif isinstance(sample['answer'], list):
            answer_list = sample['answer']
            result = {}

            for block in answer_list:
                for step_name, step_content in block.items():
                    # 去掉键里的空格  
                    step_name = step_name.replace(" ", "")
                    step_content = clean_keys(step_content)

                    # 处理 precondition 字段
                    if isinstance(step_content, dict):
                        precond = step_content.get("precondition", [])
                        if not precond or (len(precond) == 1 and precond[0] == ""):
                            precond = []
                        step_content["precondition"] = precond
                        result[step_name] = step_content
            result = clean_keys(result)
            sample['answer'] = result
            sample = remove_all_postaction(sample)
        elif isinstance(sample['answer'], dict):
            sample = remove_all_postaction(sample)

    # output_path = r'test.json'
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(samples, f_out, ensure_ascii=False, indent=4)

def parse_qwen32b_answer_from_str(input_str):
    # 去除 markdown 符号和 <ans> 包裹部分
    match = re.search(r'```json\s*(\{.*?\})\s*```', input_str, re.DOTALL)
    if not match:
        print("未找到JSON代码块")
        return None
    input_str = match.group(1)
    
    if input_str.strip().lower().startswith("json"):
        input_str = input_str.strip()[4:].lstrip()

    if '<ans>' in input_str:
        input_str = input_str.split('<ans>')[0]
    # input_str = input_str.replace("<ans>", "").replace("</ans>", "")
    input_str = input_str.strip()
    input_str = fix_json_quotes(input_str)
    # print(input_str)

    # 尝试解析 JSON
    try:
        parsed = ast.literal_eval(input_str)
    except json.JSONDecodeError as e:
        print("解析失败:", e)
        return None

    # 提取 answer 列表
    if 'steps' in parsed.keys():
        if isinstance(parsed['steps'], list):
            result = {}
            for i, step in enumerate(parsed['steps']):
                result[f'step{i+1}'] = step
    elif 'answer' in parsed.keys():
        result = parsed['answer']
    else:
        result = parsed

    return result
        
    
def run_with_timeout(target_func, timeout_seconds=30):
    p = multiprocessing.Process(target=target_func)  # 创建子进程，运行你的主程序逻辑
    p.start()                                             # 启动子进程
    p.join(timeout_seconds)                               # 等待子进程最多 timeout_seconds 秒

    if p.is_alive():                                      # 如果超时后子进程还活着（没退出）
        print("程序超时未响应，终止重启中...")
        p.terminate()                                     # 强制杀死子进程
        p.join()                                          # 等待子进程完全退出，避免僵尸进程

def max_match_binary_matrix(matrix):
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    # 转换为cost矩阵（因为linear_sum_assignment是最小化代价）
    cost_matrix = 1 - matrix  # 1变成0，0变成1
    
    # 匈牙利算法最小化 cost_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 统计真正匹配成功的数量（原矩阵中对应为1的项）
    matched = [(i, j) for i, j in zip(row_ind, col_ind) if matrix[i][j] == 1]
    
    return matched

def check_step_keys(answer_dict):
    expected_keys = {"content", "precondition"}
    invalid = False

    for step_name, step_content in answer_dict.items():
        actual_keys = set(step_content.keys())
        if actual_keys != expected_keys:
            invalid = True
            break

    return invalid

def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible 
    # https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize(s: str) -> str:
    # 大小写不敏感 + 折叠空白
    return " ".join(s.strip().lower().split())

def replace_preconditions(steps):
    # 1) 建立 content -> step 的反查表（规范化后）
    content2step = {}
    for step_key, node in steps.items():
        content = node.get("content", "")
        content2step[normalize(content)] = step_key

    # 2) 逐步替换 precondition
    out = {}
    for step_key, node in steps.items():
        preconds = node.get("precondition", [])
        # 允许 precondition 是字符串或列表
        if isinstance(preconds, str):
            precond_items = [preconds]
        elif isinstance(preconds, list):
            precond_items = preconds
        else:
            precond_items = []

        mapped: List[str] = []
        for item in precond_items:
            if not isinstance(item, str):
                mapped.append(f"<unmatched: {item!r}>")
                continue

            norm = normalize(item)

            # 情况A：precondition 已经是 step 名称
            if norm in (k.lower() for k in steps.keys()):
                # 找到真实大小写的键名
                for real_key in steps.keys():
                    if real_key.lower() == norm:
                        mapped.append(real_key)
                        break
                continue

            # 情况B：根据内容反查对应 step
            step_match = content2step.get(norm)
            if step_match:
                mapped.append(step_match)
            else:
                mapped.append(f"<unmatched: {item}>")  # 保留原始文本以便排查

        # 写回
        new_node = dict(node)
        new_node["precondition"] = mapped
        out[step_key] = new_node
    return out

def parse_blip2_steps(s: str):
    """
    将类似：
      "step1: 'content': 'remove the bottle from the table', 'precondition': 'remove the bottle from the table', 
       'step2: 'content': 'remove the power strip from the table', 'precondition': 'remove the power strip from the table', 
       'step3: 'content': 'pick up the power strip', 'precondition: 'remove the bottle from the table'"
    的字符串解析为：
      {
        'step1': {'content': '...', 'precondition': '...'},
        'step2': {'content': '...', 'precondition': '...'},
        'step3': {'content': '...', 'precondition': '...'},
      }
    """
    # 统一奇怪引号
    s = s.replace("“", "'").replace("”", "'").replace("’", "'").replace("‘", "'").strip()

    # 允许 step 前面带可选引号；非贪婪取到下一个 step 或结尾
    step_iter = re.finditer(
        r"(?:^|,)\s*['\"]?\s*step\s*(\d+)\s*['\"]?\s*:\s*(.*?)(?=(?:,\s*['\"]?\s*step\s*\d+\s*['\"]?\s*:)|$)",
        s, flags=re.IGNORECASE | re.DOTALL
    )

    def grab(block: str, key: str):
        # 尝试几种常见脏格式
        patterns = [
            rf"(?:['\"]?\s*{key}\s*['\"]?)\s*:\s*(['\"])(.*?)\1",    # 'key': '...'
            rf"{key}\s*:\s*(['\"])(.*?)\1",                         # key: '...'
            rf"['\"]\s*{key}\s*:\s*(['\"])(.*?)\1",                 # 'key: '...
        ]
        for pat in patterns:
            m = re.search(pat, block, flags=re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(2).strip()
        return None

    result = {}
    for m in step_iter:
        step_key = f"step{m.group(1)}"
        block = m.group(2).lstrip(" ',")

        content = grab(block, "content")
        precond = grab(block, "precondition")
        # print(precond)
        # 再兜一层：有些是 precondition: '...（键名引号缺失/错位）
        if precond is None:
            m_pre_loose = re.search(r"precondition\s*:\s*(['\"])(.*?)\1", block, flags=re.IGNORECASE|re.DOTALL)
            if m_pre_loose:
                precond = m_pre_loose.group(2).strip()

        if content is not None or precond is not None:
            node = {}
            if content is not None:
                node["content"] = content
            else:
                node["content"] = ""
            if precond is not None:
                node["precondition"] = [precond]
            else:
                node["precondition"] = []
            result[step_key] = node

    result = replace_preconditions(result)

    return result

def parse_cosmos_steps(steps):
    raw = steps

    # 如需，保证结尾有 ```
    if not re.search(r"```\s*\Z", raw):
        raw = raw.rstrip() + "\n```"

    # 0) 若整段是一个 Python/JSON 字符串字面量（外层引号+转义），先解包
    # 例："```json\n<ans> ... ```"
    raw_strip = raw.strip()
    if (raw_strip.startswith('"') and raw_strip.endswith('"')) or \
       (raw_strip.startswith("'") and raw_strip.endswith("'")):
        try:
            # ast.literal_eval 比 json.loads 更宽容于引号样式
            raw = ast.literal_eval(raw_strip)
        except Exception:
            # 如果失败，尝试按 JSON 字符串解
            try:
                raw = json.loads(raw_strip)
            except Exception:
                pass  # 保持原样，继续后续清洗

    # 1) 去掉 <ans> 和 </ans>
    raw = re.sub(r"<\s*/?\s*ans\s*>", "", raw, flags=re.IGNORECASE)
    # print(raw)

    result = parse_qwen7b_answer_from_str(raw)

    if result is None:
        # 提取代码块
        m = re.search(r"```(?:\w+)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
        payload = m.group(1) if m else raw
        # print(payload)
        data = json.loads(payload.strip())

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("JSON 顶层应为非空列表。")

        first = data[0]

        # 情况A：列表里是一个对象，里面直接是 stepN 键
        if isinstance(first, dict) and any(k.startswith("step") for k in first.keys()):
            # 如果列表只有一个元素，直接返回它；否则合并（后者很少见）
            result = {}
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                for k, v in obj.items():
                    if k.startswith("step") and isinstance(v, dict):
                        result[k] = v
            return result

        # 情况B：列表是步骤条目，每条都有 content / precondition
        if isinstance(first, dict) and ("content" in first or "precondition" in first):
            result = {}
            for i, item in enumerate(data, start=1):
                step_key = f"step{i}"
                result[step_key] = {
                    "content": item.get("content", ""),
                    "precondition": item.get("precondition", []),
                }
            return result

        raise ValueError("无法识别的数据结构。")

    
    return result

def to_step_dict_from_singleton_list(
    data,
    lower_content: bool = False,
    map_preconditions_by_content: bool = False
):
    """
    将形如：
        [
          {"step1": {"content": "...", "precondition": [...]}},
          {"step2": {"content": "...", "precondition": [...]}},
          {"step3": {"content": "...", "precondition": [...]}},
        ]
    合并为：
        {
          "step1": {"content": "...", "precondition": [...]},
          "step2": {"content": "...", "precondition": [...]},
          "step3": {"content": "...", "precondition": [...]},
        }

    参数
    ----
    lower_content: 是否把 content 转为小写（默认 False）
    map_preconditions_by_content: 若 precondition 中含“句子”而非 stepX，
        则依据 content 反查对应的 stepX 并替换（默认 False）
    """
    if not isinstance(data, list):
        raise TypeError("data 应为 list")

    # 先收集（用于可选的 content -> step 映射）
    step_to_payload: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"项必须是仅含一个 stepN 键的字典，收到：{item!r}")
        (step_key, payload), = item.items()
        if not isinstance(payload, dict):
            raise ValueError(f"{step_key} 的值必须是字典，收到：{type(payload)}")

        content = payload.get("content", "")
        precond = payload.get("precondition", [])

        if lower_content and isinstance(content, str):
            content = content.lower()

        # 规范 precondition 为列表[str]
        if isinstance(precond, str):
            precond = [precond]
        elif not isinstance(precond, list):
            precond = []

        step_to_payload[step_key] = {"content": content, "precondition": precond}

    if map_preconditions_by_content:
        # 建立 content -> step 反查表（小写+折叠空白）
        def _norm(s: str) -> str:
            return " ".join(s.strip().lower().split())

        content2step = {
            _norm(v["content"]): k for k, v in step_to_payload.items()
            if isinstance(v.get("content"), str)
        }

        for k, v in step_to_payload.items():
            new_pre = []
            for p in v.get("precondition", []):
                if not isinstance(p, str):
                    continue
                np = p.strip()
                # 已经是 stepX
                if np.lower().startswith("step"):
                    new_pre.append(next((kk for kk in step_to_payload if kk.lower() == np.lower()), np))
                    continue
                # 尝试按内容映射
                mapped = content2step.get(_norm(np))
                new_pre.append(mapped if mapped else np)
            v["precondition"] = new_pre

    return step_to_payload

def parse_llava1_5_steps(s: str) -> Dict[str, Dict[str, Any]]:
    """
    输入：形如 " <ans></ans>\\n{...}" 的字符串，其中 JSON 结构包含:
      {
        "question": "...",
        "answer": {
          "step1": {"content": "...", "precondition": [...], "dependency": [...]?},
          "step2": {...},
          ...
        }
      }
    输出：
      {
        "step1": {"content": "...", "precondition": [...]},
        "step2": {"content": "...", "precondition": [...]},
        ...
      }
    说明：若存在 dependency，则与 precondition 合并（去重保序）。
    """

    # 1) 去 <ans> 标签与多余空白
    s = re.sub(r"<\s*/?\s*ans\s*>", "", s, flags=re.IGNORECASE).strip()

    # 2) 若包在三引号中，取代码块内容（防御性处理）
    m = re.search(r"```(?:\w+)?\s*(.*?)\s*```", s, flags=re.DOTALL)
    payload = m.group(1) if m else s

    try:
        res = parse_internvl_steps(payload)
    except SyntaxError as e:
        if "was never closed" in str(e):
            # 在结尾补一个 '}' 再试一次（仅补一次，防止死循环）
            res = parse_internvl_steps(payload + "}")

    return res

def parse_internvl_steps(s: str) -> Dict[str, Any]:
    """
    从任意字符串中找到最左边的 '{' 和最右边的 '}'，
    取中间（含括号）片段并解析成 Python 字典。
    """
    left = s.find("{")
    right = s.rfind("}")
    if left == -1 or right == -1 or right <= left:
        raise ValueError("未找到成对的大括号。")

    segment = s[left:right+1].strip()

    # 1) 先按 JSON 解析
    try:
        obj = json.loads(segment)
    except json.JSONDecodeError:
        # 2) 回退：把 JSON 关键字换成 Python，再用 ast.literal_eval
        seg2 = re.sub(r"\btrue\b", "True", segment)
        seg2 = re.sub(r"\bfalse\b", "False", seg2)
        seg2 = re.sub(r"\bnull\b", "None", seg2)
        obj = ast.literal_eval(seg2)

    if not isinstance(obj, dict):
        raise ValueError("片段解析成功，但结果不是字典。")
    
    if 'answer' in obj:
        res = obj['answer']
    else:
        res = obj

    return res

def xmlish_ans_to_step_dict(s: str, ensure_closing_backticks: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    将形如:
      "```json
       <ans>
         <step1>
           <content>...</content>
           <precondition>[ "stepX", ... ]</precondition>
         </step1>
         <step2>...</step2>
       </ans>
       ```"
    的字符串解析为:
      { "step1": {"content": "...", "precondition": [...]}, ... }
    """

    raw = s

    # 若整段被引号包裹（且含转义），先解包
    rs = raw.strip()
    if (rs.startswith('"') and rs.endswith('"')) or (rs.startswith("'") and rs.endswith("'")):
        try:
            raw = ast.literal_eval(rs)
        except Exception:
            pass  # 解不掉也没关系，继续走后续流程

    # 如需，保证末尾有 ```
    if ensure_closing_backticks and ("```" in raw) and not re.search(r"```\s*\Z", raw):
        raw = raw.rstrip() + "\n```"

    # 去掉 <ans> 标签
    raw = re.sub(r"<\s*/?\s*ans\s*>", "", raw, flags=re.IGNORECASE)

    # 提取三引号中的内容（若存在）
    m = re.search(r"```(?:\w+)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    payload = m.group(1) if m else raw
    payload = payload.strip()

    # 解析函数：把 <precondition> 内文本转成列表
    def parse_pre_list(txt: str) -> List[str]:
        if txt is None:
            return []
        t = txt.strip()
        if t == "" or t == "[]":
            return []
        # 优先按 JSON 列表解析
        try:
            val = json.loads(t)
            if isinstance(val, list):
                return [str(x) for x in val]
            # 如果不是列表，就包成列表
            return [str(val)]
        except Exception:
            pass
        # 宽松回退：去掉首尾方括号后按逗号拆分
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1]
        parts = []
        for x in t.split(","):
            x = x.strip().strip('"').strip("'")
            if x:
                parts.append(x)
        return parts

    # 抓取每个 <stepN>...</stepN> 块
    step_blocks = list(re.finditer(r"<\s*step(\d+)\s*>(.*?)</\s*step\1\s*>",
                                   payload, flags=re.IGNORECASE | re.DOTALL))
    result: Dict[str, Dict[str, Any]] = {}

    for m in step_blocks:
        idx = m.group(1)
        block = m.group(2)

        # 提取 content
        cm = re.search(r"<\s*content\s*>(.*?)</\s*content\s*>",
                       block, flags=re.IGNORECASE | re.DOTALL)
        content = cm.group(1).strip() if cm else ""

        # 提取 precondition
        pm = re.search(r"<\s*precondition\s*>(.*?)</\s*precondition\s*>",
                       block, flags=re.IGNORECASE | re.DOTALL)
        pre_txt = pm.group(1) if pm else ""
        pre_list = parse_pre_list(pre_txt)

        result[f"step{idx}"] = {
            "content": content,
            "precondition": pre_list
        }

    if not result:
        raise ValueError("未在文本中找到任何 <stepN> ... </stepN> 块。")

    return result

def xmlish_attr_steps_to_dict(s: str) -> Dict[str, Dict[str, Any]]:
    """
    解析形如：
      " <ans>
          <step1 content=\"Remove the red box\" precondition=\"[]\"/>
          <step2 content=\"Remove the green box\" precondition=\"[]\"/>
          <step3 content=\"Remove the orange box\" precondition=\"[]\"/>
          <step4 content=\"Put the green box on the orange box\" precondition=[\"step2\", \"step3\"]/>
        </ans> "
    为：
      {
        "step1": {"content": "...", "precondition": [...]},
        ...
      }
    兼容：
      - 外层整串被引号包裹并带转义（先 literal_eval 解包）
      - precondition 属性值缺少引号但为 [ ... ] 形式
      - 大小写与空白的轻微变体
    """
    raw = s.strip()

    # 若整段被引号包住（且含 \n、\" 等转义），先解包
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            pass

    # 去掉 <ans> 容器标签
    raw = re.sub(r"</?\s*ans\s*>", "", raw, flags=re.IGNORECASE).strip()

    # 抓取每个 <stepN .../> 自闭合标签；也兼容非自闭合写法（>... </stepN>）的开头部分
    tag_iter = re.finditer(r"<\s*step(\d+)\b([^>]*)/?>", raw, flags=re.IGNORECASE | re.DOTALL)

    def get_attr(attrs: str, name: str) -> str | None:
        # name="..."; name='...'; name=[ ... ]（无引号也行）
        m = re.search(rf"{name}\s*=\s*\"([^\"]*)\"", attrs, flags=re.IGNORECASE)
        if not m:
            m = re.search(rf"{name}\s*=\s*'([^']*)'", attrs, flags=re.IGNORECASE)
        if not m:
            m = re.search(rf"{name}\s*=\s*(\[[^\]]*\])", attrs, flags=re.IGNORECASE)  # 无引号但方括号包裹
        return m.group(1).strip() if m else None

    def parse_pre_list(txt: str | None) -> List[str]:
        if not txt:
            return []
        t = txt.strip()
        if t == "" or t == "[]":
            return []
        # 优先按 JSON 解析
        try:
            val = json.loads(t)
            if isinstance(val, list):
                return [str(x) for x in val]
            return [str(val)]
        except Exception:
            pass
        # 宽松回退：去掉外层 [] 后按逗号拆分
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1]
        parts = [p.strip().strip('"').strip("'") for p in t.split(",")]
        return [p for p in parts if p]

    result: Dict[str, Dict[str, Any]] = {}
    for m in tag_iter:
        idx = m.group(1)
        attrs = m.group(2) or ""
        content = get_attr(attrs, "content") or ""
        pre_raw = get_attr(attrs, "precondition")
        pre_list = parse_pre_list(pre_raw)
        result[f"step{idx}"] = {"content": content, "precondition": pre_list}

    # 如果没匹配到任何 step，自检一下是否是 <stepN> ... </stepN> 的块状写法，做个兜底
    if not result:
        blocks = re.finditer(r"<\s*step(\d+)[^>]*>(.*?)</\s*step\1\s*>",
                             raw, flags=re.IGNORECASE | re.DOTALL)
        for b in blocks:
            idx, body = b.group(1), b.group(2)
            cm = re.search(r"<\s*content\s*>(.*?)</\s*content\s*>", body, flags=re.IGNORECASE | re.DOTALL)
            pm = re.search(r"<\s*precondition\s*>(.*?)</\s*precondition\s*>", body, flags=re.IGNORECASE | re.DOTALL)
            content = (cm.group(1).strip() if cm else "")
            pre_list = parse_pre_list(pm.group(1) if pm else "")
            result[f"step{idx}"] = {"content": content, "precondition": pre_list}

    if not result:
        raise ValueError("未找到任何 step 标签。")

    return result
