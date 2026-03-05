from utils import *
from tqdm import tqdm
from itertools import islice

def compute_precondition_and_content_scores(gt_steps, pred_steps, matches, mode):
    """
    gt_steps: dict of ground truth steps, e.g., {"step1": {"content": ..., "precondition": [...]}, ...}
    pred_steps: dict of predicted steps, same format
    matches: list of (gt_index, pred_index) tuples
    """

    # step1 -> index 0
    # print(gt_steps)
    gt_id2idx = {f"step{i+1}": i for i in range(len(gt_steps))}
    # if len(pred_steps) > 0:
    pred_id2idx = {f"step{i+1}": i for i in range(len(pred_steps))}


    # 匹配映射
    gt2pred = {gt: pred for gt, pred in matches}
    pred2gt = {f"step{pred+1}": f"step{gt+1}" for gt, pred in matches}
    # print('pred2gt:', pred2gt)

    ### --- Step 1: Content-level 精确度 ---
    content_TP = len(matches)
    content_precision = content_TP / len(pred_steps) if pred_steps else 0
    content_recall = content_TP / len(gt_steps) if gt_steps else 0
    content_FN = len(gt_steps) - content_TP
    content_FP = len(pred_steps) - content_TP
    assert(len(gt_steps) >= content_TP)
    assert(len(pred_steps) >= content_TP)
    content_f1 = 2 * content_precision * content_recall / (content_precision + content_recall) if (content_precision + content_recall) else 0

    ### --- Step 2: Precondition-level 精确度 ---
    precond_TP = precond_FP = precond_FN = 0
    # print('matches:', matches)

    all_gt_precond_steps = set()
    all_pred2gt_precond_steps = set()

    for gt_step in gt_steps.keys():
        gt_precond_steps = gt_steps[gt_step]['precondition']
        if gt_precond_steps != []:
            for gt_precond_step in gt_precond_steps:
                all_gt_precond_steps.add(f'{gt_step}-{gt_precond_step}')
    
    for pred_step in pred_steps.keys():
        try:
            pred_precond_steps = pred_steps[pred_step]['precondition']
        except:
            pred_precond_steps = []
        if pred_precond_steps != []:
            try:
                pred2gt_step = pred2gt[pred_step]
            except:
                pred2gt_step = 'x'
            for pred_precond_step in pred_precond_steps:
                try:
                    pred2gt_precond_step = pred2gt[pred_precond_step]
                except:
                    pred2gt_precond_step = 'x'
                all_pred2gt_precond_steps.add(f'{pred2gt_step}-{pred2gt_precond_step}')
    
    # print('all_gt_precond_steps:', all_gt_precond_steps)
    # print('all_pred2gt_precond_steps:', all_pred2gt_precond_steps)
    for all_gt_precond_step in all_gt_precond_steps:
        if all_gt_precond_step in all_pred2gt_precond_steps:
            precond_TP += 1
        else:
            precond_FN += 1
    
    for all_pred2gt_precond_step in all_pred2gt_precond_steps:
        if all_pred2gt_precond_step not in all_gt_precond_steps:
            precond_FP += 1

    precond_precision = precond_TP / (precond_TP + precond_FP) if (precond_TP + precond_FP) else 0
    precond_recall = precond_TP / (precond_TP + precond_FN) if (precond_TP + precond_FN) else 0
    precond_f1 = 2 * precond_precision * precond_recall / (precond_precision + precond_recall) if (precond_precision + precond_recall) else 0
    # print('precond_precision:', precond_precision)
    # print('precond_recall:', precond_recall)

    ### --- 输出 ---
    if mode == 'split':
        return {
            "content": {
                "precision": content_precision,
                "recall": content_recall,
                "f1": content_f1
            },
            "precondition": {
                "precision": precond_precision,
                "recall": precond_recall,
                "f1": precond_f1
            }
        }
    elif mode == 'total':
        return content_TP, content_FP, content_FN, precond_TP, precond_FP, precond_FN

def compute_average_metrics(data):
    # with open(json_path, 'r') as f:
    #     data = json.load(f)

    content_p = []
    content_r = []
    content_f1 = []

    precond_p = []
    precond_r = []
    precond_f1 = []

    for sample_id, metrics in data.items():
        c = metrics.get("content", {})
        p = metrics.get("precondition", {})

        if "precision" in c:
            content_p.append(c["precision"])
        if "recall" in c:
            content_r.append(c["recall"])
        if "f1" in c:
            content_f1.append(c["f1"])

        if "precision" in p:
            precond_p.append(p["precision"])
        if "recall" in p:
            precond_r.append(p["recall"])
        if "f1" in p:
            precond_f1.append(p["f1"])

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    result = {
        "content_avg": {
            "recall": safe_avg(content_r)*100,
            "precision": safe_avg(content_p)*100,
            "f1": safe_avg(content_f1)*100,
        },
        "precondition_avg": {
            "recall": safe_avg(precond_r)*100,
            "precision": safe_avg(precond_p)*100,
            "f1": safe_avg(precond_f1)*100,
        }
    }

    return result

def cal_metric(model, batch, mode='split'):
    ground_truth_path = f'annotation/{batch}/annotation_all.json'
    prediction_path = f'result/{batch}/{model}/0_end.json'
    matrix_path = f'result/{batch}/{model}/0_end_matrix.json'
    save_path = f'result/{batch}/{model}/0_end_metric.json'

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt_samples = json.load(f)

    with open(prediction_path, 'r', encoding='utf-8') as f:
        pre_samples = json.load(f)

    with open(matrix_path, 'r', encoding='utf-8') as f:
        mats = json.load(f)
    
    print(len(gt_samples), len(mats))
    # assert(len(gt_samples) == len(mats))

    gt_id_ans = {}
    for sample in gt_samples:
        answer = sample.get("answer", {})
        id = sample.get("id", "")
        gt_id_ans[id] = answer

    pre_id_ans = {}
    for sample in pre_samples:
        answer = sample.get("answer", {})
        id = sample.get("id", "")
        pre_id_ans[id] = answer

    id_matches = {}
    for id, mat in mats.items():
        # print(id)
        if mat != None:
            matches = max_match_binary_matrix(mat)
        else:
            matches = None
        id_matches[id] = matches

    if mode == 'split':
        id_metric = {}
        total_content_recall = 0
        total_content_precision = 0
        total_content_f1 = 0
        total_precond_recall = 0
        total_precond_precision = 0
        total_precond_f1 = 0
        # gt_samples = gt_samples[:3]
        for sample in tqdm(gt_samples):
            # print(id)
            id = sample.get("id", "")
            gt_ans = gt_id_ans[id]
            pre_ans = pre_id_ans[id]
            mat = mats[id]
            matches = id_matches[id]

            if mat != None:
                metric = compute_precondition_and_content_scores(gt_ans, pre_ans, matches, mode)
                id_metric[id] = metric
        print(len(id_metric))
        res = compute_average_metrics(data=id_metric)
        print(res)
    elif mode == 'total':
        total_content_TP = 0
        total_content_FP = 0
        total_content_FN = 0
        total_precond_TP = 0
        total_precond_FP = 0
        total_precond_FN = 0
        
        for sample in tqdm(gt_samples):
            id = sample.get("id", "")
            # print(id)
            gt_ans = gt_id_ans[id]
            pre_ans = pre_id_ans[id]
            mat = mats[id]
            matches = id_matches[id]

            if mat != None and np.array(mat).sum() != 0:
                content_TP, content_FP, content_FN, precond_TP, precond_FP, precond_FN = compute_precondition_and_content_scores(gt_ans, pre_ans, matches, mode)

                total_content_TP += content_TP
                total_content_FP += content_FP
                total_content_FN += content_FN
                total_precond_TP += precond_TP
                total_precond_FP += precond_FP
                total_precond_FN += precond_FN
        
        content_recall = total_content_TP / (total_content_TP + total_content_FN + 1e-8)
        content_precision = total_content_TP / (total_content_TP + total_content_FP + 1e-8)
        content_f1 = 2*content_recall*content_precision/(content_recall + content_precision + 1e-8)
        precond_recall = total_precond_TP / (total_precond_TP + total_precond_FN + 1e-8)
        precond_precision = total_precond_TP / (total_precond_TP + total_precond_FP + 1e-8)
        precond_f1 = 2*precond_recall*precond_precision/(precond_recall + precond_precision + 1e-8)
        
        return content_recall, content_precision, content_f1, precond_recall, precond_precision, precond_f1


if __name__=='__main__':
    batch = 'batch_all'
    model_list = ['gpt-4o']

    for model in model_list:
        print(model)
        mode = 'total'  # split/total，分开算指标再求平均/所有样本一起算指标
        if mode == 'split':
            cal_metric(model, batch, mode)
            metric_path = rf'result\{batch}\{model}\0_end_metric.json'
            res = compute_average_metrics(metric_path)
            print(res)
        elif mode == 'total':
            content_recall, content_precision, content_f1, precond_recall, precond_precision, precond_f1 = cal_metric(model, batch, mode)

            print({
                "content": {
                    "recall": content_recall*100,
                    "precision": content_precision*100,
                    "f1": content_f1*100,
                },
                "precondition": {
                    "recall": precond_recall*100,
                    "precision": precond_precision*100,
                    "f1": precond_f1*100,
                }
            })
