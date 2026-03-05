import json

with open(r'D:\Exp\github\SpatiaLQA\annotation\batch_all\annotation_all.json', 'r', encoding='utf-8') as f1:
    samples = json.load(f1)

s = set()
for sample in samples:
    s.add(sample['source'])

print(s)