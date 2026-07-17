import json
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(f'outputs/boss_response_{timestamp}.json', 'r') as f:
    data = json.load(f)

tasks = data['task_pipeline']
for t in tasks:
    print(t['task_score'],t['task_id'])