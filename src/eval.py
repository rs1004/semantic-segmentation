from data_generator import DataGenerator
from model import UNet
from config import Config
from datetime import date
from subprocess import check_output
import json

CONFIG = Config()
OUTPUT_FORMAT = '''
# EVALUATION REPORT

## REPORTING DATE
{date}

## RUNTIME
```
{runtime}
```

## CONFIG
{config_table}

## SCORES
{score_table}
'''


def get_label_id_name_map():
    with open(CONFIG.ROOT_DIR / 'labelmap.json', 'r') as f:
        label_map = json.load(f)
    return {v['id']: k for k, v in label_map.items()}


def report(result):
    d = date.today().isoformat()
    runtime = check_output(['nvidia-smi']).decode()
    config_table = ['|item|value|', '|-|-|']
    for k, v in CONFIG.__dict__.items():
        config_table.append(f'|{k}|{v}|')

    metrics = ['iou', 'precision', 'recall', 'f1_score']
    label_id_name_map = get_label_id_name_map()

    score_table = [f'||{"|".join(metrics)}|', f"|{'|'.join(['-' for i in range(len(metrics)+1)])}"]
    for i in range(CONFIG.CLASS_NUM):
        rows = [label_id_name_map[i]]
        for m in metrics:
            rows.append(_f2s(result[f'{m}_class_{i}']))
        score_table.append(f"|{'|'.join(rows)}|")

    average = ['average'] + [_f2s(result[m]) for m in ['mean_iou', 'mean_AP', 'mean_AR', 'f1_score']]
    score_table.append(f"|{'|'.join(average)}|")

    report = OUTPUT_FORMAT.format(
        date=d,
        runtime=runtime,
        config_table='\n'.join(config_table),
        score_table='\n'.join(score_table)
    )

    CONFIG.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG.EVAL_DIR / f'report_{d}.md', 'w') as f:
        f.write(report)


def _f2s(v, ndigits=4):
    return f'{round(v, ndigits)}'


if __name__ == '__main__':
    data_gen_val = DataGenerator('val')
    ds_val = data_gen_val.get_one_shot_iterator()

    model = UNet(input_shape=(data_gen_val.H, data_gen_val.W, 3), class_num=CONFIG.CLASS_NUM)

    result = model.evaluate(ds=ds_val, steps=data_gen_val.data_length // CONFIG.BATCH_SIZE)
    report(result=result)
