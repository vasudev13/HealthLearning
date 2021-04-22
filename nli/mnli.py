import pandas as pd
from config import CONFIG


def mnli_df(stage):
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    df = pd.read_json(f"{CONFIG['DATA_PATH']}/mli_{stage}_v1.jsonl", lines=True,)
    df = df[[CONFIG['sentence1'], CONFIG['sentence2'], CONFIG['labels']]]
    df[CONFIG['labels']] = df[CONFIG['labels']].map(label_map)
    return df
