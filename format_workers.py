import json
import numpy as np
import pandas as pd
from openai import OpenAI


client = OpenAI()
MODEL = "gpt-5-nano"


FORMAT_PROMPT = """
You are a classifier. You MUST return a JSON object with exactly these fields:

{
  "any_code_blocks": 0 or 1,
  "any_bullets": 0 or 1,
  "any_headings": 0 or 1
}

Definitions:
- any_code_blocks = 1 if any assistant message includes ```...```, code fences, or code-like blocks.
- any_bullets = 1 if any assistant message includes bullet lists (-, *, •) or numbered lists (1., 2., ...).
- any_headings = 1 if any assistant message includes markdown headings (#, ##, ### at start of line).

Return ONLY the JSON, with no explanation.
"""


def to_msg_list(conv):
    if isinstance(conv, (list, np.ndarray)):
        return list(conv)
    return []


def assistant_history_text(conv):
    msgs = to_msg_list(conv)
    texts = [
        m.get("content", "")
        for m in msgs
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    return "\n\n".join(texts) if texts else ""


def truncate_text(text: str, max_chars: int = 4000):
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= max_chars else text[-max_chars:]


def classify_format_llm(hist_text: str):
    try:
        response = client.responses.create(
            model=MODEL,
            input=(
                FORMAT_PROMPT
                + "\n\nASSISTANT HISTORY:\n\n"
                + hist_text
            ),
            max_output_tokens=100,
            reasoning={"effort": "minimal"},
        )

        out_text = response.output_text or ""

        start = out_text.find("{")
        end = out_text.rfind("}") + 1

        if start == -1 or end <= start:
            raise ValueError("No JSON detected")

        json_str = out_text[start:end]
        data = json.loads(json_str)

        return {
            "any_code_blocks": int(data.get("any_code_blocks", 0)),
            "any_bullets": int(data.get("any_bullets", 0)),
            "any_headings": int(data.get("any_headings", 0)),
        }

    except Exception as e:
        print(f"[classify_format_llm ERROR]: {e}")
        return {
            "any_code_blocks": 0,
            "any_bullets": 0,
            "any_headings": 0,
        }


def process_one(args):
    try:
        idx, side, conversation = args

        hist = assistant_history_text(conversation) or ""
        hist = truncate_text(hist)

        flags = classify_format_llm(hist)

        return idx, side, flags

    except Exception as e:
        print(f"[Worker ERROR idx={args[0]}]: {e}")
        return args[0], args[1], {
            "any_code_blocks": 0,
            "any_bullets": 0,
            "any_headings": 0,
        }


def run_parallel_format(df, side, workers=4):
    from multiprocessing import Pool
    from tqdm.auto import tqdm

    tasks = [
        (idx, side, df.loc[idx, f"conversation_{side}"])
        for idx in df.index
    ]

    with Pool(processes=workers) as pool:
        results = list(tqdm(pool.imap(process_one, tasks), total=len(tasks)))

    rows = []
    for idx, side, flags in results:

        if flags is None: 
            flags = {"any_code_blocks": 0, "any_bullets": 0, "any_headings": 0}

        rows.append({
            "row_index": idx,
            f"any_code_blocks_{side}": flags["any_code_blocks"],
            f"any_bullets_{side}": flags["any_bullets"],
            f"any_headings_{side}": flags["any_headings"],
        })

    return pd.DataFrame(rows).set_index("row_index")

