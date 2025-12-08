import numpy as np
from nltk.tokenize import word_tokenize

def to_msg_list(conv):
    if conv is None:
        return []
    if isinstance(conv, np.ndarray):
        return list(conv)
    if isinstance(conv, list):
        return conv
    return []

def extract_prompt(conv):
    msgs = to_msg_list(conv)
    prompt_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    prompt_texts = [m.get("content", "") or "" for m in prompt_msgs]
    
    return prompt_texts[0] if prompt_texts else ""

def convo_turn_length_features(conv):
    msgs = to_msg_list(conv)
    num_turns = len(msgs)

    user_lens = []
    assistant_lens = []

    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = m.get("content", "") or ""
        
        try:
            tokens = word_tokenize(text)
        except LookupError:
            import nltk
            nltk.download('punkt')
            tokens = word_tokenize(text)

        if role == "user":
            user_lens.append(len(tokens))
        elif role == "assistant":
            assistant_lens.append(len(tokens))

    num_user_turns = len(user_lens)
    num_assistant_turns = len(assistant_lens)

    total_user_tokens = sum(user_lens)
    total_assistant_tokens = sum(assistant_lens)

    avg_user_len = total_user_tokens / num_user_turns if num_user_turns > 0 else 0.0
    avg_assistant_len = total_assistant_tokens / num_assistant_turns if num_assistant_turns > 0 else 0.0

    return {
        "num_turns": num_turns,
        "num_user_turns": num_user_turns,
        "num_assistant_turns": num_assistant_turns,
        "total_user_tokens": total_user_tokens,
        "total_assistant_tokens": total_assistant_tokens,
        "avg_user_turn_len_tokens": avg_user_len,
        "avg_assistant_turn_len_tokens": avg_assistant_len,
    }