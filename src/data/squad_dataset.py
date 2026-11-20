from datasets import load_dataset
from torch.utils.data import Dataset


def encode_squad_example(tokenizer, context, question, max_len, answer=None):
    """Trim context first, keeping the question, answer cue and supervision."""
    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prefix = encode("Context:\n")
    context_ids = encode(context)
    suffix = encode(f"\n\nQuestion:\n{question}\n\nAnswer: ")
    answer_ids = encode(answer) if answer is not None else []
    space = max_len - len(prefix) - len(suffix)
    if space < 0 or (space == 0 and answer_ids):
        raise ValueError(f"max_len={max_len} cannot fit the question, answer cue and required answer tokens")
    if len(context_ids) + len(answer_ids) > space:
        answer_ids = answer_ids[:min(space, max(1, max_len // 4))]
        context_ids = context_ids[:space - len(answer_ids)]
    return {"prompt_ids": prefix + context_ids + suffix, "answer_ids": answer_ids}


class SquadGPT2Dataset(Dataset):
    """
    Returns token lists for prompt and answer separately to enable label-masking.
    """
    def __init__(self, split="train", max_samples=None, max_len=512, tokenizer=None, seed=0):
        self.ds = load_dataset("rajpurkar/squad", split=split)
        if max_samples is not None:
            self.ds = self.ds.shuffle(seed=seed).select(range(min(max_samples, len(self.ds))))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        r = self.ds[idx]
        answer = r["answers"]["text"][0] if r["answers"]["text"] else "unknown"
        return encode_squad_example(self.tokenizer, r["context"], r["question"], self.max_len, answer)
