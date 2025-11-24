from datasets import load_dataset
from torch.utils.data import Dataset

TEMPLATE = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

class SquadGPT2Dataset(Dataset):
    """
    Returns token lists for prompt and answer separately to enable label-masking.
    """
    def __init__(self, split="train", max_samples=None, max_len=512, tokenizer=None, seed=0):
        self.ds = load_dataset("rajpurkar/squad")[split]
        if max_samples is not None:
            self.ds = self.ds.shuffle(seed=seed).select(range(max_samples))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        r = self.ds[idx]
        prompt = TEMPLATE.format(context=r["context"], question=r["question"]) + " "
        answer = r["answers"]["text"][0] if r["answers"]["text"] else "unknown"
        # Encode separately (no special tokens for GPT-2)
        enc_p = self.tokenizer(prompt, add_special_tokens=False)
        enc_a = self.tokenizer(answer, add_special_tokens=False)
        p_ids = enc_p["input_ids"]
        a_ids = enc_a["input_ids"]
        # Truncate to fit max_len while keeping part of the answer
        if len(p_ids) + len(a_ids) > self.max_len:
            keep_ans = max(1, min(len(a_ids), self.max_len // 4))  # keep at least some answer tokens
            rem = self.max_len - keep_ans
            p_ids = p_ids[:max(1, rem)]
            a_ids = a_ids[:keep_ans]
        return {"prompt_ids": p_ids, "answer_ids": a_ids}