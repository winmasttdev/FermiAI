import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m='Qwen/Qwen2.5-0.5B-Instruct'
tok=AutoTokenizer.from_pretrained(m)
model=AutoModelForCausalLM.from_pretrained(m, dtype=torch.float16, device_map='auto')
print("LOADED", model.device)
msgs=[{"role":"user","content":"Reply as a friendly helpful friend. The user says: I am sad today. Respond as Friend:"}]
inp=tok.apply_chat_template(msgs, return_tensors='pt', return_attention_mask=True)
print("inp type", type(inp), list(inp.keys()) if hasattr(inp,'keys') else None)
input_ids=inp['input_ids'].to(model.device)
out=model.generate(input_ids, max_new_tokens=40, do_sample=True, temperature=0.9)
gen=out[0][input_ids.shape[1]:]
print("REPLY:", tok.decode(gen, skip_special_tokens=True))
