from transformers import BitsAndBytesConfig, pipeline
import torch
from huggingface_hub import login
import json
import os
from clearml import Task

task = Task.init(
    project_name="pershin-medailab/LLM_verification_risk_profiles",
    task_name="MedGemma Risk Scores",
    output_uri="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab"
)
HF_TOKEN = None
model_variant = "medgemma-27b-it"
model_id = f"google/{model_variant}"
use_quantization = True
max_new_tokens = 128

config_params = {
    "model": model_variant,
    "quantization": use_quantization,
    "max_new_tokens": max_new_tokens,
    "HF_TOKEN": ""
}
task.connect(config_params) 

if config_params.get("HF_TOKEN"):
    HF_TOKEN = config_params["HF_TOKEN"]

if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN")


if not HF_TOKEN:
    try:
        HF_TOKEN = task.get_parameter("General/HF_TOKEN") or task.get_parameter("Args/HF_TOKEN")
    except:
        pass

if not HF_TOKEN:
    raise ValueError("No HF_TOKEN")
print(f"HF_TOKEN found: {HF_TOKEN[:15]}...")
login(HF_TOKEN)

model_kwargs = dict(
    dtype=torch.bfloat16,
    device_map="auto"
)

if use_quantization:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

if "text" in model_variant:
    pipe = pipeline("text-generation", model=model_id, model_kwargs=model_kwargs)
else:
    pipe = pipeline("image-text-to-text", model=model_id, model_kwargs=model_kwargs)


role_instruction = """You are estimating a patient's probability of 30-day hospital readmission.

Rules:
- Use ONLY the provided patient information.
- Do NOT diagnose.
- Do NOT recommend treatment.
- Do NOT explain your reasoning.
- Use the examples below to understand how risk scores should be assigned.
- Estimate the probability of 30-day readmission as a number between 0 and 1.
- If there is insufficient information, return null.

Return ONLY one valid JSON object.

Schema:

{
    "risk_score": <number or null>
}
"""

with open('data/patients_sample.json', 'r', encoding='utf-8') as f:
    patient_jsons = json.load(f)
with open('data/fewshot.json', 'r', encoding='utf-8') as f:
    fewshots = json.load(f)


fewshot_prompt = ""
for ex in fewshots:
    rs = ex["json_context"]["risk_score"]
    ex["json_context"].pop("risk_score", None)
    ex["json_context"].pop("readmission", None)

    fewshot_prompt += f"""
Example

Patient:
{json.dumps(ex["json_context"], ensure_ascii=False)}

Output:
{{"risk_score": {rs}}}

"""
    
results = []
total_patients = len(patient_jsons)
for idx, p in enumerate(patient_jsons):
    sid = p['subject_id']
    hid = p['hadm_id']
    p = p['json_context']
    # p.pop('subject_id', None)
    # p.pop('hadm_id', None)

    prompt = f"""
    Below are examples.

    {fewshot_prompt}

    Now estimate the probability of 30-day hospital readmission for a new patient.

    Patient:

    {json.dumps(p, ensure_ascii=False)}

    Return ONLY:

    {{"risk_score": number}}
    """
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": role_instruction}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    task.get_logger().report_text(f"Processing patient (upd.) {idx+1}/{total_patients}: {sid}")
    
    output = pipe(text=messages, max_new_tokens=max_new_tokens, do_sample=False)
    response = output[0]["generated_text"][-1]["content"]

    results.append({
        "subject_id": sid,
        "hadm_id": hid,
        "explanation": response
    })

with open('llm_risk_scores.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

task.upload_artifact(
    name="medgemma_risk_scores_test",
    artifact_object="llm_risk_scores.json"
)
print(f"Done")