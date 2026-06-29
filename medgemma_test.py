from transformers import BitsAndBytesConfig, pipeline
import torch
from huggingface_hub import login
import json
import os
from clearml import Task

task = Task.init(
    project_name="pershin-medailab/LLM_verification_risk_profiles",
    task_name="MedGemma Inference",
    output_uri="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab"
)
HF_TOKEN = None

config_params = {
    "model": "medgemma-27b-it",
    "quantization": True,
    "max_new_tokens": 512,
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

model_variant = "medgemma-27b-it"
model_id = f"google/{model_variant}"
use_quantization = True

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


role_instruction = """
You are a medical assistant who analyzes patient data and provides explanations for the risk of readmission.

Your task:
1. Assess the risk of 30-day readmission based on the data provided.
2. Explain which factors influence the risk.
3. Indicate the importance of each factor (high/moderate/low).
4. Provide numerical values ​​for the factors.

Rules:
- DO NOT make diagnoses or prescribe treatment.
- If the data is insufficient, state this honestly.
- Be objective and base your answer only on the data provided.
- Use clear medical language.
- Structure your answer clearly by points.

Answer format:
1. RISK ASSESSMENT: [number from 0 to 1] - [brief explanation]
2. FACTORS INCREASING RISK (high/moderate/low):
- [factor]: [value] - [importance] - [why it influences]
3. FACTORS REDUCE RISK (high/moderate/low):
- [factor]: [value] - [importance] - [why it influences]
4. CONCLUSION: [brief summary]
"""

max_new_tokens = 512

with open('all_patients.json', 'r', encoding='utf-8') as f:
    patient_jsons = json.load(f)

results = []
total_patients = len(patient_jsons['patients'])
for idx, p in enumerate(patient_jsons['patients']):
    sid = p['subject_id']
    hid = p['hadm_id']
    p.pop('subject_id', None)
    p.pop('hadm_id', None)

    prompt = f"Describe this risk-profile: {p}"
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": role_instruction}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    task.get_logger().report_text(f"Processing patient {idx+1}/{total_patients}: {sid}")
    
    output = pipe(text=messages, max_new_tokens=512, do_sample=False)
    response = output[0]["generated_text"][-1]["content"]

    results.append({
        "subject_id": sid,
        "hadm_id": hid,
        "explanation": response
    })

with open('inference_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

task.upload_artifact(
    name="medgemma_inference_results_test",
    artifact_object="inference_results.json"
)
print(f"Done")