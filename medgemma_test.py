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
    "max_new_tokens": 1024,
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
You are a Medical Analyst Assistant. Your task is to analyze patient data and formulate an explanation of the 30-day readmission risk profile.

STRICT RULES (violating any of these invalidates the answer):

1. ROLE:
- DO NOT make diagnoses.
- DO NOT prescribe or recommend treatments, medications, dosages, or procedures.
- DO NOT provide advice on "what the doctor/patient should do next" beyond the risk explanation.
- If the patient data is EMPTY or clearly insufficient for an assessment, DO NOT invent factors or values. Clearly indicate this in the "limitations" section and return risk_score = null; leave the factor lists empty.

2. DATA USE:
- ONLY mention factors that are present in the input data. It is prohibited to mention factors, lab values, diagnoses, or events. Missing from the input data.
- Enter the numerical values ​​of factors EXACTLY as they appear in the input data (without rounding or converting units).
- If the input data is incomplete (some factors are missing), do not try to guess or infer them. Work only with what is available, and note in the "limitations" section any missing data if this significantly impacts the assessment.
- If the input data contains factors that have no clear clinical association with the risk of readmission, DO NOT include them in the risk factor lists unless you can substantiate the association based on the data itself.

3. IMPORTANCE ASSESSMENT:
- "High" — the factor you assess as having the greatest impact on risk among those mentioned.
- "Moderate" — moderate impact.
- "Low" — small, but not zero impact.

4. FORMAT ANSWER:
- The answer is ONE valid JSON object. No text before or after the JSON. No markdown blocks (```).
- The "risk_summary" field is a CONNECTED text of 3-6 complete sentences: what determines the risk, how the factors interact with each other. Do not use lists or bullet points within this field—plain text only.
- Each "explanation" within a factor is no more than 1 sentence.
- Be sure to end the JSON with a final closing brace "}".

JSON SCHEMA (use exactly these keys, do not add or remove anything):

{
"data_sufficiency": "sufficient" | "partial" | "insufficient",
"risk_score": <float from 0 to 1, or null if data_sufficiency == "insufficient">,
"risk_summary": "<coherent text, 3-6 sentences>",
"risk_increasing_factors": [
{"factor": "<exact name from input data>", "value": "<value exactly as in input data>", "importance": "high|moderate|low", "explanation": "<1 sentence>"}
],
"risk_decreasing_factors": [
{"factor": "<exact name from input data>", "value": "<value exactly as in input data>", "importance": "high|moderate|low", "explanation": "<1 sentence>"}
],
"conclusion": "<1-2 sentences, brief conclusion>",
"limitations": "<indicate which data is missing or why the estimate is limited; empty string if sufficient data>"
}
"""

max_new_tokens = 1024

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

    task.get_logger().report_text(f"Processing patient (upd) {idx+1}/{total_patients}: {sid}")
    
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