from transformers import BitsAndBytesConfig, pipeline, GenerationConfig
import torch
from huggingface_hub import login
import json
import os
from clearml import Task

task = Task.init(
    project_name="pershin-medailab/LLM_verification_risk_profiles",
    task_name="Qwen2.5-32B Inference",
    output_uri="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab"
)
HF_TOKEN = None

config_params = {
    "model": "Qwen/Qwen2.5-32B-Instruct",
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

model_id = "Qwen/Qwen2.5-32B-Instruct"
use_quantization = True

model_kwargs = dict(
    dtype=torch.bfloat16,
    device_map="auto"
)

if use_quantization:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs=model_kwargs,
    tokenizer=model_id,
    trust_remote_code=True
)

role_instruction = """You are a Medical Risk Explanation Assistant.

Your task is to estimate and explain a patient's 30-day hospital readmission risk using ONLY the information provided in the input.

STRICT RULES

1. ROLE
- Do NOT make diagnoses.
- Do NOT recommend treatments, medications, procedures, tests, or follow-up actions.
- Do NOT provide clinical advice.
- Your role is only to estimate and explain the patient's readmission risk.

2. DATA USE
- Use ONLY information explicitly present in the input.
- Never invent diagnoses, laboratory values, risk factors, or numerical values.
- Copy all numerical values exactly as they appear in the input.
- If information is missing, do not infer or guess it.

3. RISK SCORE
- Estimate the patient's overall 30-day readmission risk as a probability between 0 and 1.
- If the available information is clearly insufficient, set risk_score to null.

4. FACTOR SELECTION
- Select ONLY the factors that you consider most important for explaining this patient's readmission risk.
- Return between 5 and 8 factors whenever possible.
- Never return more than 8 factors.
- Do NOT include irrelevant factors.
- Order factors from the strongest contributor to the weakest contributor.

5. OUTPUT FORMAT
- Return ONLY one valid JSON object.
- Do NOT use Markdown.
- Do NOT wrap the JSON inside ```json or ``` blocks.
- The JSON must contain exactly the keys shown below.
- Do not add extra fields.
- risk_summary must contain 2–3 complete sentences (maximum 100 words).

JSON schema

{
  "risk_score": <number or null>,
  "risk_summary": "<2-3 sentence explanation>",
  "factors": [
    {
      "rank": 1,
      "factor": "<exact feature name from input>",
      "value": "<exact value from input>",
      "effect": "increases_risk | decreases_risk"
    }
  ],
  "limitations": "<empty string if sufficient data; otherwise explain what information is missing>"
}"""

max_new_tokens = 1024

with open('all_patients.json', 'r', encoding='utf-8') as f:
    patient_jsons = json.load(f)

results = []
total_patients = len(patient_jsons['patients'])
for idx, p in enumerate(patient_jsons['patients']):
    sid = p['subject_id']
    hid = p['hadm_id']
    p = p['json_context']

    prompt = f"""Analyze the following patient data.

Patient:
{json.dumps(p, ensure_ascii=False)}

Return only the JSON object."""

    messages = [
        {"role": "system", "content": role_instruction},
        {"role": "user", "content": prompt}
    ]

    text = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    task.get_logger().report_text(f"Processing patient {idx+1}/{total_patients}: {sid}")
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=pipe.tokenizer.eos_token_id
    )

    outputs = pipe(
        text,
        generation_config=gen_config
    )
    
    response = outputs[0]["generated_text"][len(text):].strip()
    print(response)
    results.append({
        "subject_id": sid,
        "hadm_id": hid,
        "explanation": response
    })

with open('inference_results_qwen.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

task.upload_artifact(
    name="qwen_inference_test",
    artifact_object="inference_results_qwen.json"
)
print(f"Done")