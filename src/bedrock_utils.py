import json
import logging
import time
import traceback

import boto3.session as boto3_session
import botocore.config


def extract_xml_tag(generation: str, tag):
    begin = generation.rfind(f"<{tag}>")
    if begin == -1:
        return
    begin = begin + len(f"<{tag}>")
    end = generation.rfind(f"</{tag}>", begin)
    if end == -1:
        return
    value = generation[begin:end].strip()
    return value


def predict_one_eg_mistral(x):
    current_session = boto3_session.Session()
    bedrock = current_session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        config=botocore.config.Config(
            read_timeout=120,  # corresponds to inference time limit set for Bedrock
            connect_timeout=120,
            retries={
                "max_attempts": 5,
            },
        ),
    )
    api_template = {
        "modelId": "mistral.mistral-7b-instruct-v0:2",
        "contentType": "application/json",
        "accept": "*/*",
        "body": "",
    }

    body = {"max_tokens": 512, "temperature": 1.0, "top_p": 0.8, "top_k": 10, "prompt": x["prompt_input"]}

    api_template["body"] = json.dumps(body)

    success = False
    response = None
    for i in range(10):
        try:
            response = bedrock.invoke_model(**api_template)
            success = True
            break
        except:
            traceback.print_exc()
            time.sleep(5)

    if success:
        response_body = json.loads(response.get("body").read())
        logging.info(response_body)
        return response_body["outputs"][0]["text"]
    else:
        return ""


def predict_one_eg_claude_instant(x):
    current_session = boto3_session.Session()
    bedrock = current_session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        config=botocore.config.Config(
            read_timeout=120,  # corresponds to inference time limit set for Bedrock
            connect_timeout=120,
            retries={"max_attempts": 5},
        ),
    )

    # ✅ CHANGE 1: use a supported Claude model (Claude Instant is EOL)
    # Try the first; if you get ResourceNotFound, switch to the "us." cross-region id.
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    # model_id = "us.anthropic.claude-3-haiku-20240307-v1:0"

    api_template = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",  # ✅ CHANGE 2: Claude 3 returns JSON
        "body": "",
    }

    # ✅ CHANGE 3: Claude 3 uses "messages" format (not "prompt"/"completion")
    prompt_text = x["prompt_input"].strip() + "\nWrite your summary in <summary> XML tags."

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,      # ✅ CHANGE 4: new param name
        "temperature": 1.0,
        "top_p": 0.8,
        # top_k is optional; remove if your account rejects it
        "top_k": 10,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ],
    }

    api_template["body"] = json.dumps(body)

    success = False
    response = None
    for i in range(10):
        try:
            response = bedrock.invoke_model(**api_template)
            success = True
            break
        except:
            traceback.print_exc()
            time.sleep(20)

    if success:
        response_body = json.loads(response.get("body").read())

        # ✅ CHANGE 5: Claude 3 output lives in response_body["content"] blocks
        text = "".join([c.get("text", "") for c in response_body.get("content", [])]).strip()

        summary = extract_xml_tag(text, "summary")
        logging.info(summary or text)
        return summary or text
    else:
        return ""

