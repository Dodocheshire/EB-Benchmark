import json
import sys
import os
import base64
import logging
import time
import copy
import anthropic
import google.generativeai as genai
from openai import OpenAI
import typing_extensions as typing
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_config.generation_guide_manip import llm_generation_guide_manip, vlm_generation_guide_manip
from embodiedbench.planner.manip_actions import parse_manipulation_plan, InvalidManipulationPlan
from embodiedbench.planner.planner_utils import convert_format_2claude, convert_format_2gemini, ActionPlan_1, ActionPlan, ActionPlan_lang, \
                                             ActionPlan_1_manip, ActionPlan_manip, ActionPlan_lang_manip, fix_json

temperature = 0
max_completion_tokens = 4096
remote_url = os.environ.get('remote_url')
logger = logging.getLogger(__name__)

class RemoteModel:
    def __init__(
        self,
        model_name,
        model_type='remote',
        language_only=False,
        tp=1,
        task_type=None # used to distinguish between manipulation and other environments
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.language_only = language_only
        self.task_type = task_type
        self.api_extra_body = {}
        self.api_timeout = 120
        self.reasoning_effort = None
        self.api_max_tokens = max_completion_tokens

        if self.model_type == 'local':
            from lmdeploy import pipeline, PytorchEngineConfig
            backend_config = PytorchEngineConfig(session_len=12000, dtype='float16', tp=tp)
            self.model = pipeline(self.model_name, backend_config=backend_config)
        else:
            if "claude" in self.model_name:
                self.model = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
            elif "gemini" in self.model_name:
                self.model = OpenAI(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
            elif "gpt" in self.model_name:
                self.model = OpenAI()
            elif self.model_name.startswith("deepseek"):
                self.model = OpenAI(
                    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                )
            elif 'qwen' in self.model_name:
                custom_base_url = os.getenv("OPENAI_BASE_URL")
                self.model = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY") if custom_base_url else os.getenv("DASHSCOPE_API_KEY"),
                    base_url=custom_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif "Qwen3-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Qwen2-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Qwen2.5-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "OpenGVLab/InternVL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "90b-vision-instruct" in self.model_name: # you can use fireworks to inference
                self.model = OpenAI(base_url='https://api.fireworks.ai/inference/v1',
                                    api_key=os.environ.get("firework_API_KEY"))
            else:
                try:
                    self.model = OpenAI(base_url = remote_url)
                except:
                    raise ValueError(f"Unsupported model name: {model_name}")


    def respond(self, message_history: list):
        if self.model_type == 'local':
            return self._call_local(message_history)
        else:
            if "claude" in self.model_name:
                return self._call_claude(message_history)
            elif "gemini" in self.model_name:
                return self._call_gemini(message_history)
            elif "gpt" in self.model_name:
                return self._call_gpt(message_history)
            elif self.model_name.startswith("deepseek"):
                return self._call_gpt(message_history)
            elif 'qwen' in self.model_name:
                return self._call_gpt(message_history)
            elif "Qwen3-VL" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2.5-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Qwen2.5-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                return self._call_llama11b(message_history)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "90b-vision-instruct" in self.model_name:
                return self._call_llama90(message_history)
            elif "OpenGVLab/InternVL" in self.model_name:
                return self._call_intern38b(message_history)
            # elif "OpenGVLab/InternVL2_5-38B" in self.model_name:
            #     return self._call_intern38b(message_history)
            # elif "OpenGVLab/InternVL2_5-78B" in self.model_name:
            #     return self._call_intern38b(message_history)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

    def _completion_token_kwargs(self, token_limit=None):
        if token_limit is None:
            token_limit = getattr(self, "api_max_tokens", max_completion_tokens)
        if self.model_name.startswith(("gpt-5", "gpt-6")):
            return dict(max_completion_tokens=token_limit)
        return dict(max_tokens=token_limit)

    def _temperature_kwargs(self):
        if self.model_name.startswith(("gpt-5", "gpt-6", "deepseek")):
            return {}
        return dict(temperature=temperature)

    def _reasoning_kwargs(self):
        effort = getattr(self, "reasoning_effort", None)
        return {"reasoning_effort": effort} if effort is not None else {}

    def _call_local(self, message_history: list):
        from lmdeploy import GenerationConfig
        if self.task_type == 'manip':
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": llm_generation_guide_manip if self.language_only else vlm_generation_guide_manip
                }
            }
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": llm_generation_guide if self.language_only else vlm_generation_guide
                }
            }
        response = self.model(
            message_history,
            gen_config=GenerationConfig(
                temperature=temperature,
                response_format=response_format,
                max_new_tokens=max_completion_tokens,
            )
        )
        out = response.text
        out = fix_json(out)
        return out

    def _call_claude(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2claude(message_history)

        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            messages=message_history
        )

        return response.content[0].text 

    def _call_gemini(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if self.task_type == 'manip':
            response = self.model.beta.chat.completions.parse(
                model=self.model_name, 
                messages=message_history,
                response_format= ActionPlan_lang_manip if self.language_only else ActionPlan_manip,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
        else:
            response = self.model.beta.chat.completions.parse(
                model=self.model_name, 
                messages=message_history,
                response_format= ActionPlan_lang if self.language_only else ActionPlan,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
        tokens = response.usage.prompt_tokens

        return str(response.choices[0].message.parsed.model_dump_json())

    def _call_gpt(self, message_history: list):

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        if self.task_type == "manip" and "qwen" in self.model_name.lower():
            response_format = copy.deepcopy(response_format)
            action_schema = response_format["json_schema"]["schema"]["properties"]["executable_plan"]["items"]["properties"]["action"]
            action_schema.update(type="array", items={"type": "integer", "minimum": 0, "maximum": 119},
                                 minItems=7, maxItems=7,
                                 description="Exactly [x,y,z,rx,ry,rz,gripper]. Position: 0..100; rotation: 0..119; gripper: 0 or 1. No function calls or prose.")
            message_history = copy.deepcopy(message_history)
            message_history.insert(0, {"role": "system", "content":
                "For Manipulation, executable_plan must contain objects whose action is a JSON array of exactly seven discrete numbers "
                "[x,y,z,rx,ry,rz,gripper], following the task's coordinate and rotation examples. "
                "Do not output move_to(), open_gripper(), other function calls, variable names, or natural language as actions."})

        if self.model_name.startswith("deepseek"):
            schema = response_format["json_schema"]["schema"]
            message_history = copy.deepcopy(message_history)
            message_history.insert(0, {
                "role": "system",
                "content": "Return only a JSON object matching this schema: " + json.dumps(schema),
            })
            response_format = {"type": "json_object"}

        token_limit = self.api_max_tokens
        token_ceiling = getattr(self, "api_max_tokens_limit", token_limit * 4)
        for attempt in range(3):
            request_started = time.time()
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=message_history,
                response_format=response_format,
                **self._temperature_kwargs(),
                **self._completion_token_kwargs(token_limit),
                **self._reasoning_kwargs(),
                extra_body=self.api_extra_body or None,
                timeout=self.api_timeout,
            )
            choice = response.choices[0] if response.choices else None
            if getattr(self, "request_log_path", None):
                record = {
                    **getattr(self, "request_context", {}),
                    "timestamp": time.time(),
                    "model": self.model_name,
                    "attempt": attempt + 1,
                    "request_options": {
                        **self._completion_token_kwargs(token_limit),
                        **self._reasoning_kwargs(),
                        "response_format": response_format,
                        "extra_body": self.api_extra_body,
                        "timeout": self.api_timeout,
                    },
                    "messages": [
                        {"role": message["role"], "content": message["content"] if isinstance(message["content"], str) else [
                            item if item.get("type") == "text" else
                            {"type": item.get("type"), "image_path": getattr(self, "request_context", {}).get("image_path")}
                            for item in message["content"]
                        ]} for message in message_history
                    ],
                    "elapsed_seconds": time.time() - request_started,
                    "response_id": response.id,
                    "finish_reason": choice.finish_reason if choice else "no_choices",
                    "usage": response.usage.model_dump() if response.usage else None,
                    "message": choice.message.model_dump() if choice else None,
                }
                with open(self.request_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            finish_reason = choice.finish_reason if choice is not None else "no_choices"
            if choice is not None and finish_reason != "length":
                out = choice.message.content
                if isinstance(out, str) and out.strip():
                    if self.task_type != "manip":
                        return out
                    try:
                        parse_manipulation_plan(out)
                        return out
                    except InvalidManipulationPlan as exc:
                        logger.warning("Model %s returned an invalid Manipulation plan: %s", self.model_name, exc)
                        if attempt == 2:
                            raise
                        message_history = copy.deepcopy(message_history)
                        message_history.extend([
                            {"role": "assistant", "content": out},
                            {"role": "user", "content": "Your action plan cannot be parsed: " + str(exc) +
                             ". Regenerate the complete JSON response. Every action must have seven discrete numeric values "
                             "[x,y,z,rx,ry,rz,gripper], using either a JSON array or its quoted list representation. "
                             "Use the provided object coordinates and rotation examples; no function calls, variable names, or prose. "
                             "Position range 0..100, rotation range 0..119, gripper 0 or 1."},
                        ])
                        continue
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            refusal = getattr(choice.message, "refusal", None) if choice is not None else None
            diagnostic = (
                f"Model {self.model_name} returned {'truncated output' if finish_reason == 'length' else 'empty content'} "
                f"(finish_reason={finish_reason}, completion_tokens={completion_tokens}, "
                f"refusal={bool(refusal)})"
            )
            if refusal or finish_reason == "content_filter":
                raise RuntimeError(diagnostic)
            if attempt < 2:
                if finish_reason == "length":
                    if token_limit >= token_ceiling:
                        raise RuntimeError(diagnostic + f"; token ceiling {token_ceiling} reached")
                    token_limit = min(token_limit * 2, token_ceiling)
                logger.warning("%s; retrying with token budget %s", diagnostic, token_limit)
        raise RuntimeError(diagnostic)
    
    def _call_qwen7b(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        out = response.choices[0].message.content
        return out
    
    def _call_llama90(self, message_history: list):
        if self.task_type == "manip":
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                response_format={"type": "json_object", "schema": ActionPlan_1_manip.model_json_schema()},
                temperature = temperature
            )
            out = response.choices[0].message.content
            
        else:
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                response_format={"type": "json_object", "schema": ActionPlan_1.model_json_schema()},
                temperature = temperature
            )
            out = response.choices[0].message.content
        return out
    
    def _call_llama11b(self, message_history):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content
        return out
    

    def _call_qwen72b(self, message_history):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))
        
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out
    
    def _call_intern38b(self, message_history):

        # if not self.language_only:
        #     message_history = convert_format_2gemini(message_history)

        # no use, lmdeploy use support json schema only if it is pytorch-backended
        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out



if __name__ == "__main__":

    model = RemoteModel(
        'Qwen/Qwen2-VL-72B-Instruct', #'meta-llama/Llama-3.2-11B-Vision-Instruct',
        True #False
    )#'claude-3-5-sonnet-20241022, Qwen/Qwen2-VL-72B-Instruct, meta-llama/Llama-3.2-11B-Vision-Instruct


    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        

    base64_image = encode_image("../../evaluator/midlevel/output.png")
        
    messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                }
            },
            {
                "type": "text",
                "text":f"What do you think for this picture?? {template}?"
            },
            ],
        }
    ]

    response = model.respond(messages)
    print(response)
