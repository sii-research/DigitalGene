import torch
import re
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2Tokenizer, DynamicCache
from qwen_vl_utils import process_vision_info
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field

class RespState(Enum):
    WAIT_CODE = "wait_code"                         # wait '<code>{"' (27, 1851, 88863)
    ADD_CATEGORY_KEY = "add_category_key"           # add 'category": "'
    GEN_CATEGORY_VALUE = "gen_category_value"       # generate category!
    ADD_POSITION_KEY = "add_position_key"           # add '", "pose": {"global_position":'
    GEN_POSITION_VALUE = "gen_position_value"       # generate number or ',' or '-', or ' ' or '],' (means end) ' [' or ' [-' (means start)
    ADD_ROTATION_KEY = "add_rotation_key"           # add ' "global_rotation":'
    GEN_ROTATION_VALUE = "gen_rotation_value"       # generate number or ',' or '-', or ' ' or ']},' (means end) ' [' or ' [-' (means start)
    ADD_CONCEPT_KEY = "add_concept_key"             # add ' "conceptualization": [{"template": "'
    GEN_TEMPLATE_NAME = "gen_template_name"         # generate template!
    ADD_PARAM_CON = "add_param_con"                 # add '", "parameters": {"'
    GEN_PARAM_KEY = "gen_param_key"                 # generate param key
    ADD_PARAM_KV_CON = "add_param_kv_con"           # add '": ['
    GEN_PARAM_VALUE = "gen_param_value"             # generate param value, range [100000, 101024) or ',' add ' '
    ADD_PARAM_VALUE_CON = "add_param_value_con"     # add '], "'
    GEN_TEMPLATE_OR_END = "gen_template_or_end"     # gen ']}' (token 13989, next template) or ']' (token 60, end)
    ADD_NEXT_TEMPLATE_CON = "add_next_template_con" # add '}, {"template": "'
    ADD_END = "add_end"                             # add '}}]}</code>'
    END = "end"                                     # nothing, just end

@dataclass
class GenerationContext:
    current_state: RespState = RespState.WAIT_CODE
    current_category: Optional[List] = field(default_factory=list)
    current_position: Optional[List] = field(default_factory=list)
    current_rotation: Optional[List] = field(default_factory=list)
    current_pose_value: Optional[List] = field(default_factory=list)
    current_commas: Optional[int] = 0
    current_template: Optional[List] = field(default_factory=list)
    current_param: Optional[List] = field(default_factory=list)
    current_param_list: Optional[List[List]] = field(default_factory=list)
    current_param_value: Optional[List] = field(default_factory=list)
    current_param_index: Optional[int] = 0

class StateTransition:
    def __init__(self):
        self.transitions: Dict = self._build_transition_table()
    
    def _build_transition_table(self) -> Dict[RespState, Dict[str, RespState]]:
        return {
            RespState.WAIT_CODE: {
                "remain": RespState.WAIT_CODE,
                "continue": RespState.ADD_CATEGORY_KEY
            },
            RespState.ADD_CATEGORY_KEY: {
                "continue": RespState.GEN_CATEGORY_VALUE
            },
            RespState.GEN_CATEGORY_VALUE: {
                "remain": RespState.GEN_CATEGORY_VALUE,
                "continue": RespState.ADD_POSITION_KEY
            },
            RespState.ADD_POSITION_KEY: {
                "continue": RespState.GEN_POSITION_VALUE
            },
            RespState.GEN_POSITION_VALUE: {
                "remain": RespState.GEN_POSITION_VALUE,
                "continue": RespState.ADD_ROTATION_KEY
            },
            RespState.ADD_ROTATION_KEY: {
                "continue": RespState.GEN_ROTATION_VALUE
            },
            RespState.GEN_ROTATION_VALUE: {
                "remain": RespState.GEN_ROTATION_VALUE,
                "continue": RespState.ADD_CONCEPT_KEY
            },
            RespState.ADD_CONCEPT_KEY: {
                "continue": RespState.GEN_TEMPLATE_NAME
            },
            RespState.GEN_TEMPLATE_NAME: {
                "remain": RespState.GEN_TEMPLATE_NAME,
                "continue": RespState.ADD_PARAM_CON
            },
            RespState.ADD_PARAM_CON: {
                "continue": RespState.GEN_PARAM_KEY
            },
            RespState.GEN_PARAM_KEY: {
                "remain": RespState.GEN_PARAM_KEY,
                "continue": RespState.ADD_PARAM_KV_CON
            },
            RespState.ADD_PARAM_KV_CON: {
                "continue": RespState.GEN_PARAM_VALUE
            },
            RespState.GEN_PARAM_VALUE: {
                "remain": RespState.GEN_PARAM_VALUE,
                "continue": RespState.ADD_PARAM_VALUE_CON,
                "wait": RespState.GEN_TEMPLATE_OR_END
            },
            RespState.ADD_PARAM_VALUE_CON: {
                "continue": RespState.GEN_PARAM_KEY
            },
            RespState.GEN_TEMPLATE_OR_END: {
                "continue": RespState.ADD_NEXT_TEMPLATE_CON,
                "end": RespState.ADD_END
            },
            RespState.ADD_NEXT_TEMPLATE_CON: {
                "continue": RespState.GEN_TEMPLATE_NAME
            },
            RespState.ADD_END: {
                "continue": RespState.END
            }
        }
    
    def get_next_state(self, current_state: RespState, action: str) -> RespState:
        state_transitions = self.transitions.get(current_state, {})
        # state transitions
        return state_transitions.get(action, current_state)

class ConstrainedGenerator:
    def __init__(
            self,
            model_path: str,
            categories: List[str],
            param_dims: Dict,
            float_token_start: int = 100000,
            float_token_num: int = 1024,
            float_value_start: int = 2048
        ):
        self.model_path: str = model_path
        self.model: Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
        self.processor: Qwen2_5_VLProcessor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.tokenizer: Qwen2Tokenizer = self.processor.tokenizer
        self.categories: List[str] = categories
        self.param_dims: Dict = param_dims
        self.float_token_start: int = float_token_start
        self.float_token_num: int = float_token_num
        self.value_range = (float_value_start, float_value_start + float_token_num)
        # state transition
        self.state_transition = StateTransition()
        # pre compute
        self._precompute_all_tokens()
    
    def _precompute_all_tokens(self):
        # 1. add tokens
        self.add_tokens = {}
        add_segments: Dict[str, str] = {
            "add_category_key": 'category": "',
            "add_position_key": '", "pose": {"global_position":',
            "add_rotation_key": ' "global_rotation":',
            "add_concept_key": ' "conceptualization": [{"template": "',
            "add_param_con": '", "parameters": {"',
            "add_param_kv_con": '": [',
            "add_param_value_con": '], "',
            "add_next_template_con": '}, {"template": "',
            "add_end": '}}]}</code>'
        }
        for add_tag, segment in add_segments.items():
            encoded = self.tokenizer.encode(segment, add_special_tokens=False)
            self.add_tokens[add_tag] = encoded
        # 2. categories id
        self.category_valid_tokens: dict = {}
        for category in self.categories:
            encoded = self.tokenizer.encode(category, add_special_tokens=False)
            for index, token in enumerate(encoded):
                seq: tuple = tuple(encoded[:index])
                if not seq in self.category_valid_tokens:
                    self.category_valid_tokens[seq] = set()
                self.category_valid_tokens[seq].add(token)
        # 3. template info
        # template name (given category token)
        # param name (given category and template token)
        # param dims
        self.template_valid_tokens: dict = {}
        self.param_valid_tokens: dict = {}
        self.param_valid_dims: dict = {}
        for category, params in self.param_dims.items():
            # init template and param
            category_id: tuple = tuple(self.tokenizer.encode(category, add_special_tokens=False))
            self.template_valid_tokens[category_id] = {}
            self.param_valid_tokens[category_id] = {}
            self.param_valid_dims[category_id] = {}
            # template
            template_names: List[str] = params.keys()
            for template_name in template_names:
                template_name_id: tuple = tuple(self.tokenizer.encode(template_name, add_special_tokens=False))
                for index, token in enumerate(template_name_id):
                    seq: tuple = tuple(template_name_id[:index])
                    if not seq in self.template_valid_tokens[category_id]:
                        self.template_valid_tokens[category_id][seq] = set()
                    self.template_valid_tokens[category_id][seq].add(token)
                # init param
                self.param_valid_tokens[category_id][template_name_id] = {}
                self.param_valid_dims[category_id][template_name_id] = {}
                for param_name, param_dims in params[template_name].items():
                    param_name_id: tuple = tuple(self.tokenizer.encode(param_name, add_special_tokens=False))
                    for index, token in enumerate(param_name_id):
                        seq: tuple = tuple(param_name_id[:index])
                        if not seq in self.param_valid_tokens[category_id][template_name_id]:
                            self.param_valid_tokens[category_id][template_name_id][seq] = set()
                        self.param_valid_tokens[category_id][template_name_id][seq].add(token)
                    self.param_valid_dims[category_id][template_name_id][param_name_id] = param_dims[-1]    # the last, TODO: all array
        # 4. param value
        self.value_tokens = list(range(self.float_token_start, self.float_token_start + self.float_token_num))
        # 5. wait tokens
        self.wait_tokens: dict = {}
        wait_segments: dict = {
            "wait_code": '<code>{"'
        }
        for wait_tag, segment in wait_segments.items():
            encoded = self.tokenizer.encode(segment, add_special_tokens=False)
            self.wait_tokens[wait_tag] = encoded
        # 6. gen template or end
        self.other_tokens: dict = {}
        others_segments: dict = {
            "pose_start": [' [-', ' ['],
            "position_end": ['],'],
            "rotation_end": [']},'],
            "zero": ['0'],
            "natural": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            "positive": ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
            "comma": [','],
            "negative": [' -'],
            "blank": [' '],
            "next_template": [']}'], # ']}' means next template
            "end": [']']    # ']' means end
        }
        for other_tag, segment in others_segments.items():
            self.other_tokens[other_tag] = []
            for seg in segment:
                encoded = self.tokenizer.encode(seg, add_special_tokens=False)
                self.other_tokens[other_tag].extend(encoded)
    
    def _get_allowed_tokens(self, state: RespState, context: GenerationContext) -> List[int]:
        """generate token"""
        if state == RespState.GEN_CATEGORY_VALUE:   # catetory
            # get current context
            current_category: tuple = tuple(context.current_category)
            return list(self.category_valid_tokens.get(tuple(current_category), []))
        elif state == RespState.GEN_POSITION_VALUE or state == RespState.GEN_ROTATION_VALUE:
            # get current context (token list)
            current_pose: list = context.current_position if state == RespState.GEN_POSITION_VALUE else context.current_rotation
            # import pdb; pdb.set_trace()
            if len(current_pose) == 0:
                # start
                return self.other_tokens["pose_start"]
            elif current_pose[-1] in self.other_tokens["pose_start"]:
                # first number
                return self.other_tokens["natural"]
            elif current_pose[-1] in self.other_tokens["blank"]:
                return self.other_tokens["natural"]
            elif current_pose[-1] in self.other_tokens["comma"]:
                return self.other_tokens["blank"] + self.other_tokens["negative"]
            elif current_pose[-1] in self.other_tokens["negative"]:
                return self.other_tokens["positive"]
            else:
                allowed_tokens: list = []
                # current pose value
                not_last: bool = len(context.current_pose_value) < 3 and not (len(context.current_pose_value) == 1 and context.current_pose_value[0] in self.other_tokens["zero"])
                if not_last:
                    allowed_tokens += self.other_tokens["natural"]
                # if last:
                if context.current_commas == 2:
                    allowed_tokens += self.other_tokens["position_end"] if state == RespState.GEN_POSITION_VALUE else self.other_tokens["rotation_end"]
                else:
                    allowed_tokens += self.other_tokens["comma"]
                return allowed_tokens
        elif state == RespState.GEN_TEMPLATE_NAME:
            # get current context
            current_category: tuple = tuple(context.current_category)
            assert current_category in self.template_valid_tokens
            current_template: tuple = tuple(context.current_template)
            return list(self.template_valid_tokens[current_category].get(current_template, []))
        elif state == RespState.GEN_PARAM_KEY:
            # get current conetxt
            current_category: tuple = tuple(context.current_category)
            assert current_category in self.param_valid_tokens
            current_template: tuple = tuple(context.current_template)
            assert current_template in self.param_valid_tokens[current_category]
            current_param: tuple = tuple(context.current_param)
            return list(self.param_valid_tokens[current_category][current_template].get(current_param, []))
        elif state == RespState.GEN_PARAM_VALUE:
            # get current param
            current_param_value: list = context.current_param_value
            if len(current_param_value) == 0 or current_param_value[-1] in self.other_tokens["blank"]:
                return self.value_tokens
            elif self.float_token_start <= current_param_value[-1] < self.float_token_start + self.float_token_num:
                return self.other_tokens["comma"]
            elif current_param_value[-1] in self.other_tokens["comma"]:
                return self.other_tokens["blank"]
            else:
                raise ValueError(f"Error param value: {current_param_value}")
        elif state == RespState.GEN_TEMPLATE_OR_END:
            return self.other_tokens["next_template"] + self.other_tokens["end"]
        else:
            raise ValueError(f"Error state: {state}")
    
    def _update_context(self, token_ids: torch.Tensor, context: GenerationContext):
        current_state: RespState = context.current_state
        state_transition_action: str = "continue"   # default transition
        if current_state == RespState.WAIT_CODE:
            length: int = len(self.wait_tokens["wait_code"])
            if token_ids[0, -length:].tolist() != self.wait_tokens["wait_code"]:
                state_transition_action = "remain"
        elif current_state == RespState.GEN_CATEGORY_VALUE:
            # append
            context.current_category.append(token_ids[0, -1].item())  # (update current category)
            current_category: tuple = tuple(context.current_category)
            if len(self.category_valid_tokens.get(current_category, [])) != 0:
                state_transition_action = "remain"
        elif current_state == RespState.ADD_POSITION_KEY or current_state == RespState.ADD_ROTATION_KEY:
            context.current_commas = 0
            context.current_pose_value = []
        elif current_state == RespState.GEN_POSITION_VALUE or current_state == RespState.GEN_ROTATION_VALUE:
            # append
            if current_state == RespState.GEN_POSITION_VALUE:
                context.current_position.append(token_ids[0, -1].item())  # update current position
            else:
                context.current_rotation.append(token_ids[0, -1].item())  # update current rotation
            # update commas if needed
            if token_ids[0, -1].item() in self.other_tokens["comma"]:
                context.current_commas += 1
                context.current_pose_value = []
            elif token_ids[0, -1].item() in self.other_tokens["natural"]:
                context.current_pose_value.append(token_ids[0, -1].item())
            number_tokens: list = self.other_tokens["position_end"] if current_state == RespState.GEN_POSITION_VALUE else self.other_tokens["rotation_end"]
            if not token_ids[0, -1].item() in number_tokens:
                state_transition_action = "remain"
        elif current_state == RespState.GEN_TEMPLATE_NAME:
            # generate
            context.current_template.append(token_ids[0, -1].item())      # update current template name
            current_category: tuple = tuple(context.current_category)
            current_template: tuple = tuple(context.current_template)
            assert current_category in self.template_valid_tokens
            if len(self.template_valid_tokens[current_category].get(current_template, [])) != 0:
                state_transition_action = "remain"
        elif current_state == RespState.GEN_PARAM_KEY:
            # generate
            context.current_param.append(token_ids[0, -1].item())
            current_category: tuple = tuple(context.current_category)
            current_template: tuple = tuple(context.current_template)
            assert current_category in self.param_valid_tokens
            assert current_template in self.param_valid_tokens[current_category]
            current_param: tuple = tuple(context.current_param)
            if len(self.param_valid_tokens[current_category][current_template].get(current_param, [])) != 0:
                state_transition_action = "remain"
            else:
                # update param list
                context.current_param_list.append(current_param)
        elif current_state == RespState.GEN_PARAM_VALUE:
            # generate
            context.current_param_value.append(token_ids[0, -1].item())
            if self.float_token_start <= context.current_param_value[-1] < self.float_token_start + self.float_token_num:
                context.current_param_index += 1
            current_category: tuple = tuple(context.current_category)
            current_template: tuple = tuple(context.current_template)
            current_param: tuple = tuple(context.current_param)
            if context.current_param_index >= self.param_valid_dims[current_category][current_template][current_param]:
                # reset param value
                context.current_param_value.clear()
                context.current_param_index = 0
                # next param or wait
                if len(context.current_param_list) == len(self.param_valid_dims[current_category][current_template].keys()):
                    state_transition_action = "wait"
                # next param: 'continue' (default)
            else:
                state_transition_action = "remain"
        elif current_state == RespState.GEN_TEMPLATE_OR_END:
            # update state
            if token_ids[0, -1].item() in self.other_tokens["end"]:
                state_transition_action = "end"
        elif current_state == RespState.ADD_PARAM_CON or current_state == RespState.ADD_PARAM_VALUE_CON:
            # clear current_param (value and name)
            context.current_param.clear()
            context.current_param_value.clear()
        elif current_state == RespState.ADD_PARAM_KV_CON:
            # clear current_param_value
            context.current_param_value.clear()
        elif current_state == RespState.ADD_NEXT_TEMPLATE_CON:
            context.current_template.clear()
            context.current_param.clear()
            context.current_param_list.clear()
            context.current_param_value.clear()
            context.current_param_index = 0
        # update state
        context.current_state = self.state_transition.get_next_state(current_state, state_transition_action)
        
    def _prepare_model_inputs(
            self,
            generated_ids: torch.Tensor, 
            attention_mask: torch.Tensor, 
            original_inputs: Dict,
            use_cache: bool = False,
            past_key_values = None
        ) -> Dict:
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            cache_position = torch.arange(past_length, past_length + 1, device=generated_ids.device)
            # position ids couldn't be (seq_len - 1) because of the multimodal pos
            current_inputs = {
                "input_ids": generated_ids[:, -1:],
                "attention_mask": attention_mask,
                "past_key_values": past_key_values, # generated_ids[:, :-1]
                "use_cache": use_cache,
                "cache_position": cache_position
            }
        else:
            current_inputs = {
                "input_ids": generated_ids,
                "attention_mask": attention_mask,
                "use_cache": use_cache
            }
            if "pixel_values" in original_inputs:
                current_inputs["pixel_values"] = original_inputs["pixel_values"]
            
            if "image_grid_thw" in original_inputs:
                current_inputs["image_grid_thw"] = original_inputs["image_grid_thw"]
        
        return current_inputs
    
    def _constrained_generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values: torch.Tensor,
            image_grid_thw: torch.Tensor,
            max_new_tokens: int = 1024,
            use_cache: bool = True
        ) -> str:
        generated_ids = input_ids.clone()
        context = GenerationContext()
        # get tempearture
        temperature: float = self.model.generation_config.temperature
        # kv cache
        past_key_values: DynamicCache = None
        original_inputs = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw
        }
        # init attention mask
        current_attention_mask = attention_mask.clone()
        # loop invariant: generated_ids.length == current_attention_mask.length == pixel_values.length + 1
        for step in range(max_new_tokens):
            if context.current_state == RespState.END:
                break
            # wait or generate or add
            current_state: RespState = context.current_state
            state_value: str = current_state.value
            state_action: str = state_value.split('_')[0]
            if state_action == "wait":
                # generate without constraint
                model_inputs = self._prepare_model_inputs(
                    generated_ids, current_attention_mask, original_inputs, 
                    use_cache=use_cache, past_key_values=past_key_values
                )
                # forward
                with torch.no_grad():
                    outputs = self.model(**model_inputs)
                    logits = outputs.logits[0, -1, :]
                    if use_cache:
                        past_key_values = outputs.past_key_values
                next_tokens = torch.tensor([torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)]).unsqueeze(dim=0)
                generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
                # update attention_mask
                new_token_num = len(next_tokens[0])
                new_attention = torch.ones((1, new_token_num), device=current_attention_mask.device, dtype=current_attention_mask.dtype)
                current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=-1)
                assert generated_ids.shape[-1] == current_attention_mask.shape[-1]
            elif state_action == "add":
                # update kv cache
                if use_cache:
                    model_inputs = self._prepare_model_inputs(
                        generated_ids, current_attention_mask, original_inputs, 
                        use_cache=use_cache, past_key_values=past_key_values
                    )
                    # forward
                    with torch.no_grad():
                        outputs = self.model(**model_inputs)
                        past_key_values = outputs.past_key_values
                # append add_ids directly
                add_token_list = self.add_tokens[state_value]
                # assert state_value in self.add_kv_cache
                for i, token_id in enumerate(add_token_list[:-1]):
                    # update
                    generated_ids = torch.cat([generated_ids, torch.tensor([[token_id]], device=generated_ids.device)], dim=-1)
                    current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_attention_mask.device)], dim=-1)
                    if use_cache:
                        model_inputs = self._prepare_model_inputs(
                            generated_ids, current_attention_mask, original_inputs, 
                            use_cache=use_cache, past_key_values=past_key_values
                        )
                        # forward
                        with torch.no_grad():
                            outputs = self.model(**model_inputs)
                            past_key_values = outputs.past_key_values
                token_id = add_token_list[-1]
                # update
                generated_ids = torch.cat([generated_ids, torch.tensor([[token_id]], device=generated_ids.device)], dim=-1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_attention_mask.device)], dim=-1)
            else:
                assert state_action == "gen"
                # get allowed tokens
                allowed_tokens = self._get_allowed_tokens(context.current_state, context)
                if not allowed_tokens:
                    raise ValueError(f"No allowed tokens for state: {context.current_state}")
                # else:
                model_inputs = self._prepare_model_inputs(
                    generated_ids, current_attention_mask, original_inputs, 
                    use_cache=use_cache, past_key_values=past_key_values
                )
                with torch.no_grad():
                    outputs = self.model(**model_inputs)
                    logits = outputs.logits[0, -1, :]
                    if use_cache:
                        past_key_values = outputs.past_key_values
                # apply constraint
                masked_logits = torch.full_like(logits, float('-inf'))
                valid_tokens = [t for t in allowed_tokens if 0 <= t < len(logits)]
                if valid_tokens:
                    masked_logits[valid_tokens] = logits[valid_tokens]
                    next_tokens = torch.tensor([torch.multinomial(torch.softmax(masked_logits / temperature, dim=-1), 1)]).unsqueeze(dim=0)
                else:
                    raise ValueError(f"No valid tokens in vocabulary for state: {context.current_state}")
                # append generated ids
                generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
                # update attention_mask
                new_token_num = len(next_tokens[0])
                new_attention = torch.ones((1, new_token_num), device=current_attention_mask.device, dtype=current_attention_mask.dtype)
                current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=-1)
                assert generated_ids.shape[-1] == current_attention_mask.shape[-1]
            # update context
            token_ids = generated_ids.clone()
            self._update_context(token_ids, context)  # update
        return self.decode_generated_result(input_ids, generated_ids)
    
    def decode_generated_result(self, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
        """decode"""
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else ""
    
    @staticmethod
    def _prompt_template(question: str, image_path: str, system_prompt: str = "") -> List[Dict]:
        if system_prompt:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}",
                        },
                        {"type": "text", "text": question}
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}",
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
        return messages
    
    def constrained_generate(self, question: str, image_path: str, system_prompt: str = "") -> str:
        messages = ConstrainedGenerator._prompt_template(question, image_path, system_prompt)
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        # generate
        generated_text: str = self._constrained_generate(**inputs, use_cache=True)
        return generated_text
