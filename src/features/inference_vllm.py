from typing import Type, List
from vllm import LLM, SamplingParams

from conversation_vllm import get_conv_template


def preprocess_instance(source):
    conv = get_conv_template("ie_as_qa")
    for j, sentence in enumerate(source):
        value = sentence['value']
        if j == len(source) - 1:
            value = None
        conv.append_message(conv.roles[j % 2], value)
    prompt = conv.get_prompt()
    return prompt


def get_response(responses):
    responses = [r.split('ASSISTANT:')[-1].strip() for r in responses]
    return responses


def inference(
    model: Type[LLM],
    examples: List[dict],
    max_new_tokens: int = 256,
):
    prompts = [preprocess_instance(example['conversations']) for example in examples]
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=['</s>'])
    responses = model.generate(prompts, sampling_params)
    responses_corret_order = []
    response_set = {response.prompt: response for response in responses}
    for prompt in prompts:
        assert prompt in response_set
        responses_corret_order.append(response_set[prompt])
    responses = responses_corret_order
    outputs = get_response([output.outputs[0].text for output in responses])
    return outputs
