import json

from vllm import LLM, SamplingParams

from typing import Callable, List

from drgrpo_grader import r1_zero_reward_fn

def test_reward_function():
    response = "asjfklawer"
    ground_truth = "asdfas"
    print(r1_zero_reward_fn(response, ground_truth))
    

def simple_demo():
    
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # Create an LLM.
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        
def eval_zero_shot():
    # load model and parameters
    vllm_model = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )
    
    # load datasets
    dataset_path = "/home/exouser/cs336/assignment5-alignment/data/gsm8k/train.jsonl"
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    
    # prepare prompts from dataset
    propmt_template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {question}\nAssistant: <think>"""
    prompts_questions = [propmt_template.format(question=item['question']) for item in dataset]
    prompts_answers = [item['answer'] for item in dataset]
    
    # only select the first 50 question-answer pairs
    N = 50
    prompts_questions = prompts_questions[:N]
    prompts_answers = prompts_answers[:N]
    
    # generate outputs
    outputs = vllm_model.generate(prompts_questions, eval_sampling_params)
    results_to_dump = []
    for output, ground_truth in zip(outputs, prompts_answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ground_truth = ground_truth
        
        eval_result = r1_zero_reward_fn(generated_text, ground_truth)
        results_to_dump.append(
            {
                "prompt": prompt,
                "response": generated_text,
                "truth": ground_truth,
                "eval_result": eval_result
            }
        )
        
    # dump the results to local
    with open("/home/exouser/cs336/assignment5-alignment/results/zero_shot_eval_result.json", "w", encoding="utf-8") as f:
        json.dump(results_to_dump, f, indent=4)
    
    
        
if  __name__ == '__main__':
    eval_zero_shot()