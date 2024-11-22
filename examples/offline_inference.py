import datetime
from tqdm import tqdm
from typing import List

from sarathi.config import ModelConfig, ParallelConfig, SarathiSchedulerConfig, MetricsConfig, SystemConfig, ReplicaConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput

from line_profiler import profile

def split_text_file(filename: str, prompt_length: int, num_prompts: int):
    chunks = []
    current_chunk = []
    words_in_chunk = 0

    # Read the file and process non-empty lines
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Ignore empty lines
            
            words = line.split()
            for word in words:
                current_chunk.append(word)
                words_in_chunk += 1
                
                # Once current chunk reaches the desired word count
                if words_in_chunk == prompt_length:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    words_in_chunk = 0

                    # Stop if we have reached the required number of prompts
                    if len(chunks) == num_prompts:
                        return chunks

    # If there are leftover words, add them as the last chunk
    if current_chunk and len(chunks) < num_prompts:
        chunks.append(' '.join(current_chunk))

    return chunks

BASE_OUTPUT_DIR = "./offline_inference_output"

# Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]

num_gpus = 1
num_prompts = 128
num_prefill_tokens = 256
num_decode_tokens = 256
batch_size = 128

prompts = split_text_file("/home/hice1/kagidi6/scratch/smr/vajra/examples/holmes.txt", prompt_length=num_prefill_tokens, num_prompts=num_prompts)  
# prompts = [
#     "",
# ]

# Create a sampling params object.
# Ignore the stop token to ensure we generate max_tokens each time.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=num_decode_tokens, ignore_eos=True)

output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

replica_config = ReplicaConfig(
    output_dir=output_dir,
)

model_config = ModelConfig(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
)

parallel_config = ParallelConfig(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
)

scheduler_config = SarathiSchedulerConfig(
    chunk_size=100,
    max_num_seqs=batch_size, #batch size
)

metrics_config = MetricsConfig(
    write_metrics=True,
    enable_chrome_trace=True,
)

system_config = SystemConfig(
    replica_config=replica_config,
    model_config=model_config,
    parallel_config=parallel_config,
    scheduler_config=scheduler_config,
    metrics_config=metrics_config,
)

llm_engine = LLMEngine.from_system_config(system_config)

@profile
def generate(
    llm_engine: LLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
) -> List[RequestOutput]:
    for prompt in prompts:
        llm_engine.add_request(prompt, sampling_params)

    num_requests = llm_engine.get_num_unfinished_requests()
    pbar = tqdm(total=num_requests, desc="Processed prompts")

    # Run the engine
    outputs: List[RequestOutput] = []
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                pbar.update(1)

    pbar.close()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.seq_id))
    return outputs


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = generate(llm_engine, prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.text
    print("===========================================================")
    print(f"Prompt: {prompt!r}")
    print("-----------------------------------------------------------")
    print(f"Generated text: {generated_text!r}")
    print("===========================================================")

llm_engine.pull_worker_metrics()
llm_engine.plot_metrics()