# NaniKanji
# Checks for Kanji Information with Llama3.1
# Help from https://llama-cpp-python.readthedocs.io/en/latest/

# Brendan Apple
# August 17, 2024

from llama_cpp import Llama
import os
import sys
import argparse
import time
import platform
import GPUtil


# System Info
current_directory = os.getcwd()
processor = platform.processor()
cpu_count = os.cpu_count()
gpu_present = GPUtil.getAvailable()
gpus = None
if gpu_present:
    gpus = GPUtil.getGPUs()

# Arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--verbose", help="display more info about program processes", action="store_true")
args = arg_parser.parse_args()

# Operational Settings
main_gpu = 0 if gpu_present else None
gpu_acceleration = -1 if gpu_present else 0  # 32 Tokens: 7.016 without GPU, 7.1229894161224365 with -1
cpu_threads = cpu_count - 2 if cpu_count > 2 else 1

# Statistics
prompt_tokens = 0
response_tokens = 0
total_tokens = 0

print('''NaniKanji
Llama3.1 Kanji Info

Brendan Apple
August 17, 2024''')

if args.verbose:
    print("Arguments: " + str(args))
    print()
    print('Python version:', platform.python_version())
    print("Model:" + current_directory + "\\Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf")
    print("Processor: " + processor)
    print("CPU Cores: ", cpu_threads, "/", cpu_count)
    print("GPUs" + ((": " + str(gpus[main_gpu]) + " from " + str(gpus)) if gpu_present else ": None"))
    print("GPU Acceleration: " + str(gpu_acceleration))
    print()
else:
    print()


# Output Suppression
class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


# Llama Object
class LlamaModel:
    def __init__(self):
        print("Load Llama3.1 8B Instruct ... ")
        start_time = time.time()

        if args.verbose:
            self.llm = self.load_llama()
        else:
            with suppress_stdout_stderr():
                self.llm = self.load_llama()

        load_time = time.time() - start_time
        if args.verbose:
            print("Time: ", load_time)

        time.sleep(0.2)
        os.system("cls||clear")
        print("-- NANI KANJI --")
        print("    何  漢字    ")

    @staticmethod
    def load_llama():
        return Llama(
            model_path=(current_directory + "\\Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"),
            n_gpu_layers=gpu_acceleration,  # Uncomment to use GPU acceleration
            # seed=1337,  # Uncomment to set a specific seed
            n_threads=cpu_threads,
            n_ctx=2048,  # Uncomment to increase the context window
            verbose=False
        )

    def complete(self, prompt: str, temp: float = 0.8, top_p: float = 1.0, max_tokens: int = 250,
                 freq_penalty: float = 0.1, stop_conditions=None):
        echo_prompt = False  # Echo the prompt back in the output

        if args.verbose:
            print("Generating Response ... ")
        start_time = time.time()

        with suppress_stdout_stderr():
            response = self.llm.create_completion(
                prompt,  # Prompt
                max_tokens=max_tokens,
                stop=stop_conditions,
                echo=echo_prompt,
                temperature=temp,
                top_p=top_p,
                frequency_penalty=freq_penalty
            )

        global total_tokens
        global prompt_tokens
        global response_tokens
        total_tokens += response['usage']['total_tokens']
        prompt_tokens += response['usage']['prompt_tokens']
        response_tokens += response['usage']['completion_tokens']

        load_time = time.time() - start_time
        if args.verbose:
            print("Time: ", load_time)
            print(response)
            print("tokens: " + str(total_tokens))
            print("prompt tokens: " + str(prompt_tokens))
            print("response tokens: " + str(response_tokens))
            print(response['choices'][-1]['finish_reason'])

        return response['choices'][-1]['text'].strip()

    def kanji_meaning(self, kanji: str):
        stop_words = ["What", "?", "Note: ", "\n"]
        return self.complete("Briefly, what does the Japanese kanji " + kanji + " mean? Answer: ",
                             temp=0.8, stop_conditions=stop_words)

    def kanji_description(self, kanji: str):
        return self.complete("Describe the Japanese kanji " + kanji + ". Answer: ",
                             temp=0.8)


# Main Sequential Instructions
llama3 = LlamaModel()

while True:
    print()
    user_request = input("Kanji: ")

    # Special Requests
    exit_words = ["", " ", "\n", "exit", "close", "quit", "leave", "escape"]
    if user_request in exit_words:
        break
    if ord(user_request[:1]) < 19968 or ord(user_request[:1]) > 40959:
        print("Japanese Kanji Only")
        continue
    if len(user_request) > 1:
        print("One Kanji at a Time")
        continue

    output_meaning = llama3.kanji_meaning(user_request)
    output_description = llama3.kanji_description(user_request)

    print()
    print("Meaning: ", output_meaning)
    print()
    print(output_description)
    print("--------------------------------------")
