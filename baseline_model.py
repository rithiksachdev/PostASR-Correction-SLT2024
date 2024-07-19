import os
import requests
import json
import re
from dotenv import load_dotenv
import jiwer
import time
import anthropic
from whisper.normalizers import EnglishTextNormalizer

# Load environment variables from .env file
load_dotenv()

client = anthropic.Anthropic()

def read_json(file_path):
    """Reads a JSON file and returns the data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as jsonFile:
        content = jsonFile.read().strip()
        if not content:
            raise ValueError(f"The file {file_path} is empty.")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from the file {file_path}: {e}")
        return data

def get_example_contexts(example_file, n, train_data):
    """Retrieve n unique example contexts from the training data."""
    contexts = []
    examples = set()
    
    while len(contexts) < n:
        line = example_file.readline().strip()
        if line:
            t = int(line)
            if t not in examples:
                examples.add(t)
                for i in range(5):
                    examples.add(t + i + 1)
                    examples.add(t - i - 1)
                contexts.append(train_data[t])
        else:
            break
    
    return contexts

def construct_prompt(user_messages, assistant_messages, question_messages):
    """Construct the full prompt for the API request."""
    prompt = ""
    for i , (user_message, assistant_message, question_message) in enumerate(zip(user_messages, assistant_messages, question_messages)):
        # prompt += f"Human: {user_message}\n\nAssistant: {assistant_message}\n\n"
        prompt += f"{i+1} {question_message}\n\n {i+1} Assistant: ??\n\n"
    prompt += "Fill in the question marks and return the complete 20 results like this example: The true transcription from the 5-best hypotheses is: \"\". Make sure to return the response in the order of questions with each one inside the quotes."
    return prompt

def extract_text_between_quotes(text):
    """Extract and return the text between the first pair of quotes."""
    start = text.find('"') + 1
    end = text.find('"', start)
    return text[start:end]

def call_api(system_prompt, full_prompt):
    """Call the Claude API with the given prompt and return the response."""

    try:
        print("-------Prompt------")
        print(full_prompt[:1000])    
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1048,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_prompt
                        }
                    ]
                }
            ]
        )
        print(response)
        time.sleep(1)
        with open('api_response.txt', 'a') as f:
            f.write(response.content[0].text + '\n')
        return response.content[0].text
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as e:
        print(f"Error: {e}")
    return None

def chat_rescore(system_prompt, batched_prompts, example_files, n, train_data):
    """Perform language model rescoring using the Claude API."""
    user_messages = []
    assistant_messages = []
    question_messages = []
    for i, (test_txt, example_file) in enumerate(zip(batched_prompts, example_files)):
        contexts = get_example_contexts(example_file, n, train_data)
        if contexts:
            context = contexts[ n - (i % 5) - 1]
            if 'source' in context and 'target' in context:
                hypotheses = context['source'].split('. ')
                txt = "\n".join(hypotheses)
                user_message = f"The 5-best hypothesis are:\n{txt}"
                assistant_message = f"The true transcription from the 5-best hypotheses is: \"{context['target']}\""
                question_text = f"The 5-best hypothesis are:\n{test_txt}"
                user_messages.append(user_message)
                assistant_messages.append(assistant_message)
                question_messages.append(question_text)
    
    if user_messages and assistant_messages:
        full_prompt = construct_prompt(user_messages, assistant_messages, question_messages)
        return call_api(system_prompt,full_prompt)
    return None

def normalize_text(text, normalizer):
    """Normalize text using EnglishTextNormalizer."""
    return normalizer(text)

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER) using jiwer."""
    return jiwer.wer(reference, hypothesis)

def calculate_wer_with_system_prompt(system_prompt, test_data_path, train_data_path, example_dir, result_file_path, n, batch_num):
    # Load test data
    test_data = read_json(test_data_path)
    
    # Load training data
    train_data = read_json(train_data_path)
    
    result_file = open(result_file_path, 'w+')
    
    all_references = []
    all_hypotheses = []
    
    normalizer = EnglishTextNormalizer()
    batched_prompts = []
    example_files = []
    count = 0
    for i, question in enumerate(test_data):
        hypotheses = question['input']
        test_txt = "\n".join(hypotheses)
        example_file_path = os.path.join(example_dir, f"{i}.txt")
        if os.path.exists(example_file_path):
            example_file = open(example_file_path, 'r')
            batched_prompts.append(test_txt)
            example_files.append(example_file)            
            if len(batched_prompts) == batch_num:
                response = chat_rescore(system_prompt, batched_prompts, example_files, n, train_data)
                if response:
                    print("------RESP--------")
                    print(response)
                    results = [part.strip() for part in response.split('\n') if part.strip()]
                    if not re.match(r'^\d+\.', results[0]):
                        results = results[1:]
                    print("-------RESULT------")
                    print(results)
                    for j, (result, question) in enumerate(zip(results, test_data[i-batch_num+1:i+1])):
                        hypothesis = extract_text_between_quotes(result)
                        if hypothesis:
                            print(hypothesis + f'({j+i-18})', file=result_file)
                            normalized_reference = normalize_text(question['output'], normalizer)
                            normalized_hypothesis = normalize_text(hypothesis, normalizer)
                            if normalized_reference and normalized_hypothesis:
                                all_references.append(normalized_reference)
                                all_hypotheses.append(normalized_hypothesis)
                        else:
                            print(f"Error processing question {j+i-8}", file=result_file)
                # Reset batched prompts and example files for the next batch
                batched_prompts = []
                example_files = []
                # break
        count = i
    
    # Process any remaining prompts
    if batched_prompts:
        response = chat_rescore(system_prompt, batched_prompts, example_files, n, train_data)
        if response:
            # results = response.split("The true transcription from the 5-best hypotheses is: ")[1:]
            results = [part.strip() for part in response.split('\n') if part.strip()]
            if not re.match(r'^\d+\.', results[0]):
                results = results[1:]
            for k, (result, question) in enumerate(zip(results, test_data[-len(batched_prompts):])):
                hypothesis = extract_text_between_quotes(result)
                if hypothesis:
                    print(hypothesis + f'({k+count-14})', file=result_file)
                    normalized_reference = normalize_text(question['output'], normalizer)
                    normalized_hypothesis = normalize_text(hypothesis, normalizer)
                    if normalized_reference and normalized_hypothesis:
                        all_references.append(normalized_reference)
                        all_hypotheses.append(normalized_hypothesis)
                else:
                    print(f"Error processing question {k+count-8}", file=result_file)
    
    # Calculate overall WER
    wer = jiwer.wer(
        all_references,
        all_hypotheses
    )
    
    print(f"Overall WER: {wer}")
    
    result_file.write(f"\nOverall WER: {wer}")
    
    result_file.close()

    return wer

if __name__ == "__main__":
    system_prompt = "You are faced with a complex linguistic challenge: meticulously examine five diverse transcription hypotheses for a specific audio recording. Your task is to synthesize these interpretations into a single, comprehensive sentence that accurately captures the essence of the audio content. Employ impeccable English grammar, all sentences are mostly in simple present tense and style to craft a cohesive and precise representation of the true transcription, without referencing the multiple sources or the process of analysis. These transcriptions have financial transcriptions and might have initialisms that need to be seperated into letters."
    test_data_path = "./HyPoradise-v0/test/test_wsj_score.json"
    train_data_path = "./HyPoradise-v0/train/train_chime4.json"
    example_dir = "Hyporadise-icl/examples/knn/"
    result_file = "knn_result.txt"
    batch_num = 20
    wer = calculate_wer_with_system_prompt(system_prompt, test_data_path, train_data_path, example_dir, result_file, 5, batch_num)
    print(f"WER: {wer}")