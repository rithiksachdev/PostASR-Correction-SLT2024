import os
import json
import pandas as pd
from tqdm import tqdm
import time
import random
import anthropic
from baseline_model2 import calculate_wer_with_system_prompt

client = anthropic.Anthropic()

def get_final_prompt(text):
    parts = text.split("<prompt>")
    if len(parts) > 1:
        prompt = parts[-1].split("</prompt>")[0]
        prompt = prompt.strip()
        return prompt
    else:
        if text.startswith("\"") and text.endswith("\""):
            text = text[1:-1]
        return text

def llm_query(full_prompt):
    try:
        print("-----Prompt-------")
        print(full_prompt)
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=650,
            temperature=0,
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
        result = response.content[0].text
        print("----RESPONSE---")
        print(result)
        time.sleep(2)
        with open('api_response.txt', 'a') as f:
            f.write(result + '\n')

        return result
    except Exception as e:
        print(f"Error: {e}")
    return "<prompt>None</prompt>"

def load_initial_prompts(file_path, num_prompts=5):
    df = pd.read_csv(file_path)
    if 'prompt' not in df.columns or 'score' not in df.columns:
        raise ValueError("CSV file must contain 'prompt' and 'score' columns")
    return df.head(num_prompts)

def dummy_calculate_wer_with_system_prompt(prompt, test_data_path, train_data_path, example_dir, result_file, n, batch):
    """Dummy function to replace calculate_wer_with_system_prompt"""
    result_file = open(result_file, 'w+')
    result_file.write("temp")
    return random.uniform(0.03, 0.15)

def evaluate_prompts(generation, prompts_df, csv_file):
    generation_file = f"generation_{generation}.csv"
    evaluated_prompts = {}

    if os.path.exists(generation_file):
        # Load existing scores
        
        existing_df = pd.read_csv(generation_file)
        print(existing_df)
        for index, row in existing_df.iterrows():
            if(row['score']==None):
                print("----NONE----")
                print(row['prompt'])
                evaluated_prompts[row['prompt']] = calculate_wer_with_system_prompt(prompt, test_data_path, train_data_path, example_dir, result_file, n, 20)
            else:
                evaluated_prompts[row['prompt']] = row['score']
        return evaluated_prompts, existing_df

    test_data_path = "./HyPoradise-v0/test/test_chime4.json"
    train_data_path = "./HyPoradise-v0/train/train_chime4.json"
    example_dir = "Hyporadise-icl/examples/knn/"
    n = 5

    for index, row in prompts_df.iterrows():
        prompt = row['prompt']
        result_file = f"knn_result{index+(generation*5)}.txt"
        if pd.isna(row['score']):
            score = calculate_wer_with_system_prompt(prompt, test_data_path, train_data_path, example_dir, result_file, n, 20)
            evaluated_prompts[prompt] = score
            prompts_df.at[index, 'score'] = score
        else:
            evaluated_prompts[prompt] = row['score']

    prompts_df.to_csv(generation_file, index=False)
    return evaluated_prompts, prompts_df

def select_top_k(prompts, evaluated_prompts, k=5):
    sorted_prompts = sorted(prompts, key=lambda x: evaluated_prompts[x])
    return sorted_prompts[:k]

def generate_new_prompt(template, prompt1, prompt2):
    request_content = template.replace("<prompt1>", prompt1).replace("<prompt2>", prompt2)
    child_prompt = llm_query(request_content)
    return get_final_prompt(child_prompt)

def write_population(step, population, evaluated_prompts, output_dir, prefix):
    with open(os.path.join(output_dir, f"{prefix}_step{step}.txt"), "w") as wf:
        for prompt in population:
            score_str = str(round(evaluated_prompts[prompt], 4))
            wf.write(f"{prompt}\t{score_str}\n\n")

def main():
    prompt_file = "prompts.csv"
    final_solution = "results.csv"
    ga_template = """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """
    
    num_initial_prompts = 5
    population_size = 5
    num_generations = 3
    top_k = 4
    output_dir = "./output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    initial_prompts_df = load_initial_prompts(prompt_file, num_initial_prompts)
    evaluated_prompts, updated_prompts_df = evaluate_prompts(0, initial_prompts_df, prompt_file)
    
    for generation in range(num_generations):
        print(f"Generation {generation + 1}")
        
        top_prompts = select_top_k(initial_prompts_df['prompt'].tolist(), evaluated_prompts, top_k)
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(top_prompts, 2)
            new_prompt = generate_new_prompt(ga_template, parent1, parent2)
            new_population.append(new_prompt)
        
        new_prompts_df = pd.DataFrame(new_population, columns=['prompt'])
        new_prompts_df['score'] = None
        new_evaluations, new_prompts_df = evaluate_prompts(generation + 1, new_prompts_df, prompt_file)
        
        updated_prompts_df = pd.concat([updated_prompts_df, new_prompts_df])
        evaluated_prompts.update(new_evaluations)
        
        # Select the top K prompts from the new (n=pop) genetic modified prompts and initial prompts
        initial_prompts_df = updated_prompts_df.loc[updated_prompts_df['prompt'].isin(select_top_k(list(evaluated_prompts.keys()), evaluated_prompts, top_k))]
        
        write_population(generation, initial_prompts_df['prompt'].tolist(), evaluated_prompts, output_dir, "pop")

    final_top_prompts = select_top_k(list(evaluated_prompts.keys()), evaluated_prompts, top_k)
    with open(os.path.join(output_dir, "final_prompts.txt"), "w") as f:
        for prompt in final_top_prompts:
            f.write(f"{prompt}\t{evaluated_prompts[prompt]}\n")
    
    # Save the updated prompts with scores back to CSV
    updated_prompts_df.to_csv(final_solution, index=False)

    print("Optimization completed. Final prompts saved to final_prompts.txt")

if __name__ == "__main__":
    main()
