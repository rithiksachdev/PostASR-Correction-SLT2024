import os
import argparse
import pandas as pd # type: ignore
from tqdm import tqdm # type: ignore
import time
import random
import anthropic # type: ignore
from baseline_model import calculate_wer_with_system_prompt

client = anthropic.Anthropic()

def get_final_prompt(text):
    """
    Extracts the final prompt from the API response.

    Args:
    text (str): API response containing the prompt within <prompt> tags.

    Returns:
    str: Extracted prompt.
    """
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
    """
    Queries the Claude API with the given prompt.

    Args:
    full_prompt (str): Full prompt for the API call.

    Returns:
    str: Response from the API.
    """
    try:
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
        time.sleep(2)
        with open('api_response.txt', 'a') as f:
            f.write(result + '\n')

        return result
    except Exception as e:
        print(f"Error: {e}")
    return "<prompt>None</prompt>"

def load_initial_prompts(file_path, num_prompts=5):
    """
    Loads initial prompts from a CSV file.

    Args:
    file_path (str): Path to the CSV file.
    num_prompts (int): Number of initial prompts to load.

    Returns:
    DataFrame: DataFrame containing the loaded prompts and scores.
    """
    df = pd.read_csv(file_path)
    if 'prompt' not in df.columns or 'score' not in df.columns:
        raise ValueError("CSV file must contain 'prompt' and 'score' columns")
    return df.head(num_prompts)

def dummy_calculate_wer_with_system_prompt(prompt, test_data_path, train_data_path, example_dir, result_file, n, batch):
    """
    Dummy function to replace calculate_wer_with_system_prompt.

    Args:
    prompt (str): Prompt to evaluate.
    test_data_path (str): Path to the test data JSON file.
    train_data_path (str): Path to the training data JSON file.
    example_dir (str): Directory containing example files.
    result_file (str): Path to the result file.
    n (int): Number of examples to retrieve.
    batch (int): Number of prompts to process in each batch.

    Returns:
    float: Randomly generated WER for testing purposes.
    """
    result_file = open(result_file, 'w+')
    result_file.write("temp")
    return random.uniform(0.03, 0.15)

def evaluate_prompts(generation, prompts_df, csv_file):
    """
    Evaluates prompts and updates scores.

    Args:
    generation (int): Current generation number.
    prompts_df (DataFrame): DataFrame containing prompts and scores.
    csv_file (str): Path to the CSV file with initial prompts.

    Returns:
    dict: Dictionary of evaluated prompts and their scores.
    DataFrame: Updated DataFrame with evaluated scores.
    """
    generation_file = f"generation_{generation}.csv"
    evaluated_prompts = {}

    if os.path.exists(generation_file):
        # Load existing scores
        existing_df = pd.read_csv(generation_file)
        for index, row in existing_df.iterrows():
            if pd.isna(row['score']):
                prompt = row['prompt']
                evaluated_prompts[prompt] = calculate_wer_with_system_prompt(prompt, test_data_path, train_data_path, example_dir, result_file, n, 20)
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
    """
    Selects the top k prompts based on their scores.

    Args:
    prompts (list): List of prompts.
    evaluated_prompts (dict): Dictionary of evaluated prompts and their scores.
    k (int): Number of top prompts to select.

    Returns:
    list: List of top k prompts.
    """
    sorted_prompts = sorted(prompts, key=lambda x: evaluated_prompts[x])
    return sorted_prompts[:k]

def generate_new_prompt(template, prompt1, prompt2):
    """
    Generates a new prompt by combining two parent prompts.

    Args:
    template (str): Template for generating new prompts.
    prompt1 (str): First parent prompt.
    prompt2 (str): Second parent prompt.

    Returns:
    str: Generated new prompt.
    """
    request_content = template.replace("<prompt1>", prompt1).replace("<prompt2>", prompt2)
    child_prompt = llm_query(request_content)
    return get_final_prompt(child_prompt)

def write_population(step, population, evaluated_prompts, output_dir, prefix):
    """
    Writes the population of prompts and their scores to a file.

    Args:
    step (int): Current generation step.
    population (list): List of prompts in the population.
    evaluated_prompts (dict): Dictionary of evaluated prompts and their scores.
    output_dir (str): Directory to save the output file.
    prefix (str): Prefix for the output file name.
    """
    with open(os.path.join(output_dir, f"{prefix}_step{step}.txt"), "w") as wf:
        for prompt in population:
            score_str = str(round(evaluated_prompts[prompt], 4))
            wf.write(f"{prompt}\t{score_str}\n\n")

def main(prompt_file, final_solution, ga_template, num_initial_prompts, population_size, num_generations, top_k, output_dir):
    """
    Main function to run the genetic algorithm for prompt optimization.

    Args:
    prompt_file (str): Path to the file containing initial prompts.
    final_solution (str): Path to save the final solutions.
    ga_template (str): Template for generating new prompts.
    num_initial_prompts (int): Number of initial prompts to load.
    population_size (int): Size of the population in each generation.
    num_generations (int): Number of generations to run the optimization for.
    top_k (int): Number of top prompts to select.
    output_dir (str): Directory to save output files.

    Returns:
    None
    """

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
    
    ga_template = """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.
1. """
    
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Prompt Optimization")
    parser.add_argument("--prompt_file", type=str, default="prompts.csv", help="Path to the file containing initial prompts")
    parser.add_argument("--final_solution", type=str, default="results.csv", help="Path to save the final solutions")
    parser.add_argument("--num_initial_prompts", type=int, default=5, help="Number of initial prompts to load")
    parser.add_argument("--population_size", type=int, default=5, help="Size of the population in each generation")
    parser.add_argument("--num_generations", type=int, default=3, help="Number of generations to run the optimization for")
    parser.add_argument("--top_k", type=int, default=4, help="Number of top prompts to select")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output files")
    
    args = parser.parse_args()
    main(
        args.prompt_file,
        args.final_solution,
        args.ga_template,
        args.num_initial_prompts,
        args.population_size,
        args.num_generations,
        args.top_k,
        args.output_dir
    )