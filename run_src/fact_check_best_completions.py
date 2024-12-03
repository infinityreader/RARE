import os
from typing import Set

from eval.safe import colbert_thread_sentence_search_augmented_factuality_eval as safe
from eval.safe import config as safe_config
from safe_common import open_modeling
import json
import os.path
import argparse

answer_dict = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}

def get_quality(fact_check, skip_option=True):
    num_support = 0
    num_not_support = 0
    for jdx in range(len(fact_check['checked_statements'])):
        statement = fact_check['checked_statements'][jdx]
        if skip_option:
            if (jdx > 0) and ('(A)' in statement['sentence'] or '(B)' in statement['sentence'] or '(C)' in statement[
                'sentence'] or '(D)' in statement['sentence']
                    or '(E)' in statement['sentence'] or 'option' in statement['sentence'].lower() or 'choice' in
                    statement['sentence'].lower()):
                continue
        if fact_check['checked_statements'][jdx]['annotation'] == 'Supported':
            num_support += 1
        elif fact_check['checked_statements'][jdx]['annotation'] == 'Not Supported':
            num_not_support += 1
    if num_not_support + num_support == 0:
        import ipdb; ipdb.set_trace()
    rate = num_support / (num_support + num_not_support)
    return rate, len(fact_check['checked_statements'])


def rank_with_fact_check(gen_file_path, eval_task, eval_split, model_name, suffix):
    with open(gen_file_path, "r") as f:
        save_generations = json.load(f)
        num_rollouts = len(save_generations[-1]['model_solutions'])
    fact_check_path = "../save/fact_check_" + eval_task + "_" + eval_split+ "_" + model_name +"_" +str(num_rollouts)+"_"+ suffix +".json"
    if os.path.isfile(fact_check_path):
        with open(fact_check_path, "r") as f:
            all_fact_check = json.load(f)
            num_correct = all_fact_check[-1]['num_correct']
    else:
        all_fact_check = []
        num_correct = 0
    rater_model = open_modeling.Model(
        safe_config.model,
        temperature=safe_config.model_temp,
        max_tokens=safe_config.max_tokens,
    )
    for idx in range(len(save_generations)):
        if idx < len(all_fact_check):
            # accuracy = all_fact_check[idx]['num_correct'] / (idx + 1)
            # print(idx+1, accuracy)
            continue
        print(idx, "/", len(save_generations))
        # import ipdb; ipdb.set_trace()
        gen_obj = save_generations[idx]
        prompt = gen_obj['question']
        row_fact_check = []
        best_rate = 0.0
        best_choice = "F"
        scores = []
        choices_count = {"F": 0}
        for choice in gen_obj["model_choices"]:
            if choice in choices_count.keys():
                choices_count[choice] += 1
            else:
                choices_count[choice] = 1
        for jdx in range(num_rollouts):
            response = gen_obj['model_solutions'][jdx]
            choice = gen_obj['model_choices'][jdx]
            answer = gen_obj['model_answers'][jdx]
            if len(choices_count) > 2:
                fact_check_result = safe.main(prompt, response, rater_model, False)
                row_fact_check.append(fact_check_result)
                rate, length = get_quality(fact_check_result, skip_option=False)
                if rate > best_rate or (rate == best_rate and choices_count[choice] > choices_count[best_choice]):
                    best_rate = rate
                    best_response = response
                    best_answer = answer
                    best_choice = choice
            else:
                fact_check_result = None
                best_rate = 1.0
                rate = 1.0
                best_response = response
                best_answer = answer
                best_choice = choice
                row_fact_check.append(fact_check_result)

            print(jdx, "/ ", num_rollouts)
            scores.append([jdx, rate])
        item = {}
        item["factchecks"] = row_fact_check
        item["best_rate"] = best_rate
        item["best_answer"] = best_answer
        item["best_choice"] = best_choice
        item["best_response"] = best_response
        item["scores"] = scores
        if best_choice == gen_obj["gold_answer"]:
            num_correct += 1
        item["num_correct"] = num_correct
        all_fact_check.append(item)
        print("Accuracy: ", num_correct/len(all_fact_check), len(all_fact_check))
        with open(fact_check_path, "w") as f:
            json.dump(all_fact_check, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval pipeline')
    parser.add_argument('--data_file_path', type=str, default='', help='data file path')
    parser.add_argument('--eval_task', type=str, default='MedQA', help='eval task')
    parser.add_argument('--eval_split', type=str, default='test', help='eval split')
    parser.add_argument('--gen_model_name', type=str, default='llama3_8b', help='gen model name')
    parser.add_argument('--suffix', type=str, default='', help='suffix')
    args = parser.parse_args()
    rank_with_fact_check(args.data_file_path, args.eval_task, args.eval_split, args.gen_model_name, args.suffix)

