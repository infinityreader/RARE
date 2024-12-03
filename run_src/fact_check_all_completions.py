import os
from typing import Set

# from eval.safe import colbert_thread_sentence_search_augmented_factuality_eval as safe
from eval.safe import async_safe as safe
from eval.safe import config as safe_config
from safe_common import open_modeling
import json
import os.path
import argparse
import random
from fuzzywuzzy import fuzz, process
from thefuzz import fuzz
import asyncio

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


def filter_first_sentence(completion):
    sentences = completion.split('.')
    if 'step by step' in sentences[0]:
        filtered = '.'.join(sentences[1:])
        return filtered
    else:
        return completion

def get_best_choices(choice_info, gen_obj, rater_model, num_votes):
    if len(choice_info) == 1:
        best_choice = next(iter(choice_info))
        return best_choice, 1.0, [None]
    else:
        prompt = gen_obj['question']
        best_score = -1
        fact_check_results = []
        for choice in choice_info.keys():
            choice_completions = choice_info[choice]['completions']
            if len(choice_completions) > num_votes:
                check_completions = random.sample(choice_completions, num_votes)
            else:
                check_completions = choice_completions
            num_pass = 0
            for completion in check_completions:
                filtered_completion = filter_first_sentence(completion)
                # fact_check_result = safe.main(prompt, filtered_completion, rater_model, False)
                fact_check_result = asyncio.run(safe.main(prompt, filtered_completion, rater_model, False))
                rate, length = get_quality(fact_check_result, skip_option=False)
                fact_check_result['score'] = rate
                fact_check_result['length'] = length
                if rate == 1:
                    num_pass += 1
                fact_check_results.append(fact_check_result)
                print(choice, rate, "------------------------------")
            if num_pass == len(check_completions) and num_pass > 1:
                score = 2
            elif num_pass >= 1:
                score = 1
            else:
                score = 0
            if score > best_score:
                best_score = score
                best_choice = choice
            elif score == best_score:
                if choice_info[choice]["score"] > choice_info[best_choice]["score"]:
                    best_choice = choice
                    best_score = score
            print(choice, " score: ", score, "------------------------------")
        return best_choice, best_score, fact_check_results


def check_answers_equiv(answer_a: str, answer_b: str):
    score = fuzz.ratio(answer_a, answer_b)
    correct = score >= 90
    return correct


def rank_with_fact_check(gen_file_path, model_name, suffix, num_votes):
    with open(gen_file_path, "r") as f:
        save_generations = json.load(f)
    setting_name = gen_file_path.split("/")[-1].replace(".json","")
    fact_check_path = "../save/fact_check_"  + model_name + "_" + setting_name + "_" + str(num_votes)  +"_"+ suffix +".json"
    if os.path.isfile(fact_check_path):
        with open(fact_check_path, "r") as f:
            all_fact_check = json.load(f)
            # fact_check_result = all_fact_check[0]
            # fact_check_result['num_fact_check_correct'] = 1
            # fact_check_result['num_star_correct'] = 0
            # fact_check_result['num_gen_correct'] = 0
            # all_fact_check = [fact_check_result]
            # with open(fact_check_path, "w") as f:
            #     json.dump(all_fact_check, f)
            # import ipdb; ipdb.set_trace()
            num_correct = all_fact_check[-1]['num_fact_check_correct']
    else:
        all_fact_check = []
        num_correct = 0
    rater_model = open_modeling.Model(
        safe_config.model,
        temperature=safe_config.model_temp,
        max_tokens=safe_config.max_tokens,
    )
    for idx in range(len(save_generations)):
        gen_obj = save_generations[idx]
        num_star_correct = gen_obj['num_correct']
        num_gen_correct = gen_obj['num_gen_correct']
        gold_choice = gen_obj['gold_answer']

        if idx < len(all_fact_check):
            all_fact_check[idx]['gold_choice'] = gold_choice
            print(idx, "/", len(save_generations), " FC acc: ", all_fact_check[idx]['num_fact_check_correct'] / (idx+1), "Star acc: ", num_star_correct / (idx+1),
                  "Gen acc: ", num_gen_correct / (idx+1))
            continue
        choices_info = gen_obj['choices_info']
        # all_choices = gen_obj['model_all_choices']
        # options = gen_obj['options']
        # all_answers = gen_obj['model_all_answers']
        # for jdx in range(len(all_choices)):
        #     choice = all_choices[jdx]
        #     if not isinstance(all_answers[jdx], str):
        #         continue
        #     ans = all_answers[jdx].replace(choice+": ", "")
        #     if ans[-1] == ".":
        #         ans = ans[:-1]
        #     valid_choice = check_answers_equiv(ans, options[choice])
        #     if not valid_choice:
        #         continue
        #     if choice not in choices_info.keys():
        #         choices_info[choice] = [jdx]
        #     else:
        #         choices_info[choice].append(jdx)
        # if len(choices_info) == 0:
        #     import ipdb; ipdb.set_trace()
        best_choice, best_score, fact_check_results = get_best_choices(choices_info, gen_obj, rater_model, num_votes)
        if best_choice == gold_choice:
            num_correct += 1
        item = {
            "best_choice": best_choice,
            "best_score": best_score,
            "fact_check_results": fact_check_results,
            "num_fact_check_correct": num_correct,
            "num_star_correct": num_star_correct,
            "num_gen_correct": num_gen_correct,
            "gold_choice": gold_choice
        }
        all_fact_check.append(item)
        print(idx, "FC acc: ", num_correct/len(all_fact_check), "Score acc: ", num_star_correct/len(all_fact_check), "Frequent acc: ", num_gen_correct/len(all_fact_check))
        with open(fact_check_path, "w") as f:
            json.dump(all_fact_check, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval pipeline')
    parser.add_argument('--data_file_path', type=str, default='', help='data file path')
    parser.add_argument('--rate_model_name', type=str, default='llama3_8b', help='rate model name')
    parser.add_argument('--suffix', type=str, default='', help='suffix')
    parser.add_argument('--num_votes', type=int, default=1, help='number of completions to check each choices')
    args = parser.parse_args()
    rank_with_fact_check(args.data_file_path, args.rate_model_name, args.suffix, args.num_votes)

