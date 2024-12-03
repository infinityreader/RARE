import os
import json
import argparse
from eval_util import *
from tqdm import tqdm
import gc
import re
import logging
import sys
import evaluate
sys.path.append(".")
from models.IO_System import IO_System
import torch
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import GenerationConfig, set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score
from alignscore import AlignScore

sys_message = "Imagine you are a doctor, please answer the medical questions based on the patient's description.\n"

class Evaluator():
    def __init__(self, args, model=None, tokenizer=None) -> None:
        self.args = args

        self.io = IO_System(args, None, args.model_ckpt)

        # get task info
        self.eval_data_path = args.eval_data_path
        self.eval_task = args.eval_task

        # get running setings
        self.eval_sample_size = args.eval_sample_size
        self.eval_batch_size = args.eval_batch_size
        self.eval_save = args.eval_save
        self.eval_save_path = args.eval_save_path
        self.max_new_tokens = args.max_new_tokens
        self.eval_task = args.eval_task
        self.num_shot = args.num_shot
        self.eval_steps = args.eval_steps


    def prep_data(self):
        tmp_data_path = self.eval_data_path
        print('=> Loading data from:', tmp_data_path)
        eval_data = load_json(tmp_data_path)
        if self.eval_sample_size != -1:
            eval_data = eval_data[:self.eval_sample_size]
        for idx, dp in enumerate(eval_data):
            dp['prompt'] = self.make_prompt(dp)
        return eval_data

    def make_prompt(self, dp):
        prompt = sys_message + "Question: " + dp['problem']
        return prompt

    def eval_one_split(self, data):
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load("bleu")
        # self.mauve = evaluate.load("mauve")
        # self.align_scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0', ckpt_path='/home/htran/generation/med_preferences/AlignScore/checkpoints/AlignScore-large.ckpt',
        #                     evaluation_mode='nli_sp')

        if self.eval_save and os.path.exists(self.eval_save_path):
            with open(self.eval_save_path, "r") as f:
                save_obj = json.load(f)
                all_target = save_obj["all_target"]
                all_prompts = save_obj["all_prompts"]
                raw_results = save_obj["all_responses"]
        else:
            raw_results = []
            all_target = []
            all_prompts = []
        pbar = tqdm(range(0, len(data), self.eval_batch_size))
        for i in pbar:
            if i < len(raw_results):
                continue
            batch_data = data[i: i + self.eval_batch_size]
            batch_prompts = [dp['prompt'] for dp in batch_data]
            prompt = batch_prompts[0]
            response = self.io.generate(prompt, max_tokens=self.max_new_tokens, num_return=1, stop_tokens=["Question"])
            lm_outputs = response
            raw_results.extend(lm_outputs)
            all_prompts.extend(batch_prompts)
            all_target.append(batch_data[0]['answer'])

            # Evaluate metrics after each batch and update progress bar
            if (i+1) % self.eval_steps == 0:
                eval_result = self.eval_metrics(all_target, raw_results)
                tmp_data = {
                    "bleu": eval_result['bleu'],
                    "rouge": eval_result['rouge'],
                    # "mauve": eval_result['mauve'],
                    # "alignscore": eval_result['alignscore'],
                    "test_size": eval_result['test_size'],
                }
                pbar.set_postfix({"Current Score": tmp_data}, refresh=True)
            if self.eval_save:
                save_path = self.eval_save_path
                save_obj = {}
                save_obj["all_target"] = all_target
                save_obj["all_prompts"] = all_prompts
                save_obj["all_responses"] = raw_results
                with open(save_path, "w") as f:
                    json.dump(save_obj, f)

        # Evaluation Metrics
        eval_result = self.eval_metrics(all_target, raw_results)
        return eval_result

    def _eval(self):
        eval_data = self.prep_data()
        eval_result = self.eval_one_split(eval_data)
        logging.info(' => Evaluation Results: {}'.format(eval_result))
        return eval_result

    def eval_metrics(self, true_answ, pred_answ):
        eval_result = {}
        rouge_results = self.rouge.compute(predictions=pred_answ, references=true_answ)
        bleu_results = self.bleu.compute(predictions=pred_answ, references=true_answ)
        eval_result["bleu"] = bleu_results["precisions"][0]
        eval_result["rouge"] = rouge_results["rouge1"]
        # mauve_results = self.mauve.compute(predictions=pred_answ, references=true_answ)
        # eval_result["mauve"] = mauve_results.mauve
        # align_score = self.align_scorer.score(contexts=true_answ, claims=pred_answ)
        # eval_result["alignscore"] = sum(align_score)/len(align_score)
        eval_result["test_size"] = len(true_answ)
        return eval_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation pipeline')
    parser.add_argument('--model_ckpt', type=str, default='gpt-35-turbo',
                        help='specific model to eval')
    parser.add_argument('--eval_sample_size', type=int, default=10,
                        help='number of samples to run for debug runs, default -1 mean all samples')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='inference batch size, default 8')
    parser.add_argument('--eval_steps', type=int, default=1, help='number of steps to eval')
    parser.add_argument('--eval_task', type=str, default='usmle', help='specific task to eval')
    parser.add_argument('--eval_split', type=str, default='test', help='specific dataset split to eval')
    parser.add_argument('--eval_data_path', type=str, default='/data/41_text_qa_cleaned/evaluation')
    parser.add_argument('--eval_save_path', type=str, default='results', help='path to save generation responses')
    parser.add_argument('--eval_save', type=bool, default=True, help='whether to save generation responses')
    parser.add_argument('--max_new_tokens', type=int, default=10,
                        help='max new tokens')
    parser.add_argument('--num_shot', type=int, default=0,
                        help='number of in-context examples')
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--top_k", type=int, default=40, help="top_k")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p")
    allowed_apis = ["together", "huggingface", "llama", "vllm", "debug", "gpt3.5-turbo"]
    parser.add_argument(
        "--api", type=str, choices=allowed_apis, default="gpt3.5-turbo", help=f"API to use: Choose from {allowed_apis}."
    )
    args = parser.parse_args()

    set_seed(42)
    EV = Evaluator(args)
    print(EV._eval())
