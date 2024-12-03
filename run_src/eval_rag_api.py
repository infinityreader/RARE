import os
import json
import argparse
from eval_util import *
from tqdm import tqdm
import gc
import re
import logging
import sys
sys.path.append(".")
from models.IO_System import IO_System
import torch
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import GenerationConfig, set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fuzzywuzzy import fuzz
import requests

usmle_in_context_examples = [
'''
Question: 
A 46-year-old woman comes to the physician because of a 2-week history of diplopia and ocular pain when reading the newspaper. She also has a 3-month history of amenorrhea, hot flashes, and increased sweating. She reports that she has been overweight all her adult life and is happy to have lost 6.8 kg (15 lb) of weight in the past 2 months. Her pulse is 110/min, and blood pressure is 148/98 mm Hg. Physical examination shows moist palms and a nontender thyroid gland that is enlarged to two times its normal size. Ophthalmologic examination shows prominence of the globes of the eyes, bilateral lid retraction, conjunctival injection, and an inability to converge the eyes. There is no pain on movement of the extraocular muscles. Visual acuity is 20/20 bilaterally. Neurologic examination shows a fine resting tremor of the hands. Deep tendon reflexes are 3+ with a shortened relaxation phase. Which of the following is the most likely cause of this patient's ocular complaints? 
Choices:
(A) Granulomatous inflammation of the cavernous sinus
(B) Abnormal communication between the cavernous sinus and the internal carotid artery
(C) Glycosaminoglycan accumulation in the orbit
(D) Bacterial infection of the orbital contents
(E) Sympathetic hyperactivity of levator palpebrae superioris
Answer:
The correct answer is [C] Glycosaminoglycan accumulation in the orbit.
''',
'''
Question: 
A 1-year-old boy presents to the emergency department with weakness and a change in his behavior. His parents state that they first noticed the change in his behavior this morning and it has been getting worse. They noticed the patient was initially weak in his upper body and arms, but now he won’t move his legs with as much strength or vigor as he used to. Physical exam is notable for bilateral ptosis with a sluggish pupillary response, a very weak sucking and gag reflex, and shallow respirations. The patient is currently drooling and his diaper is dry. The parents state he has not had a bowel movement in over 1 day. Which of the following is the pathophysiology of this patient’s condition? A: Antibodies against postsynaptic nicotinic cholinergic ion channels, B: Autoantibodies against the presynaptic voltage-gated calcium channels, C: Autoimmune demyelination of peripheral nerves, D: Blockade of presynaptic acetylcholine release at the neuromuscular junction, E: Lower motor neuron destruction in the anterior horn
Choices:
(A) Antibodies against postsynaptic nicotinic cholinergic ion channels
(B) Autoantibodies against the presynaptic voltage-gated calcium channels
(C) Autoimmune demyelination of peripheral nerves
(D) Blockade of presynaptic acetylcholine release at the neuromuscular junction
(E) Lower motor neuron destruction in the anterior horn
Answer:
The correct answer is [D] Blockade of presynaptic acetylcholine release at the neuromuscular junction.
''',
'''
Question: 
A 9-month-old female is brought to the emergency department after experiencing a seizure. She was born at home and was normal at birth according to her parents. Since then, they have noticed that she does not appear to be achieving developmental milestones as quickly as her siblings, and often appears lethargic. Physical exam reveals microcephaly, very light pigmentation (as compared to her family), and a "musty" body odor. The varied manifestations of this disease can most likely be attributed to which of the following genetic principles?
Choices:
(A) Anticipation
(B) Incomplete penetrance 
(C) Multiple gene mutations 
(D) Pleiotropy 
(E) Variable expressivity
Answer:
The correct answer is [D] Pleiotropy.
''',
'''
Question: 
A 23-year-old man comes to the physician for evaluation of decreased hearing, dizziness, and ringing in his right ear for the past 6 months. Physical examination shows multiple soft, yellow plaques and papules on his arms, chest, and back. There is sensorineural hearing loss and weakness of facial muscles bilaterally. His gait is unsteady. An MRI of the brain shows a 3-cm mass near the right internal auditory meatus and a 2-cm mass at the left cerebellopontine angle. The abnormal cells in these masses are most likely derived from which of the following embryological structures?
Choices:
(A) Neural tube 
(B) Surface ectoderm 
(C) Neural crest 
(D) Notochord 
(E) Mesoderm
Answer:
The correct answer is [C] Neural crest.
''',
'''
Question: 
A 62-year-old woman comes to the physician because of coughing and fatigue during the past 2 years. In the morning, the cough is productive of white phlegm. She becomes short of breath walking up a flight of stairs. She has hypertension and hyperlipidemia. She has recently retired from working as a nurse at a homeless shelter. She has smoked 1 pack of cigarettes daily for 40 years. Current medications include ramipril and fenofibrate. Her temperature is 36.5°C (97.7°F), respirations are 24/min, pulse is 85/min, and blood pressure is 140/90 mm Hg. Scattered wheezing and rhonchi are heard throughout both lung fields. There are no murmurs, rubs, or gallops but heart sounds are distant. Which of the following is the most likely underlying cause of this patient's symptoms? A: Chronic decrease in pulmonary compliance, B: Local accumulation of kinins, C: Mycobacterial invasion of pulmonary parenchyma, D: Progressive obstruction of expiratory airflow, E: Incremental loss of functional residual capacity
Choices:
(A) Chronic decrease in pulmonary compliance 
(B) Local accumulation of kinins
(C) Mycobacterial invasion of pulmonary parenchyma 
(D) Progressive obstruction of expiratory airflow 
(E) Incremental loss of functional residual capacity
Answer:
The correct answer is [D] Progressive obstruction of expiratory airflow.
''',
]

usmle_template = '''
Question:
{question}
Choices:
{answer_choices}
Relevant documents:
{documents}
Answer:
The correct answer is
'''.strip()
# sys_message = "\nImagine you are a medical doctor answering questions on a medical entrance test. For each of the following multiple-choice questions, please select the correct answer.\n"
sys_message = "\nYou are a helpful assistant. For each of the following multiple-choice questions, please select the correct answer.\n"

class Evaluator():
    data_path_dict = {
        'usmle': './data/usmle_public/questions/US'
    }

    usmle_answer_dict = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4
    }

    def __init__(self, args, model=None, tokenizer=None) -> None:
        # Create and configure logger
        os.makedirs(args.logging_dir, exist_ok=True)
        logging.basicConfig(
            filename="{}logging.log".format(args.logging_dir),
            format='%(asctime)s %(message)s',
            filemode='a',
            level=logging.INFO
        )
        logging.info("\n\n\n")
        logging.info("=============== Evaluation ===============")
        logging.info('Initiating...')
        self.args = args
        self.eval_home_dir = args.eval_home_dir
        self.eval_debug = args.eval_debug

        # set device config
        # max_memory_dict = {}
        # for device_id in range(torch.cuda.device_count()):
        #     max_memory_dict[device_id] = "{}GB".format(vars(args)['memory_{}'.format(device_id)])
        # logging.info('=> max_memory_dict:\n' + str(max_memory_dict))

        self.io = IO_System(args, None, args.model_ckpt)

        # get task info
        self.eval_data_path = args.eval_data_path
        self.eval_task = args.eval_task
        if args.eval_split == 'all':
            self.eval_split = ['train', 'dev', 'test']
        else:
            self.eval_split = [args.eval_split]

        # get running setings
        self.eval_sample_size = args.eval_sample_size
        self.eval_batch_size = args.eval_batch_size
        self.eval_save = args.eval_save
        self.eval_save_path = args.eval_save_path
        self.max_new_tokens = args.max_new_tokens
        self.eval_task = args.eval_task
        self.num_shot = args.num_shot


    def prep_data(self, task_name, split):
        all_data = {}
        for subset in split:
            tmp_data_path = '{}/{}/{}.json'.format(self.eval_data_path, self.eval_task, subset)
            print('=> Loading data from:', tmp_data_path)
            all_data[subset] = load_json(tmp_data_path)
            if self.eval_sample_size != -1:
                all_data[subset] = all_data[subset][:self.eval_sample_size]
            for idx, dp in enumerate(all_data[subset]):
                dp['true_answ'] = self.usmle_answer_dict[dp['answer']]
                dp['prompt'] = self.make_prompt(dp, num_shot=self.num_shot)
        return all_data

    def make_prompt(self, dp, num_shot=0):
        answer_choices = ''
        for choice in dp['options']:
            answer_choices += '({}) {}\n'.format(choice, dp['options'][choice])
        in_context = ''
        for idx in range(num_shot):
            in_context = in_context + usmle_in_context_examples[idx]
        request_query = "http://172.16.34.22:6000/api/search?query=" + dp['question'] + "&k=" + str(3)
        x = requests.get(request_query)
        jsobj = json.loads(x.text)
        document = ""
        for idx in range(len(jsobj['topk'])):
            document = document + jsobj['topk'][idx]['text'] + "\n"
        prompt = usmle_template.format(
            question=dp['question'],
            answer_choices=answer_choices,
            documents= document
        )
        prompt = sys_message + in_context + prompt
        return prompt

    def parse_raw_result(self, raw_result):
        pred_text = raw_result.strip()
        sq_idx = pred_text.find("[")
        cl_idx = pred_text.find("]")
        par_idx = pred_text.find("(")
        then_idx = pred_text.find(")")
        if sq_idx != -1 and cl_idx != -1:
            answer = pred_text[sq_idx + 1:cl_idx]
            if answer.upper() in self.usmle_answer_dict.keys():
                return self.usmle_answer_dict[answer]
        if then_idx != -1:
            answer = pred_text[then_idx-1:then_idx]
            if answer.upper() in self.usmle_answer_dict.keys():
                return self.usmle_answer_dict[answer.upper()]
        print(raw_result)
        return -1

    def parse_raw_result_with_options(self, raw_result, options):
        pred_text = raw_result.strip()
        choice = self.check_valid_answer(pred_text, options)
        return self.usmle_answer_dict[choice.upper()]

    def check_valid_answer(self, answer: str, options: dict):
        highest_score = -1
        for char, opt in options.items():
            choice = char +" "+ opt
            score = fuzz.ratio(answer.lower(), choice.lower())
            if score > highest_score:
                highest_score = score
                highest_option = char
        return highest_option


    def eval_one_split(self, data):
        if self.eval_save and os.path.exists(self.eval_save_path):
            with open(self.eval_save_path, "r") as f:
                save_obj = json.load(f)
                all_target = save_obj["all_target"]
                all_pred_answ = save_obj["all_pred"]
                all_prompts = save_obj["all_prompts"]
                raw_results = save_obj["all_responses"]
        else:
            raw_results = []
            all_pred_answ = []
            all_target = []
            all_prompts = []
        counter = 0
        pbar = tqdm(range(0, len(data), self.eval_batch_size))
        for i in pbar:
            if i < len(raw_results):
                continue
            batch_data = data[i: i + self.eval_batch_size]
            batch_prompts = [dp['prompt'] for dp in batch_data]
            batch_answ = [dp['true_answ'] for dp in batch_data]
            batch_options = [dp['options'] for dp in batch_data]
            prompt = batch_prompts[0]
            response = self.io.generate(prompt, max_tokens=self.max_new_tokens, num_return=1, stop_tokens=["Question"])
            lm_outputs = response
            # pred_answ = [self.parse_raw_result(r) for r in lm_outputs]#, dp in zip(lm_outputs, batch_data)]
            pred_answ = [self.parse_raw_result_with_options(r, o) for r, o in zip(lm_outputs, batch_options)]  # , dp in zip(lm_outputs, batch_data)]
            # import ipdb; ipdb.set_trace()
            raw_results.extend(lm_outputs)
            all_pred_answ.extend(pred_answ)
            all_target.extend(batch_answ)
            all_prompts.extend(batch_prompts)
            ### Cleaning up ###
            # input_ids.to('cpu')
            # del input_ids
            gc.collect()
            torch.cuda.empty_cache()

            ### Debug ###
            counter += len(pred_answ)
            if self.eval_debug and counter > self.eval_sample_size:
                break

            # Evaluate metrics after each batch and update progress bar
            eval_result = self.eval_metrics(all_target, all_pred_answ)
            tmp_data = {
                "all": eval_result['acc'],
                "all_test_size": eval_result['test_size'],
            }
            pbar.set_postfix({"Current Score": tmp_data}, refresh=True)
            if self.eval_save:
                save_path = self.eval_save_path
                save_obj = {}
                save_obj["all_target"] = all_target
                save_obj["all_pred"] = all_pred_answ
                save_obj["all_prompts"] = all_prompts
                save_obj["all_responses"] = raw_results
                with open(save_path, "w") as f:
                    json.dump(save_obj, f)

        # Evaluation Metrics
        eval_result = self.eval_metrics(all_target, all_pred_answ)
        return eval_result
    def _eval(self):
        all_data = self.prep_data(self.eval_task, self.eval_split)
        all_results = {}
        for split in all_data:
            eval_result = self.eval_one_split(all_data[split])
            all_results[split] = eval_result


        # logging.info(' => Running configurations: {}'.format(self.args))
        logging.info(' => Evaluation Results: {}'.format(all_results))
        return all_results

    def eval_metrics(self, true_answ, pred_answ, metrics=['acc']):
        if len(true_answ) == 0:
            return {
                'acc': 0,
                'invalid': 0,
                'test_size': 0
            }
        eval_result = {}
        for m in metrics:
            if m == 'acc':
                eval_result[m] = round(accuracy_score(true_answ, pred_answ), 5)
            else:
                # TODO: add more eval metrics here
                raise NotImplementedError

        invalid_prediction = sum([i == -1 for i in pred_answ])
        eval_result['invalid'] = round(invalid_prediction / len(pred_answ), 5)
        eval_result['test_size'] = len(true_answ)
        return eval_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation pipeline')
    parser.add_argument('--model_ckpt', type=str, default='gpt-35-turbo',
                        help='specific model to eval')
    parser.add_argument('--eval_adapter_name', type=str, default='')  # by default, there is no adapter
    parser.add_argument('--eval_debug', action="store_true")
    parser.add_argument('--eval_sample_size', type=int, default=10,
                        help='number of samples to run for debug runs, default -1 mean all samples')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='inference batch size, default 8')
    parser.add_argument('--eval_task', type=str, default='usmle', help='specific task to eval')
    parser.add_argument('--eval_split', type=str, default='test', help='specific dataset split to eval')
    parser.add_argument('--eval_home_dir', type=str, default='./', help='home directory')
    parser.add_argument('--eval_w_deepspeed', action="store_true",
                        help='whether or not running in deepspeed environment')
    parser.add_argument('--eval_data_path', type=str, default='/data/41_text_qa_cleaned/evaluation')
    parser.add_argument('--logging_dir', type=str, default='./log/')
    parser.add_argument('--eval_save_path', type=str, default='results', help='path to save generation responses')
    parser.add_argument('--eval_save', type=bool, default=True, help='whether to save generation responses')
    # parser.add_argument('--provide_reasoning', type=bool, default=False, help='whether to generate reasoning in the response')
    parser.add_argument('--retry_invalid_samples', type=bool, default=False,
                        help='whether to retry with invalid samples')
    parser.add_argument('--previous_generation_path', type=str, default=None,
                        help='the path to json object save the previous generation')
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

    parser.add_argument("--enable_chat_template", type=bool, default=False)
    # set device max memory
    for device_id in range(8):
        parser.add_argument('--memory_{}'.format(device_id), type=int, default=15,
                            help='specify max memory requirements for gpu '.format(device_id))
    args = parser.parse_args()

    set_seed(1)
    EV = Evaluator(args)
    print(EV._eval())
