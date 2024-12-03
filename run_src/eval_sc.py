import os
import json
import argparse
from eval_util import *
from tqdm import tqdm
import gc
import re
import logging
from thefuzz import fuzz
import torch
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import GenerationConfig, set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score
import itertools
from collections import Counter
import random


usmle_in_context_examples = [
'''
**Question**: A 46-year-old woman comes to the physician because of a 2-week history of diplopia and ocular pain when reading the newspaper. She also has a 3-month history of amenorrhea, hot flashes, and increased sweating. She reports that she has been overweight all her adult life and is happy to have lost 6.8 kg (15 lb) of weight in the past 2 months. Her pulse is 110/min, and blood pressure is 148/98 mm Hg. Physical examination shows moist palms and a nontender thyroid gland that is enlarged to two times its normal size. Ophthalmologic examination shows prominence of the globes of the eyes, bilateral lid retraction, conjunctival injection, and an inability to converge the eyes. There is no pain on movement of the extraocular muscles. Visual acuity is 20/20 bilaterally. Neurologic examination shows a fine resting tremor of the hands. Deep tendon reflexes are 3+ with a shortened relaxation phase. Which of the following is the most likely cause of this patient's ocular complaints? 
**Choices**:
(A) Granulomatous inflammation of the cavernous sinus
(B) Abnormal communication between the cavernous sinus and the internal carotid artery
(C) Glycosaminoglycan accumulation in the orbit
(D) Bacterial infection of the orbital contents
(E) Sympathetic hyperactivity of levator palpebrae superioris
**Answer**:
The answer is (C) Glycosaminoglycan accumulation in the orbit.
''',
'''
**Question**: A 1-year-old boy presents to the emergency department with weakness and a change in his behavior. His parents state that they first noticed the change in his behavior this morning and it has been getting worse. They noticed the patient was initially weak in his upper body and arms, but now he won’t move his legs with as much strength or vigor as he used to. Physical exam is notable for bilateral ptosis with a sluggish pupillary response, a very weak sucking and gag reflex, and shallow respirations. The patient is currently drooling and his diaper is dry. The parents state he has not had a bowel movement in over 1 day. Which of the following is the pathophysiology of this patient’s condition? A: Antibodies against postsynaptic nicotinic cholinergic ion channels, B: Autoantibodies against the presynaptic voltage-gated calcium channels, C: Autoimmune demyelination of peripheral nerves, D: Blockade of presynaptic acetylcholine release at the neuromuscular junction, E: Lower motor neuron destruction in the anterior horn
**Choices**:
(A) Antibodies against postsynaptic nicotinic cholinergic ion channels
(B) Autoantibodies against the presynaptic voltage-gated calcium channels
(C) Autoimmune demyelination of peripheral nerves
(D) Blockade of presynaptic acetylcholine release at the neuromuscular junction
(E) Lower motor neuron destruction in the anterior horn
**Answer**:
The answer is (D) Blockade of presynaptic acetylcholine release at the neuromuscular junction.
''',
'''
**Question**: A 9-month-old female is brought to the emergency department after experiencing a seizure. She was born at home and was normal at birth according to her parents. Since then, they have noticed that she does not appear to be achieving developmental milestones as quickly as her siblings, and often appears lethargic. Physical exam reveals microcephaly, very light pigmentation (as compared to her family), and a "musty" body odor. The varied manifestations of this disease can most likely be attributed to which of the following genetic principles?
**Choices**:
(A) Anticipation
(B) Incomplete penetrance 
(C) Multiple gene mutations 
(D) Pleiotropy 
(E) Variable expressivity
**Answer**:
The answer is (D) Pleiotropy.
''',
'''
**Question**: A 23-year-old man comes to the physician for evaluation of decreased hearing, dizziness, and ringing in his right ear for the past 6 months. Physical examination shows multiple soft, yellow plaques and papules on his arms, chest, and back. There is sensorineural hearing loss and weakness of facial muscles bilaterally. His gait is unsteady. An MRI of the brain shows a 3-cm mass near the right internal auditory meatus and a 2-cm mass at the left cerebellopontine angle. The abnormal cells in these masses are most likely derived from which of the following embryological structures?
**Choices**:
(A) Neural tube 
(B) Surface ectoderm 
(C) Neural crest 
(D) Notochord 
(E) Mesoderm
**Answer**:
The answer is (C) Neural crest.
''',
'''
**Question**: A 62-year-old woman comes to the physician because of coughing and fatigue during the past 2 years. In the morning, the cough is productive of white phlegm. She becomes short of breath walking up a flight of stairs. She has hypertension and hyperlipidemia. She has recently retired from working as a nurse at a homeless shelter. She has smoked 1 pack of cigarettes daily for 40 years. Current medications include ramipril and fenofibrate. Her temperature is 36.5°C (97.7°F), respirations are 24/min, pulse is 85/min, and blood pressure is 140/90 mm Hg. Scattered wheezing and rhonchi are heard throughout both lung fields. There are no murmurs, rubs, or gallops but heart sounds are distant. Which of the following is the most likely underlying cause of this patient's symptoms? A: Chronic decrease in pulmonary compliance, B: Local accumulation of kinins, C: Mycobacterial invasion of pulmonary parenchyma, D: Progressive obstruction of expiratory airflow, E: Incremental loss of functional residual capacity
**Choices**:
(A) Chronic decrease in pulmonary compliance 
(B) Local accumulation of kinins
(C) Mycobacterial invasion of pulmonary parenchyma 
(D) Progressive obstruction of expiratory airflow 
(E) Incremental loss of functional residual capacity
**Answer**:
The answer is (D) Progressive obstruction of expiratory airflow.
''',
]


siqa_in_context_examples =[
'''
Cameron decided to have a barbecue and gathered her friends together. How would Others feel as a result?
(A) like attending 
(B) like staying home
(C) a good friend to have
Answer: The answer is [A] like attending.
''',
'''
Jan needed to give out jobs for an upcoming project at work. What will Others want to do next?
(A) disagree with Jan
(B) get to work
(C) argue with the assignments
Answer: The answer is [B] get to work.
''',
'''
Bailey found Carson's cleaning solution after not being able to find it by himself. What does Bailey need to do before this?
(A) give Carson their solution
(B) return it to Carson
(C) follow directions to the location
Answer: The answer is [C] follow directions to the location.
''',
'''
Addison gave a hug to Skylar's son when they were feeling down. Why did Addison do this?
(A) better
(B) wrong
(C) keep hugging the son
Answer: The answer is [A] better.
''',
'''
Kai found one for sale online but it was too much money for her. What does Kai need to do before this? A: cheaper, B: Open up her laptop, C: save money
(A) cheaper
(B) Open up her laptop 
(C) save money
Answer: The answer is [B] Open up her laptop.
''',
'''
Cameron promised not to tell anyone about Ash's surprise party. Cameron told someone the secret. What does Cameron need to do before this?
(A) tell Ash about the party
(B) Go to Ash's party
(C) tell someone about the party
Answer: The answer is [C] tell someone about the party.
''',
'''
Kendall worked the weekend at the steakhouse and made bank on tips. What will Kendall want to do next?
(A) Save the money
(B) get hired at the steakhouse
(C) Quit her job
Answer: The answer is [A] Save the money.
''',
'''
Sydney was a school teacher and made sure their students learned well. How would you describe Sydney?
(A) As someone that asked for a job
(B) As someone that takes teaching seriously
(C) Like a leader
Answer: The answer is [B] As someone that takes teaching seriously.
''',
'''
Quinn wanted to help me clean my room up because it was so messy. What will Quinn want to do next?
(A) Eat messy snacks
(B) help out a friend
(C) Pick up the dirty clothes
Answer: The answer is [C] Pick up the dirty clothes.
''',
'''
Kendall ran back and thanked Lee for helping her find the dog. How would you describe Kendall?
(A) grateful
(B) super
(C) amazing
Answer: The answer is [A] grateful.
''',
]

usmle_template = '''
{question}
{answer_choices}
'''.strip()


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

        self.eval_model_name = args.eval_model_name

        # get model info
        self.tokenizer = AutoTokenizer.from_pretrained(self.eval_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.eval_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

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
        self.num_votes = args.num_votes
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
                dp['prompts'] = []
                for jdx in range(self.num_votes):
                    prompt = self.make_prompt(dp)
                    dp['prompts'].append(prompt)

        return all_data

    def make_prompt(self, dp):
        answer_choices = ''
        for choice in dp['options']:
            answer_choices += '({}) {}\n'.format(choice, dp['options'][choice])
        in_context = ''
        # for idx in example_indexes:
        #     in_context = in_context + siqa_in_context_examples[idx]
        prompt = usmle_template.format(
            question=dp['question'],
            answer_choices=answer_choices
        )
        prompt = in_context + prompt
        return prompt

    def parse_raw_result(self, raw_result):
        pred_text = raw_result.strip()
        sq_idx = pred_text.rfind("(")
        cl_idx = pred_text.rfind(")")
        # par_idx = pred_text.find("(")
        # then_idx = pred_text.find(")")
        if sq_idx != -1 and cl_idx != -1:
            answer = pred_text[sq_idx + 1:cl_idx]
            if answer.upper() in self.usmle_answer_dict.keys():
                return self.usmle_answer_dict[answer]
        # if then_idx != -1:
        #     answer = pred_text[then_idx-1:then_idx]
        #     if answer.upper() in self.usmle_answer_dict.keys():
        #         return self.usmle_answer_dict[answer.upper()]
        print(raw_result)
        return -1

    def parse_raw_result_with_options(self, raw_result, options):
        pred_text = raw_result.strip()
        sentences = pred_text.split(".")
        ans_sent = ""
        for sent in sentences:
            if "answer" in sent.lower():
                ans_sent = sent
        if len(ans_sent) == 0:
            print(pred_text)
            ans_sent = pred_text
        choice = self.check_valid_answer(ans_sent, options)
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
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
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
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        pbar = tqdm(range(0, len(data), self.eval_batch_size))
        for i in pbar:
            if i < len(raw_results):
                continue
            batch_data = data[i: i + self.eval_batch_size]
            batch_prompts = data[i]['prompts']
            batch_answ = [dp['true_answ'] for dp in batch_data]
            options = batch_data[0]['options']

            batch_chat_prompts = []
            for jdx in range(len(batch_prompts)):
                messages = [
                    {"role": "system",
                     # "content": "You are a helpful medical expert, and your task is to answer a multi-choice medical question. For each of the following multiple-choice questions, please select the correct answer."},
                     "content": "You are a social commonsense expert, and your task is to answer a social commonsense multi-choice commonsense question. For each of the following multiple-choice questions, please select the correct answer."},
                    # "content": "You are a helpful medical expert, and your task is to answer a multi-choice medical question. For each of the following multiple-choice questions, please not only select the correct answer but also provide a detailed reasoning for your choice."},
                ]
                in_context_indexes = range(0, len(siqa_in_context_examples))
                chosen_indexes = random.sample(in_context_indexes, self.num_shot)
                for kdx in chosen_indexes:
                    question = siqa_in_context_examples[kdx].split("Answer:")[0]
                    answer = "Answer:" + siqa_in_context_examples[kdx].split("Answer:")[1]
                    messages.append({"role": "user", "content": question})
                    messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user", "content": batch_prompts[jdx]})
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                chat_prompt = chat_prompt + "Answer: The answer is"
                batch_chat_prompts.append(chat_prompt)
            batch_inputs = self.tokenizer(
                batch_chat_prompts,
                padding=True,
                return_tensors="pt"
            )
            batch_inputs['input_ids'] = batch_inputs['input_ids'].to('cuda')
            batch_inputs['attention_mask'] = batch_inputs['attention_mask'].to('cuda')
            outputs = self.model.generate(
                input_ids=batch_inputs['input_ids'],
                attention_mask=batch_inputs['attention_mask'],
                max_new_tokens=self.max_new_tokens,
                eos_token_id=terminators,
                # do_sample=False,
                # top_p=1,
                do_sample=True,
                temperature=0.01,
                top_k=40,
                top_p=0.95,
                return_dict_in_generate=True,
            )
            lm_outputs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            lm_outputs = [r.split('assistant')[-1] for r in lm_outputs]
            lm_outputs = [r.split('Answer:')[-1] for r in lm_outputs]
            lm_outputs = [r.split('Question:')[0] for r in lm_outputs]

            # pred_answ = [self.parse_raw_result(r) for r in lm_outputs]#, dp in zip(lm_outputs, batch_data)]
            pred_answ = [self.parse_raw_result_with_options(r, options) for r in lm_outputs]  # , dp in zip(lm_outputs, batch_data)]
            # import ipdb; ipdb.set_trace()
            maj_answ = Counter(pred_answ).most_common(1)[0][0]
            raw_results.append(lm_outputs[0])
            all_pred_answ.append(maj_answ)
            all_target.extend(batch_answ)
            all_prompts.append(batch_prompts[0])
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
    parser.add_argument('--eval_model_name', type=str, default='/data/30_LLaMa_model_weights_HF/7B',
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
    parser.add_argument('--num_votes', type=int, default=1,
                        help='number of votes')

    parser.add_argument('--num_shot', type=int, default=0,
                        help='number of in-context examples')


    # set device max memory
    for device_id in range(8):
        parser.add_argument('--memory_{}'.format(device_id), type=int, default=15,
                            help='specify max memory requirements for gpu '.format(device_id))
    args = parser.parse_args()

    set_seed(1)
    EV = Evaluator(args)
    print(EV._eval())
