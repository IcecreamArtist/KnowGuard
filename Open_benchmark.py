import json
import os
import time
import logging
from patient import Patient
import importlib
from LLM_judge import compare_answer_to_options, judge_answer_yes_no

def setup_logger(name, file):
    if not file: return None
    logger = logging.getLogger(name)
    handler = logging.FileHandler(file, mode='a')
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def log_info(message, logger="history_logger", print_to_std=False):
    if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(logger)
    if type(logger) != str:
        if logger: logger.info(message)
    if print_to_std: print(message + "\n")

def load_data(filename):
    with open(filename, "r") as json_file:
        json_list = list(json_file)
    data = [json.loads(line) for line in json_list]
    data = {item['id']: item for item in data}
    return data

def main():
    if os.path.exists(args.output_filename):
        with open(args.output_filename, "r") as f:
            lines = f.readlines()
        output_data = [json.loads(line) for line in lines]
        if len(lines) == 0: processed_ids = []
        else: processed_ids = {sample["id"]: {"correct": sample["interactive_system"]["closest_option"] == sample["info"]["correct_answer_idx"],
                                              "timeout": len(sample["interactive_system"]["intermediate_answers"]) > args.max_questions,
                                              "turns": sample["interactive_system"]["num_questions"]}
                                for sample in output_data}
    else:
        processed_ids = []

    expert_module = importlib.import_module(args.expert_module)
    expert_class = getattr(expert_module, args.expert_class)
    patient_module = importlib.import_module(args.patient_module)
    patient_class = getattr(patient_module, args.patient_class)
    
    patient_data_path = os.path.join(args.data_dir, args.dev_filename)
    patient_data = load_data(patient_data_path)

    num_processed = 0
    correct_history, timeout_history, turn_lengths = [], [], []

    for pid, sample in patient_data.items():
        if pid in processed_ids:
            print(f"Skipping patient {pid} as it has already been processed.")
            correct_history.append(processed_ids[pid]["correct"])
            timeout_history.append(processed_ids[pid]["timeout"])
            turn_lengths.append(processed_ids[pid]["turns"])
            continue

        log_info(f"|||||||||||||||||||| PATIENT #{pid} ||||||||||||||||||||")
        free_text_answer, closest_option, questions, answers, temp_choice_list, temp_additional_info, sample_info = run_patient_interaction(expert_class, patient_class, sample)
        log_info(f"|||||||||||||||||||| Interaction ended for patient #{pid} ||||||||||||||||||||\n\n\n")

        output_dict = {
            "id": pid,
            "interactive_system": {
                "correct": closest_option == sample["answer_idx"],
                "free_text_answer": free_text_answer,
                "closest_option": closest_option,
                "questions": questions,
                "answers": answers,
                "num_questions": len(questions),
                "intermediate_answers": temp_choice_list,
                "temp_additional_info": temp_additional_info
            },
            "info": sample_info,
        }

        os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)
        with open(args.output_filename, 'a+') as f:
            f.write(json.dumps(output_dict) + '\n')

        correct_history.append(closest_option == sample["answer_idx"])
        timeout_history.append(len(temp_choice_list) > args.max_questions)
        turn_lengths.append(len(temp_choice_list))
        num_processed += 1
        accuracy = sum(correct_history) / len(correct_history) if len(correct_history) > 0 else None
        timeout_rate = sum(timeout_history) / len(timeout_history) if len(timeout_history) > 0 else None
        avg_turns = sum(turn_lengths) / len(turn_lengths) if len(turn_lengths) > 0 else None

        results_logger.info(f'Processed {num_processed}/{len(patient_data)} patients | Accuracy: {accuracy}')
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processed {num_processed}/{len(patient_data)} patients | Accuracy: {accuracy} | Timeout Rate: {timeout_rate} | Avg. Turns: {avg_turns}")
    print(f"Accuracy: {sum(correct_history)} / {len(correct_history)} = {accuracy}")
    print(f"Timeout Rate: {sum(timeout_history)} / {len(timeout_history)} = {timeout_rate}")
    print(f"Avg. Turns: {avg_turns}")
    

def parse_options(options_str):
    if isinstance(options_str, dict):
        return options_str

    if not isinstance(options_str, str):
        return {}

    try:
        return json.loads(options_str)
    except:
        try:
            import ast
            return ast.literal_eval(options_str)
        except:
            try:
                clean_str = options_str.strip()
                if clean_str.startswith('{') and clean_str.endswith('}'):
                    clean_str = clean_str[1:-1]

                result = {}
                pairs = clean_str.split(',')

                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                    elif ';' in pair:
                        key, value = pair.split(';', 1)
                    else:
                        continue

                    key = key.strip().strip("'").strip('"')
                    value = value.strip().strip("'").strip('"')

                    result[key] = value

                return result
            except:
                print(f"Warning: Failed to parse options string: {options_str}")
                return {}

def run_patient_interaction(expert_class, patient_class, sample):

    if args.question_type == 'multiple_choice':
        expert_system = expert_class(args, sample["question"], sample["options"])
    elif args.question_type == 'open-ended':
        expert_system = expert_class(args, sample["question"], None)
    else:
        raise NotImplementedError
    patient_system = patient_class(args, sample)
    options = parse_options(sample["options"])
    temp_answer_list = []
    temp_choice_list = []
    temp_additional_info = [] 
    patient_state = patient_system.get_state()

    while len(patient_system.get_questions()) < args.max_questions:
        current_stage = len(patient_system.get_questions())
        log_info(f"==================== Turn {len(patient_system.get_questions()) + 1} ====================",print_to_std=False)
        patient_state = patient_system.get_state()
        response_dict = expert_system.respond(patient_state, current_stage, False)
        log_info(f"[Expert System]: {response_dict}")

        if args.question_type == 'multiple_choice':
            temp_additional_info.append({k: v for k, v in response_dict.items() if k not in ["type", "letter_choice", "question"]})
        else:
            temp_additional_info.append({k: v for k, v in response_dict.items() if k not in ["type", "free_text_answer", "question"]})

        if response_dict["type"] == "question":
            if args.question_type == 'multiple_choice':
                temp_choice_list.append(response_dict["letter_choice"])
            else:
                if 'type' in response_dict:
                    temp_answer_list.append(response_dict["type"])
            patient_response = patient_system.respond(response_dict["question"])
            log_info(f"[Patient System]: {patient_response}", print_to_std=False)

        elif response_dict["type"] == "choice":
            expert_decision = response_dict["letter_choice"]
            temp_choice_list.append(expert_decision)
            sample_info = {
                "initial_info": patient_system.initial_info,
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "question": sample["question"],
                "options": sample["options"],
                "context": sample["context"],
                "facts": patient_system.facts,
            }
            return expert_decision, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list, temp_additional_info, sample_info

        elif response_dict["type"] == "answer":
            expert_answer = response_dict["free_text_answer"]
            if "type" in response_dict:
                temp_answer_list.append(response_dict["type"])

            if sample['question_type'] == 'mcq':
                closest_option = compare_answer_to_options(expert_answer, options, args.judge_model)
            else:
                closest_option = judge_answer_yes_no(expert_answer, sample["answer_rationale"], args.judge_model)
            sample_info = {
                "initial_info": patient_system.initial_info,
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "question": sample["question"],
                "options": options,
                "context": sample["context"],
                "facts": patient_system.facts,
                "closest_option": closest_option,
            }
            
            return expert_answer, closest_option, patient_system.get_questions(), patient_system.get_answers(), temp_answer_list, temp_additional_info, sample_info
        else:
            raise ValueError("Invalid response type from expert_system.")

    log_info(f"==================== Max Interaction Length ({args.max_questions} turns) Reached --> Force Final Answer ====================")
    patient_state = patient_system.get_state()
    response_dict = expert_system.respond(patient_state, args.max_questions, True)
    log_info(f"[Expert System]: {response_dict}")
    if args.question_type == 'multiple_choice':
        stuck_response = response_dict["letter_choice"]
    else:
        expert_answer = response_dict["free_text_answer"]

        if sample['question_type'] == 'mcq':
            closest_option = compare_answer_to_options(expert_answer, options, args.judge_model)
        else:
            closest_option = judge_answer_yes_no(expert_answer, sample["answer_rationale"], args.judge_model)

    temp_additional_info.append({k: v for k, v in response_dict.items() if (k != "letter_choice" and k != "free_text_answer")})

    sample_info = {
        "initial_info": patient_system.initial_info,
        "correct_answer": sample["answer"],
        "correct_answer_idx": sample["answer_idx"],
        "question": sample["question"],
        "options": options,
        "context": sample["context"],
        "facts": patient_system.facts,
    }
    if args.question_type == 'multiple_choice':
        return stuck_response, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list + [stuck_response], temp_additional_info, sample_info
    else:
        return expert_answer, closest_option, patient_system.get_questions(), patient_system.get_answers(), temp_answer_list, temp_additional_info, sample_info
