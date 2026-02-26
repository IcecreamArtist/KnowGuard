import prompts
import expert_basics
import logging
import helper

PROB_THRESHOLD = 0.8
SCALE_THRESHOLD = 4.0
KG_THRESHOLD = 4.0

def answer_to_idx(answer):
    return ord(answer) - ord("A")

def log_info(message, logger="detail_logger", print_to_std=False):
    if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(logger)
    if logger: logger.info(message)
    if print_to_std: print(message + "\n")


def numcutoff_abstention_decision_open(patient_state, rationale_generation, inquiry, abstain_threshold, **kwargs):
    """
    Numcutoff abstention strategy based on the current patient state.
    This function prompts the model to produce a numerical confidence score of how confident it is in its decision, then decide abstention based on arbitrarily set threshold
    """
    if not abstain_threshold: abstain_threshold = PROB_THRESHOLD

    # Get the response from the expert system
    prompt_key = "numcutoff_RG" if rationale_generation else "numcutoff"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])

    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["open_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, conf_score, log_probs, num_tokens = expert_basics.expert_response_confidence_score(messages, abstain_threshold=abstain_threshold, **kwargs)
    abstain_decision = conf_score < abstain_threshold
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    prompt_answer = prompts.expert_system["open_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, prompts.expert_system["open_long_answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, num_tokens_answer = expert_basics.expert_response_free_text(messages_answer, **kwargs)

    num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[NUMCUTOFF ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, free_text_answer: {response_text}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "free_text_answer": response_text,
    }


def scale_abstention_decision_open(patient_state, rationale_generation, inquiry, abstain_threshold, kg_threshold, encoded_images=None, text_knowledge="", is_know_expert=0, know_mode="multimodal", max_round=False, prev_messages=None, **kwargs):

    if not abstain_threshold: abstain_threshold = SCALE_THRESHOLD
    
    prompt_key = "scale_RG" if rationale_generation else "scale"
    
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]

    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])

    prompt_abstain = prompts.expert_system["open_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, abstain_task_prompt)

    if prev_messages:
        messages = prev_messages
        messages.append({"role": "user", "content": prompt_abstain})
    else:
        messages = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user", "content": prompt_abstain}
        ]

    log_msgs = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, conf_score, log_probs, num_tokens = expert_basics.expert_response_scale_score(messages, abstain_threshold=abstain_threshold, **kwargs)
    abstain_decision = conf_score < abstain_threshold

    log_info(f"[ABSTENTION PROMPT]: {log_msgs}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n",print_to_std=False)
    messages.append({"role": "assistant", "content": response_text})

    if max_round or abstain_decision == False:
        task_prompt = prompts.expert_system["open_long_answer"]
        patient_info = patient_state["initial_info"]
        conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])

        prompt = prompts.expert_system["open_template"].format(patient_info, conv_log, inquiry, task_prompt)

        messages = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user", "content": prompt}
        ]

        num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
        num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]
        final_decision = 'answer'
        log_info(f"[FREE TEXT ANSWER]: {response_text}", print_to_std=False)
    else:
        response_text = 'No abstention.'
        final_decision = 'question'
    log_info(f"[SCALE ABSTAIN OPEN RETURN]: abstain: {final_decision}, confidence: {conf_score}, free_text_answer: {response_text}, usage: {num_tokens}\n")

    return {
        "abstain": final_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "free_text_answer": response_text,
    }

def knowguard_abstention_decision_open(patient_state, rationale_generation, inquiry, abstain_threshold, kg_threshold, encoded_images=None, text_knowledge="", is_know_expert=0, know_mode="multimodal", max_round=False, prev_messages=None, **kwargs):
    if not kg_threshold: kg_threshold = KG_THRESHOLD

    if know_mode == "text_only" or (text_knowledge != None and encoded_images == []):
        prompt_key = "know_text_scale_RG" if rationale_generation else "know_scale"
    elif know_mode == "image_only" or (text_knowledge == None and encoded_images != []):
        prompt_key = "know_image_scale_RG" if rationale_generation else "know_scale"
    elif text_knowledge != None and encoded_images != []:
        prompt_key = "know_multimodal_scale_RG" if rationale_generation else "know_scale"
    else:
        prompt_key = "scale_KG_RG" if rationale_generation else "scale"
    
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]

    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])

    if text_knowledge and (know_mode in ['text_only', 'multimodal']):
        prompt_abstain = prompts.expert_system['know_template'].format(patient_info, conv_log if conv_log != '' else 'None', text_knowledge, inquiry, abstain_task_prompt)
    else:
        prompt_abstain = prompts.expert_system["open_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, abstain_task_prompt)

    if encoded_images != [] and encoded_images and (know_mode in ['image_only', 'multimodal']):
        content = []
        content.append({"type": "text", "text": prompt_abstain})

        for encoded_image in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })
        if prev_messages:
            messages = prev_messages
            messages.append({"role": "user", "content": content})
        else:
            messages = [
                {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
                {"role": "user", "content": content}
            ]

    else:
        if prev_messages:
            messages = prev_messages
            messages.append({"role": "user", "content": prompt_abstain})
        else:
            messages = [
                {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
                {"role": "user", "content": prompt_abstain}
            ]

    log_msgs = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, conf_score, log_probs, num_tokens = expert_basics.expert_response_scale_score(messages, abstain_threshold=abstain_threshold, **kwargs)
    # print('confidence score', conf_score)
    abstain_decision = conf_score < kg_threshold

    # print(abstain_decision, kg_threshold, abstain_threshold, conf_score)

    log_info(f"[ABSTENTION PROMPT]: {log_msgs}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n",print_to_std=False) # print
    messages.append({"role": "assistant", "content": response_text})

    if max_round or abstain_decision == False:
        if is_know_expert == 1:
            if know_mode == "text_only" or (text_knowledge != None and encoded_images == []):
                task_prompt = prompts.expert_system["know_text_open_answer"]
            elif know_mode == "image_only" or (text_knowledge == None and encoded_images != []):
                task_prompt = prompts.expert_system["know_image_open_answer"]
            elif text_knowledge != None and encoded_images != []:
                task_prompt = prompts.expert_system["know_multimodal_open_answer"]
            else:
                task_prompt = prompts.expert_system["open_long_answer"]
        elif is_know_expert == 0:
            task_prompt = prompts.expert_system["open_long_answer"]
        patient_info = patient_state["initial_info"]
        conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])

        prompt = prompts.expert_system["open_template"].format(patient_info, conv_log, inquiry, task_prompt)

        if text_knowledge and (know_mode in ['text_only', 'multimodal']):
            prompt += f"\n\n--- RETRIEVED MEDICAL KNOWLEDGE ---\n{text_knowledge}"

        if encoded_images != [] and encoded_images and (know_mode in ['image_only', 'multimodal']):
            content = []
            content.append({"type": "text", "text": prompt})
            for encoded_image in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                })

            messages = [
                {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
                {"role": "user", "content": content}
            ]

        else:
            messages = [
                {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
                {"role": "user", "content": prompt}
            ]
        response_text, num_tokens_answer = expert_basics.expert_response_free_text(messages, **kwargs)

        num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
        num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]
        final_decision = 'answer'
        log_info(f"[FREE TEXT ANSWER]: {response_text}", print_to_std=False) # print
    else:
        response_text = 'No abstention.'
        final_decision = 'question'
    log_info(f"[SCALE ABSTAIN OPEN RETURN]: abstain: {final_decision}, confidence: {conf_score}, free_text_answer: {response_text}, usage: {num_tokens}\n")
    # print(final_decision)
    return {
        "abstain": final_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "free_text_answer": response_text,
    }


def question_generation(patient_state, inquiry, options_dict, messages, independent_modules, is_know_expert=False, **kwargs):

    task_prompt = prompts.expert_system['atomic_question_improved']
    if independent_modules:
        patient_info = patient_state["initial_info"]
        conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
        options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
        prompt = prompts.expert_system["curr_template"].format(patient_info, conv_log, inquiry, options_text, task_prompt)

        messages = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user", "content": prompt}
        ]
    else:
        messages.append({"role": "user", "content": task_prompt})

    response_text, atomic_question, num_tokens = expert_basics.expert_response_question(messages, **kwargs)
    log_info(f"[ATOMIC QUESTION PROMPT]: {messages}")
    log_info(f"[ATOMIC QUESTION RESPONSE]: {atomic_question}\n")
    messages.append({"role": "assistant", "content": atomic_question})

    log_info(f"[ATOMIC QUESTION RETURN]: {atomic_question}, usage: {num_tokens}\n",print_to_std=False) # print
    return {
        "atomic_question": atomic_question,
        "messages": messages,
        "usage": num_tokens,
    }


def generate_relevant_query(patient_info, inquiry, **kwargs):
    task_prompt = prompts.expert_system["relevant_kg_query"].format(patient_info, inquiry)
    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": task_prompt}
    ]
    response_text, log_probs, num_tokens = helper.get_response(messages, **kwargs)
    if not response_text:
        log_info("[<RELEVANT KG QUERY LM RES>]: " + "No response.")
        return patient_info, None, num_tokens
    log_info("[<RELEVANT KG QUERY LM RES>]: " + response_text, print_to_std=False) # print

    return response_text