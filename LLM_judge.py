from helper import get_response

def compare_answer_to_options(answer, options, model_name):
    """
    Compare the generated answer to the options using a third agent
    Return the closest option letter
    """
    # Check if option E exists
    has_option_e = "E" in options

    # Construct options part of the prompt
    options_text = ""
    for key in options:
        options_text += f"{key}: {options[key]}\n"

    # Implement the comparison logic here
    # Use LLM as a judge to decide which option is closest
    prompt = f"""You are a medical evaluation expert. Please analyze the free answer and its match with the options, and determine which option is closest to the answer.

Free Answer: {answer}

Options:
{options_text}
Please output only the option letter ({', '.join(options.keys())}) that has the highest match.
"""
    # In actual implementation, you need to call LLM API to get the response
    # This is just an example
    messages = [
        {"role": "system", "content": "You are a medical evaluation expert, tasked with evaluating the match between an answer and options."},
        {"role": "user", "content": prompt}
    ]
    response_text, _, _ = get_response(messages, model_name)
    closest_option = response_text.strip()

    # Extract the option letter (if the response contains more text)
    for option_letter in options.keys():
        if option_letter in closest_option and len(closest_option) > 1:
            closest_option = option_letter
            break

    return closest_option

def judge_answer_yes_no(answer, ground_truth, model_name):
    """
    Compare the generated answer to the ground truth answer using a third agent
    Return 'A' if the answer is correct, 'B' if the answer is incorrect
    """
    # Implement the comparison logic here
    # Use LLM as a judge to decide which option is closest
    prompt = f"""You are a medical evaluation expert. Please analyze the free answer and the ground truth answer, and determine if the answer is similar to the ground truth answer.

Free Answer: {answer}

Ground Truth Answer: {ground_truth}

Please output only 'A' if the answer is similar to the ground truth answer, 'B' if the answer is completely different. The output should not contain the quotation marks.
"""
    # In actual implementation, you need to call LLM API to get the response
    # This is just an example
    messages = [
        {"role": "system", "content": "You are a medical evaluation expert, tasked with evaluating the match between an answer and the ground truth answer."},
        {"role": "user", "content": prompt}
    ]
    response_text, _, _ = get_response(messages, model_name)
    result = response_text.strip()

    if result == 'A':
        return 'A'
    else:
        return 'B'
