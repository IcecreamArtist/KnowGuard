expert_system = {
    
    "meditron_system_msg": "You are a medical doctor trying to reason through a real-life clinical case. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, respond according to the task specified by the user. Base your response on the current and standard practices referenced in medical guidelines.",
    
    "question_word": "Doctor Question",
    "answer_word": "Patient Response",

    "scale_KG_RG": """Medical conditions are complex, so you should seek to understand their situations across many features. First, given the interactions between the patient and the doctor, and the medical knowledge provided to you, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment, such as features mentioned in the medical knowledge; finally, consider whether all necessary information is given in the conversation above. How confident are you to answer correctly to the problem factually using the conversation log and medical knowledge? Choose between the following ratings:
"Very Confident" - The correct answer is supported by all evidence, and there is enough evidence to eliminate the other possible answers, so the option can be confirmed conclusively. There is no extra information or any test results needed for confirming the diagnosis.
"Somewhat Confident" - I have reasonably enough information to tell that the correct answer is more likely than other possible answers, more information like physical examination results, test results, and medical history, etc. is helpful to make a conclusive decision.
"Neither Confident or Unconfident" - There are evident supporting the correct answer, but further evidence is needed to be sure about the answer.
"Somewhat Unconfident" - There are evidence supporting more than one possible answers, therefore more questions are needed to further distinguish the answers.
"Very Unconfident" - There are not enough evidence supporting any answers, the likelihood of giving the correct answer at this point is near random guessing.\n\nAnswer in the following format:\nREASON: a one-sentence explanation of why you are or are not confident and what other information is needed.\nDECISION: chosen rating from the above list.
    For exmample:
    Patient information: A 19-year-old man is brought to the physician by his mother because she is worried about his strange behavior. He does not have any friends and spends most of his time in his room playing online games. 3. Rather than doing his coursework, he spends most of his time reading up on paranormal phenomena, especially demons. 4. He says that he has never seen any demons, but sometimes there are signs of their presence. 5. For example, a breeze in an enclosed room is likely the “breath of a demon”.
    Medical knowledge: Psychotic symptoms including delusions, hallucinations, bizarre behaviour, illogic ideas, thought blocking, deficiencies in speech, neologisms, incoherence and loose thought associations may be effectively treated with antipsychotic medicines.
    Answer:
    \nREASON: The patient's behavior and the medical knowledge both suggest that the patient might be suffering from a psychotic disorder, such as schizophrenia. However, the patient's mother is worried about his strange behavior, and the patient himself says that he has never seen any demons, but sometimes there are signs of their presence. So, the correct answer is not schizophrenia, but Schizotypal personality disorder.
    \nDECISION: Somewhat Confident
""",

    "yes_no": "Now, are you confident to pick the correct option to the inquiry factually using the conversation log? Answer with YES or NO and NOTHING ELSE.",

    "atomic_question": "If there are missing features that prevent you from picking a confident and factual answer to the inquiry, consider which features are not yet asked about in the conversation log; then, consider which missing feature is the most important to ask the patient in order to provide the most helpful information toward a correct medical decision. Ask ONE SPECIFIC ATOMIC QUESTION to address this feature. The question should be bite-sized, and NOT ask for too much at once. Generate the atomic question and NOTHING ELSE.",
    
    "atomic_question_improved": "If there are missing features that prevent you from picking a confident and factual answer to the inquiry, consider which features are not yet asked about in the conversation log; then, consider which missing feature is the most important to ask the patient in order to provide the most helpful information toward a correct medical decision. You can ask about any relevant information about the patient’s case, such as patient profile and medical history (e.g., HIV, pregnant, diabetes), family history, tests and exams results, treatments already done, etc. Consider what are the common questions asked in the specific subject relating to the patient’s known symptoms, and what the best and most intuitive doctor would ask. Ask ONE SPECIFIC ATOMIC QUESTION to address this feature. The question should be bite-sized, and NOT ask for too much at once. Make sure to NOT repeat any questions from the above conversation log. Answer in the following format:\nATOMIC QUESTION: the atomic question and NOTHING ELSE.\nATOMIC QUESTION: ",

    "open_long_answer": "Assume that you already have enough information from the above question-answer pairs to answer the patient inquiry, use the above information to produce a factual conclusion. Respond with a comprehensive and well-reasoned answer.",

    "curr_template": """A patient comes into the clinic presenting with a symptom as described in the conversation log below:
    
PATIENT INFORMATION: {}
CONVERSATION LOG:
{}
QUESTION: {}
OPTIONS: {}
YOUR TASK: {}""",

    "open_template": """A patient comes into the clinic presenting with a symptom as described in the conversation log below:

PATIENT INFORMATION: {}
CONVERSATION LOG:
{}
QUESTION: {}
YOUR TASK: {}""",

    "know_template": """A patient comes into the clinic presenting with a symptom as described in the conversation log below:

PATIENT INFORMATION: {}
CONVERSATION LOG:
{}
RETRIEVED KNOWLEDGE: {}
QUESTION: {}
YOUR TASK: {}""",

    # 添加新的RAG相关提示
    "know_scale": """Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above.

You have been provided with relevant medical knowledge and images (the following IMAGES) that might help with this case. While this reference material is valuable, it may not contain all patient-specific information needed for a complete diagnosis.

How confident are you to answer to the problem factually using the conversation log and provided reference materials? Choose between the following ratings:
"Very Confident" - The answer is supported by all evidence, and there is enough evidence to eliminate other possible situations, so the answer can be confirmed conclusively.
"Somewhat Confident" - I have reasonably enough information to tell the answer, more information is helpful to make a conclusive decision.
"Neither Confident or Unconfident" - There are evident supporting the answer, but further evidence is needed to be sure about the answer.
"Somewhat Unconfident" - There are evidence supporting more than one possible answers, therefore more questions are needed to further distinguish possible situations.
"Very Unconfident" - There are not enough evidence supporting any answer, the likelihood of answer correctly at this point is near random guessing.\n\nThink carefully step by step, respond with the chosen confidence rating ONLY and NOTHING ELSE.""",

    "know_image_scale_RG": """Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above.

You have been provided with relevant medical knowledge images (the following IMAGES) that might help with this case. While this reference material is valuable, it may not contain all patient-specific information needed for a complete diagnosis.

How confident are you to answer to the problem factually using the conversation log and provided reference materials? Choose between the following ratings:
"Very Confident" - The answer is supported by all evidence, and there is enough evidence to eliminate other possible situations, so the answer can be confirmed conclusively.
"Somewhat Confident" - I have reasonably enough information to tell the answer, more information is helpful to make a conclusive decision.
"Neither Confident or Unconfident" - There are evident supporting the answer, but further evidence is needed to be sure about the answer.
"Somewhat Unconfident" - There are evidence supporting more than one possible answers, therefore more questions are needed to further distinguish possible situations.
"Very Unconfident" - There are not enough evidence supporting any answer, the likelihood of answer correctly at this point is near random guessing.\n\nAnswer in the following format:\nREASON: a one-sentence explanation of why you are or are not confident and what other information is needed.\nDECISION: chosen rating from the above list.""",

    "know_text_scale_RG": """Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above.

You have been provided with relevant medical knowledge (the following TEXT) that might help with this case. While this reference material is valuable, it may not contain all patient-specific information needed for a complete diagnosis.

How confident are you to answer to the problem factually using the conversation log and provided reference materials? Choose between the following ratings:
"Very Confident" - The answer is supported by all evidence, and there is enough evidence to eliminate other possible situations, so the answer can be confirmed conclusively.
"Somewhat Confident" - I have reasonably enough information to tell the answer, more information is helpful to make a conclusive decision.
"Neither Confident or Unconfident" - There are evident supporting the answer, but further evidence is needed to be sure about the answer.
"Somewhat Unconfident" - There are evidence supporting more than one possible answers, therefore more questions are needed to further distinguish possible situations.
"Very Unconfident" - There are not enough evidence supporting any answer, the likelihood of answer correctly at this point is near random guessing.\n\nAnswer in the following format:\nREASON: a one-sentence explanation of why you are or are not confident and what other information is needed.\nDECISION: chosen rating from the above list.""",

    "know_multimodal_scale_RG": """Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment. We provide some relevant medical knowledge in text and image (attached in this request) format, which may remind you of possible answers to consider, and provide information that help eliminate incorrect possibilities.
IMPORTANT: Information marked as "The patient cannot answer this question" should be considered unanswerable. We should not expect to obtain this information, and there is no need to ask them again.
Based on the conversation history and the medical knowledge available to you, assess your confidence in providing a factual and helpful answer to the specific inquiry at this point. Choose between the following ratings:
"Very Confident" - The answer is supported by all evidence, and there is enough evidence to eliminate other possible situations, so the answer can be confirmed conclusively.
"Somewhat Confident" - I have reasonably enough information to tell the answer, for other possibilities, I have tried asking questions related to them, such as recent laboratory results, but the patient did not give answers to them.
"Neither Confident or Unconfident" - There are evident supporting the answer, but further evidence is needed to be asked to be sure about the answer, such as recent laboratory results, or imaging studies. I have not asked those questions yet.
"Somewhat Unconfident" - There are evidence supporting more than one possible answers, therefore more questions are needed to further distinguish possible situations.
"Very Unconfident" - There are not enough evidence supporting any answer, the likelihood of answer correctly at this point is near random guessing.\n\nAnswer in the following format:\nEXPLANATION: a one-sentence explanation of why you are or are not confident and what other information is needed, such as family medical history, patient living hobbies, examination results.\nDECISION: chosen rating from the above list.""",

   "know_atomic_question": """If there are missing features that prevent you from picking a confident and factual answer to the inquiry, consider which features are not yet asked about in the conversation log; then, consider which missing feature is the most important to ask the patient in order to provide the most helpful information toward a correct medical decision.
Using the available case information, conversation history, and reference medical knowledge materials, formulate ONE SPECIFIC ATOMIC QUESTION to address this feature.
The question should be bite-sized, and NOT ask for too much at once.
Make sure to NOT repeat any questions from the above conversation log.
Answer in the following format:\nATOMIC QUESTION: the atomic question and NOTHING ELSE.\nATOMIC QUESTION: """,

    "know_image_open_answer": """Assume that you already have enough information from the above question-answer pairs and the provided medical knowledge images (the following IMAGES) to answer the patient inquiry. Use all available information to produce a factual conclusion.

Base your answer on both the patient's specific information from the conversation log AND the medical knowledge provided. When appropriate, include relevant insights from the medical reference materials in your explanation.

Respond with a comprehensive and well-reasoned answer.""",

    "know_text_open_answer": """Assume that you already have enough information from the above question-answer pairs and the provided medical knowledge (the following TEXT) to answer the patient inquiry. Use all available information to produce a factual conclusion.

Base your answer on both the patient's specific information from the conversation log AND the medical knowledge provided. When appropriate, include relevant insights from the medical reference materials in your explanation.

Respond with a comprehensive and well-reasoned answer.""",

    "know_multimodal_open_answer": """Assume that you already have enough information from the above question-answer pairs and the provided medical knowledge and images (the following IMAGES) to answer the patient inquiry. Use all available information to produce a factual conclusion.

Base your answer on both the patient's specific information from the conversation log AND the medical knowledge provided. When appropriate, include relevant insights from the medical reference materials in your explanation.

Respond with a comprehensive and well-reasoned answer.""",

    "know_text_relevance": """Evaluate the relevance of following medical knowledge to the patient's information and the question.

Patient information:
{}

Question:
{}

Medical knowledge:
{}

Please assess the relevance considering:
How relevant is this medical knowledge to this specific patient's situation?
Does this knowledge contain potential diagnoses applicable to this patient?
Does this knowledge help answer the question?

Return only the score ranging from 1 (completely irrelevant) to 10 (helpful for answer), without any additional text.""",

    "relevant_kg_query": """Given the current patient information and the inquiry, generate a knowledge query sentence that can be used to search relevant knowledge in the knowledge base.

Patient information:
{}

Question:
{}

Example:
Patient information:
He reports symptoms of malaise, anorexia, and abdominal cramps followed by watery diarrhea.

Question:
What is the most likely diagnosis?

Knowledge query:
Viral gastroenteritis, Traveler's diarrhea, Food poisoning

Respond with the query sentence and NOTHING ELSE: """
}

patient_system = {
    "system": "You are a truthful assistant that understands the patient's information, and you are trying to answer questions from a medical doctor about the patient. ",
    "header": "Below is a list of factual statements about the patient:\n",
    "prompt": 'Which of the above atomic factual statements answers the question? If no statement answers the question, simply say "The patient cannot answer this question, please do not ask this question again." Answer only what the question asks for. Do not provide any analysis, inference, or implications. Respond by selecting all statements that answer the question from above ONLY and NOTHING ELSE.',

    "prompt_new": """Below is a list of factual statements about the patient:\n
{}\n
Which of the above atomic factual statements answers the question? If no statement answers the question, simply say "The patient cannot answer this question, please do not ask this question again." Answer only what the question asks for. Do not provide any analysis, inference, or implications. Respond with all statements that directly answer the question from above verbatim ONLY and NOTHING ELSE, with one statement on each line.

Example:
Question from the doctor: [some question]
STATEMENTS:\n[example statement: she reports that...]\n[example statement: she has a history of...]

Question from the doctor: {}
""",

    "system_first_person": "You are a patient with a list of symptoms, and you task is to truthfully answer questions from a medical doctor. ",
    "header_first_person": "Below is a list of atomic facts about you, use ONLY the information in this list and answer the doctor's question.",
    "prompt_first_person": """Which of the above atomic factual statements are the best answer to the question? Select at most two statements. If no statement answers the question, simply say "The patient cannot answer this question, please do not ask this question again." Do not provide any analysis, inference, or implications. Respond by reciting the matching statements, then convert the selected statements into first person perspective as if you are the patient but keep the same information. Generate your answer in this format:

STATEMENTS: 
FIRST PERSON: """
}
