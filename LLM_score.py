from typing import List, Dict, Optional
import re
import logging
from helper import get_response

def log_info(message, logger="detail_logger", print_to_std=False):
    if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(logger)
    if type(logger) != str:
        if logger: logger.info(message)
    if print_to_std: print(message + "\n")


class EvidenceWeightingEngine:
    
    def __init__(self, model_name: str = None, max_batch_size: int = 10):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.relevance_cache = {}
    
    def evaluate_relevance_with_context(self, patient_info: str, inquiry: str, triplets: List[Dict]) -> List[Dict]:
        """batch evaluate triplets' relevance, considering context information"""
        if not triplets:
            return []
        
        log_info(f"start batch evaluating knowledge relevance, {len(triplets)} items...", print_to_std=False)
        
        relevance_results = []
        
        # batch processing
        for batch_start in range(0, len(triplets), self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, len(triplets))
            batch_triplets = triplets[batch_start:batch_end]
            
            # check cache
            cached_results, uncached_indices = self._check_cache(
                patient_info, inquiry, batch_triplets, batch_start
            )
            relevance_results.extend(cached_results)
            
            if not uncached_indices:
                continue
            
            # batch evaluate uncached triplets
            uncached_triplets = [batch_triplets[i] for i in uncached_indices]
            batch_results = self._evaluate_batch_uncached(
                patient_info, inquiry, uncached_triplets, batch_start, uncached_indices
            )
            relevance_results.extend(batch_results)
        
        # sort by index
        relevance_results.sort(key=lambda x: x['index'])
        return relevance_results
    
    def _check_cache(self, patient_info: str, inquiry: str, 
                     batch_triplets: List[Dict], batch_start: int) -> tuple:
        """check cache and return cache results and uncached indices"""
        cached_results = []
        uncached_indices = []
        
        for i, triplet in enumerate(batch_triplets):
            content = triplet['content']
            cache_key = f"{patient_info}_{inquiry}_{content}"
            
            if cache_key in self.relevance_cache:
                score = self.relevance_cache[cache_key]
                cached_results.append({
                    "type": "text",
                    "content": content,
                    "score": score,
                    "index": batch_start + i
                })
            else:
                uncached_indices.append(i)
        
        return cached_results, uncached_indices
    
    def _evaluate_batch_uncached(self, patient_info: str, inquiry: str,
                                uncached_triplets: List[Dict], batch_start: int,
                                uncached_indices: List[int]) -> List[Dict]:
        """evaluate uncached triplets"""
        if not uncached_triplets:
            return []
        
        # build batch evaluation prompt
        relations_text = self._format_relations_for_prompt(uncached_triplets)
        
        # get few-shot examples
        few_shot_examples = self._get_few_shot_examples(patient_info, inquiry)
        
        # build prompt
        prompt = self._build_batch_evaluation_prompt(
            patient_info, inquiry, relations_text, few_shot_examples, len(uncached_triplets)
        )
        
        messages = [
            {"role": "system", "content": "You are a medical expert with extensive experience in clinical diagnosis and treatment."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response_text, _, _ = get_response(messages, self.model_name)
            scores = self._parse_batch_scores(response_text, len(uncached_triplets))
            
            results = []
            for i, score in enumerate(scores):
                if i < len(uncached_indices):
                    original_idx = uncached_indices[i]
                    global_idx = batch_start + original_idx
                    
                    # update cache
                    content = uncached_triplets[i]['content']
                    cache_key = f"{patient_info}_{inquiry}_{content}"
                    self.relevance_cache[cache_key] = score
                    
                    results.append({
                        "type": "text",
                        "content": content,
                        "score": score,
                        "index": global_idx
                    })
            
            log_info(f"[Know] batch evaluation completed, scores: {[f'{s:.3f}' for s in scores]}", print_to_std=False)
            return results
            
        except Exception as e:
            log_info(f"[Know] batch evaluation error: {str(e)}, using default scores", print_to_std=True)
            
            # return default scores
            results = []
            for i in range(len(uncached_triplets)):
                if i < len(uncached_indices):
                    original_idx = uncached_indices[i]
                    global_idx = batch_start + original_idx
                    
                    results.append({
                        "type": "text",
                        "content": uncached_triplets[i]['content'],
                        "score": 0.5,
                        "index": global_idx
                    })
            
            return results
    
    def _format_relations_for_prompt(self, triplets: List[Dict]) -> str:
        """format relations text for prompt"""
        relations_text = ""
        for i, triplet in enumerate(triplets):
            content = triplet['content']
            relations_text += f"[{i}] {content}\n"
        
        return relations_text.rstrip('\n')
    
    def _build_batch_evaluation_prompt(self, patient_info: str, inquiry: str,
                                     relations_text: str, few_shot_examples: str,
                                     num_triplets: int) -> str:
        """build batch evaluation prompt"""
        return f"""You are a medical expert evaluating the relevance of medical knowledge to patient cases. Rate each relation's relevance independently on a scale from 0 to 1.

Scoring Guidelines:
- 0.0-0.2: Completely irrelevant or contradictory
- 0.3-0.4: Low relevance, tangentially related
- 0.5-0.6: Medium relevance, somewhat helpful
- 0.7-0.8: High relevance, directly applicable
- 0.9-1.0: Extremely relevant, critical for diagnosis/treatment

{few_shot_examples}

Now evaluate the following case:

Patient Information: {patient_info}
Current Inquiry: {inquiry}

Relations to evaluate:
{relations_text}

Please respond with only the scores in the format: [score1, score2, score3, ...] where each score is independently rated between 0 and 1."""
    
    def _parse_batch_scores(self, response_text: str, expected_count: int) -> List[float]:
        """parse batch evaluation scores"""
        try:
            # find pattern like [0.1, 0.2, 0.3]
            pattern = r'\[([\d\.,\s]+)\]'
            match = re.search(pattern, response_text)
            
            if match:
                scores_str = match.group(1)
                scores = [float(s.strip()) for s in scores_str.split(',')]
            else:
                # try to find single numbers
                numbers = re.findall(r'\d+\.?\d*', response_text)
                scores = [float(num) for num in numbers[:expected_count]]
            
            # ensure correct number of scores
            if len(scores) != expected_count:
                log_info(f"parsed scores ({len(scores)}) do not match expected number ({expected_count})", print_to_std=False)
                while len(scores) < expected_count:
                    scores.append(0.5)
                scores = scores[:expected_count]
            
            # ensure scores are in range 0-1
            scores = [max(0.0, min(1.0, score)) for score in scores]
            return scores
            
        except Exception as e:
            log_info(f"error parsing batch scores: {str(e)}", print_to_std=True)
            return [0.5] * expected_count
    
    def _get_few_shot_examples(self, patient_info: str, inquiry: str) -> str:
        """get few-shot examples"""
        # can dynamically select examples based on patient information and inquiry content
        return self._get_general_examples()
    
    def _get_general_examples(self) -> str:
        """get general few-shot examples"""
        return """
Examples:

Example 1:
Patient Information: 45-year-old female with irregular menstrual periods, hot flashes, and mood changes
Current Inquiry: Evaluating menopausal transition

Relations:
[0] Perimenopause typically begins in the 40s with irregular cycles and vasomotor symptoms
[1] Polycystic ovary syndrome causes irregular periods and insulin resistance
[2] Hormone replacement therapy can alleviate menopausal symptoms but has cardiovascular risks

Scores: [0.95, 0.6, 0.8]
Explanation: [0] is extremely relevant (0.95) as it perfectly describes perimenopause in the patient's age group. [1] has moderate relevance (0.6) as it explains irregular periods but doesn't fit the age/symptom pattern. [2] is relevant (0.8) as it addresses menopausal symptom management.

Example 2:
Patient Information: 22-year-old male with acute abdominal pain, nausea, and fever
Current Inquiry: Diagnosing acute abdomen

Relations:
[0] Appendicitis commonly presents with right lower quadrant pain, nausea, and fever
[1] Gastroesophageal reflux disease causes chronic heartburn and regurgitation
[2] Inflammatory bowel disease can cause abdominal pain and diarrhea

Scores: [0.9, 0.1, 0.4]
Explanation: [0] is highly relevant (0.9) as it matches the acute presentation perfectly. [1] is irrelevant (0.1) as it describes chronic symptoms, not acute abdomen. [2] has low-moderate relevance (0.4) as it can cause abdominal pain but typically with different characteristics.

Example 3:
Patient Information: 8-year-old child with persistent cough, wheezing, and exercise intolerance
Current Inquiry: Investigating pediatric respiratory symptoms

Relations:
[0] Childhood asthma frequently presents with cough, wheezing, and activity limitations
[1] Pneumonia in children causes fever, cough, and respiratory distress
[2] Congestive heart failure management includes diuretics and ACE inhibitors

Scores: [0.9, 0.5, 0.1]
Explanation: [0] is highly relevant (0.9) as it perfectly describes pediatric asthma symptoms. [1] has moderate relevance (0.5) as it addresses pediatric respiratory symptoms but lacks fever. [2] is irrelevant (0.1) as it discusses adult heart failure treatment, not pediatric respiratory issues.
"""
    def infer_patient_population(self, patient_profile: str, all_demographics: List[str], all_diseases: List[str]):
        """infer patient population from patient_profile"""
        demographic_info = None
        disease_info = None

        try:
            all_demographics_str = ", ".join(all_demographics)
            all_diseases_str = ", ".join(all_diseases)

            prompt = self._build_demographic_disease_info_prompt(patient_profile, all_demographics_str, all_diseases_str)

            messages = [
                {"role": "system", "content": "You are a medical expert with extensive experience in clinical diagnosis and treatment."},
                {"role": "user", "content": prompt}
            ]

            response_text, _, _ = get_response(messages, self.model_name)
            
            # parse generated demographic and disease information
            demographic_info, disease_info = self._parse_demographic_disease_info(response_text)

            if demographic_info:
                log_info(f"extracted demographic information: {demographic_info}", print_to_std=False)
            if disease_info:
                log_info(f"extracted disease information: {disease_info}", print_to_std=False)

            return demographic_info, disease_info

        except Exception as e:
            log_info(f"error extracting demographic and disease information: {str(e)}", print_to_std=True)
            return None, None
        
    def _build_demographic_disease_info_prompt(self, patient_profile: str, all_demographics: str, all_diseases: str) -> str:
        """build prompt to extract demographic and disease information"""
        return f"""
Given the patient profile and all conditions, please extract the demographic information and disease information from the patient profile that belong to all conditions.
Please ensure the information is accurate. Response with the exact demographic information and disease information, separated by a new line. If there is no demographic information or disease information, please return "None".

Example 1:
Patient Profile: 35-year-old male with chest pain, shortness of breath, and family history of heart disease. He suffers from high blood pressure
All Demographics: Pregnant woman, people with HIV, Adults, Elderly
All Diseases: heart disease, diabetes, hypertension, cancer

Answer:
Adults
heart disease, hypertension

Example 2:
Patient Profile: 28-year-old female with fatigue, weight gain, and cold intolerance. She is a smoker.
All Demographics: Pregnant woman, people with HIV, Adults, Elderly
All Diseases: heart disease, diabetes, hypertension, cancer

Answer:
Adults
None

Now evaluate the following case:

Patient Profile: {patient_profile}
All Demographics: {all_demographics}
All Diseases: {all_diseases}

Answer:
"""

    def _parse_demographic_disease_info(self, response_text: str):
        """parse generated demographic and disease information"""
        demographic_info = None
        disease_info = None

        try:
            demographic_line, disease_line = response_text.split('\n')
            demographic_line = demographic_line.strip()
            disease_line = disease_line.strip()
            
            # parse demographic information
            if demographic_line:
                demographic_info = [item.strip() for item in demographic_line.split(',') if item.strip()]
                if not demographic_info:  # if list is empty, set to None
                    demographic_info = None
            
            # parse disease information
            if disease_line and disease_line != 'None':
                disease_info = [item.strip() for item in disease_line.split(',') if item.strip()]
                if not disease_info:  # if list is empty, set to None
                    disease_info = None
                    
        except Exception as e:
            log_info(f"error parsing demographic and disease information: {str(e)}", print_to_std=True)
            # print(f"response_text: {response_text}")
            return None, None

        return demographic_info, disease_info