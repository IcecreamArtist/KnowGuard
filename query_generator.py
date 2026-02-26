from typing import List, Tuple
import logging
from helper import get_response

def log_info(message, logger="detail_logger", print_to_std=False):
    if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(logger)
    if type(logger) != str:
        if logger: logger.info(message)
    if print_to_std: print(message + "\n")


class QueryGenerator:
    """query generator, used to generate optimized search queries from patient information or doctor's question"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.query_cache = {}
    
    def generate_search_queries(self, text: str, inquiry: str, context_type: str = "patient_info") -> List[str]:
        """
        generate 2 optimized search queries from input text
        
        Parameters:
        - text: input text (patient information or doctor's question)
        - inquiry: question
        - context_type: context type, "patient_info" or "doctor_question"
        
        Returns:
        - generated query list
        """
        # check cache
        cache_key = f"{context_type}_{text}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        try:
            # build prompt
            prompt = self._build_query_generation_prompt(text, inquiry, context_type)
            
            messages = [
                {"role": "system", "content": "You are a medical expert specializing in clinical knowledge retrieval."},
                {"role": "user", "content": prompt}
            ]
            
            response_text, _, _ = get_response(messages, self.model_name)
            
            # parse generated queries
            queries = self._parse_generated_queries(response_text)
            
            # update cache
            self.query_cache[cache_key] = queries
            
            log_info(f"generate queries for {inquiry} and {context_type}: {queries}", print_to_std=False)
            return queries
            
        except Exception as e:
            log_info(f"error generating queries: {str(e)}, using original text", print_to_std=False)
            # revert to original text
            return [text]
            
    def _parse_generated_queries(self, response_text: str) -> List[str]:
        """parse generated queries"""
        import re
        
        queries = []
        
        # try to match "Query X: ..." format
        pattern = r'Query\s*\d+\s*:\s*(.+?)(?=Query\s*\d+\s*:|$)'
        matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            query = match.strip()
            if query:
                queries.append(query)
        
        # if parsing fails, try to split by line
        if len(queries) < 2:
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            queries = []
            for line in lines:
                # remove possible prefixes
                cleaned = re.sub(r'^(Query\s*\d+\s*:\s*|[\d\.\-\*]\s*)', '', line, flags=re.IGNORECASE).strip()
                if cleaned and len(cleaned) > 5:  # ensure the query is meaningful
                    queries.append(cleaned)
                if len(queries) >= 2:
                    break
        
        # ensure at least one query
        if not queries:
            queries = ["medical knowledge query"]
        
        # limit to 2 queries
        return queries[:2]
    
    # add method to QueryGenerator class
    def _build_query_generation_prompt(self, text: str, inquiry: str, context_type: str) -> str:
        """build query generation prompt"""
        if context_type == "patient_info":
            return f"""Based on the following patient information, generate 2 optimized search queries to retrieve relevant medical knowledge from a knowledge base.

    The queries should:
    1. Focus on key symptoms, conditions, or medical concerns
    2. Use medical terminology when appropriate
    3. Be specific enough to find relevant information
    4. Cover different aspects of the patient's condition

    Patient Information: {text}

    Please respond with exactly 2 queries in the format:
    Query 1: [your first query]
    Query 2: [your second query]

    Example:
    Patient Information: 28-year-old male with sudden onset severe abdominal pain
    Query 1: acute abdominal pain young male appendicitis peritonitis
    Query 2: sudden onset severe abdominal pain differential diagnosis emergency"""
            
        elif context_type == "doctor_question":
            return f"""Based on the following doctor's question, generate 2 optimized search queries to retrieve relevant medical knowledge that would help answer this question.

    The queries should:
    1. Focus on the medical concepts being asked about
    2. Include relevant diagnostic or treatment information
    3. Use appropriate medical terminology
    4. Cover different aspects of the question

    Doctor's Question: {text}

    Please respond with exactly 2 queries in the format:
    Query 1: [your first query]
    Query 2: [your second query]

    Example:
    Patient Information: 35-year-old pregnant woman at 32 weeks gestation with severe headache and visual changes
    Query 1: preeclampsia pregnancy headache visual disturbances 32 weeks
    Query 2: headache pregnancy third trimester hypertensive disorders"""
            
        elif context_type == "qa_pair":  # qa_pair
            return f"""Based on the following question, and a conversation of doctor's question and patient's response, generate 2 optimized search queries to retrieve relevant medical knowledge.

    The queries should:
    1. Integrate both the question context and patient's specific response
    2. Focus on relevant symptoms, conditions, or medical concerns mentioned
    3. Use medical terminology when appropriate
    4. Cover different diagnostic or clinical aspects

    Question: {inquiry}
    Conversation: {text}

    Please respond with exactly 2 queries in the format:
    Query 1: [your first query]
    Query 2: [your second query]

    Example:
    Question: What is the most likely diagnosis?
    Conversation: Question: How long have you had this cough?\nPatient Response: About 3 weeks now, and I've been coughing up blood-tinged sputum
    Query 1: chronic cough hemoptysis 3 weeks duration tuberculosis
    Query 2: hemoptysis differential diagnosis lung cancer bronchitis"""
        
        elif context_type == "patient_profile":
            return f"""Based on the inquiry and the following patient profile, generate 2 optimized search queries to retrieve relevant medical knowledge.

    The queries should:
    1. Focus on relevant symptoms, conditions, or medical concerns mentioned
    2. Use medical terminology when appropriate
    3. Cover different diagnostic or clinical aspects
    
    Question: {inquiry}
    Patient Profile: {text}

    Please respond with exactly 2 queries in the format:
    Query 1: [your first query]
    Query 2: [your second query]

    Example:
    Question: What could be causing her shortness of breath?
    Patient Profile: 65-year-old female with diabetes and hypertension, presenting with dyspnea and ankle swelling
    Query 1: heart failure dyspnea ankle swelling elderly diabetes
    Query 2: shortness of breath differential diagnosis women elderly comorbidities"""
        