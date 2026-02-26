import random
import expert_functions
from graph_reason import *
import logging
import graph_reason
from helper import get_embeddings
import pandas as pd
from know_storage import get_faiss_db, get_embeddings_model, get_demographics_info

def log_info(message, logger="detail_logger", print_to_std=False):
    if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(logger)
    if type(logger) != str:
        if logger: logger.info(message)
    if print_to_std: print(message + "\n")

class Expert:
    """
    Expert system skeleton
    """
    def __init__(self, args, inquiry, options):
        # Initialize the expert with necessary parameters and the initial context or inquiry
        self.args = args
        self.inquiry = inquiry
        self.options = options

    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        raise NotImplementedError
    
    def ask_question(self, patient_state, prev_messages, is_know_expert, text_knowledge, encoded_images):
        # Generate a question based on the current patient state
        kwargs = {
            "patient_state": patient_state,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "messages": prev_messages,
            "independent_modules": self.args.independent_modules,
            "model_name": self.args.expert_model_question_generator,
            "use_vllm": self.args.use_vllm,
            "use_api": self.args.use_api,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
            "api_account": self.args.api_account,
            "text_knowledge": text_knowledge,
            "encoded_images": encoded_images,
            "is_know_expert": is_know_expert
        }
        return expert_functions.question_generation(**kwargs)
    
    def get_abstain_kwargs(self, patient_state):
        kwargs = {
            "max_depth": self.args.max_questions,
            "patient_state": patient_state,
            "rationale_generation": self.args.rationale_generation,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "abstain_threshold": self.args.abstain_threshold,
            "self_consistency": self.args.self_consistency,
            "model_name": self.args.expert_model,
            "use_vllm": self.args.use_vllm,
            "use_api": self.args.use_api,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
            "api_account": self.args.api_account,
            "kg_threshold": self.args.kg_threshold
        }
        return kwargs


class OpenEndedNumericalCutOffExpert(Expert):
    def respond(self, patient_state, current_round, max_round=False):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.numcutoff_abstention_decision_open(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "answer",
                "free_text_answer": abstain_response_dict["free_text_answer"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"], is_know_expert=0, text_knowledge=None, encoded_images=None)
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "free_text_answer": abstain_response_dict["free_text_answer"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"],
            "intermediate_answer": abstain_response_dict["free_text_answer"]
        }


class OpenEndedScaleExpert(Expert):
    def respond(self, patient_state, current_round, max_round=False):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.scale_abstention_decision_open(max_round=max_round, **kwargs)
        if abstain_response_dict["abstain"] == 'knowledge':
            return {
                "type": "answer",
                "free_text_answer": abstain_response_dict["free_text_answer"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"], is_know_expert=0, text_knowledge=None, encoded_images=None)
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "free_text_answer": abstain_response_dict["free_text_answer"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"],
            "intermediate_answer": abstain_response_dict["free_text_answer"]
        }


class KnowGuardExpert(Expert):

    def __init__(self, args, inquiry, options):
        super().__init__(args, inquiry, options)
        self.inquiry = inquiry
        self.know_mode = args.know_mode

        # retrieval
        self.dynamic_kg = None
        self.retrieved_images = []
        self.knowledge_descriptions = []
        self.formatted_knowledge = None
        self.encoded_images = []
        self.knowledge_retrieved = False

        # relevance
        self.relevance_scores = []
        self.relevance_threshold = args.relevance_threshold
        self.llm_relevance_threshold = args.llm_relevance_threshold
        self.relevance_modality = args.relevance_modality
        self.initial_triplets = args.initial_triplets
        self.direct_query_new = args.direct_query_new
        self.max_queue_size = args.max_queue_size

        # hyper-parameters
        self.embedding_weight = args.embedding_weight
        self.llm_weight = args.llm_weight
        self.coherence_weight = args.coherence_weight
        self.decay_weight = args.decay_weight

        # new parameters
        self.use_question_query = getattr(args, 'use_question_query', False)  # whether to query from question
        self.multi_hop = getattr(args, 'multi_hop', False)                    # whether to use multi-hop extension
        self.max_hop_depth = getattr(args, 'max_hop_depth', 2)                # maximum hop depth
        self.beam_size = getattr(args, 'beam_size', 3)                        # Beam Search size
        self.hop_decay_factor = getattr(args, 'hop_decay_factor', 0.7)        # hop decay factor
        self.use_query_generation = getattr(args, 'use_query_generation', True)  # whether to use query generation
        self.subgraph_weight = getattr(args, 'subgraph_weight', 1.0)          # subgraph weight

        # else
        self.ensemble = args.ensemble
        self.model_name = args.expert_model

        log_info(f"[Know] Know mode: {self.know_mode}", print_to_std=False)
        log_info(f"[Know] KG configuration: multi_hop={self.multi_hop}, use_question_query={self.use_question_query}, use_query_generation={self.use_query_generation}, subgraph_weight={self.subgraph_weight}",
                       print_to_std=False)

    def _init_knowledge_graph(self, patient_state):

        doctor_questions, patient_responses = self._collect_qa_pairs(patient_state)
        
        faiss_db = get_faiss_db()
        embeddings = get_embeddings_model()
        all_demographics, all_diseases, demographic2pdf, disease2pdf, demo_dis2pdf = get_demographics_info()
        
        if faiss_db is None or embeddings is None:
            raise RuntimeError("Database not initialized, please call initialize_db()")

        self.dynamic_kg = graph_reason.InvestigationEngine(
            faiss_db=faiss_db,
            embeddings=embeddings,
            max_queue_size=self.max_queue_size,
            relevance_threshold=self.relevance_threshold,
            llm_relevance_threshold=self.llm_relevance_threshold,
            weights={
                'embedding': self.embedding_weight,
                'llm': self.llm_weight,
                'coherence': self.coherence_weight,
                'decay': self.decay_weight
            },
            image_base_path='WHO/',
            initial_triplets=self.initial_triplets,
            direct_query_new=self.direct_query_new,
            # new parameters
            use_question_query=self.use_question_query,
            multi_hop=self.multi_hop,
            max_hop_depth=self.max_hop_depth,
            beam_size=self.beam_size,
            hop_decay_factor=self.hop_decay_factor,
            model_name=self.model_name,
            use_query_generation=self.use_query_generation,  # pass query generation configuration
            all_demographics=all_demographics,
            all_diseases=all_diseases,
            demographic2pdf=demographic2pdf,
            disease2pdf=disease2pdf,
            demo_dis2pdf=demo_dis2pdf,
            subgraph_weight=self.subgraph_weight  # pass subgraph weight
        )

        # initialize knowledge graph with question-answer pairs and doctor questions
        self.dynamic_kg.initialize_with_query(
            doctor_questions=doctor_questions,
            patient_responses=patient_responses
        )

    def _collect_qa_pairs(self, patient_state):
        doctor_questions = []
        patient_responses = []
        
        # add inquiry and initial patient information
        doctor_questions.append(self.inquiry)
        patient_responses.append(patient_state['initial_info'])
        
        # process interaction history
        for interaction in patient_state['interaction_history']:
            doctor_questions.append(interaction['question'])
            patient_responses.append(interaction['answer'])
        
        log_info(f"[Know] Collected {len(doctor_questions)} doctor questions and {len(patient_responses)} patient responses", print_to_std=False) # print
        
        return doctor_questions, patient_responses

    def _update_knowledge_graph(self, patient_state):
        if not patient_state['interaction_history']:
            return
            
        last_interaction = patient_state['interaction_history'][-1]
        
        # update knowledge graph (DynamicKnowledgeGraph will automatically handle "cannot answer" cases)
        self.dynamic_kg.update_with_new_information(
            new_patient_info=last_interaction['answer'],
            doctor_question=last_interaction['question']
        )

    def _get_current_knowledge(self):
        current_triplets, encoded_images = self.dynamic_kg.get_current_triplets()
        image_paths = [t.get('image_path', '') for t in current_triplets]
        text_knowledge = [t['content'] for t in current_triplets]

        return text_knowledge, image_paths, encoded_images


    def respond(self, patient_state, current_round, max_round=False):

        if init is False:
            if len(patient_state['interaction_history']) > 0:
                self._update_knowledge_boundary(patient_state)
        else:
            self._init_knowledge_graph(patient_state)
            init = True

        text_knowledge, image_paths, encoded_images = self._get_current_knowledge()
        is_know_expert = True

        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.scale_abstention_decision_open(
            encoded_images=encoded_images,
            text_knowledge=text_knowledge,
            is_know_expert=is_know_expert,
            know_mode=self.know_mode,
            max_round=max_round,
            ensemble=self.ensemble,
            **kwargs
        )

        if abstain_response_dict["abstain"] == 'answer':
            return {
                "type": "answer",
                "free_text_answer": abstain_response_dict["free_text_answer"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"],
                "text_knowledge": text_knowledge,
                "encoded_images": image_paths
            }

        question_response_dict = self.ask_question(
            patient_state,
            abstain_response_dict["messages"],
            is_know_expert=is_know_expert,
            encoded_images=encoded_images,
            text_knowledge=text_knowledge
        )

        # merge usage statistics
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]

        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "free_text_answer": abstain_response_dict["free_text_answer"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"],
            "intermediate_answer": f"XX"
        }