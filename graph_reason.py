import heapq
import numpy as np
import base64
from PIL import Image
import io
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any, Optional
import logging
import pandas as pd
from collections import defaultdict

from LLM_score import LLMKnowledgeEvaluator
from helper import get_response
from query_generator import QueryGenerator

def log_info(message, logger="detail_logger", print_to_std=False):
    if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(logger)
    if type(logger) != str:
        if logger: logger.info(message)
    if print_to_std: print(message + "\n")

@dataclass(order=True)
class EvidenceTriplet:
    """knowledge boundary triplet data structure"""
    priority: float
    last_updated_round: int = field(compare=False)
    triplet: Dict = field(compare=False)
    decay_factor: float = field(compare=False, default=0.9)
    llm_relevance_score: float = field(compare=False, default=0.5)
    embedding_similarity: float = field(compare=False, default=0.5)
    coherence_score: float = field(compare=False, default=0.0)
    encoded_image: Optional[str] = field(compare=False, default=None)
    subgraph_inside: bool = field(compare=False, default=False)  # mark if from subgraph

    def update_priority(self, embedding_similarity: float, llm_relevance: float,
                        coherence_score: float, current_round: int,
                        weights: Dict[str, float], subgraph_weight: float = 1.0):
        """update priority, considering multiple factors"""
        rounds_passed = current_round - self.last_updated_round
        temporal_decay = self.decay_factor ** rounds_passed

        self.embedding_similarity = embedding_similarity
        self.llm_relevance_score = llm_relevance
        self.coherence_score = coherence_score

        new_priority = (
            weights['semantic'] * embedding_similarity +
            weights['clinical'] * llm_relevance +
            weights['coherence'] * coherence_score
        )

        # if from subgraph, apply weight
        if self.subgraph_inside:
            new_priority *= subgraph_weight

        self.priority = new_priority * (1 - weights['temporal']) + self.priority * temporal_decay * weights['temporal']
        self.last_updated_round = current_round


class ImageProcessor:
    """image processing class"""
    
    def __init__(self, base_path: str = 'WHO/', max_size: Tuple[int, int] = (800, 800)):
        self.base_path = base_path
        self.max_size = max_size
        self.cache = {}
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """encode image to base64 string"""
        if not image_path:
            return None

        if image_path in self.cache:
            return self.cache[image_path]

        try:
            full_path = self.base_path + image_path if not image_path.startswith(self.base_path) else image_path

            with Image.open(full_path) as img:
                img.thumbnail(self.max_size, Image.LANCZOS)
                
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                self.cache[image_path] = encoded_image
                return encoded_image

        except Exception as e:
            log_info(f"error processing image {image_path}: {e}", print_to_std=True)
            return None


class GraphInvestigator:
    """knowledge boundary expansion class"""
    
    def __init__(self, faiss_db, embeddings, max_hop_depth: int = 2, 
                 beam_size: int = 3, hop_decay_factor: float = 0.7):
        self.faiss_db = faiss_db
        self.embeddings = embeddings
        self.max_hop_depth = max_hop_depth
        self.beam_size = beam_size
        self.hop_decay_factor = hop_decay_factor
        self.neighbor_cache = {}
    
    def expand_boundary_local(self, priority_queue: List[EvidenceTriplet], 
                       visited_edges: Set[str], k: int = 10) -> List[Dict]:
        candidate_triplets = []
        expanded_nodes = set()
        num_expanded = 0

        for prioritized_triplet in sorted(priority_queue, key=lambda x: x.priority, reverse=True):
            triplet = prioritized_triplet.triplet
            source_node = triplet['source_node']
            target_node = triplet['target_node']

            for node in [source_node, target_node]:
                if node in expanded_nodes:
                    continue
                
                expanded_nodes.add(node)
                neighbors = self._get_node_neighbors(node)

                for idx, neighbor in neighbors.iterrows():
                    edge_id = f"{neighbor['source_node']}_{neighbor['edge_type']}_{neighbor['target_node']}"
                    if edge_id in visited_edges:
                        continue
                    
                    candidate_triplets.append(neighbor.to_dict())
                
                num_expanded += len(neighbors)
                if num_expanded > k:
                    return candidate_triplets

        return candidate_triplets
    
    def expand_boundary_global(self, priority_queue: List[EvidenceTriplet], 
                         visited_edges: Set[str], k: int = 10) -> List[Dict]:
        candidate_triplets = []
        expanded_nodes = set()
        
        # initialize beam queue
        beam_queue = []
        for prioritized_triplet in sorted(priority_queue, key=lambda x: x.priority, reverse=True)[:self.beam_size]:
            triplet = prioritized_triplet.triplet
            source_node = triplet['source_node']
            target_node = triplet['target_node']
            
            beam_queue.append((source_node, 0, prioritized_triplet.priority, None))
            beam_queue.append((target_node, 0, prioritized_triplet.priority, None))
        
        beam_queue = list(set(beam_queue))

        for current_hop in range(self.max_hop_depth):
            log_info(f"expanding {current_hop+1} hop, current beam queue size: {len(beam_queue)}", print_to_std=False)
            
            next_beam = []
            
            for node_id, hop_count, score, parent_triplet in beam_queue:
                if node_id in expanded_nodes:
                    continue
                
                expanded_nodes.add(node_id)
                hop_weight = self.hop_decay_factor ** hop_count
                neighbors = self._get_node_neighbors(node_id)

                for idx, neighbor in neighbors.iterrows():
                    edge_id = f"{neighbor['source_node']}_{neighbor['edge_type']}_{neighbor['target_node']}"
                    if edge_id in visited_edges:
                        continue
                    
                    neighbor_score = score * hop_weight
                    candidate_triplets.append(neighbor.to_dict())
                    
                    next_node = neighbor['target_node'] if node_id == neighbor['source_node'] else neighbor['source_node']
                    next_beam.append((next_node, hop_count + 1, neighbor_score, neighbor.to_dict()))

            next_beam.sort(key=lambda x: x[2], reverse=True)
            beam_queue = next_beam[:self.beam_size]

            if len(candidate_triplets) >= k or not beam_queue:
                break

        log_info(f"multi-hop expansion completed, found {len(candidate_triplets)} candidate triplets", print_to_std=False)
        return candidate_triplets
    
    def _get_node_neighbors(self, node_id: str) -> pd.DataFrame:
        """get node neighbors"""
        if node_id in self.neighbor_cache:
            return self.neighbor_cache[node_id]
        
        try:
            neighbors = self.faiss_db.find_neighbors(node_id, max_hops=1, max_neighbors_per_hop=10)
            
            data = []
            for hop, hop_neighbors in neighbors.items():
                for neighbor in hop_neighbors:
                    neighbor['hop'] = hop
                    data.append(neighbor)
            
            df = pd.DataFrame(data)
            self.neighbor_cache[node_id] = df
            return df

        except Exception as e:
            log_info(f"error getting node {node_id} neighbors: {e}", print_to_std=True)
            return pd.DataFrame()


class EvidenceWeightingEngine:
    """context decision module - handling temporal decay and population reasoning"""
    
    def __init__(self, model_name: str = None, max_batch_size: int = 10):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.weights = {
            'temporal': 0.8,    # temporal decay weight
            'population': 0.2   # population context weight
        }
        # keep the same decay factor as the original code
        self.decay_factor = 0.9
        
    def compute_temporal_decay(self, current_round: int, last_updated_round: int) -> float:
        """compute temporal decay score"""
        rounds_passed = current_round - last_updated_round
        return self.decay_factor ** rounds_passed
    
    def compute_population_score(self, triplet: Dict, subgraph_inside: bool, subgraph_weight: float = 1.0) -> float:
        """compute population relevance score"""
        # keep the same population weighting logic
        return subgraph_weight if subgraph_inside else 1.0
    
    def apply_context(self, base_priority: float, old_priority: float, 
                     current_round: int, last_updated_round: int,
                     subgraph_inside: bool, subgraph_weight: float = 1.0) -> float:
        """apply context factors to priority calculation"""
        # compute temporal decay
        temporal_decay = self.compute_temporal_decay(current_round, last_updated_round)
        
        # compute population score
        population_score = self.compute_population_score(None, subgraph_inside, subgraph_weight)
        
        # keep the same calculation logic as the original code
        new_priority = base_priority * population_score
        final_priority = new_priority * (1 - self.weights['temporal']) + old_priority * temporal_decay * self.weights['temporal']
        
        return final_priority

class InvestigationEngine:
    """knowledge boundary detector main class"""
    
    def __init__(self,
                 faiss_db,
                 embeddings,
                 max_queue_size: int = 15,
                 relevance_threshold: float = 0.6,
                 llm_relevance_threshold: float = 0.2,
                 weights: Dict[str, float] = None,
                 image_base_path: str = 'WHO/',
                 max_image_size: Tuple[int, int] = (800, 800),
                 initial_triplets: int = 2,
                 direct_query_new: bool = False,
                 use_question_query: bool = False,
                 multi_hop: bool = False,
                 max_hop_depth: int = 2,
                 beam_size: int = 3,
                 hop_decay_factor: float = 0.7,
                 model_name: str = None,
                 max_batch_size: int = 10,
                 use_query_generation: bool = True,
                 all_demographics: List[str] = [],
                 all_diseases: List[str] = [],
                 demographic2pdf: Dict[str, List[str]] = {},
                 disease2pdf: Dict[str, List[str]] = {},
                 demo_dis2pdf: Dict[str, Dict[str, List[str]]] = {},
                 subgraph_weight: float = 1.0):
        
        # core components
        self.faiss_db = faiss_db
        self.embeddings = embeddings
        
        # configuration parameters
        self.max_queue_size = max_queue_size
        self.relevance_threshold = relevance_threshold
        self.llm_relevance_threshold = llm_relevance_threshold
        self.initial_triplets = initial_triplets
        self.direct_query_new = direct_query_new
        self.use_question_query = use_question_query
        self.multi_hop = multi_hop
        self.model_name = model_name
        
        # knowledge boundary evaluation weights
        self.weights = weights or {
            'semantic': 0.4,   # semantic similarity weight
            'clinical': 0.4,   # clinical relevance weight
            'coherence': 0.2,  # graph structure coherence weight
        }
        
        # initialize context decision module
        self.context_maker = EvidenceWeightingEngine(model_name, max_batch_size)
        
        # initialize other components
        self.image_processor = ImageProcessor(image_base_path, max_image_size)
        self.graph_expander = GraphInvestigator(faiss_db, embeddings, max_hop_depth, beam_size, hop_decay_factor)
        self.llm_evaluator = LLMKnowledgeEvaluator(model_name, max_batch_size)
        self.query_generator = QueryGenerator(model_name)
        self.use_query_generation = use_query_generation

        # state management
        self.priority_queue: List[EvidenceTriplet] = []
        self.visited_nodes: Set[str] = set()
        self.visited_edges: Set[str] = set()
        
        # conversation information
        self.current_round = 0
        self.current_query = ""
        self.inquiry = ""
        self.all_demographics = all_demographics
        self.all_diseases = all_diseases
        self.demographic2pdf = demographic2pdf
        self.disease2pdf = disease2pdf
        self.demo_dis2pdf = demo_dis2pdf
        self.subgraph_weight = subgraph_weight

        # node frequency tracking
        self.node_frequency: Dict[str, int] = defaultdict(int)

    def initialize_with_query(self, doctor_questions: List[str] = None,
                              patient_responses: List[str] = None):
        """use initial query to initialize knowledge graph"""
        self.inquiry = doctor_questions[0]
        
        # collect all query results
        all_query_results = self._collect_initial_query_results(doctor_questions, patient_responses)
        
        # batch evaluate all triplets
        evaluated_triplets = self._batch_evaluate_triplets(all_query_results)
        
        # select and build priority queue
        self._establish_evidence_pool(evaluated_triplets)
        
        self._log_initialization_results()
    
    def update_with_new_information(self, new_patient_info: str, doctor_question: str = None):
        """update knowledge graph with new information"""
        if "cannot answer" in new_patient_info.lower() or "can't answer" in new_patient_info.lower():
            log_info("patient cannot answer, skip knowledge graph update", print_to_std=False)
            return
        
        self._update_round_info(new_patient_info, doctor_question)
        
        # generate update queries (if there is a doctor question, generate queries; otherwise, directly use patient information)
        update_queries, evaluation_context = self._generate_update_queries(new_patient_info, doctor_question)
        
        # use generated queries to update existing triplets
        self._update_existing_triplets_with_queries(update_queries, evaluation_context)
        
        # use generated queries to collect new candidate triplets
        new_candidates = self._collect_new_candidates_with_queries(update_queries, evaluation_context, new_patient_info, doctor_question)
        
        # update priority queue
        self._update_priority_queue(new_candidates)
        
        self._log_update_results()
    
    def _generate_update_queries(self, new_patient_info: str, doctor_question: str = None) -> Tuple[List[str], str]:
        """generate update queries and evaluation context"""
        if doctor_question:
            # if there is a doctor question, create QA pair
            qa_context = f"Question: {doctor_question}\nPatient Response: {new_patient_info}"
            
            if self.use_query_generation:
                # generate optimized queries
                generated_queries = self.query_generator.generate_search_queries(
                    qa_context, self.inquiry, "qa_pair"
                )
                log_info(f"generate queries for QA pair: {generated_queries}", print_to_std=False) # print
                return generated_queries, qa_context
            else:
                # directly use QA pair text
                return [qa_context], qa_context
        else:
            # if there is no doctor question, directly process patient information
            if self.use_query_generation:
                generated_queries = self.query_generator.generate_search_queries(
                    new_patient_info, self.inquiry, "patient_info"
                )
                log_info(f"generate queries for patient information: {generated_queries}", print_to_std=False) # print
                return generated_queries, new_patient_info
            else:
                return [new_patient_info], new_patient_info
    
    def get_current_triplets(self) -> Tuple[List[Dict], List[Optional[str]]]:
        """get current priority queue of triplets, sorted by priority"""
        sorted_triplets = sorted(self.priority_queue, key=lambda x: x.priority, reverse=True)
        return [item.triplet for item in sorted_triplets], [item.encoded_image for item in sorted_triplets]
    
    def _collect_initial_query_results(self, doctor_questions: List[str], patient_responses: List[str]) -> List[Tuple[str, List[Tuple]]]:
        """collect all retrieval results for initial queries - reformatted as question-answer pairs"""
        all_results = []
        patient_profile = ""
        
        # process question-answer pairs
        for i, (question, response) in enumerate(zip(doctor_questions, patient_responses)):
            # skip cases where the answer is "cannot answer" or "can't answer"
            if "cannot answer" in response.lower() or "can't answer" in response.lower():
                log_info(f"skipping question that cannot be answered: {question}", print_to_std=False)
                continue
            
            # create context for question-answer pairs
            qa_context = f"Question: {question}\nPatient Response: {response}"
            patient_profile += response
            
            # select query strategy based on configuration
            if self.use_query_generation:
                # generate optimized queries
                generated_queries = self.query_generator.generate_search_queries(
                    qa_context, self.inquiry, "qa_pair"
                )
                
                for query in generated_queries:
                    query_results = self.faiss_db.similarity_search_with_relevance_scores(
                        query, k=self.initial_triplets
                    )
                    if query_results:
                        all_results.append((qa_context, query_results))
                        log_info(f"QA pair query {i+1}: {query} -> {len(query_results)} results", print_to_std=False) # print
            else:
                # directly use QA pair text query
                query_results = self.faiss_db.similarity_search_with_relevance_scores(
                    qa_context, k=self.initial_triplets
                )
                if query_results:
                    all_results.append((qa_context, query_results))
                    log_info(f"QA pair direct query {i+1}: {len(query_results)} results", print_to_std=False) # print

        # use LLM to extract demographic and disease information from patient_profile (if applicable)
        demographic_info, disease_info = self.llm_evaluator.extract_demographic_disease_info(patient_profile, self.all_demographics, self.all_diseases)
        
        condition_pdfs = []
        if demographic_info:
            for demographic in demographic_info:
                if demographic in self.demographic2pdf: condition_pdfs.extend(self.demographic2pdf[demographic])
        if disease_info:
            for disease in disease_info:
                if disease in self.disease2pdf: condition_pdfs.extend(self.disease2pdf[disease])
        if len(condition_pdfs) > 0:
            # use LLM to generate optimized queries for all patient_profiles
            qa_context = f"Question: {self.inquiry}\nPatient Response: {patient_profile}\nPatient Demographic: {demographic_info}\nPatient Disease: {disease_info}"
            generated_queries = self.query_generator.generate_search_queries(
                qa_context, self.inquiry, "patient_profile"
            )

            # use generated queries to retrieve relevant knowledge
            for query in generated_queries:
                query_results = self.faiss_db.similarity_search_with_subgraph_condition(
                    query, times=10, max_search=200, k=self.initial_triplets, subgraph_conditions=condition_pdfs
                )
                if query_results:
                    all_results.append((qa_context, query_results))
                    log_info(f"generated query with demographic_info and disease_info: {query} -> {len(query_results)} results", print_to_std=False) # print

        return all_results

    def _batch_evaluate_triplets(self, query_results: List[Tuple[str, List[Tuple]]]) -> List[Dict]:
        """batch evaluate all triplets"""
        all_evaluated = []
        
        for context, results in query_results:
            if not results:
                continue
                
            # prepare triplets for evaluation
            triplets_to_evaluate = []
            for doc, emb_score in results:
                triplet_info = self._extract_triplet_info(doc)
                triplet_info['embedding_score'] = emb_score
                
                # check if from subgraph retrieval (by checking if context contains demographic/disease information)
                is_subgraph = "Patient Demographic:" in context or "Patient Disease:" in context
                triplet_info['subgraph_inside'] = is_subgraph
                
                triplets_to_evaluate.append(triplet_info)
            
            # batch LLM evaluation
            relevance_results = self.llm_evaluator.evaluate_batch(context, self.inquiry, triplets_to_evaluate)
            
            # merge results
            for i, triplet in enumerate(triplets_to_evaluate):
                llm_score = next((r['score'] for r in relevance_results if r['index'] == i), 0.5)
                
                if llm_score >= self.llm_relevance_threshold:
                    triplet['llm_score'] = llm_score
                    triplet['context'] = context
                    all_evaluated.append(triplet)
        
        return all_evaluated
    
    def _establish_evidence_pool(self, evaluated_triplets: List[Dict]):
        """build initial priority queue"""
        seen_content = set()
        selected_triplets = []
        
        # calculate combined score and sort
        for triplet in evaluated_triplets:
            combined_score = (
                self.weights['semantic'] * triplet['embedding_score'] +
                self.weights['clinical'] * triplet['llm_score']
            )
            triplet['combined_score'] = combined_score
        
        # sort by combined score
        evaluated_triplets.sort(key=lambda x: x['combined_score'], reverse=True)
        
        for triplet in evaluated_triplets:
            content = triplet['content']
            if content in seen_content or len(selected_triplets) >= self.max_queue_size:
                continue
            
            # create priority triplet
            priority_triplet = self._create_prioritized_triplet(triplet)
            selected_triplets.append(priority_triplet)
            seen_content.add(content)
            
            # update visited records
            self._update_visited_records(triplet)
        
        self.priority_queue = selected_triplets
        heapq.heapify(self.priority_queue)
    
    def _update_round_info(self, new_patient_info: str, new_doctor_question: str):
        """update round information - modified to handle question-answer pairs"""
        self.current_round += 1
        
        log_info("\n" + "="*80, print_to_std=False)
        log_info(f"start updating knowledge graph - conversation round {self.current_round}", print_to_std=False)
        log_info(f"new patient information: {new_patient_info[:100]}..." if len(new_patient_info) > 100 else new_patient_info, print_to_std=False)
        log_info("="*80, print_to_std=False)
        
        # if patient cannot answer the question, set special mark
        self.current_query = f"Question: {new_doctor_question}\nPatient Response: {new_patient_info}"
    
    def _update_existing_triplets_with_queries(self, update_queries: List[str], evaluation_context: str):
        """use generated queries to update the priority of existing triplets"""
        if not self.priority_queue:
            return
        
        log_info(f"use {len(update_queries)} queries to update the relevance of existing knowledge", print_to_std=False)
        
        # prepare existing triplets for re-evaluation
        existing_triplets = [pt.triplet for pt in self.priority_queue]
        
        # batch re-evaluation - use evaluation context
        relevance_results = self.llm_evaluator.evaluate_batch(
            evaluation_context, self.inquiry, existing_triplets
        )
        
        # calculate average embedding of all queries (or use the first query)
        if len(update_queries) == 1:
            query_embedding = self.embeddings.embed_query(update_queries[0])
        else:
            # use average embedding of multiple queries
            query_embeddings = [self.embeddings.embed_query(q) for q in update_queries]
            query_embedding = [
                sum(emb[i] for emb in query_embeddings) / len(query_embeddings)
                for i in range(len(query_embeddings[0]))
            ]
        
        # update the priority of each triplet
        for i, prioritized_triplet in enumerate(self.priority_queue):
            llm_score = next((r['score'] for r in relevance_results if r['index'] == i), 0.5)
            
            # calculate new embedding similarity
            triplet_content = prioritized_triplet.triplet['content']
            triplet_embedding = self.embeddings.embed_query(triplet_content)
            similarity = self._calculate_similarity(query_embedding, triplet_embedding)
            
            # calculate coherence score
            coherence_score = self._calculate_coherence_score(prioritized_triplet.triplet)
            
            # update priority
            prioritized_triplet.update_priority(
                embedding_similarity=similarity,
                llm_relevance=llm_score,
                coherence_score=coherence_score,
                current_round=self.current_round,
                weights=self.weights,
                subgraph_weight=self.subgraph_weight
            )
        
        heapq.heapify(self.priority_queue)
        log_info(f"completed updating priority of {len(self.priority_queue)} knowledge", print_to_std=False) # print
    
    def _collect_new_candidates_with_queries(self, update_queries: List[str], evaluation_context: str, 
                                       new_patient_info: str, doctor_question: str = None) -> List[EvidenceTriplet]:
        """use generated queries to collect new candidate triplets"""
        all_candidates = []
        
        # 1. direct retrieval candidate (if enabled)
        if self.direct_query_new:
            direct_candidates = self._get_direct_query_candidates_with_queries(update_queries, evaluation_context)
            all_candidates.extend(direct_candidates)
        
        # 2. question retrieval candidate (if enabled and there is a doctor question)
        if self.use_question_query and doctor_question:
            question_candidates = self._get_question_query_candidates_with_queries(
                update_queries, evaluation_context, doctor_question
            )
            all_candidates.extend(question_candidates)
        
        # 3. graph expansion candidate
        graph_candidates = self._get_graph_expansion_candidates_with_queries(update_queries, evaluation_context)
        all_candidates.extend(graph_candidates)
        
        return all_candidates

    def _get_direct_query_candidates_with_queries(self, update_queries: List[str], evaluation_context: str) -> List[EvidenceTriplet]:
        """use updated queries to get direct retrieval candidates"""
        log_info("use updated queries to get direct retrieval candidates...", print_to_std=False) # print
        
        all_candidates = []
        
        for i, query in enumerate(update_queries):
            direct_results = self.faiss_db.similarity_search_with_relevance_scores(
                query, k=max(self.initial_triplets//2, 2)
            )
            
            if direct_results:
                candidates = self._process_query_candidates(
                    evaluation_context, direct_results, f"direct-query-{i+1}"
                )
                all_candidates.extend(candidates)
        
        return all_candidates

    def _get_question_query_candidates_with_queries(self, update_queries: List[str], evaluation_context: str, 
                                               doctor_question: str) -> List[EvidenceTriplet]:
        """use doctor question to get retrieval candidates"""
        log_info(f"use doctor question to get retrieval candidates: {doctor_question}", print_to_std=False) # print
        
        all_candidates = []
        
        if self.use_query_generation:
            # generate queries for doctor question separately
            question_queries = self.query_generator.generate_search_queries(
                doctor_question, self.inquiry, "doctor_question"
            )
            
            for i, query in enumerate(question_queries):
                question_results = self.faiss_db.similarity_search_with_relevance_scores(
                    query, k=max(1, self.initial_triplets//2)
                )
                
                if question_results:
                    candidates = self._process_query_candidates(
                        evaluation_context, question_results, f"question-query-{i+1}"
                    )
                    all_candidates.extend(candidates)
        else:
            # use doctor question directly
            question_results = self.faiss_db.similarity_search_with_relevance_scores(
                doctor_question, k=max(1, self.initial_triplets//2)
            )
            
            if question_results:
                candidates = self._process_query_candidates(
                    evaluation_context, question_results, "question-direct-query"
                )
                all_candidates.extend(candidates)
        
        # weight for question retrieval is slightly higher
        for candidate in all_candidates:
            candidate.priority *= 1.1
        
        return all_candidates

    def _get_graph_expansion_candidates_with_queries(self, update_queries: List[str], evaluation_context: str) -> List[EvidenceTriplet]:
        """use updated queries to get graph expansion candidates"""
        if self.multi_hop:
            candidate_triplets = self.graph_expander.expand_multi_hop(
                self.priority_queue, self.visited_edges
            )
            log_info(f"found {len(candidate_triplets)} multi-hop candidate knowledge nodes", print_to_std=False)
        else:
            candidate_triplets = self.graph_expander.expand_one_hop(
                self.priority_queue, self.visited_edges
            )
            log_info(f"found {len(candidate_triplets)} 1-hop candidate knowledge nodes", print_to_std=False)
        
        if not candidate_triplets:
            return []
        
        # pre-filter: use embedding similarity
        if len(candidate_triplets) > 10:
            # calculate average embedding of all queries for filtering
            if len(update_queries) == 1:
                query_embedding = self.embeddings.embed_query(update_queries[0])
            else:
                query_embeddings = [self.embeddings.embed_query(q) for q in update_queries]
                query_embedding = [
                    sum(emb[i] for emb in query_embeddings) / len(query_embeddings)
                    for i in range(len(query_embeddings[0]))
                ]
            
            embedding_scores = []
            for triplet in candidate_triplets:
                content = triplet['content']
                triplet_embedding = self.embeddings.embed_query(content)
                similarity = self._calculate_similarity(query_embedding, triplet_embedding)
                embedding_scores.append((triplet, similarity))
            
            embedding_scores.sort(key=lambda x: x[1], reverse=True)
            filtered_candidates = [item[0] for item in embedding_scores[:20]]
        else:
            filtered_candidates = candidate_triplets
        
        # use evaluation context for LLM evaluation
        return self._process_query_candidates(
            evaluation_context, 
            [(triplet, 0.5) for triplet in filtered_candidates], 
            "graph-expansion", 
            is_triplet_dict=True
        )

    def _process_query_candidates(self, context: str, results: List[Tuple], 
                                candidate_type: str, is_triplet_dict: bool = False) -> List[EvidenceTriplet]:
        """process query candidate results"""
        if not results:
            return []
        
        # prepare triplets for evaluation
        triplets_to_evaluate = []
        embedding_scores = []
        
        for item, score in results:
            if is_triplet_dict:
                triplet_info = item
                embedding_score = score
            else:
                triplet_info = self._extract_triplet_info(item)
                embedding_score = score
            
            # skip visited edges
            edge_id = f"{triplet_info['source_node']}_{triplet_info['edge_type']}_{triplet_info['target_node']}"
            if edge_id in self.visited_edges:
                continue
            
            # check if from subgraph (by checking condition_pdfs)
            subgraph_inside = False
            if candidate_type == "graph-expansion":
                # for graph expansion, check if satisfies subgraph condition
                image_path = triplet_info.get('image_path', '')
                if image_path:
                    pdf_name = 'WHO/' + '/'.join(image_path.split('/')[:-1])
                    # check if in demographic or disease related PDFs
                    for demographic in self.all_demographics:
                        if demographic in self.demographic2pdf and pdf_name in self.demographic2pdf[demographic]:
                            subgraph_inside = True
                            break
                    if not subgraph_inside:
                        for disease in self.all_diseases:
                            if disease in self.disease2pdf and pdf_name in self.disease2pdf[disease]:
                                subgraph_inside = True
                                break
            
            triplet_info['subgraph_inside'] = subgraph_inside
            triplets_to_evaluate.append(triplet_info)
            embedding_scores.append(embedding_score)
        
        if not triplets_to_evaluate:
            return []
        
        # batch LLM evaluation
        relevance_results = self.llm_evaluator.evaluate_batch(context, self.inquiry, triplets_to_evaluate)
        
        # create candidate triplets
        candidates = []
        query_embedding = self.embeddings.embed_query(context)
        
        for i, triplet in enumerate(triplets_to_evaluate):
            llm_score = next((r['score'] for r in relevance_results if r['index'] == i), 0.5)
            
            if llm_score < self.llm_relevance_threshold:
                continue
            
            # calculate embedding similarity
            if not is_triplet_dict:
                content = triplet['content']
                triplet_embedding = self.embeddings.embed_query(content)
                similarity = self._calculate_similarity(query_embedding, triplet_embedding)
            else:
                similarity = embedding_scores[i]
            
            # create candidate triplet
            triplet['embedding_score'] = similarity
            triplet['llm_score'] = llm_score
            
            priority_triplet = self._create_prioritized_triplet(triplet)
            candidates.append(priority_triplet)
            
            # log_info(f"{candidate_type} knowledge #{i+1}/{len(triplets_to_evaluate)}:", print_to_std=True)
            # log_info(f"  - content: {triplet['content']}", print_to_std=True)
            # log_info(f"  - embedding similarity: {similarity:.4f} - LLM relevance: {llm_score:.4f}", print_to_std=True)
            # log_info(f"  - priority: {priority_triplet.priority:.4f}", print_to_std=True)
        
        return candidates
    
    def _consolidate_evidence(self, new_candidates: List[EvidenceTriplet]):
        if not new_candidates:
            return
        
        new_candidates.sort(key=lambda x: x.priority, reverse=True)
        
        original_queue_size = len(self.priority_queue)
        added_count = 0
        replaced_count = 0
        
        for candidate in new_candidates:
            if len(self.priority_queue) < self.max_queue_size:
                heapq.heappush(self.priority_queue, candidate)
                added_count += 1
                self._update_visited_records(candidate.triplet)
            else:
                if candidate.priority > self.priority_queue[0].priority:
                    heapq.heappop(self.priority_queue)
                    heapq.heappush(self.priority_queue, candidate)
                    replaced_count += 1
                    self._update_visited_records(candidate.triplet)
        
        log_info(f"queue updated: added {added_count} new knowledge, replaced {replaced_count} old knowledge", print_to_std=False) # print
        log_info(f"queue size changed: {original_queue_size} -> {len(self.priority_queue)}", print_to_std=False) # print
    
    def _extract_triplet_info(self, doc) -> Dict:
        """extract triplet information from document"""
        return {
            'content': doc.page_content,
            'source_node': doc.metadata.get('source_node', ''),
            'edge_type': doc.metadata.get('edge_type', ''),
            'target_node': doc.metadata.get('target_node', ''),
            'image_path': doc.metadata.get('image_path', ''),
            'original_index': doc.metadata.get('original_index', ''),
            'content_hash': doc.metadata.get('content_hash', '')
        }
    
    def _create_prioritized_triplet(self, triplet_info: Dict) -> EvidenceTriplet:
        """create prioritized triplet"""
        # calculate base knowledge boundary score
        base_score = self.compute_boundary_score(triplet_info)
        
        # get image encoding
        image_path = triplet_info.get('image_path', '')
        encoded_image = self.image_processor.encode_image(image_path) if image_path else None
        
        # get subgraph mark
        subgraph_inside = triplet_info.get('subgraph_inside', False)
        
        # apply context factors (keep original logic)
        final_priority = self.context_maker.apply_context(
            base_priority=base_score,
            old_priority=0.0,  # new created triplet has no history priority
            current_round=self.current_round,
            last_updated_round=self.current_round,
            subgraph_inside=subgraph_inside,
            subgraph_weight=self.subgraph_weight
        )
        
        return EvidenceTriplet(
            priority=final_priority,
            last_updated_round=self.current_round,
            triplet=triplet_info,
            llm_relevance_score=triplet_info['llm_score'],
            embedding_similarity=triplet_info['embedding_score'],
            coherence_score=self._calculate_coherence_score(triplet_info),
            encoded_image=encoded_image,
            subgraph_inside=subgraph_inside
        )
    
    def _update_visited_records(self, triplet: Dict):
        """update visited records and node frequency"""
        source_node = triplet['source_node']
        target_node = triplet['target_node']
        
        # update visited records
        self.visited_nodes.add(source_node)
        self.visited_nodes.add(target_node)
        
        # update node frequency
        self.node_frequency[source_node] += 1
        self.node_frequency[target_node] += 1
        
        # update visited edges
        edge_id = f"{source_node}_{triplet['edge_type']}_{target_node}"
        self.visited_edges.add(edge_id)
    
    def _calculate_similarity(self, query_embedding, triplet_embedding):
        """calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(query_embedding, triplet_embedding))
        query_norm = sum(a * a for a in query_embedding) ** 0.5
        triplet_norm = sum(b * b for b in triplet_embedding) ** 0.5
        
        if query_norm * triplet_norm == 0:
            return 0
        
        return dot_product / (query_norm * triplet_norm)
    
    # def _calculate_coherence_score(self, triplet: Dict) -> float:
    #     """calculate coherence score of triplet with existing knowledge"""
    #     source_node = triplet['source_node']
    #     target_node = triplet['target_node']
        
    #     connections = 0
    #     for node in [source_node, target_node]:
    #         if node in self.visited_nodes:
    #             connections += 1
        
    #     if len(self.visited_nodes) == 0:
    #         return 0
        
    #     return connections / 2
    def _calculate_coherence_score(self, triplet: Dict) -> float:
        if len(self.visited_nodes) == 0:
            return 0.1  # give a small base score to avoid cold start
        
        source_node = triplet['source_node']
        target_node = triplet['target_node']
        
        # consider node frequency
        source_freq = self.node_frequency.get(source_node, 0)
        target_freq = self.node_frequency.get(target_node, 0)
        
        # weighted coherence score
        coherence = (min(source_freq, 1) + min(target_freq, 1)) / 2
        return coherence

    def _log_initialization_results(self):
        """log initialization results"""
        log_info(f"knowledge graph initialized, selected {len(self.priority_queue)} triplets", print_to_std=False) # print
        log_info("\nknowledge queue details after initialization:", print_to_std=False) # print
        
        for i, pt in enumerate(sorted(self.priority_queue, key=lambda x: x.priority, reverse=True)):
            subgraph_mark = " [Subgraph]" if pt.subgraph_inside else ""
            log_info(f"knowledge #{i+1} - priority: {pt.priority:.4f}{subgraph_mark}", print_to_std=False) # print
            log_info(f"  - content: {pt.triplet['content']}", print_to_std=False) # print
            log_info(f"  - source node: {pt.triplet['source_node']}, target node: {pt.triplet['target_node']}", print_to_std=False) # print
            log_info(f"  - embedding similarity: {pt.embedding_similarity:.4f} - LLM relevance: {pt.llm_relevance_score:.4f}", print_to_std=False) # print
            log_info(f"  - coherence score: {pt.coherence_score:.4f} - round: {pt.last_updated_round}", print_to_std=False) # print
    
    def _log_update_results(self):
        """log update results"""
        log_info("\nknowledge graph queue status after update:", print_to_std=False) # print
        
        for i, pt in enumerate(sorted(self.priority_queue, key=lambda x: x.priority, reverse=True)):
            rounds_decayed = self.current_round - pt.last_updated_round
            subgraph_mark = " [Subgraph]" if pt.subgraph_inside else ""
            log_info(f"knowledge #{i+1} - priority: {pt.priority:.4f} - rounds decayed: {rounds_decayed}{subgraph_mark}", print_to_std=False) # print
            log_info(f"  - content: {pt.triplet['content']}", print_to_std=False) # print
            log_info(f"  - source node: {pt.triplet['source_node']}, target node: {pt.triplet['target_node']}", print_to_std=False) # print
            log_info(f"  - embedding similarity: {pt.embedding_similarity:.4f} - LLM relevance: {pt.llm_relevance_score:.4f}", print_to_std=False) # print
            log_info(f"  - coherence score: {pt.coherence_score:.4f}", print_to_std=False) # print


class KnowRetriever:
    def __init__(self, faiss_db, know_mode, model_name, use_query_generation, k, image_base_path, max_image_size, inquiry):
        self.know_mode = know_mode
        self.model_name = model_name
        self.use_query_generation = use_query_generation
        self.k = k
        self.image_processor = ImageProcessor(image_base_path, max_image_size)
        self.query_generator = QueryGenerator(model_name)
        self.inquiry = inquiry
        self.faiss_db = faiss_db
        
    def get_text_knowledge(self, doctor_question, patient_response):
        # selection:
        qa_context = f"Question: {doctor_question}\nPatient Response: {patient_response}"
        if self.use_query_generation:
            # generate optimized queries
            generated_queries = self.query_generator.generate_search_queries(
                qa_context, self.inquiry, "qa_pair"
            )
            
            for query in generated_queries:
                query_results = self.faiss_db.similarity_search_with_relevance_scores(
                    query, k=self.k
                )
                if query_results:
                    log_info(f"QA query {len(query_results)} results", print_to_std=False) # print
        else:
            # directly use QA text query
            query_results = self.faiss_db.similarity_search_with_relevance_scores(
                qa_context, k=self.k
            )
            if query_results:
                log_info(f"QA direct query {len(query_results)} results", print_to_std=False) # print

        if query_results:
            text_knowledge = []
            for doc, score in query_results:
              text_knowledge.append(doc.page_content)
            
            return text_knowledge
        else:
            return None
    
    def get_multimodal_knowledge(self, doctor_question, patient_response):
        qa_context = f"Question: {doctor_question}\nPatient Response: {patient_response}"
        if self.use_query_generation:
            generated_queries = self.query_generator.generate_search_queries(
                qa_context, self.inquiry, "qa_pair"
            )

            for query in generated_queries:
                query_results = self.faiss_db.similarity_search_with_relevance_scores(
                    query, k=self.k
                )
                if query_results:
                    log_info(f"QA query {len(query_results)} results", print_to_std=False) # print
        else:
            # directly use QA text query
            query_results = self.faiss_db.similarity_search_with_relevance_scores(
                qa_context, k=self.k
            )
            if query_results:
                log_info(f"QA direct query {len(query_results)} results", print_to_std=False) # print

        if query_results:
            text_knowledge = []
            image_paths = []
            encoded_images = []
            for doc, score in query_results:
                  text_knowledge.append(doc.page_content)
                  image_path = doc.metadata.get('image_path', '')
                  image_paths.append(image_path)
                  encoded_image = self.image_processor.encode_image(image_path) if image_path else None
                  encoded_images.append(encoded_image)
        
            return text_knowledge, image_paths, encoded_images
        else:
            return None, None, None
            
