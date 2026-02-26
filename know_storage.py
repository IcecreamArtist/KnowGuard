
import hashlib
import os
import numpy as np
import faiss
import pickle
import pandas as pd
from langchain_core.documents import Document
from tqdm import tqdm
from helper import get_embeddings
import pandas as pd
import os
import json
import re


faiss_db = None
embeddings = None

def parse_pdf_names(pdf_string):
    matches = re.findall(r'\{([^}]+)\}', pdf_string)
    return [match.strip() for match in matches]


def create_faiss_vector_db(all_data_clean, embeddings, persist_directory="faiss_db_text-embedding-3-small"):
    """
    Create a FAISS vector database, using content hash instead of index to identify documents, and save the graph structure
    """
    os.makedirs(persist_directory, exist_ok=True)

    index_path = os.path.join(persist_directory, "index.faiss")
    documents_path = os.path.join(persist_directory, "documents.pkl")
    metadata_path = os.path.join(persist_directory, "metadata.pkl")
    content_hash_path = os.path.join(persist_directory, "content_hashes.pkl")
    graph_path = os.path.join(persist_directory, "graph_structure.pkl")

    existing_content_hashes = set()
    graph_structure = {}

    if os.path.exists(index_path) and os.path.exists(documents_path) and os.path.exists(metadata_path):
        print("Loading existing FAISS database...")

        if os.path.exists(content_hash_path):
            with open(content_hash_path, 'rb') as f:
                existing_content_hashes = pickle.load(f)

        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                graph_structure = pickle.load(f)

        with open(documents_path, 'rb') as f:
            existing_documents = pickle.load(f)

        with open(metadata_path, 'rb') as f:
            existing_metadata = pickle.load(f)

        existing_index = faiss.read_index(index_path)

        print(f"Existing vector number: {existing_index.ntotal}")
        print(f"Existing unique content number: {len(existing_content_hashes)}")
        print(f"Existing node number: {len(graph_structure)}")

        new_vectors = []
        new_documents = []
        new_metadata = []
        hash_to_index_map = {existing_metadata[i]['content_hash']: i for i in range(len(existing_metadata))}

        added_count = 0
        skipped_count = 0

        for idx, row in tqdm(all_data_clean.iterrows(), total=len(all_data_clean), desc="Processing data"):
            try:
                if pd.isna(row['relevant_description']) or row['relevant_description'] is None:
                    text = ""
                else:
                    text = str(row['relevant_description']).strip()

                if not text:
                    skipped_count += 1
                    continue

                source_node = str(row['x_name']) if pd.notna(row['x_name']) else ""
                edge_type = str(row['y_name']) if pd.notna(row['y_name']) else ""
                target_node = str(row['relationship']) if pd.notna(row['relationship']) else ""

                if not (source_node and target_node):
                    skipped_count += 1
                    continue

                content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

                if content_hash in existing_content_hashes:
                    if source_node not in graph_structure:
                        graph_structure[source_node] = []

                    edge_info = (edge_type, target_node, content_hash)
                    if edge_info not in graph_structure[source_node]:
                        graph_structure[source_node].append(edge_info)

                    if target_node not in graph_structure:
                        graph_structure[target_node] = []

                    skipped_count += 1
                    continue

                image_path = row['image_path'] if pd.notna(row['image_path']) else ""

                vector = embeddings.embed_query(text)

                metadata = {
                    "content_hash": content_hash,
                    "original_index": idx,
                    "image_path": image_path,
                    "source_node": source_node,
                    "edge_type": edge_type,
                    "target_node": target_node
                }

                new_vectors.append(vector)
                new_documents.append(text)
                new_metadata.append(metadata)

                existing_content_hashes.add(content_hash)

                if source_node not in graph_structure:
                    graph_structure[source_node] = []

                edge_info = (edge_type, target_node, content_hash)
                if edge_info not in graph_structure[source_node]:
                    graph_structure[source_node].append(edge_info)

                if target_node not in graph_structure:
                    graph_structure[target_node] = []

                added_count += 1

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        if new_vectors:
            print(f"Added {added_count} new documents, skipped {skipped_count} duplicate or empty documents...")

            new_vectors_np = np.array(new_vectors).astype('float32')
            existing_index.add(new_vectors_np)

            all_documents = existing_documents + new_documents
            all_metadata = existing_metadata + new_metadata

            for i in range(len(existing_metadata), len(all_metadata)):
                hash_to_index_map[all_metadata[i]['content_hash']] = i

            faiss.write_index(existing_index, index_path)

            with open(documents_path, 'wb') as f:
                pickle.dump(all_documents, f)

            with open(metadata_path, 'wb') as f:
                pickle.dump(all_metadata, f)

            with open(content_hash_path, 'wb') as f:
                pickle.dump(existing_content_hashes, f)

            with open(graph_path, 'wb') as f:
                pickle.dump(graph_structure, f)

            print(f"Database updated, now {existing_index.ntotal} vectors, {len(graph_structure)} nodes")

            def search_function(query, k=10, expand_nodes=False, max_neighbors=5):
                """
                Enhanced search function
                - query: query text
                - k: number of returned matches
                - expand_nodes: whether to expand related nodes
                - max_neighbors: maximum number of neighbors returned for each node
                """
                query_vector = embeddings.embed_query(query)
                query_vector_np = np.array([query_vector]).astype('float32')

                scores, indices = existing_index.search(query_vector_np, k)

                results = []
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(all_documents):
                        result = {
                            'content': all_documents[idx],
                            'metadata': all_metadata[idx],
                            'relevance_score': float(scores[0][i]),
                            'original_index': all_metadata[idx]['original_index'],
                            'image_path': all_metadata[idx]['image_path'],
                            'source_node': all_metadata[idx]['source_node'],
                            'edge_type': all_metadata[idx]['edge_type'],
                            'target_node': all_metadata[idx]['target_node']
                        }

                        if expand_nodes:
                            source = all_metadata[idx]['source_node']
                            target = all_metadata[idx]['target_node']

                            neighbors = []
                            if source in graph_structure:
                                for edge_type, target_node, content_hash in graph_structure[source][:max_neighbors]:
                                    if content_hash in hash_to_index_map:
                                        neighbor_idx = hash_to_index_map[content_hash]
                                        neighbors.append({
                                            'content': all_documents[neighbor_idx],
                                            'metadata': all_metadata[neighbor_idx],
                                            'relation': f"Edges from the same source node: {source} --{edge_type}--> {target_node}"
                                        })

                            for node, edges in graph_structure.items():
                                for edge_type, node_target, content_hash in edges:
                                    if node_target == target and content_hash in hash_to_index_map:
                                        neighbor_idx = hash_to_index_map[content_hash]
                                        neighbors.append({
                                            'content': all_documents[neighbor_idx],
                                            'metadata': all_metadata[neighbor_idx],
                                            'relation': f"Edges to the same target node: {node} --{edge_type}--> {target}"
                                        })
                                        if len(neighbors) >= max_neighbors * 2:
                                            break
                                if len(neighbors) >= max_neighbors * 2:
                                    break

                            result['neighbors'] = neighbors

                        results.append(result)

                return pd.DataFrame(results)

            return search_function

        else:
            print(f"No new documents to add, skipped {skipped_count} duplicate or empty documents")

            with open(graph_path, 'wb') as f:
                pickle.dump(graph_structure, f)

            def search_function(query, k=10, expand_nodes=False, max_neighbors=5):
                """
                Enhanced search function
                - query: query text
                - k: number of returned matches
                - expand_nodes: whether to expand related nodes
                - max_neighbors: maximum number of neighbors returned for each node
                """
                query_vector = embeddings.embed_query(query)
                query_vector_np = np.array([query_vector]).astype('float32')

                scores, indices = existing_index.search(query_vector_np, k)

                results = []
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(existing_documents):
                        result = {
                            'content': existing_documents[idx],
                            'metadata': existing_metadata[idx],
                            'relevance_score': float(scores[0][i]),
                            'original_index': existing_metadata[idx]['original_index'],
                            'image_path': existing_metadata[idx]['image_path'],
                            'source_node': existing_metadata[idx]['source_node'],
                            'edge_type': existing_metadata[idx]['edge_type'],
                            'target_node': existing_metadata[idx]['target_node']
                        }

                        if expand_nodes:
                            source = existing_metadata[idx]['source_node']
                            target = existing_metadata[idx]['target_node']

                            neighbors = []
                            if source in graph_structure:
                                for edge_type, target_node, content_hash in graph_structure[source][:max_neighbors]:
                                    if content_hash in hash_to_index_map:
                                        neighbor_idx = hash_to_index_map[content_hash]
                                        neighbors.append({
                                            'content': existing_documents[neighbor_idx],
                                            'metadata': existing_metadata[neighbor_idx],
                                            'relation': f"Edges from the same source node: {source} --{edge_type}--> {target_node}"
                                        })

                            for node, edges in graph_structure.items():
                                for edge_type, node_target, content_hash in edges:
                                    if node_target == target and content_hash in hash_to_index_map:
                                        neighbor_idx = hash_to_index_map[content_hash]
                                        neighbors.append({
                                            'content': existing_documents[neighbor_idx],
                                            'metadata': existing_metadata[neighbor_idx],
                                            'relation': f"Edges to the same target node: {node} --{edge_type}--> {target}"
                                        })
                                        if len(neighbors) >= max_neighbors * 2:
                                            break
                                if len(neighbors) >= max_neighbors * 2:
                                    break

                            result['neighbors'] = neighbors

                        results.append(result)

                return pd.DataFrame(results)

            return search_function

    else:
        print("Creating new FAISS database...")

        vectors = []
        documents = []
        metadata_list = []
        content_hashes = set()
        hash_to_index_map = {}

        added_count = 0
        skipped_count = 0

        for idx, row in tqdm(all_data_clean.iterrows(), total=len(all_data_clean), desc="Processing data"):
            try:
                if pd.isna(row['relevant_description']) or row['relevant_description'] is None:
                    text = ""
                else:
                    text = str(row['relevant_description']).strip()

                if not text:
                    skipped_count += 1
                    continue

                source_node = str(row['x_name']) if pd.notna(row['x_name']) else ""
                edge_type = str(row['relationship']) if pd.notna(row['relationship']) else ""
                target_node = str(row['y_name']) if pd.notna(row['y_name']) else ""

                if not (source_node and target_node):
                    skipped_count += 1
                    continue

                content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

                if content_hash in content_hashes:
                    if source_node not in graph_structure:
                        graph_structure[source_node] = []

                    edge_info = (edge_type, target_node, content_hash)
                    if edge_info not in graph_structure[source_node]:
                        graph_structure[source_node].append(edge_info)

                    if target_node not in graph_structure:
                        graph_structure[target_node] = []

                    skipped_count += 1
                    continue

                image_path = row['image_path'] if pd.notna(row['image_path']) else ""

                vector = embeddings.embed_query(text)

                metadata = {
                    "content_hash": content_hash,
                    "original_index": idx,
                    "image_path": image_path,
                    "source_node": source_node,
                    "edge_type": edge_type,
                    "target_node": target_node
                }

                if source_node not in graph_structure:
                    graph_structure[source_node] = []

                edge_info = (edge_type, target_node, content_hash)
                if edge_info not in graph_structure[source_node]:
                    graph_structure[source_node].append(edge_info)

                if target_node not in graph_structure:
                    graph_structure[target_node] = []

                vectors.append(vector)
                documents.append(text)
                metadata_list.append(metadata)
                hash_to_index_map[content_hash] = len(documents) - 1

                content_hashes.add(content_hash)

                added_count += 1

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        if vectors:
            print(f"Added {added_count} new documents, skipped {skipped_count} duplicate or empty documents...")
            print(f"Graph structure contains {len(graph_structure)} nodes")

            dimension = len(vectors[0])
            index = faiss.IndexFlatIP(dimension)

            vectors_np = np.array(vectors).astype('float32')
            index.add(vectors_np)

            faiss.write_index(index, index_path)

            with open(documents_path, 'wb') as f:
                pickle.dump(documents, f)

            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_list, f)

            with open(content_hash_path, 'wb') as f:
                pickle.dump(content_hashes, f)

            with open(graph_path, 'wb') as f:
                pickle.dump(graph_structure, f)

            print(f"Database created, now {index.ntotal} vectors")

            def find_node_neighbors(node_id, max_hops, max_neighbors_per_hop):
                """
                Find the neighbors of a node

                Parameters:
                - node_id: node ID
                - max_hops: maximum number of hops
                - max_neighbors_per_hop: maximum number of neighbors per hop

                Returns:
                - DataFrame containing neighbor information
                """
                results = []
                visited = set()
                current_hop = 1
                current_level = [node_id]

                while current_hop <= max_hops and current_level:
                    next_level = []

                    for current_node in current_level:
                        if current_node in visited:
                            continue

                        visited.add(current_node)

                        if current_node in graph_structure:
                            edges = graph_structure[current_node][:max_neighbors_per_hop]

                            for edge_type, target_node, content_hash in edges:
                                if content_hash in hash_to_index_map:
                                    idx = hash_to_index_map[content_hash]

                                    results.append({
                                        'hop': current_hop,
                                        'node_id': target_node,
                                        'source_node': current_node,
                                        'target_node': target_node,
                                        'edge_type': edge_type,
                                        'relation': f"{current_node} --{edge_type}--> {target_node}",
                                        'content': documents[idx],
                                        'content_hash': content_hash
                                    })

                                    next_level.append(target_node)

                        for source, edges in graph_structure.items():
                            if source == current_node or source in visited:
                                continue

                            for edge_type, target, content_hash in edges:
                                if target == current_node and content_hash in hash_to_index_map:
                                    idx = hash_to_index_map[content_hash]

                                    results.append({
                                        'hop': current_hop,
                                        'node_id': source,
                                        'source_node': source,
                                        'target_node': current_node,
                                        'edge_type': edge_type,
                                        'relation': f"{source} --{edge_type}--> {current_node}",
                                        'content': documents[idx],
                                        'content_hash': content_hash
                                    })

                                    next_level.append(source)

                                    if len(results) >= max_neighbors_per_hop * current_hop:
                                        break

                            if len(results) >= max_neighbors_per_hop * current_hop:
                                break

                    current_level = next_level
                    current_hop += 1

                return pd.DataFrame(results)

            def search_function(query, k=10, expand_nodes=False, max_neighbors=5):
                """
                Enhanced search function, supports graph structure queries
                Parameters:
                - query: query text or special command
                - k: number of returned matches
                - expand_nodes: whether to expand related nodes' edges
                - max_neighbors: maximum number of neighbors returned for each node
                """
                # check if it is a special neighbor query
                if query.startswith("__find_neighbors__"):
                    # parse the special query format
                    parts = query.replace("__find_neighbors__", "").split("|")
                    if len(parts) >= 3:
                        node_id = parts[0]
                        max_hops = int(parts[1])
                        max_neighbors_per_hop = int(parts[2])
                        return find_node_neighbors(node_id, max_hops, max_neighbors_per_hop)

                # normal vector search logic...
                query_vector = embeddings.embed_query(query)
                query_vector_np = np.array([query_vector]).astype('float32')

                scores, indices = index.search(query_vector_np, k)

                results = []
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(documents):
                        result = {
                            'content': documents[idx],
                            'metadata': metadata_list[idx],
                            'relevance_score': float(scores[0][i]),
                            'original_index': metadata_list[idx]['original_index'],
                            'image_path': metadata_list[idx]['image_path'],
                            'source_node': metadata_list[idx]['source_node'],
                            'edge_type': metadata_list[idx]['edge_type'],
                            'target_node': metadata_list[idx]['target_node']
                        }

                        # if expand related nodes
                        if expand_nodes:
                            source = metadata_list[idx]['source_node']
                            target = metadata_list[idx]['target_node']

                            # add other edges related to the source node
                            neighbors = []
                            if source in graph_structure:
                                for edge_type, target_node, content_hash in graph_structure[source][:max_neighbors]:
                                    if content_hash in hash_to_index_map:
                                        neighbor_idx = hash_to_index_map[content_hash]
                                        neighbors.append({
                                            'content': documents[neighbor_idx],
                                            'metadata': metadata_list[neighbor_idx],
                                            'relation': f"Edges from the same source node: {source} --{edge_type}--> {target_node}"
                                        })

                            # add other edges to the target node
                            for node, edges in graph_structure.items():
                                for edge_type, node_target, content_hash in edges:
                                    if node_target == target and content_hash in hash_to_index_map:
                                        neighbor_idx = hash_to_index_map[content_hash]
                                        neighbors.append({
                                            'content': documents[neighbor_idx],
                                            'metadata': metadata_list[neighbor_idx],
                                            'relation': f"Edges to the same target node: {node} --{edge_type}--> {target}"
                                        })
                                        if len(neighbors) >= max_neighbors * 2:
                                            break
                                if len(neighbors) >= max_neighbors * 2:
                                    break

                            result['neighbors'] = neighbors

                        results.append(result)

                return pd.DataFrame(results)

            return search_function

        else:
            print("No valid documents to add, cannot create database")
            return None

class MedicalKnowledgeStore:
    """Chroma compatible wrapper for medical knowledge storage, supports graph structure queries"""

    def __init__(self, search_function):
        self.search_function = search_function

        # read JSON file
        with open('WHO/overview_328_updated.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        # create a mapping dictionary from name to png_dir_path
        self.doc_name_to_pdf_path = {item['name']: item['png_dir_path'] for item in data}


    def search_with_boundary_scores(self, query, k=10, expand_nodes=False, max_neighbors=5):
        """Search with boundary scores"""
        results = self.search_function(query, k=k, expand_nodes=expand_nodes, max_neighbors=max_neighbors)

        if results is None or len(results) == 0:
            return []

        # convert to Chroma compatible format
        output = []
        for _, row in results.iterrows():
            content = row['content']
            metadata = row['metadata']
            score = row['relevance_score']

            # if expand nodes, add neighbor information to metadata
            if expand_nodes and 'neighbors' in row:
                metadata['neighbors'] = row['neighbors']

            # create Document object
            doc = Document(
                page_content=content,
                metadata=metadata
            )

            # add to results
            output.append((doc, score))

        return output

    # demographic: limit the similarity search to the subgraph, if subgraph_conditions is None, no filter is applied
    def search_with_population_context(self, query, times=5, max_search=100, k=10, expand_nodes=False, max_neighbors=5, subgraph_conditions=None):
        '''
        subgraph_conditions: a list of document names that are used to filter the subgraph, if None, no filter is applied
        '''

        # convert document names to pdf names, if the document name is not in the doc_name_to_pdf_path, skip it
        subgraph_pdf_names = [
            self.doc_name_to_pdf_path[pdf_name] 
            for pdf_name in subgraph_conditions 
            if pdf_name in self.doc_name_to_pdf_path
        ]
        # print('subgraph_conditions: ', subgraph_conditions, 'subgraph_pdf_names: ', subgraph_pdf_names)

        search_k = min(k * times, max_search)  # dynamically adjust the number of search results
        results = self.search_function(query, k=search_k, expand_nodes=expand_nodes, max_neighbors=max_neighbors)

        if results is None or len(results) == 0:
            return []

        # convert to Chroma compatible format
        output = []
        for _, row in results.iterrows():
            image_path = row['image_path']
            pdf_name = 'WHO/' + '/'.join(image_path.split('/')[:-1])
            if pdf_name in subgraph_pdf_names:
                content = row['content']
                metadata = row['metadata']
                score = row['relevance_score']

                # create Document object
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )

                output.append((doc, score))

                # stop if the number of results reaches the target
                if len(output) >= k:
                    break

        return output

    def similarity_search(self, query, k=10, expand_nodes=False, max_neighbors=5):
        """
        Chroma compatible similarity search interface (no scores returned)

        Parameters:
        - query: query text
        - k: number of returned matches
        - expand_nodes: whether to expand related nodes' edges
        - max_neighbors: maximum number of neighbors returned for each node
        """
        results_with_scores = self.search_with_boundary_scores(
            query, k, expand_nodes=expand_nodes, max_neighbors=max_neighbors
        )
        return [doc for doc, _ in results_with_scores]

    def find_neighbors(self, node_id, max_hops=1, max_neighbors_per_hop=5):
        """
        Find the neighbors of a given node

        Parameters:
        - node_id: node ID to find neighbors
        - max_hops: maximum number of hops (1 for direct neighbors, 2 for neighbors of neighbors, etc.)
        - max_neighbors_per_hop: maximum number of neighbors returned for each hop

        Returns:
        - dictionary containing neighbor information, organized by hop
        """
        # retrieve the neighbors of a given node through a custom query
        # here we assume search_function can access graph_structure
        # note: this requires modifying search_function to support this query mode

        # an example implementation, assume we add a special query format to find neighbors
        special_query = f"__find_neighbors__{node_id}|{max_hops}|{max_neighbors_per_hop}"
        neighbors_df = self.search_function(special_query, k=max_neighbors_per_hop * max_hops)

        # organize the results into a structure grouped by hop
        neighbors_by_hop = {}

        for _, row in neighbors_df.iterrows():
            hop = row.get('hop', 1)  # default is 1
            if hop not in neighbors_by_hop:
                neighbors_by_hop[hop] = []

            neighbor_info = {
                'original_index': row.get('original_index'),
                'source_node': row.get('source_node', ''),
                'target_node': row.get('target_node', ''),
                'edge_type': row.get('edge_type', ''),
                'content': row.get('content', ''),
                'image_path': row.get('image_path', '')
            }

            neighbors_by_hop[hop].append(neighbor_info)

        return neighbors_by_hop


# create a compatible interface
def get_topk_similar(query, expand_nodes=False, k=10, max_neighbors=5):
    """compatible search function with graph structure query option"""
    global faiss_db  # use global variable, ensure it can be accessed in other places

    results = faiss_db.search_with_boundary_scores(
        query, k=k, expand_nodes=expand_nodes, max_neighbors=max_neighbors
    )

    # convert to DataFrame format
    data = []
    for doc, score in results:
        item = {
            'original_index': doc.metadata.get('original_index', ''),
            'image_path': doc.metadata.get('image_path', ''),
            'relevance_score': score,
            'content': doc.page_content,
            'source_node': doc.metadata.get('source_node', ''),
            'edge_type': doc.metadata.get('edge_type', ''),
            'target_node': doc.metadata.get('target_node', '')
        }

        # if there is neighbor information, also add it to the results
        if 'neighbors' in doc.metadata:
            item['neighbors'] = doc.metadata['neighbors']

        data.append(item)

    return pd.DataFrame(data)

def initialize_db(args):
    global faiss_db, embeddings, all_demographics, all_diseases, demographic2pdf, disease2pdf, demo_dis2pdf

    # Initialize AzureOpenAIEmbeddings with your client
    if args.use_api == 'azureopenai':
        from openai import AzureOpenAI
        from langchain.embeddings import AzureOpenAIEmbeddings  # Import AzureOpenAIEmbeddings
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-10-21"
        )
        embeddings = AzureOpenAIEmbeddings(
          client=client,chunk_size=1000,
          azure_deployment='text-embedding-3-small')
    else:
        embeddings = get_embeddings(args.expert_model)
    all_data_clean = pd.read_csv('filtered_data_v1.csv')

    # use example
    search_function = create_faiss_vector_db(all_data_clean, embeddings)

    # create a compatible layer
    faiss_db = MedicalKnowledgeStore(search_function)

    # read disease2demo(1).csv
    disease2demographic = pd.read_csv('baseline_dataset/Disease2demo(1).csv')
    all_demographics = disease2demographic['demographic'].unique()
    all_diseases = disease2demographic['disease'].unique()
    disease2demographic['pdf_name'] = disease2demographic['pdf_name'].apply(parse_pdf_names)
    demographic2pdf = disease2demographic.groupby('demographic')['pdf_name'].apply(lambda x: sum(x, [])).to_dict()
    disease2pdf = disease2demographic.groupby('disease')['pdf_name'].apply(lambda x: sum(x, [])).to_dict()
    demo_dis2pdf = disease2demographic.groupby(['demographic', 'disease'])['pdf_name'].apply(lambda x: sum(x, [])).to_dict()

    return faiss_db, embeddings

def get_demographics_info():
    return all_demographics, all_diseases, demographic2pdf, disease2pdf, demo_dis2pdf

def get_faiss_db():
    """get the initialized database"""
    return faiss_db

def get_embeddings_model():
    """get the initialized embeddings model"""
    return embeddings
