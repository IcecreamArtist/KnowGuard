import torch
import logging
# from keys import mykey
import torch
import logging
import numpy as np
# from keys import mykey

# A dictionary to cache models and tokenizers to avoid reloading
import os

os.environ["AZURE_OPENAI_ENDPOINT"] = None
os.environ['AZURE_OPENAI_API_KEY'] = None
os.environ['XTY_API_KEY'] = None

global models
models = {}

def log_info(message, logger_name="message_logger", print_to_std=False, mode="info"):
    logger = logging.getLogger(logger_name)
    if logger: 
        if mode == "error": logger.error(message)
        if mode == "warning": logger.warning(message)
        else: logger.info(message)
    if print_to_std: print(message + "\n")

class ModelCache:
    def __init__(self, model_name, use_vllm=False, use_api=None, **kwargs):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_api = use_api
        self.model = None
        self.tokenizer = None
        self.terminators = None
        self.client = None
        self.args = kwargs
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        if self.use_api == "openai":
            from openai import OpenAI
            self.api_account = self.args.get("api_account", "openai")
            self.client = OpenAI(api_key=mykey[self.api_account]) # Setup API key appropriately in keys.py
        elif self.use_api == 'azureopenai':
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="XX"
            )
        elif self.use_vllm:
            try:
                from vllm import LLM
                enable_prefix_caching = self.args.get("enable_prefix_caching", False)
                self.model = LLM(model=self.model_name, enable_prefix_caching=enable_prefix_caching)
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            except Exception as e:
                log_info(f"[ERROR] [{self.model_name}]: If using a custom local model, it is not compatible with VLLM, will load using Huggingfcae and you can ignore this error: {str(e)}", mode="error")
                self.use_vllm = False
        if not self.use_vllm and self.use_api != "openai" and self.use_api != 'azureopenai':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()  # Set the model to evaluation mode
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            try:
                eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_id is not None:
                    self.terminators = [self.tokenizer.eos_token_id, eot_id]
                else:
                    self.terminators = [self.tokenizer.eos_token_id]
            except:
                self.terminators = [self.tokenizer.eos_token_id]
    
    def generate(self, messages):

        self.temperature = self.args.get("temperature", 0.6)
        self.max_tokens = self.args.get("max_tokens", 256)
        self.top_p = self.args.get("top_p", 0.9)
        self.top_logprobs = self.args.get("top_logprobs", 0)

        if self.use_api == "openai" or self.use_api == 'azureopenai': 
            return self.openai_generate(messages)
        elif self.use_vllm: return self.vllm_generate(messages)
        else: return self.huggingface_generate(messages)
    
    def huggingface_generate(self, messages):
        try:
            template_output = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if hasattr(self.model, 'device'):
                inputs = template_output.to(self.model.device)
            else:
                inputs = template_output
        except Exception as e:
            log_info(f"[{self.model_name}]: Could not apply chat template to messages: {str(e)}", mode="warning")
            prompt = "\n\n".join([m['content'] for m in messages])
            tokenized = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            if hasattr(self.model, 'device'):
                inputs = tokenized['input_ids'].to(self.model.device)
            else:
                inputs = tokenized['input_ids']

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                do_sample=True,
                max_new_tokens=self.max_tokens, 
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.terminators
            )
        
        input_length = inputs.shape[-1]
        response_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        usage = {"input_tokens": input_length, "output_tokens": outputs.shape[-1] - input_length}
        output_dict = {'response_text': response_text, 'usage': usage}

        # log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, None, usage
        
        
    def vllm_generate(self, messages):
        try:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            inputs = "\n\n".join([m['content'] for m in messages])
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        from vllm import SamplingParams
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presense_penalty", 0)
        sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p, logprobs=self.top_logprobs, 
                                        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        outputs = self.model.generate(inputs, sampling_params)
        response_text = outputs[0].outputs[0].text
        logprobs = outputs[0].outputs[0].cumulative_logprob
        # TODO: If top_logprobs > 0, return logprobs of generation
        # if self.top_logprobs > 0: logprobs = outputs[0].outputs[0].logprobs
        usage = {"input_tokens": len(outputs[0].prompt_token_ids), "output_tokens": len(outputs[0].outputs[0].token_ids)}
        output_dict = {'response_text': response_text, 'usage': usage}

        # log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, logprobs, usage

    def openai_generate(self, messages):
        if self.top_logprobs == 0:
            response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p
                    )
        else:
            response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        logprobs=True, 
                        top_logprobs=self.top_logprobs
                    )
        
        num_input_tokens = response.usage.prompt_tokens
        num_output_tokens = response.usage.completion_tokens
        response_text = response.choices[0].message.content
        if response_text is not None:
            response_text = response_text.strip()
        else:
            response_text = ""
        log_probs = response.choices[0].logprobs.top_logprobs if self.top_logprobs > 0 else None
        # log_info(f"[{self.model_name}][OUTPUT]: {response}")
        return response_text, log_probs, {"input_tokens": num_input_tokens, "output_tokens": num_output_tokens}


def get_response(messages, model_name, use_vllm=False, use_api=None, **kwargs):
    # if 'gpt2' not in model_name:
    #     if 'gpt' in model_name or 'o1' in model_name: use_api = "openai"
    # else:
    #     use_api = None
    model_cache = models.get(model_name, None)
    if model_cache is None:
        model_cache = ModelCache(model_name, use_vllm=use_vllm, use_api=use_api, **kwargs)
        models[model_name] = model_cache
    return model_cache.generate(messages)





class EmbeddingModel:

    def __init__(self, model_name, use_api=None, chunk_size=1000, **kwargs):
        self.model_name = model_name
        self.use_api = use_api
        self.chunk_size = chunk_size
        self.model = None
        self.tokenizer = None
        self.client = None
        self.args = kwargs
        self.load_model()
    
    def load_model(self):
        # Try to load with sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.model_type = "sentence_transformer"
            log_info(f"[{self.model_name}]: Loaded embedding model using sentence-transformers")
        except ImportError:
            log_info(f"[{self.model_name}]: sentence-transformers not available, using transformers", mode="warning")
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.model_type = "transformers"
            log_info(f"[{self.model_name}]: Loaded embedding model using transformers")
    
    def embed_query(self, text):
        """
        Embed a single query text
        Args:
            text: String to embed
        Returns:
            List of floats representing the embedding
        """
        return self._embed_texts([text])[0]
    
    def embed_documents(self, texts):
        """
        Embed multiple documents
        Args:
            texts: List of strings to embed
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        return self._embed_texts(texts)
    
    def _embed_texts(self, texts):
        """Internal method to embed texts"""
        if self.use_api == "openai":
            return self._openai_embeddings(texts)
        elif self.model_type == "sentence_transformer":
            return self._sentence_transformer_embeddings(texts)
        else:  # transformers
            return self._transformers_embeddings(texts)

    def _sentence_transformer_embeddings(self, texts):
        """Get embeddings using sentence-transformers"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i:i + self.chunk_size]
            try:
                embeddings = self.model.encode(chunk, convert_to_numpy=True)
                all_embeddings.extend(embeddings.tolist())
            except Exception as e:
                log_info(f"[ERROR] Sentence transformer embedding failed: {str(e)}", mode="error")
                raise e
        
        return all_embeddings
    
    def _transformers_embeddings(self, texts):
        """Get embeddings using transformers AutoModel"""
        import torch
        from torch.nn.functional import normalize
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i:i + self.chunk_size]
            try:
                # Tokenize
                inputs = self.tokenizer(chunk, padding=True, truncation=True, 
                                      return_tensors="pt", max_length=512)
                
                # Move to device if model has device
                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Mean pooling
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Normalize embeddings
                    embeddings = normalize(embeddings, p=2, dim=1)
                    
                    all_embeddings.extend(embeddings.cpu().numpy().tolist())
                    
            except Exception as e:
                log_info(f"[ERROR] Transformers embedding failed: {str(e)}", mode="error")
                raise e
        
        return all_embeddings


def get_embeddings(model_name, use_api=None, chunk_size=1000, **kwargs):
    """
    Get an embedding model wrapper similar to AzureOpenAIEmbeddings
    Args:
        model_name: Name of the embedding model
        use_api: API to use ("openai" or None for local models)
        chunk_size: Batch size for processing
    Returns:
        EmbeddingModel instance with embed_query and embed_documents methods
    """
    # Determine API usage for embedding models
    if use_api == 'openai':
        pass
    elif use_api == 'azureopenai':
        from openai import AzureOpenAI
        from langchain.embeddings import AzureOpenAIEmbeddings  # Import AzureOpenAIEmbeddings
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-10-21"
        )
        embedding_model = AzureOpenAIEmbeddings(
            client=client,
            chunk_size=1000,
            azure_deployment="text-embedding-3-small"
        )
        print(embedding_model)
    else:
        # Create a unique cache key for embedding models
        embedding_model_key = f"{model_name}_embedding"
        
        embedding_model = models.get(embedding_model_key, None)
        if embedding_model is None:
            embedding_model = EmbeddingModel(
                model_name=model_name,
                use_api=use_api,
                chunk_size=chunk_size,
                **kwargs
            )
            models[embedding_model_key] = embedding_model
    
    return embedding_model
