"""
TikTok Viral Caption Generator - Model Functions
Handles the loading and operation of ML models for caption generation
"""

import torch
import numpy as np
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoModelForCausalLM
from constants import SETTINGS, SUCCESSFUL_CAPTIONS, VIRAL_PROMPT_TEMPLATE, TIKTOK_TEMPLATES
from utils import embed_text, preprocess_tiktok_text, get_template_caption
import random


def load_caption_model():
    """Initialize language model and tokenizer for caption generation"""
    try:
        # Initialize language model and tokenizer
        caption_model_name = SETTINGS.get('models', {}).get('caption', {}).get('primary', "google/flan-t5-base")
        tokenizer = AutoTokenizer.from_pretrained(caption_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(caption_model_name)
        model_status_msg = f"Loaded {caption_model_name} model successfully"
    except Exception as e:
        # Fallback to an even smaller model as backup
        try:
            caption_model_name = SETTINGS.get('models', {}).get('caption', {}).get('fallback', "google/flan-t5-small")
            tokenizer = AutoTokenizer.from_pretrained(caption_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(caption_model_name)
            model_status_msg = f"Loaded fallback model {caption_model_name} successfully"
        except Exception as e:
            # Last resort fallback
            caption_model_name = SETTINGS.get('models', {}).get('caption', {}).get('emergency', "distilgpt2")
            tokenizer = AutoTokenizer.from_pretrained(caption_model_name, padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(caption_model_name)
            model_status_msg = f"Loaded emergency fallback model {caption_model_name}"

    return model, tokenizer, model_status_msg


def load_embedding_model():
    """Initialize embedding model for similarity search"""
    try:
        embedding_model_name = SETTINGS.get('models', {}).get('embedding', {}).get('primary',
                                                                                   "sentence-transformers/all-MiniLM-L6-v2")
        embedding_model = AutoModel.from_pretrained(embedding_model_name)
        embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        embedding_status_msg = f"Loaded embedding model {embedding_model_name} successfully"
    except Exception as e:
        # Create dummy embedding capabilities as fallback
        class DummyModel:
            def __call__(self, **kwargs):
                class Output:
                    def __init__(self):
                        self.last_hidden_state = torch.ones((1, 1, 384))

                return Output()

        embedding_model = DummyModel()
        embedding_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        embedding_status_msg = "Using fallback embedding model"

    return embedding_model, embedding_tokenizer, embedding_status_msg


def build_caption_index(embedding_model, embedding_tokenizer):
    """Create a searchable index of successful caption embeddings"""
    try:
        # Get embeddings for all captions
        caption_embeddings_list = []
        for caption in SUCCESSFUL_CAPTIONS:
            try:
                embedding = embed_text(caption, embedding_model, embedding_tokenizer)
                caption_embeddings_list.append(embedding)
            except Exception as e:
                print(f"Error embedding caption: {str(e)}")
                # Add a dummy embedding to maintain indices
                caption_embeddings_list.append(np.ones(384))

        caption_embeddings = np.array(caption_embeddings_list)
        dim = caption_embeddings.shape[1]

        # Make sure all embeddings have the same dimension
        if not all(emb.shape[0] == dim for emb in caption_embeddings_list):
            # Normalize shapes if needed
            max_dim = max(emb.shape[0] for emb in caption_embeddings_list)
            caption_embeddings = np.array([
                np.pad(emb, (0, max(0, max_dim - emb.shape[0])))
                for emb in caption_embeddings_list
            ])

        # Create and populate the FAISS index
        index = faiss.IndexFlatL2(caption_embeddings.shape[1])
        index.add(caption_embeddings)
        return index, caption_embeddings
    except Exception as e:
        print(f"Error building caption index: {str(e)}")
        # Return a dummy index and embeddings as fallback
        dummy_embeddings = np.ones((len(SUCCESSFUL_CAPTIONS), 384))
        dummy_index = faiss.IndexFlatL2(384)
        dummy_index.add(dummy_embeddings)
        return dummy_index, dummy_embeddings


def generate_caption_with_model(topic, model, tokenizer):
    """Generate caption using language model"""
    try:
        # Format the prompt with the topic
        prompt = VIRAL_PROMPT_TEMPLATE.format(topic=topic)

        # Handle different model types (encoder-decoder vs decoder-only)
        if hasattr(model, 'config') and hasattr(model.config, 'model_type') and model.config.model_type == 'gpt2':
            # GPT-style models are decoder-only
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
        else:
            # T5/FLAN style models are encoder-decoder
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)

        base_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up the output if it's too verbose or contains artifacts
        if len(base_caption) > 300:
            base_caption = base_caption[:300] + "..."

        if len(base_caption) < 50 or "caption" in base_caption.lower():
            base_caption = get_template_caption(topic)

        return base_caption

    except Exception as e:
        print(f"Error generating caption with model: {str(e)}")
        # Fallback to template-based generation
        return get_template_caption(topic)


def find_similar_captions(topic, embedding_model, embedding_tokenizer, caption_index):
    """Find similar successful captions for inspiration"""
    try:
        topic_embedding = embed_text(topic, embedding_model, embedding_tokenizer).reshape(1, -1)
        _, retrieved_indices = caption_index.search(topic_embedding, 2)
        similar_caption = SUCCESSFUL_CAPTIONS[retrieved_indices[0][0]]
        return similar_caption
    except Exception as e:
        print(f"Error finding similar captions: {str(e)}")
        # Choose a random caption as fallback
        return random.choice(SUCCESSFUL_CAPTIONS)