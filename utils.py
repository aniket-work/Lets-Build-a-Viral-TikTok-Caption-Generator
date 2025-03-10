"""
TikTok Viral Caption Generator - Utility Functions
Contains helper functions for text processing and caption generation
"""

import re
import random
import numpy as np
from constants import (
    TRENDY_HASHTAGS,
    POPULAR_EMOJIS,
    TIKTOK_TEMPLATES,
    CALLS_TO_ACTION
)


def preprocess_tiktok_text(text):
    """Clean and prepare text for TikTok caption generation"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s#@\'\"\.\!\?\,]', '', text)  # Keep hashtags, mentions, and essential punctuation
    return text.strip()


def detect_category(topic):
    """Auto-detect the most relevant category based on topic keywords"""
    topic_lower = topic.lower()
    best_category = "generic"
    best_score = 0

    for category, hashtags in TRENDY_HASHTAGS.items():
        if category == "generic":
            continue

        score = 0
        for tag in hashtags:
            clean_tag = tag.replace('#', '')
            if clean_tag in topic_lower:
                score += 2

        # Check category name directly
        if category in topic_lower:
            score += 3

        if score > best_score:
            best_score = score
            best_category = category

    return best_category


def get_random_hashtags(category, count=3):
    """Get random relevant hashtags for a specific category"""
    # If category is not in our list, use generic
    if category not in TRENDY_HASHTAGS:
        category = "generic"

    # Get hashtags from the specified category and add some generic ones
    category_tags = TRENDY_HASHTAGS[category].copy()
    generic_tags = TRENDY_HASHTAGS["generic"].copy()

    # Combine and select random hashtags
    combined_tags = list(set(category_tags + generic_tags))
    if len(combined_tags) <= count:
        return combined_tags
    return random.sample(combined_tags, count)


def get_random_emojis(count=2):
    """Get random popular TikTok emojis"""
    return random.sample(POPULAR_EMOJIS, min(count, len(POPULAR_EMOJIS)))


def suggest_hashtags(topic, count=5):
    """Suggest relevant hashtags for a topic"""
    # Detect the most likely category
    category = detect_category(topic)

    # Get hashtags from that category
    category_tags = TRENDY_HASHTAGS[category].copy()

    # Add some generic viral hashtags
    suggested_tags = category_tags + random.sample(TRENDY_HASHTAGS["generic"], 3)

    # Remove duplicates and limit to requested count
    suggested_tags = list(set(suggested_tags))
    if len(suggested_tags) <= count:
        return suggested_tags
    return random.sample(suggested_tags, count)


def get_template_caption(topic):
    """Get a template-based caption with the topic inserted"""
    template = random.choice(TIKTOK_TEMPLATES)
    n = random.randint(1, 100)  # For "Day {n}" templates
    return template.format(topic=topic, n=n)


def decorate_with_emojis(caption, style):
    """Add emoji decoration to a caption based on style"""
    if style == "minimal":
        return caption

    # Check if caption already has emojis
    has_emojis = any(emoji in caption for emoji in POPULAR_EMOJIS)

    if not has_emojis:
        selected_emojis = get_random_emojis(2)
        emoji_placement = random.choice(["start", "end", "both"])

        if emoji_placement in ["start", "both"]:
            caption = f"{selected_emojis[0]} {caption}"

        if emoji_placement in ["end", "both"]:
            caption = f"{caption} {selected_emojis[-1]}"

    # For emoji-heavy style, add more emojis throughout
    if style == "emoji-heavy" and len(caption) > 10:
        words = caption.split()
        for _ in range(min(len(words), 3)):
            random_position = random.randint(0, len(words) - 1)
            words[random_position] = words[random_position] + " " + random.choice(POPULAR_EMOJIS)
        caption = " ".join(words)

    return caption


def add_call_to_action(caption, style):
    """Add a call-to-action to the caption if appropriate"""
    if style != "minimal" and random.random() < 0.3:
        return f"{caption}\n{random.choice(CALLS_TO_ACTION)}"
    return caption


def format_final_caption(base_caption, hashtags, style):
    """Format the final caption with all components"""
    # Add call to action
    caption = add_call_to_action(base_caption, style)

    # Add hashtags
    hashtag_string = ' '.join(hashtags)
    return f"{caption}\n{hashtag_string}"


def embed_text(text, embedding_model, embedding_tokenizer):
    """Convert text to embeddings for similarity matching"""
    try:
        import torch
        inputs = embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = embedding_model(**inputs)
        # Get mean pooling of the last hidden state
        embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        # Handle case where embedding is not 1D (happens with batches of 1)
        if len(embedding.shape) == 0:
            embedding = embedding.reshape(1)
        return embedding
    except Exception as e:
        print(f"Error in embed_text: {str(e)}")
        # Return a simple fallback embedding if needed
        return np.ones(384)  # Standard dimension for MiniLM model