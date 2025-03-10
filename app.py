"""
TikTok Viral Caption Generator - Main Application
The main entry point for the Streamlit application
"""

import streamlit as st
import random
import time

from constants import TIKTOK_TEMPLATES
from utils import (
    preprocess_tiktok_text,
    detect_category,
    get_random_hashtags,
    suggest_hashtags,
    format_final_caption,
    decorate_with_emojis
)
from models import (
    load_caption_model,
    load_embedding_model,
    build_caption_index,
    generate_caption_with_model,
    find_similar_captions
)
from ui import (
    setup_page_config,
    apply_custom_css,
    setup_header,
    setup_sidebar,
    create_input_form,
    display_results,
    show_loading_spinner
)

# Initialize session state variables if they don't exist
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'caption_model' not in st.session_state:
    st.session_state.caption_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'embedding_tokenizer' not in st.session_state:
    st.session_state.embedding_tokenizer = None
if 'caption_index' not in st.session_state:
    st.session_state.caption_index = None
if 'caption_embeddings' not in st.session_state:
    st.session_state.caption_embeddings = None
if 'model_status_msg' not in st.session_state:
    st.session_state.model_status_msg = "Initializing models..."
if 'embedding_status_msg' not in st.session_state:
    st.session_state.embedding_status_msg = ""


def load_models():
    """Load the models in the background to avoid UI freezing"""
    # Load caption generation model
    model, tokenizer, model_status_msg = load_caption_model()

    # Load embedding model for similarity search
    embedding_model, embedding_tokenizer, embedding_status_msg = load_embedding_model()

    # Build caption index
    caption_index, caption_embeddings = build_caption_index(embedding_model, embedding_tokenizer)

    # Update session state
    st.session_state.caption_model = model
    st.session_state.tokenizer = tokenizer
    st.session_state.embedding_model = embedding_model
    st.session_state.embedding_tokenizer = embedding_tokenizer
    st.session_state.caption_index = caption_index
    st.session_state.caption_embeddings = caption_embeddings
    st.session_state.models_loaded = True
    st.session_state.model_status_msg = model_status_msg
    st.session_state.embedding_status_msg = embedding_status_msg

    return model, tokenizer, embedding_model, embedding_tokenizer, caption_index, caption_embeddings


def generate_viral_tiktok_caption(topic, category, style, model, tokenizer, embedding_model, embedding_tokenizer,
                                  caption_index):
    """
    Generate a viral TikTok caption based on the topic and preferences

    Args:
        topic (str): The main topic or content of the TikTok video
        category (str): Content category (comedy, fashion, beauty, etc.)
        style (str): Caption style (default, template, minimal, emoji-heavy)

    Returns:
        str: A TikTok-ready caption with emojis and hashtags
    """
    # Preprocess the topic
    processed_topic = preprocess_tiktok_text(topic)

    # Auto-detect category if set to auto-detect
    if category == "auto-detect":
        category = detect_category(topic)

    # Generate base caption using language model
    if style == "template":
        # Use a template-based approach
        base_caption = random.choice(TIKTOK_TEMPLATES).format(
            topic=processed_topic,
            n=random.randint(1, 100)  # For "Day {n}" templates
        )
    else:
        base_caption = generate_caption_with_model(processed_topic, model, tokenizer)

    # Find similar successful captions for inspiration
    similar_caption = find_similar_captions(processed_topic, embedding_model, embedding_tokenizer, caption_index)

    # Apply different caption styles
    if style == "minimal":
        # Keep it short and simple
        sentences = base_caption.split('.')
        base_caption = sentences[0].strip()  # Just take the first sentence
    else:
        # Add emojis based on style
        base_caption = decorate_with_emojis(base_caption, style)

    # Add hashtags
    hashtag_count = 4 if style != "minimal" else 2
    hashtags = get_random_hashtags(category, hashtag_count)

    # Format final caption
    final_caption = format_final_caption(base_caption, hashtags, style)

    return final_caption, similar_caption, hashtags


def main():
    """Main application entry point"""
    # Setup the page
    setup_page_config()
    apply_custom_css()
    setup_header()

    # Setup sidebar
    model_status = setup_sidebar(
        st.session_state.models_loaded,
        st.session_state.model_status_msg,
        st.session_state.embedding_status_msg
    )

    # Load models in background if not already loaded
    if not st.session_state.models_loaded:
        with st.spinner("Loading models, please wait..."):
            model, tokenizer, embedding_model, embedding_tokenizer, caption_index, caption_embeddings = load_models()
            model_status.success(f"{st.session_state.model_status_msg} | {st.session_state.embedding_status_msg}")
    else:
        model = st.session_state.caption_model
        tokenizer = st.session_state.tokenizer
        embedding_model = st.session_state.embedding_model
        embedding_tokenizer = st.session_state.embedding_tokenizer
        caption_index = st.session_state.caption_index
        caption_embeddings = st.session_state.caption_embeddings

    # Create input form
    form_data = create_input_form()

    # Handle form submission
    if form_data["submit_button"]:
        if not form_data["topic"]:
            st.error("Please enter a description of your TikTok video.")
        else:
            with show_loading_spinner():
                # Add a slight delay to make it feel like it's "thinking" (better UX)
                time.sleep(1)

                # Generate caption
                caption, similar_caption, hashtags = generate_viral_tiktok_caption(
                    form_data["topic"],
                    form_data["category"],
                    form_data["style"],
                    model,
                    tokenizer,
                    embedding_model,
                    embedding_tokenizer,
                    caption_index
                )

                # Get additional hashtag suggestions
                additional_tags = suggest_hashtags(form_data["topic"])

            # Display results
            display_results(caption, similar_caption, hashtags, additional_tags)


if __name__ == "__main__":
    main()