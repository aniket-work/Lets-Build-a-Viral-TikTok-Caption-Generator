# TikTok Viral Caption Generator - Streamlit UI
# A professional web interface for generating engaging, trend-aware captions for TikTok videos

import streamlit as st
import torch
import re
import random
import numpy as np
import faiss
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel

# Set page configuration
st.set_page_config(
    page_title="TikTok Viral Caption Generator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF0050;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333333;
        margin-bottom: 1rem;
    }
    .caption-output {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #FF0050;
        margin: 20px 0px;
    }
    .highlight {
        color: #FF0050;
        font-weight: bold;
    }
    .success-message {
        background-color: #d1e7dd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0px;
    }
    .loading-message {
        color: #6c757d;
        font-style: italic;
        margin: 10px 0px;
    }
    .hashtag {
        background-color: #e9ecef;
        border-radius: 15px;
        padding: 5px 10px;
        margin: 5px;
        display: inline-block;
        font-size: 0.85rem;
    }
    .emoji-container {
        font-size: 1.5rem;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.markdown("<h1 class='main-header'>✨ Context Based Agentic TikTok Viral Caption Generator ✨</h1>", unsafe_allow_html=True)
st.markdown("Create engaging, trend-aware captions that help your TikTok videos go viral Using Agentic System")

# Sidebar for model loading status and additional options
with st.sidebar:
    st.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    st.write("This tool uses AI to generate captions optimized for TikTok engagement and virality.")

    st.markdown("<h2 class='sub-header'>Model Status</h2>", unsafe_allow_html=True)
    model_status = st.empty()
    model_status.info("Initializing models...")

    st.markdown("<h2 class='sub-header'>Additional Options</h2>", unsafe_allow_html=True)
    show_methodology = st.checkbox("Show caption methodology details", value=False)

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

# Popular TikTok caption templates that tend to go viral
tiktok_templates = [
    "✨ The secret to {topic} that nobody talks about ✨ #gamechangers",
    "POV: When you discover the perfect {topic} technique 🤩 #mindblown",
    "Tell me you're obsessed with {topic} without TELLING me you're obsessed with {topic} 💯",
    "Day {n} of showing you my {topic} hacks that actually work 🔥",
    "Wait for it... this {topic} trick will change your life 🤯 #lifechanging",
    "*Everyone else's {topic}* vs *My {topic}* 👀",
    "3 {topic} mistakes you're making right now 🚫 (and how to fix them)",
    "I tested this viral {topic} trend so you don't have to 🧪 #results",
    "Nobody is talking about THIS {topic} secret 🤫 #gamechanger",
    "This is your sign to try {topic} today ✨ #justdoit"
]

# Trendy TikTok hashtags organized by content category
trendy_hashtags = {
    "comedy": ["#funny", "#comedy", "#joke", "#laughing", "#meme", "#humor", "#skit", "#comedytiktok"],
    "fashion": ["#fashion", "#style", "#outfit", "#ootd", "#aesthetic", "#fashiontiktok", "#styling", "#fashionhack"],
    "beauty": ["#makeup", "#beauty", "#skincare", "#glowup", "#makeuptutorial", "#beautyhack", "#skincareroutine"],
    "food": ["#food", "#recipe", "#cooking", "#foodie", "#yummy", "#tasty", "#foodtiktok", "#easyrecipe"],
    "dance": ["#dance", "#routine", "#choreography", "#trending", "#dancechallenge", "#dancetutorial"],
    "fitness": ["#fitness", "#workout", "#gym", "#exercise", "#fitcheck", "#motivation", "#weightloss", "#fittok"],
    "travel": ["#travel", "#adventure", "#vacation", "#wanderlust", "#exploring", "#traveldiaries", "#traveltips"],
    "lifestyle": ["#daily", "#routine", "#life", "#dayinmylife", "#relatable", "#lifehack", "#productivity"],
    "music": ["#music", "#song", "#musician", "#singer", "#playlist", "#newmusic", "#soundeffect"],
    "storytime": ["#storytime", "#story", "#mystory", "#truestory", "#confession", "#lifestory"],
    "generic": ["#fyp", "#foryou", "#foryoupage", "#viral", "#trending", "#tiktok", "#viralvideo", "#blowthisup"]
}

# Popular TikTok emojis that increase engagement
popular_emojis = ["✨", "🔥", "😂", "🥰", "👀", "🙌", "💯", "🤩", "🤪", "🤣", "💕", "❤️", "😍", "👇", "👉", "🤍",
                  "😱", "🥺", "⭐", "🌟", "👏", "💫", "✅", "🎯", "🧠", "💡", "🤯", "🔄", "🤞", "😎"]

# Sample successful TikTok captions for reference and similarity matching
successful_captions = [
    "living my best life ✨ don't forget to smile today 😊 #positivevibes #mindset",
    "POV: your dog when you open a bag of chips 🐶😂 #dogsoftiktok #funny",
    "This makeup hack will save you SO much time ⏰ #beauty #makeuphack",
    "tell me you're obsessed with coffee without telling me you're obsessed with coffee ☕️ #coffeetok",
    "day 5 of showing you my favorite outfit combos 👚👖 #fashion #ootd",
    "the way he looks at me 🥺❤️ #relationship #couplegoals",
    "trying this viral recipe and... WAIT FOR IT 🤯 #foodtiktok #viral",
    "if you know you know 😏 like for part 2! #foryou #fyp",
    "my toxic trait is thinking I can squeeze everything into one day 🤪 #relatable",
    "this song has been living in my head rent free 🎵 #catchysong #earworm",
    "POV: you finally find the perfect sound for your video 🔊 #soundeffect #trending",
    "Things in my house that just make sense ✨ #homeinspo #decor",
    "Am I the only one who does this? 👀 #relatable #fyp",
    "3 outfits that never miss 👌 #fashion #styletips",
    "Trying to do this trend but failing miserably 💀 #trendingchallenge",
    "The EASIEST way to ✨manifest✨ what you want #spiritualtok #manifestation",
    "My morning routine (realistic version) 😴 #morningroutine #reallife",
    "Why didn't anyone tell me this sooner?? 😱 #lifehack #gamechanger",
    "Reply to @user yes, I'll do a part 2! 🥰 #reply #part2",
    "How I went from 0 to 10k followers in ONE month 📈 #growthtips #creatortips"
]


# Load the models in the background to avoid UI freezing
def load_models():
    try:
        # Initialize language model and tokenizer
        caption_model_name = "google/flan-t5-base"  # More reliable and smaller model
        tokenizer = AutoTokenizer.from_pretrained(caption_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(caption_model_name)
        model_status_msg = f"Loaded {caption_model_name} model successfully"
    except Exception as e:
        # Fallback to an even smaller model as backup
        try:
            caption_model_name = "google/flan-t5-small"
            tokenizer = AutoTokenizer.from_pretrained(caption_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(caption_model_name)
            model_status_msg = f"Loaded fallback model {caption_model_name} successfully"
        except Exception as e:
            # Last resort fallback
            caption_model_name = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(caption_model_name, padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(caption_model_name)
            model_status_msg = f"Loaded emergency fallback model {caption_model_name}"

    # Initialize embedding model for similarity search
    try:
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
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

    # Update sidebar status
    model_status.success(f"{model_status_msg} | {embedding_status_msg}")
    return model, tokenizer, embedding_model, embedding_tokenizer, caption_index, caption_embeddings


# TikTok-specific text processing
def preprocess_tiktok_text(text):
    """Clean and prepare text for TikTok caption generation"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s#@\'\"\.\!\?\,]', '', text)  # Keep hashtags, mentions, and essential punctuation
    return text.strip()


# Function to embed text for similarity search
def embed_text(text, embedding_model, embedding_tokenizer):
    """Convert text to embeddings for similarity matching"""
    try:
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
        st.error(f"Error in embed_text: {str(e)}")
        # Return a simple fallback embedding if needed
        return np.ones(384)  # Standard dimension for MiniLM model


# Build a vector index for successful captions to find similar content
def build_caption_index(embedding_model, embedding_tokenizer):
    """Create a searchable index of successful caption embeddings"""
    try:
        # Get embeddings for all captions
        caption_embeddings_list = []
        for caption in successful_captions:
            try:
                embedding = embed_text(caption, embedding_model, embedding_tokenizer)
                caption_embeddings_list.append(embedding)
            except Exception as e:
                st.warning(f"Error embedding caption: {str(e)}")
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
        st.warning(f"Error building caption index: {str(e)}")
        # Return a dummy index and embeddings as fallback
        dummy_embeddings = np.ones((len(successful_captions), 384))
        dummy_index = faiss.IndexFlatL2(384)
        dummy_index.add(dummy_embeddings)
        return dummy_index, dummy_embeddings


def get_random_hashtags(category, count=3):
    """Get random relevant hashtags for a specific category"""
    # If category is not in our list, use generic
    if category not in trendy_hashtags:
        category = "generic"

    # Get hashtags from the specified category and add some generic ones
    category_tags = trendy_hashtags[category].copy()
    generic_tags = trendy_hashtags["generic"].copy()

    # Combine and select random hashtags
    combined_tags = list(set(category_tags + generic_tags))
    if len(combined_tags) <= count:
        return combined_tags
    return random.sample(combined_tags, count)


def get_random_emojis(count=2):
    """Get random popular TikTok emojis"""
    return random.sample(popular_emojis, min(count, len(popular_emojis)))


def detect_category(topic):
    """Auto-detect the most relevant category based on topic keywords"""
    topic_lower = topic.lower()
    best_category = "generic"
    best_score = 0

    for category, hashtags in trendy_hashtags.items():
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

    # Generate base caption using language model with our detailed viral prompt
    # Constructing a comprehensive prompt using our viral caption guidelines
    viral_prompt = f"""
    Create a viral TikTok caption about {processed_topic} that includes:

    1. A strong hook or opening line that immediately captures attention
    2. Emotional connection using relatability ("Who else...?", "Am I the only one...?")
    3. Strategic emoji placement (2-4 carefully selected emojis)
    4. Engagement triggers like a call-to-action ("Comment if...", "Like for Part 2")

    Use one of these viral formats:
    - "Tell me without telling me" format
    - "POV:" scenario setup
    - "Day X of [doing something]" series format
    - "The way [something happens]" observation format
    - "Things that just make sense..." list format
    - "[X] things about [Y] that..." educational format

    Style guidelines:
    - Use concise, conversational language (NOT formal)
    - Include TikTok speech patterns (shortened words, slang)
    - Keep it between 150-300 characters for optimal performance
    - Write in first person perspective when appropriate
    - Avoid complete sentences - use fragments and casual speech

    The caption should feel authentic to TikTok culture and have high viral potential.
    """

    try:
        # Handle different model types (encoder-decoder vs decoder-only)
        if hasattr(model, 'config') and hasattr(model.config, 'model_type') and model.config.model_type == 'gpt2':
            # GPT-style models are decoder-only
            inputs = tokenizer(viral_prompt, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
        else:
            # T5/FLAN style models are encoder-decoder
            inputs = tokenizer(viral_prompt, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)

        base_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up the output if it's too verbose or contains artifacts
        if len(base_caption) > 300:
            base_caption = base_caption[:300] + "..."

        if len(base_caption) < 50 or "caption" in base_caption.lower():
            template = random.choice(tiktok_templates)
            n = random.randint(1, 100)
            base_caption = template.format(topic=processed_topic, n=n)
    except Exception as e:
        st.warning(f"Error generating caption with model: {str(e)}")
        # Fallback to template-based generation
        template = random.choice(tiktok_templates)
        n = random.randint(1, 100)  # For "Day {n}" templates
        base_caption = template.format(topic=processed_topic, n=n)

    # Find similar successful captions for inspiration
    try:
        topic_embedding = embed_text(processed_topic, embedding_model, embedding_tokenizer).reshape(1, -1)
        _, retrieved_indices = caption_index.search(topic_embedding, 2)
        similar_caption = successful_captions[retrieved_indices[0][0]]
    except Exception as e:
        st.warning(f"Error finding similar captions: {str(e)}")
        # Choose a random caption as fallback
        similar_caption = random.choice(successful_captions)

    # Apply different caption styles
    if style == "template":
        # Use a template-based approach
        template = random.choice(tiktok_templates)
        n = random.randint(1, 100)  # For "Day {n}" templates
        base_caption = template.format(topic=processed_topic, n=n)
    elif style == "minimal":
        # Keep it short and simple
        sentences = re.split(r'[.!?]', base_caption)
        base_caption = sentences[0].strip()  # Just take the first sentence
    elif style == "emoji-heavy":
        # Add more emojis throughout the caption
        words = base_caption.split()
        for i in range(min(len(words), 3)):
            random_position = random.randint(0, len(words) - 1)
            words[random_position] = words[random_position] + " " + random.choice(popular_emojis)
        base_caption = " ".join(words)

    # Add opening/closing emojis if needed
    if style != "minimal" and not any(e in base_caption for e in popular_emojis):
        # Only add emojis if there aren't already some in the caption
        emoji_placement = random.choice(["start", "end", "both"])
        selected_emojis = get_random_emojis(2)

        if emoji_placement == "start" or emoji_placement == "both":
            base_caption = f"{selected_emojis[0]} {base_caption}"
        if emoji_placement == "end" or emoji_placement == "both":
            base_caption = f"{base_caption} {selected_emojis[-1]}"

    # Add call to action (sometimes)
    if random.random() < 0.3 and style != "minimal":
        ctas = [
            "Like if you agree! 👍",
            "Tag someone who needs to see this 👀",
            "Drop a ❤️ if you relate",
            "Comment your thoughts 👇",
            "Save this for later ✨",
            "Follow for more! ✅",
            "Share with a friend who needs this 🙌"
        ]
        base_caption += "\n" + random.choice(ctas)

    # Add hashtags
    hashtag_count = 4 if style != "minimal" else 2
    hashtags = get_random_hashtags(category, hashtag_count)
    hashtag_string = ' '.join(hashtags)

    # Combine everything into final caption
    final_caption = f"{base_caption}\n{hashtag_string}"

    return final_caption, similar_caption, hashtags


def suggest_hashtags(topic, count=5):
    """Suggest relevant hashtags for a topic"""
    # Detect the most likely category
    category = detect_category(topic)

    # Get hashtags from that category
    category_tags = trendy_hashtags[category].copy()

    # Add some generic viral hashtags
    suggested_tags = category_tags + random.sample(trendy_hashtags["generic"], 3)

    # Remove duplicates and limit to requested count
    suggested_tags = list(set(suggested_tags))
    if len(suggested_tags) <= count:
        return suggested_tags
    return random.sample(suggested_tags, count)


# Load the detailed viral caption generation prompt
def load_detailed_prompt():
    detailed_prompt = """
# Viral TikTok Caption Generator Prompt

## Context
This prompt guides the language model to create TikTok captions optimized for maximum engagement, discoverability, and virality potential. The model should produce captions that feel authentic, trendy, and incorporate current TikTok speech patterns and conventions.

## Input Parameters to Consider
- **Video Topic**: What is the video about? (e.g., makeup tutorial, food recipe, comedy skit)
- **Content Category**: Which niche does this belong to? (beauty, food, comedy, etc.)
- **Target Audience**: Who is the intended audience? (age group, interests)
- **Desired Style**: Casual, informative, emotional, humorous, etc.
- **Key Video Elements**: Any specific hooks, reveals, or moments to highlight

## Output Requirements

### Essential Caption Elements (ALL captions must include these)
1. **Strong Hook/Opening Line**
   - Immediately capture attention with a question, bold statement, or curiosity gap
   - Begin with phrases like "POV:", "That moment when...", or "Wait for it..."
   - Create intrigue within the first 3-5 words

2. **Emotional Connection**
   - Use language that creates relatability ("Who else...?", "Am I the only one...?")
   - Include elements of surprise, humor, or curiosity
   - Appeal to shared experiences or pain points

3. **Strategic Emoji Placement**
   - Place 2-4 carefully selected emojis (not random) that enhance meaning
   - Position emojis at points of emphasis or to break up text
   - Use trending emoji combinations (e.g., "✨🤌", "👁👄👁", "🤯👀")

4. **Discoverability Elements**
   - Include 3-5 relevant hashtags (mix of specific and trending)
   - Always include at least one trending/viral hashtag (#fyp, #foryou, #viral)
   - Prioritize niche-specific hashtags (#makeuptutorial, #recipeoftheday)

5. **Engagement Triggers**
   - Include at least one call-to-action ("Comment if...", "Like for Part 2")
   - Create opportunities for debate/discussion ("Agree or disagree?")
   - Ask a simple question or request feedback
"""
    return detailed_prompt


# Load models in the background if not already loaded
if not st.session_state.models_loaded:
    with st.spinner("Loading models, please wait..."):
        model, tokenizer, embedding_model, embedding_tokenizer, caption_index, caption_embeddings = load_models()
else:
    model = st.session_state.caption_model
    tokenizer = st.session_state.tokenizer
    embedding_model = st.session_state.embedding_model
    embedding_tokenizer = st.session_state.embedding_tokenizer
    caption_index = st.session_state.caption_index
    caption_embeddings = st.session_state.caption_embeddings

# Main input form
st.markdown("<h2 class='sub-header'>Video Information</h2>", unsafe_allow_html=True)

with st.form("caption_form"):
    # Topic input
    topic = st.text_area("What is your TikTok video about?",
                         help="Describe your video content in a few words")

    # Category selection
    categories = [cat for cat in trendy_hashtags.keys() if cat != "generic"]
    categories.append("auto-detect")
    category = st.selectbox("Choose a content category",
                            options=categories,
                            format_func=lambda x: x.capitalize() if x != "auto-detect" else "Auto-detect",
                            help="Select the category that best fits your content")

    # Style selection
    styles = ["default", "template", "minimal", "emoji-heavy"]
    style = st.selectbox("Choose a caption style",
                         options=styles,
                         format_func=lambda x: x.capitalize(),
                         help="Select how you want your caption to be styled")

    # Advanced options (collapsible)
    with st.expander("Advanced Options"):
        target_audience = st.multiselect("Target Audience",
                                         options=["Teens", "Young Adults", "Adults", "Parents", "Professionals",
                                                  "Students"],
                                         default=["Young Adults"],
                                         help="Who is your content aimed at?")

        additional_hashtags = st.text_input("Additional Custom Hashtags (comma separated)",
                                            help="Add your own hashtags (without #)")

        emotion = st.select_slider("Emotional Tone",
                                   options=["Humorous", "Informative", "Inspirational", "Controversial", "Emotional"],
                                   value="Humorous",
                                   help="What emotional response are you aiming for?")

    submit_button = st.form_submit_button("Generate Viral Caption")

# Display results and advanced options
if submit_button:
    if not topic:
        st.error("Please enter a description of your TikTok video.")
    else:
        with st.spinner("✨ Generating your viral TikTok caption..."):
            # Add a slight delay to make it feel like it's "thinking" (better UX)
            time.sleep(1)

            # Generate caption
            caption, similar_caption, hashtags = generate_viral_tiktok_caption(
                topic, category, style,
                model, tokenizer, embedding_model, embedding_tokenizer, caption_index
            )

            # Get additional hashtag suggestions
            additional_tags = suggest_hashtags(topic)

            # Pick a random engagement tip
            tips = [
                "Post when your audience is most active",
                "Respond to comments within the first hour",
                "Create a hook in the first 3 seconds of your video",
                "Use trending sounds to increase discoverability",
                "Create content that encourages sharing and saving",
                "Pair your caption with high-quality visuals",
                "Be consistent with your posting schedule",
                "Engage with comments quickly to boost algorithm performance"
            ]
            random_tip = random.choice(tips)

        # Display caption with copy button
        st.markdown("<h2 class='sub-header'>📱 Your TikTok Caption</h2>", unsafe_allow_html=True)

        st.markdown(f"<div class='caption-output'>{caption.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
