"""
TikTok Viral Caption Generator - UI Components
Handles the Streamlit UI elements and layout
"""

import streamlit as st
import time
from constants import CONFIG, TRENDY_HASHTAGS, ENGAGEMENT_TIPS, DETAILED_METHODOLOGY


def setup_page_config():
    """Configure the Streamlit page settings"""
    app_config = CONFIG.get('app', {})
    st.set_page_config(
        page_title=app_config.get('title', "TikTok Viral Caption Generator"),
        page_icon=app_config.get('icon', "✨"),
        layout=app_config.get('layout', "wide"),
        initial_sidebar_state=app_config.get('initial_sidebar_state', "expanded")
    )


def apply_custom_css():
    """Apply custom CSS styling for a more professional look"""
    styles = CONFIG.get('styles', {})
    css = """
    <style>
    """

    # Add main header styling
    main_header = styles.get('main_header', {})
    css += f"""
    .main-header {{
        font-size: {main_header.get('font_size', '2.5rem')};
        color: {main_header.get('color', '#FF0050')};
        text-align: {main_header.get('text_align', 'center')};
        margin-bottom: {main_header.get('margin_bottom', '1rem')};
    }}
    """

    # Add sub header styling
    sub_header = styles.get('sub_header', {})
    css += f"""
    .sub-header {{
        font-size: {sub_header.get('font_size', '1.5rem')};
        color: {sub_header.get('color', '#333333')};
        margin-bottom: {sub_header.get('margin_bottom', '1rem')};
    }}
    """

    # Add caption output styling
    caption_output = styles.get('caption_output', {})
    css += f"""
    .caption-output {{
        background-color: {caption_output.get('background_color', '#f8f9fa')};
        border-radius: {caption_output.get('border_radius', '10px')};
        padding: {caption_output.get('padding', '20px')};
        border-left: {caption_output.get('border_left', '5px solid #FF0050')};
        margin: {caption_output.get('margin', '20px 0px')};
    }}
    """

    # Add highlight styling
    highlight = styles.get('highlight', {})
    css += f"""
    .highlight {{
        color: {highlight.get('color', '#FF0050')};
        font-weight: {highlight.get('font_weight', 'bold')};
    }}
    """

    # Add success message styling
    success_message = styles.get('success_message', {})
    css += f"""
    .success-message {{
        background-color: {success_message.get('background_color', '#d1e7dd')};
        border-radius: {success_message.get('border_radius', '5px')};
        padding: {success_message.get('padding', '10px')};
        margin: {success_message.get('margin', '10px 0px')};
    }}
    """

    # Add loading message styling
    loading_message = styles.get('loading_message', {})
    css += f"""
    .loading-message {{
        color: {loading_message.get('color', '#6c757d')};
        font-style: {loading_message.get('font_style', 'italic')};
        margin: {loading_message.get('margin', '10px 0px')};
    }}
    """

    # Add hashtag styling
    hashtag = styles.get('hashtag', {})
    css += f"""
    .hashtag {{
        background-color: {hashtag.get('background_color', '#e9ecef')};
        border-radius: {hashtag.get('border_radius', '15px')};
        padding: {hashtag.get('padding', '5px 10px')};
        margin: {hashtag.get('margin', '5px')};
        display: {hashtag.get('display', 'inline-block')};
        font-size: {hashtag.get('font_size', '0.85rem')};
    }}
    """

    # Add emoji container styling
    emoji_container = styles.get('emoji_container', {})
    css += f"""
    .emoji-container {{
        font-size: {emoji_container.get('font_size', '1.5rem')};
        margin: {emoji_container.get('margin', '10px 0px')};
    }}
    """

    css += """
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def setup_header():
    """Display app title and introduction"""
    st.markdown("<h1 class='main-header'>✨ Context Based Agentic TikTok Viral Caption Generator ✨</h1>",
                unsafe_allow_html=True)
    st.markdown("Create engaging, trend-aware captions that help your TikTok videos go viral Using Agentic System")


def setup_sidebar(models_loaded=False, model_status_msg="Initializing models...", embedding_status_msg=""):
    """Setup sidebar with model status and options"""
    sidebar_config = CONFIG.get('sidebar', {})

    with st.sidebar:
        st.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
        st.write(sidebar_config.get('about_text',
                                    "This tool uses AI to generate captions optimized for TikTok engagement and virality."))

        st.markdown("<h2 class='sub-header'>Model Status</h2>", unsafe_allow_html=True)
        model_status = st.empty()

        if models_loaded:
            model_status.success(f"{model_status_msg} | {embedding_status_msg}")
        else:
            model_status.info("Initializing models...")

        st.markdown("<h2 class='sub-header'>Additional Options</h2>", unsafe_allow_html=True)
        show_methodology = st.checkbox("Show caption methodology details", value=False)

        if show_methodology:
            st.markdown("### Caption Generation Methodology")
            st.markdown(DETAILED_METHODOLOGY)

    return model_status


def create_input_form():
    """Create the main input form for caption generation"""
    st.markdown("<h2 class='sub-header'>Video Information</h2>", unsafe_allow_html=True)

    with st.form("caption_form"):
        # Topic input
        topic = st.text_area("What is your TikTok video about?",
                             help="Describe your video content in a few words")

        # Category selection
        categories = [cat for cat in TRENDY_HASHTAGS.keys() if cat != "generic"]
        categories.append("auto-detect")
        category = st.selectbox("Choose a content category",
                                options=categories,
                                format_func=lambda x: x.capitalize() if x != "auto-detect" else "Auto-detect",
                                help="Select the category that best fits your content")

        # Style selection
        styles = CONFIG.get('form_options', {}).get('caption_styles', ["default", "template", "minimal", "emoji-heavy"])
        style = st.selectbox("Choose a caption style",
                             options=styles,
                             format_func=lambda x: x.capitalize(),
                             help="Select how you want your caption to be styled")

        # Advanced options (collapsible)
        with st.expander("Advanced Options"):
            target_audience = st.multiselect(
                "Target Audience",
                options=CONFIG.get('form_options', {}).get('target_audience',
                                                           ["Teens", "Young Adults", "Adults", "Parents",
                                                            "Professionals", "Students"]),
                default=["Young Adults"],
                help="Who is your content aimed at?"
            )

            additional_hashtags = st.text_input(
                "Additional Custom Hashtags (comma separated)",
                help="Add your own hashtags (without #)"
            )

            emotion = st.select_slider(
                "Emotional Tone",
                options=CONFIG.get('form_options', {}).get('emotional_tones',
                                                           ["Humorous", "Informative", "Inspirational", "Controversial",
                                                            "Emotional"]),
                value="Humorous",
                help="What emotional response are you aiming for?"
            )

        submit_button = st.form_submit_button("Generate Viral Caption")

    return {
        "topic": topic,
        "category": category,
        "style": style,
        "target_audience": target_audience,
        "additional_hashtags": additional_hashtags,
        "emotion": emotion,
        "submit_button": submit_button
    }


def display_results(caption, similar_caption, hashtags, additional_tags):
    """Display the generated caption and related information"""
    # Display caption with copy button
    st.markdown("<h2 class='sub-header'>📱 Your TikTok Caption</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='caption-output'>{caption.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

    # Display additional information in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 class='sub-header'>🔍 Similar Successful Caption</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='padding: 10px; background-color: #f1f3f5; border-radius: 5px;'>{similar_caption}</div>",
            unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 class='sub-header'>🏷️ Additional Hashtag Suggestions</h3>", unsafe_allow_html=True)
        hashtag_html = ""
        for tag in additional_tags:
            hashtag_html += f"<span class='hashtag'>{tag}</span> "
        st.markdown(f"<div>{hashtag_html}</div>", unsafe_allow_html=True)

    # Display engagement tip
    st.markdown("<h3 class='sub-header'>💡 Engagement Tip</h3>", unsafe_allow_html=True)
    tip = random.choice(ENGAGEMENT_TIPS)
    st.info(tip)


def show_loading_spinner(message="Generating your viral TikTok caption..."):
    """Show a loading spinner with a message"""
    return st.spinner(f"✨ {message}")