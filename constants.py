"""
TikTok Viral Caption Generator - Constants
Contains data structures and fixed values used throughout the application
"""

import yaml
import json
import os

# Load settings from YAML file
def load_settings():
    try:
        with open("settings.yaml", "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        return {}

# Load configuration from JSON file
def load_config():
    try:
        with open("config.json", "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {}

# Settings and config
SETTINGS = load_settings()
CONFIG = load_config()

# Popular TikTok caption templates that tend to go viral
TIKTOK_TEMPLATES = [
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
TRENDY_HASHTAGS = {
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
POPULAR_EMOJIS = ["✨", "🔥", "😂", "🥰", "👀", "🙌", "💯", "🤩", "🤪", "🤣", "💕", "❤️", "😍", "👇", "👉", "🤍",
                  "😱", "🥺", "⭐", "🌟", "👏", "💫", "✅", "🎯", "🧠", "💡", "🤯", "🔄", "🤞", "😎"]

# Sample successful TikTok captions for reference and similarity matching
SUCCESSFUL_CAPTIONS = [
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

# Engagement tips and call-to-action phrases
ENGAGEMENT_TIPS = [
    "Post when your audience is most active",
    "Respond to comments within the first hour",
    "Create a hook in the first 3 seconds of your video",
    "Use trending sounds to increase discoverability",
    "Create content that encourages sharing and saving",
    "Pair your caption with high-quality visuals",
    "Be consistent with your posting schedule",
    "Engage with comments quickly to boost algorithm performance"
]

CALLS_TO_ACTION = [
    "Like if you agree! 👍",
    "Tag someone who needs to see this 👀",
    "Drop a ❤️ if you relate",
    "Comment your thoughts 👇",
    "Save this for later ✨",
    "Follow for more! ✅",
    "Share with a friend who needs this 🙌"
]

# Viral prompt template
VIRAL_PROMPT_TEMPLATE = """
Create a viral TikTok caption about {topic} that includes:

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

# Load the detailed prompt for methodology display
DETAILED_METHODOLOGY = """
# Viral TikTok Caption Generator Methodology

## Context
This tool creates TikTok captions optimized for maximum engagement, discoverability, and virality potential. The captions feel authentic, trendy, and incorporate current TikTok speech patterns and conventions.

## Key Caption Elements
1. **Strong Hook/Opening Line**
   - Immediately captures attention with a question, bold statement, or curiosity gap
   - Begins with phrases like "POV:", "That moment when...", or "Wait for it..."
   - Creates intrigue within the first 3-5 words

2. **Emotional Connection**
   - Uses language that creates relatability ("Who else...?", "Am I the only one...?")
   - Includes elements of surprise, humor, or curiosity
   - Appeals to shared experiences or pain points

3. **Strategic Emoji Placement**
   - Places 2-4 carefully selected emojis that enhance meaning
   - Positions emojis at points of emphasis or to break up text
   - Uses trending emoji combinations (e.g., "✨🤌", "👁👄👁", "🤯👀")

4. **Discoverability Elements**
   - Includes 3-5 relevant hashtags (mix of specific and trending)
   - Always includes at least one trending/viral hashtag (#fyp, #foryou, #viral)
   - Prioritizes niche-specific hashtags (#makeuptutorial, #recipeoftheday)

5. **Engagement Triggers**
   - Includes at least one call-to-action ("Comment if...", "Like for Part 2")
   - Creates opportunities for debate/discussion ("Agree or disagree?")
   - Asks a simple question or requests feedback
"""