"""Coping content: guided exercises and personality-tailored suggestions.

These are general self-help techniques (paced breathing, grounding, task-chunking,
sleep hygiene) -- not treatment. Wording is original.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------------
# Guided exercises (offered after a matching intent; user replies "yes")
# ---------------------------------------------------------------------------------
EXERCISES: dict[str, dict] = {
    "breathing": {
        "title": "2-minute paced breathing",
        "offer": "Would you like a quick 2-minute breathing exercise? (reply **yes**)",
        "text": (
            "**Paced breathing (about 2 minutes)**\n"
            "1. Sit comfortably and relax your shoulders.\n"
            "2. Breathe in through your nose for **4** counts.\n"
            "3. Hold gently for **2** counts.\n"
            "4. Breathe out slowly through your mouth for **6** counts.\n"
            "5. Repeat 8-10 times. Longer exhales help your body settle.\n\n"
            "How do you feel after trying it?"),
    },
    "grounding": {
        "title": "5-4-3-2-1 grounding",
        "offer": "Want to try a short grounding exercise to calm racing thoughts? (reply **yes**)",
        "text": (
            "**5-4-3-2-1 grounding**\n"
            "Look around and name:\n"
            "- **5** things you can see\n"
            "- **4** things you can feel (chair, floor, fabric...)\n"
            "- **3** things you can hear\n"
            "- **2** things you can smell\n"
            "- **1** thing you can taste\n\n"
            "Go slowly. This pulls your attention out of the spiral and back to the room."),
    },
    "sleep": {
        "title": "Wind-down routine",
        "offer": "Want a simple wind-down routine to try tonight? (reply **yes**)",
        "text": (
            "**Wind-down routine**\n"
            "1. Pick a wake-up time you can keep every day, including weekends.\n"
            "2. Stop screens 30-45 minutes before bed (or dim them and use night mode).\n"
            "3. Avoid caffeine after mid-afternoon.\n"
            "4. Write tomorrow's to-do list on paper so your mind can let go of it.\n"
            "5. If you're awake after ~20 minutes, get up, do something quiet in dim light, "
            "and return when sleepy.\n\n"
            "If sleep problems last for weeks, please talk to a doctor or campus health centre."),
    },
    "start_plan": {
        "title": "5-minute start plan",
        "offer": "Want a tiny plan to get started? (reply **yes**)",
        "text": (
            "**5-minute start plan**\n"
            "1. Pick ONE task and write its smallest first step (e.g. \"open the notes and write the title\").\n"
            "2. Put your phone in another room and set a timer for **25 minutes**.\n"
            "3. Work until the timer ends, then take a **5-minute** break.\n"
            "4. After the break, decide: another round, or stop. Starting is the win.\n\n"
            "Which task will you start with?"),
    },
    "reflection": {
        "title": "Three-line reflection",
        "offer": "Would a short writing exercise help you sort through this? (reply **yes**)",
        "text": (
            "**Three-line reflection** (write it down or just think it through)\n"
            "1. *What am I feeling right now, in one word?*\n"
            "2. *What is one thing that triggered it?*\n"
            "3. *What is one small kind thing I could do for myself in the next hour?*\n\n"
            "You can tell me your answers if you like. I'm listening."),
    },
    "cooldown": {
        "title": "Cool-down",
        "offer": "Want a quick way to cool down before you respond to anyone? (reply **yes**)",
        "text": (
            "**Cool-down**\n"
            "1. Step away from the situation for a few minutes.\n"
            "2. Splash water on your face or take a short walk.\n"
            "3. Breathe out slowly, longer than you breathe in, 6-8 times.\n"
            "4. Ask: *what do I actually need right now?* Then respond, or write it down first.\n"),
    },
}

# which intent offers which exercise
INTENT_EXERCISE = {
    "stress": "breathing", "anxiety": "grounding", "sleep": "sleep",
    "procrastination": "start_plan", "sadness": "reflection", "loneliness": "reflection",
    "self_esteem": "reflection", "anger": "cooldown", "family_pressure": "reflection",
    "relationships": "reflection",
}

# ---------------------------------------------------------------------------------
# Personality-tailored suggestions.
# Each entry: (required trait levels, suggestion). Traits: extraversion (E),
# agreeableness (A), conscientiousness (C), emotional_stability (S), openness (O).
# Levels are "low" / "high". The suggestion matching the most traits wins; an entry
# with {} is the neutral default.
# ---------------------------------------------------------------------------------
SUGGESTIONS: dict[str, list[tuple[dict, str]]] = {
    "stress": [
        ({"conscientiousness": "high"}, "Since you like structure: list everything due, rank by deadline, and block the next 2 days hour by hour. Seeing it on paper often shrinks it."),
        ({"conscientiousness": "low"}, "Skip the big plan. Pick just the *one* task due soonest and do 25 focused minutes on it. Momentum matters more than a perfect schedule."),
        ({"extraversion": "high"}, "Try studying with a friend or in a group for a bit. Talking through the load out loud can lighten it."),
        ({"extraversion": "low"}, "A quiet reset might suit you: 10 minutes alone with no screen (a walk, music, or tea) before going back to work."),
        ({}, "Try writing down everything on your mind, then circle the one thing you can act on today."),
    ],
    "anxiety": [
        ({"openness": "high"}, "A short guided mindfulness or body-scan audio could suit you. Notice the sensations without trying to change them."),
        ({"emotional_stability": "low"}, "Anxiety can feel very loud for you right now. Try a slow breath out and tell yourself: *this feeling is temporary and I've handled hard moments before.*"),
        ({"conscientiousness": "high"}, "Turn the worry into a plan: write the worry, then the *one* thing within your control, then the first step."),
        ({}, "Try the 'worry window': give yourself 10 minutes a day to worry on purpose, and park worries outside that time."),
    ],
    "sadness": [
        ({"extraversion": "high"}, "Reaching out could help: message or call one person you trust today, even just to say hi."),
        ({"extraversion": "low"}, "You don't have to socialise to feel better. Gentle solo things count: a short walk, sunlight, a favourite show, or journaling."),
        ({"openness": "high"}, "Creative outlets (music, drawing, writing) can give feelings somewhere to go. Even 10 minutes counts."),
        ({}, "Try one tiny act of self-care today: a shower, a meal, 10 minutes outside. Small steps count when everything feels heavy."),
    ],
    "loneliness": [
        ({"extraversion": "high"}, "You gain energy from people, so try joining one club, study group or sports session this week."),
        ({"extraversion": "low"}, "Connection doesn't have to mean crowds. One low-pressure one-on-one (a coffee, a call, a shared interest group online) can be enough."),
        ({"agreeableness": "high"}, "You're warm with people. Try being the one to start a small conversation or invite someone to lunch; others are probably hoping for it too."),
        ({}, "Consider one small step: say hi to a classmate or join one activity. Most people feel this way at some point in college."),
    ],
    "sleep": [
        ({"conscientiousness": "high"}, "A fixed schedule will work well for you: set a consistent bedtime and wake time and track it for a week."),
        ({"conscientiousness": "low"}, "Don't aim for a perfect routine. Start with just ONE habit: phone out of reach 30 minutes before bed."),
        ({}, "Try keeping the same wake-up time daily and avoiding caffeine after mid-afternoon."),
    ],
    "procrastination": [
        ({"conscientiousness": "low"}, "Make the first step ridiculously small (open the book, write one line) and use a 25-minute timer. Starting is the hard part."),
        ({"conscientiousness": "high"}, "Procrastination can be perfectionism in disguise. Allow a 'rough first draft' version of the task, then polish later."),
        ({"openness": "high"}, "Make it interesting: change your study spot, turn the topic into a question to answer, or teach it to someone."),
        ({}, "Break the task into 3 mini-steps and do only the first one for 10 minutes."),
    ],
    "self_esteem": [
        ({"emotional_stability": "low"}, "Your inner critic is being very loud. Try saying to yourself what you'd say to a friend in your exact situation."),
        ({"conscientiousness": "high"}, "You hold yourself to high standards. Try writing down 3 things you did *well* today, however small, alongside what went wrong."),
        ({}, "Try listing 3 things you handled this week, however small. Your mind tends to skip these."),
    ],
    "anger": [
        ({"emotional_stability": "low"}, "Strong feelings hit hard for you. Give yourself a few minutes away from the trigger before responding."),
        ({"agreeableness": "low"}, "Your frustration may be pointing at something unfair. Once you've cooled down, think about what boundary or request you'd like to state clearly."),
        ({}, "Move your body (a brisk walk, stretching) to burn off the adrenaline, then decide how you'd like to respond."),
    ],
    "family_pressure": [
        ({"agreeableness": "high"}, "You care a lot about not letting people down. Remember that you're allowed to share your limits kindly with family; it's a form of honesty, not disappointment."),
        ({"conscientiousness": "high"}, "Try writing down what *you* want and what's realistic, then choose one small thing to share with a family member you trust."),
        ({}, "Expectations can feel enormous when unspoken. Consider talking with one trusted person (relative, mentor, counsellor) about what you're carrying."),
    ],
    "relationships": [
        ({"extraversion": "high"}, "Lean on your other friends right now; spending time with people who care about you will help."),
        ({"extraversion": "low"}, "Give yourself quiet time to process, and keep one trusted person updated so you're not holding it all alone."),
        ({}, "Be gentle with yourself. Hurt after conflict or loss is normal and it eases with time and support."),
    ],
}
