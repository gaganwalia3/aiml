"""MindMate Streamlit app.  Run:  streamlit run app.py"""
from __future__ import annotations

import html
import importlib
import json

import altair as alt
import pandas as pd
import streamlit as st

from src import chatbot as chatbot_mod
from src import config as C
from src import mood, risk, safety, ui

importlib.reload(ui)
importlib.reload(chatbot_mod)

from src.chatbot import MindMateBot
from src.coping import EXERCISES
from src.data_prep import SLEEP_ORDER
from src.personality import ITEMS, SCALE, TRAITS, Profile

st.set_page_config(
    page_title="MindMate · Student Wellness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Session State ----------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"
if "bot" not in st.session_state:
    st.session_state.bot = MindMateBot()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "text": "Hi, I'm MindMate. I'm here to listen, share coping tools, and help you check in with yourself. How are you feeling today?",
            "intent": None,
            "confidence": None,
            "crisis": "none",
        }
    ]
if "last_risk" not in st.session_state:
    st.session_state.last_risk = None
if "confirm_delete_journal" not in st.session_state:
    st.session_state.confirm_delete_journal = False

bot: MindMateBot = st.session_state.bot
journal_df = mood.load()
helplines = safety.load_helplines()
has_profile = bool(bot.profile and bot.profile.has_data)


def _send_chat_message(prompt_text: str) -> None:
    """Process a user message through MindMateBot and append both turns."""
    st.session_state.messages.append(
        {"role": "user", "text": prompt_text, "intent": None, "confidence": None, "crisis": "none"}
    )
    reply = bot.respond(prompt_text)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": reply.text,
            "intent": reply.intent,
            "confidence": reply.confidence,
            "crisis": reply.crisis,
        }
    )


def _go(page: str) -> None:
    st.session_state.active_page = page


dark = bool(st.session_state.dark_mode)
tokens = ui.theme_tokens(dark)
ui.inject_styles(dark)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🧠 MindMate Hub")
    st.toggle("🌙 Dark mode", key="dark_mode")

    profile_badge = ui.pill_html("Personalised ON", "low") if has_profile else ui.pill_html("Default Mode", "neutral")

    if st.session_state.last_risk:
        lr = st.session_state.last_risk
        risk_badge = ui.pill_html(f"{lr['level'].title()} ({lr['probability']:.0%})", lr["level"])
    else:
        risk_badge = ui.pill_html("Not checked yet", "neutral")

    st.markdown(
        f"""
<div class="mm-side-box">
<div class="mm-side-title">Session Overview</div>
<div style="margin-bottom:0.45rem;">Chat tone: {profile_badge}</div>
<div style="margin-bottom:0.45rem;">Risk check: {risk_badge}</div>
<div>Journal logs: <strong>{len(journal_df)}</strong> entries</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if has_profile:
        with st.expander("Your Big Five snapshot", expanded=False):
            for t in TRAITS:
                label = t.replace("_", " ").title()
                st.write(f"**{label}**: {bot.profile.levels[t]} ({bot.profile.scores[t]:.1f}/7)")

    st.markdown(ui.helpline_card(helplines, title="24×7 Crisis Support"), unsafe_allow_html=True)

    st.divider()
    st.caption(
        "Model status: "
        + ("Synthetic demo data" if risk.is_synthetic() else "Real dataset")
        + " · Journal stored locally in `data/journal.csv`."
    )

# ---------------- Top Brand & Navigation ----------------
ui.render_brand_header()

NAV_ITEMS = [
    ("Home", "🏠 Home"),
    ("Chat", "💬 Companion Chat"),
    ("Risk Check", "📊 Risk Check"),
    ("Personality", "🧩 Personality"),
    ("Journal", "📓 Mood Journal"),
    ("About", "ℹ️ About & Models"),
]
nav_cols = st.columns(len(NAV_ITEMS), gap="small")
for idx, (page_key, page_label) in enumerate(NAV_ITEMS):
    btn_type = "primary" if st.session_state.active_page == page_key else "secondary"
    if nav_cols[idx].button(page_label, key=f"nav_{page_key}", type=btn_type, width="stretch"):
        _go(page_key)
        st.rerun()

page = st.session_state.active_page

# ==============================================================================
# PAGE 1: LANDING / HOME
# ==============================================================================
if page == "Home":
    ui.render_landing_hero(len(journal_df), has_profile)

    cta1, cta2, cta3 = st.columns([1.2, 1.2, 1], gap="medium")
    if cta1.button("💬 Start Companion Chat", type="primary", width="stretch", key="hero_cta_chat"):
        _go("Chat")
        st.rerun()
    if cta2.button("📊 Run Wellbeing Risk Check", width="stretch", key="hero_cta_risk"):
        _go("Risk Check")
        st.rerun()
    if cta3.button("🧩 Personalise with Quiz", width="stretch", key="hero_cta_quiz"):
        _go("Personality")
        st.rerun()

    st.markdown(
        """
<div class="mm-sec-head">
<h3>How are you feeling right now?</h3>
<p>Pick a feeling below to jump straight into a tailored conversation with MindMate.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    mood_starters = [
        ("😌 Feeling good", "I'm feeling pretty good and calm today"),
        ("📚 Exam stress", "I'm really stressed about my upcoming exams"),
        ("😰 Anxious", "I feel so anxious and overwhelmed"),
        ("😔 Lonely / low", "I've been feeling really lonely on campus"),
        ("😴 Exhausted", "I've been having trouble sleeping lately"),
    ]
    m_cols = st.columns(len(mood_starters), gap="small")
    for idx, (m_label, m_prompt) in enumerate(mood_starters):
        if m_cols[idx].button(m_label, key=f"home_mood_{idx}", width="stretch"):
            _send_chat_message(m_prompt)
            _go("Chat")
            st.rerun()

    st.markdown(
        """
<div class="mm-sec-head">
<h3>Everything in MindMate</h3>
<p>Four focused modules built around student wellbeing, transparency, and safety.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns(4, gap="medium")

    with b1:
        st.markdown(
            f"""
<div class="mm-bento">
<div class="mm-bento-top">
<div class="mm-bento-icon">💬</div>
{ui.pill_html("Safety First", "high")}
</div>
<h4>Companion Chat</h4>
<p>Topic-aware emotional support paired with 6 guided coping exercises like paced breathing and 5-4-3-2-1 grounding.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open Chat →", key="bento_chat", width="stretch"):
            _go("Chat")
            st.rerun()

    with b2:
        r_pill = (
            ui.pill_html(st.session_state.last_risk["level"].title(), st.session_state.last_risk["level"])
            if st.session_state.last_risk
            else ui.pill_html("Screening Aid", "neutral")
        )
        st.markdown(
            f"""
<div class="mm-bento">
<div class="mm-bento-top">
<div class="mm-bento-icon">📊</div>
{r_pill}
</div>
<h4>Wellbeing Risk Check</h4>
<p>Survey-based pattern check across sleep, study pressure, and lifestyle factors with a clear visual breakdown.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open Risk Check →", key="bento_risk", width="stretch"):
            _go("Risk Check")
            st.rerun()

    with b3:
        p_pill = ui.pill_html("Active", "low") if has_profile else ui.pill_html("10 Items", "neutral")
        st.markdown(
            f"""
<div class="mm-bento">
<div class="mm-bento-top">
<div class="mm-bento-icon">🧩</div>
{p_pill}
</div>
<h4>Big Five Profile</h4>
<p>Quick TIPI personality check with a polar radar chart that adapts MindMate's tone and coping suggestions.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open Personality →", key="bento_quiz", width="stretch"):
            _go("Personality")
            st.rerun()

    with b4:
        j_pill = ui.pill_html(f"{len(journal_df)} Logs", "mid" if len(journal_df) else "neutral")
        st.markdown(
            f"""
<div class="mm-bento">
<div class="mm-bento-top">
<div class="mm-bento-icon">📓</div>
{j_pill}
</div>
<h4>Mood Journal</h4>
<p>Private local CSV check-ins tracking daily mood, sleep hours, and stress with 7-day trend insights.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open Journal →", key="bento_journal", width="stretch"):
            _go("Journal")
            st.rerun()

    st.markdown(
        """
<div class="mm-sec-head">
<h3>Designed with care &amp; transparency</h3>
</div>
""",
        unsafe_allow_html=True,
    )
    p1, p2, p3 = st.columns(3, gap="medium")
    p1.markdown(
        """
<div class="mm-card">
<h4>🛡️ Safety Layer Before ML</h4>
<p>Every chat message and journal note is screened by transparent crisis-detection rules before any classifier runs, surfacing immediate helplines when needed.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    p2.markdown(
        """
<div class="mm-card">
<h4>📈 Interpretable Models</h4>
<p>Inspect model comparisons, ROC curves, confusion matrices, and feature importance directly inside the About &amp; Models view.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    p3.markdown(
        """
<div class="mm-card">
<h4>🔒 Local-First Privacy</h4>
<p>Your personality quiz stays only in active session memory, and your mood journal is saved locally in <code>data/journal.csv</code>.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="mm-disclaimer">
<strong>Educational project notice:</strong> MindMate is an AI student project and is <strong>not</strong> a medical device, therapist, or diagnostic tool.
If you or someone you know is in crisis, call <strong>Tele-MANAS 14416</strong> (24×7, free) or emergency services <strong>112</strong>.
</div>
""",
        unsafe_allow_html=True,
    )

# ==============================================================================
# PAGE 2: COMPANION CHAT
# ==============================================================================
elif page == "Chat":
    col_chat_main, col_chat_side = st.columns([2.25, 1], gap="large")

    with col_chat_side:
        st.markdown(
            """
<div class="mm-card">
<h4>How MindMate replies</h4>
<p>1. <strong>Safety rules</strong> screen for crisis language first.<br>
2. <strong>Intent + keywords</strong> identify your topic.<br>
3. <strong>Personality profile</strong> tailors suggestions &amp; tone.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="mm-breath-box">
<strong style="font-size:0.86rem;">Quick 4-2-6 Calm Breathing</strong>
<div class="mm-breath-cycle">
<div class="mm-breath-phase">4s<span>Inhale</span></div>
<div class="mm-breath-phase">2s<span>Hold</span></div>
<div class="mm-breath-phase">6s<span>Exhale</span></div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if has_profile:
            st.success("Personalised replies are **ON** using your Big Five profile.")
        else:
            st.info("Take the **Personality** quiz to tailor replies and coping ideas to your style.")

        with st.expander("🧘 Coping Exercise Library", expanded=False):
            for ex in EXERCISES.values():
                st.markdown(f"**{ex['title']}**")
                st.markdown(ex["text"])
                st.divider()

        if st.button("🔄 Start fresh conversation", width="stretch", key="btn_clear_chat"):
            st.session_state.bot = MindMateBot(profile=bot.profile)
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "text": "Fresh start. What's on your mind right now?",
                    "intent": None,
                    "confidence": None,
                    "crisis": "none",
                }
            ]
            st.rerun()

    with col_chat_main:
        st.markdown("**Quick prompts** — click one to start:")
        quick_prompts = [
            ("📚 Stress", "I'm really stressed about my upcoming exams"),
            ("😴 Sleep", "I've been having trouble sleeping lately"),
            ("😰 Anxiety", "I feel so anxious and overwhelmed"),
            ("⏳ Focus", "I keep procrastinating and can't start studying"),
            ("🫶 Lonely", "I've been feeling really lonely on campus"),
        ]
        qcols = st.columns(len(quick_prompts))
        for idx, (label, prompt_val) in enumerate(quick_prompts):
            if qcols[idx].button(label, key=f"quick_prompt_{idx}", width="stretch"):
                _send_chat_message(prompt_val)
                st.rerun()

        chat_box = st.container(height=445, border=True)
        with chat_box:
            for msg in st.session_state.messages:
                avatar = "🧠" if msg["role"] == "assistant" else "🙂"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["text"])
                    if msg["role"] == "assistant" and msg.get("intent"):
                        intent_label = msg["intent"].replace("_", " ").title()
                        conf_pct = f"{msg['confidence']:.0%}" if msg.get("confidence") is not None else ""
                        if msg.get("crisis") not in (None, "none"):
                            meta_str = f"Safety layer triggered ({msg['crisis']} priority)"
                        elif msg["intent"] == "exercise":
                            meta_str = "Guided coping exercise"
                        elif msg["intent"] == "acknowledgement":
                            meta_str = "Conversation check-in"
                        elif msg["intent"] == "unknown":
                            meta_str = f"Clarifying prompt (confidence {conf_pct})"
                        else:
                            meta_str = f"Topic: {intent_label} · confidence {conf_pct}"
                        st.markdown(f'<div class="mm-chat-meta">{html.escape(meta_str)}</div>', unsafe_allow_html=True)

        if bot.pending_exercise and bot.pending_exercise in EXERCISES:
            ex_title = EXERCISES[bot.pending_exercise]["title"]
            if st.button(f"✨ Yes, guide me through: {ex_title}", type="primary", width="stretch", key="btn_accept_ex"):
                _send_chat_message("yes")
                st.rerun()

        if prompt := st.chat_input("Share what's on your mind..."):
            _send_chat_message(prompt)
            st.rerun()

# ==============================================================================
# PAGE 3: RISK CHECK
# ==============================================================================
elif page == "Risk Check":
    st.subheader("Wellbeing Risk Check")
    st.caption(
        "A screening aid based on patterns in student survey data—this is **not** a clinical diagnosis."
    )

    if risk.is_synthetic():
        st.warning(
            "Currently running on a model trained with **synthetic placeholder data** (`data/student_depression.csv` not found). "
            "Use for demonstration only."
        )

    d = risk.defaults()
    col_form_a, col_form_b = st.columns(2, gap="large")

    with col_form_a:
        st.markdown(
            '<div class="mm-card"><h4>🎓 Academic &amp; Study Load</h4><p>Rate your current study pressures and routine.</p></div>',
            unsafe_allow_html=True,
        )
        age = st.number_input("Age", 15, 40, int(d.get("Age", 21)))
        cgpa = st.number_input("CGPA (0–10)", 0.0, 10.0, float(d.get("CGPA", 7.5)), 0.1)
        ap = st.slider("Academic pressure (1 low – 5 high)", 1, 5, 3)
        ss = st.slider("Study satisfaction (1 low – 5 high)", 1, 5, 3)
        hours = st.slider("Work / study hours per day", 0, 12, 6)

    with col_form_b:
        st.markdown(
            '<div class="mm-card"><h4>🌿 Lifestyle &amp; Background</h4><p>Sleep, nutrition, and personal context factors.</p></div>',
            unsafe_allow_html=True,
        )
        sleep = st.selectbox("Sleep duration", SLEEP_ORDER, index=2)
        diet = st.selectbox("Dietary habits", ["Healthy", "Moderate", "Unhealthy"], index=1)
        fs = st.slider("Financial stress (1 low – 5 high)", 1, 5, 3)
        gender = st.selectbox("Gender", ["Male", "Female"])
        fam = st.selectbox("Family history of mental illness", ["No", "Yes"])

    if st.button("Run Risk Check", type="primary", width="stretch", key="btn_run_risk"):
        inputs = {
            "Age": age,
            "Gender": gender,
            "Academic Pressure": ap,
            "CGPA": cgpa,
            "Study Satisfaction": ss,
            "Work/Study Hours": hours,
            "Financial Stress": fs,
            "Sleep Duration": sleep,
            "Dietary Habits": diet,
            "Family History of Mental Illness": fam,
        }
        res = risk.predict(inputs)
        st.session_state.last_risk = {**res, "inputs": inputs}
        st.rerun()

    if st.session_state.last_risk:
        r = st.session_state.last_risk
        inp = r.get("inputs", {})
        prob = float(r["probability"])
        lvl = r["level"]

        st.divider()
        res_left, res_right = st.columns([1, 1.55], gap="large")

        with res_left:
            badge_html = ui.pill_html(f"{lvl.upper()} MATCH", lvl)
            st.markdown(
                f"""
<div class="mm-card" style="text-align:center;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
<span style="font-weight:700;">Risk Pattern Gauge</span>
{badge_html}
</div>
<div class="mm-kpi-sub">Classifier: {html.escape(r['model'])}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.altair_chart(ui.build_risk_gauge(prob, lvl, dark), theme=None, width="stretch")
            st.caption(
                f"Thresholds: Low ≤ {C.RISK_LOW_MAX:.0%} · Moderate ≤ {C.RISK_MODERATE_MAX:.0%} · Elevated > {C.RISK_MODERATE_MAX:.0%}"
            )
            st.markdown(
                f"""
<div class="mm-card">
<h4>What this means</h4>
<p>{html.escape(risk.ADVICE[lvl])}</p>
</div>
""",
                unsafe_allow_html=True,
            )

        with res_right:
            st.markdown("#### Lifestyle & Study Factor Breakdown")
            st.altair_chart(ui.build_risk_factors_chart(inp, dark), theme=None, width="stretch")

            factors = []
            if inp.get("Sleep Duration") in ("Less than 5 hours", "5-6 hours"):
                factors.append("Short sleep duration (< 7 hours) can strongly amplify stress and low mood.")
            if inp.get("Academic Pressure", 0) >= 4:
                factors.append("High academic pressure reported (4–5/5) — consider chunking tasks or visiting **Companion Chat**.")
            if inp.get("Study Satisfaction", 5) <= 2:
                factors.append("Low study satisfaction — adjusting study habits or discussing course load with a mentor may help.")
            if inp.get("Financial Stress", 0) >= 4:
                factors.append("Elevated financial stress — check if campus aid or counselling services offer support.")
            if inp.get("Dietary Habits") == "Unhealthy":
                factors.append("Unhealthy dietary habits — small regular meals and hydration help stabilise energy.")
            if inp.get("Work/Study Hours", 0) >= 9:
                factors.append("Long daily study/work hours (9+ h) — scheduled breaks protect against burnout.")

            if factors:
                with st.expander("Modifiable lifestyle observations", expanded=True):
                    for f_txt in factors:
                        st.write("• " + f_txt)

            if lvl == "elevated":
                st.markdown(ui.helpline_card(helplines, title="Recommended Support Contacts"), unsafe_allow_html=True)

# ==============================================================================
# PAGE 4: PERSONALITY QUIZ
# ==============================================================================
elif page == "Personality":
    st.subheader("Big Five Personality Profile (TIPI 10-Item)")
    st.caption(
        "Optional. Used as a design heuristic to adapt MindMate's tone and coping suggestions—never stored on disk."
    )

    quiz_col, prof_col = st.columns([1.3, 1], gap="large")

    with quiz_col:
        with st.form("quiz_form"):
            st.markdown("**Rate how much you agree with each statement (1 = Disagree strongly, 7 = Agree strongly):**")
            q_c1, q_c2 = st.columns(2, gap="medium")
            answers = []
            for i, (text, _, _) in enumerate(ITEMS):
                target_col = q_c1 if i % 2 == 0 else q_c2
                val = target_col.select_slider(
                    f"{i + 1}. {text}",
                    options=list(SCALE),
                    value=4,
                    format_func=lambda k: f"{k} · {SCALE[k]}",
                    key=f"quiz_q_{i}",
                )
                answers.append(val)

            if st.form_submit_button("💾 Save Profile for This Session", type="primary", width="stretch"):
                bot.profile = Profile.from_answers(answers)
                st.success("Personality profile saved! Your Companion Chat replies are now tailored.")
                st.rerun()

    with prof_col:
        st.markdown("#### Your Trait Radar & Breakdown")
        if bot.profile and bot.profile.has_data:
            radar_fig = ui.build_personality_radar(bot.profile.scores, dark)
            st.pyplot(radar_fig, clear_figure=True, width="stretch")

            trait_hints = {
                "extraversion": "Shapes social vs. quiet solo coping suggestions.",
                "agreeableness": "Influences relational framing and boundary tips.",
                "conscientiousness": "Tunes structured schedules vs. tiny low-friction steps.",
                "emotional_stability": "Low scores add extra gentle validation first.",
                "openness": "Suggests creative, reflective, or mindfulness options.",
            }
            trait_names = {
                "extraversion": "Extraversion",
                "agreeableness": "Agreeableness",
                "conscientiousness": "Conscientiousness",
                "emotional_stability": "Emotional Stability",
                "openness": "Openness",
            }
            for t in TRAITS:
                st.markdown(
                    ui.trait_card(
                        trait_names[t],
                        bot.profile.scores[t],
                        bot.profile.levels[t],
                        trait_hints[t],
                    ),
                    unsafe_allow_html=True,
                )
            st.caption("Design heuristic based on TIPI (Gosling, Rentfrow & Swann, 2003); not a clinical assessment.")
            if st.button("Reset personality profile", width="stretch", key="btn_reset_profile"):
                bot.profile = None
                st.rerun()
        else:
            st.markdown(
                """
<div class="mm-card">
<h4>No profile active yet</h4>
<p>Answer the 10 quick statements on the left and click <strong>Save Profile</strong> to generate your Big Five radar chart and enable personalised chat suggestions.</p>
</div>
""",
                unsafe_allow_html=True,
            )

# ==============================================================================
# PAGE 5: MOOD JOURNAL
# ==============================================================================
elif page == "Journal":
    st.subheader("Mood Journal & Personal Trends")
    j_left, j_right = st.columns([1, 1.45], gap="large")

    mood_labels = {1: "1 · Very low", 2: "2 · Low", 3: "3 · Okay", 4: "4 · Good", 5: "5 · Great"}
    stress_labels = {1: "1 · Calm", 2: "2 · Mild", 3: "3 · Moderate", 4: "4 · High", 5: "5 · Overwhelmed"}

    with j_left:
        st.markdown(
            '<div class="mm-card"><h4>📝 Log a Check-in</h4><p>Saved locally in <code>data/journal.csv</code>. Notes are screened by the safety layer.</p></div>',
            unsafe_allow_html=True,
        )
        with st.form("journal_form", clear_on_submit=True):
            m = st.select_slider("Mood today", options=[1, 2, 3, 4, 5], value=3, format_func=mood_labels.get)
            sl = st.number_input("Hours slept last night", 0.0, 14.0, 7.0, 0.5)
            stv = st.select_slider("Stress level", options=[1, 2, 3, 4, 5], value=3, format_func=stress_labels.get)
            note = st.text_area("Reflection / note (optional)", placeholder="What helped or weighed on you today?")
            if st.form_submit_button("Add Journal Entry", type="primary", width="stretch"):
                mood.add_entry(m, sl, stv, note)
                st.success("Entry saved to your local journal.")
                if safety.assess(note) != "none":
                    st.error(safety.crisis_response("high"))
                st.rerun()

    with j_right:
        df = mood.load()
        if df.empty:
            st.markdown(
                """
<div class="mm-card">
<h4>Your trends will appear here</h4>
<p>Add your first check-in on the left to start tracking mood, sleep, and stress over time.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(ui.kpi_card("Entries", str(len(df_sorted)), "Total logs"), unsafe_allow_html=True)
            k2.markdown(ui.kpi_card("Avg Mood", f"{df_sorted['mood'].mean():.1f}/5", "Scale 1–5"), unsafe_allow_html=True)
            k3.markdown(ui.kpi_card("Avg Sleep", f"{df_sorted['sleep_hours'].mean():.1f}h", "Per night"), unsafe_allow_html=True)
            k4.markdown(ui.kpi_card("Avg Stress", f"{df_sorted['stress'].mean():.1f}/5", "Scale 1–5"), unsafe_allow_html=True)

            st.markdown("##### Mood & Stress Over Time")
            melted = df_sorted.melt(
                id_vars=["timestamp"],
                value_vars=["mood", "stress"],
                var_name="Metric",
                value_name="Score",
            )
            melted["Metric"] = melted["Metric"].str.title()
            chart = (
                alt.Chart(melted)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X(
                        "timestamp:T",
                        title="Date / Time",
                        axis=alt.Axis(labelColor=tokens["muted"], titleColor=tokens["muted"], gridColor=tokens["grid"]),
                    ),
                    y=alt.Y(
                        "Score:Q",
                        scale=alt.Scale(domain=[0.5, 5.5]),
                        title="Score (1–5)",
                        axis=alt.Axis(labelColor=tokens["muted"], titleColor=tokens["muted"], gridColor=tokens["grid"]),
                    ),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(domain=["Mood", "Stress"], range=[tokens["accent"], "#D97757"]),
                        legend=alt.Legend(labelColor=tokens["text"], titleColor=tokens["muted"]),
                    ),
                    tooltip=[alt.Tooltip("timestamp:T", title="Time"), "Metric:N", "Score:Q"],
                )
                .properties(height=250)
                .configure_view(strokeWidth=0, fill="transparent")
                .configure(background="transparent")
            )
            st.altair_chart(chart, theme=None, width="stretch")

            st.markdown("##### Insights (last 7 entries)")
            for tip in mood.insights(df_sorted):
                st.info(tip)

            with st.expander("View & manage journal entries", expanded=False):
                display_df = df_sorted.sort_values("timestamp", ascending=False).copy()
                st.dataframe(display_df, hide_index=True, width="stretch")

                act_c1, act_c2 = st.columns(2)
                csv_bytes = display_df.to_csv(index=False).encode("utf-8")
                act_c1.download_button(
                    "⬇️ Export CSV",
                    data=csv_bytes,
                    file_name="mindmate_journal.csv",
                    mime="text/csv",
                    width="stretch",
                )
                if not st.session_state.confirm_delete_journal:
                    if act_c2.button("🗑️ Delete journal", width="stretch", key="btn_ask_delete"):
                        st.session_state.confirm_delete_journal = True
                        st.rerun()
                else:
                    if act_c2.button("⚠️ Confirm permanent delete", type="primary", width="stretch", key="btn_confirm_delete"):
                        mood.clear()
                        st.session_state.confirm_delete_journal = False
                        st.rerun()

# ==============================================================================
# PAGE 6: ABOUT & MODELS
# ==============================================================================
elif page == "About":
    st.subheader("How MindMate Works")
    steps = [
        ("1", "Safety Layer", "Rule-based regex screening checks every message and journal note for crisis language before any ML model runs."),
        ("2", "Intent Classifier", "TF-IDF (word + char n-grams) + Logistic Regression + keyword rules identify the student's topic."),
        ("3", "Personality Module", "10-item Big Five (TIPI) heuristic adapts tone and selects coping suggestions & guided exercises."),
        ("4", "Risk Screening", "Tabular ML pipeline (Logistic Regression / RF / SVM / GB compared via 5-fold CV) estimates lifestyle pattern match."),
        ("5", "Mood Journal", "Local CSV log tracking mood, sleep, stress, and correlations over the last 7 entries."),
    ]
    scols = st.columns(5, gap="small")
    for idx, (num, title, desc) in enumerate(steps):
        scols[idx].markdown(
            f"""
<div class="mm-step">
<div style="margin-bottom:0.45rem;">
<span class="mm-step-num">{num}</span>
<strong style="font-size:0.9rem;">{html.escape(title)}</strong>
</div>
<p style="font-size:0.82rem; color:var(--mm-muted); margin:0; line-height:1.45;">{html.escape(desc)}</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Model Evaluation & Reports")

    tab_m_path = C.REPORT_DIR / "tabular_metrics.json"
    int_m_path = C.REPORT_DIR / "intent_metrics.json"

    rep_c1, rep_c2 = st.columns(2, gap="large")
    with rep_c1:
        st.markdown("#### Tabular Risk Model")
        if tab_m_path.exists():
            tm = json.loads(tab_m_path.read_text())
            data_tag = "Synthetic demo data" if tm.get("synthetic_data") else "Real Kaggle data"
            st.caption(
                f"Best model: **{tm.get('best_model')}** · Rows: **{tm.get('n_rows')}** · Source: **{data_tag}**"
            )
            if tm.get("test"):
                st.dataframe(pd.DataFrame(tm["test"]), hide_index=True, width="stretch")
        else:
            st.info("Run `python -m src.train_tabular` to generate tabular metrics.")

    with rep_c2:
        st.markdown("#### Intent Classifier")
        if int_m_path.exists():
            im = json.loads(int_m_path.read_text())
            st.caption(
                f"Deployed: **{im.get('deployed')}** · Patterns: **{im.get('n_patterns')}** · Intents: **{im.get('n_intents')}** ({im.get('cv_folds')}-fold CV)"
            )
            if im.get("results"):
                st.dataframe(pd.DataFrame(im["results"]), hide_index=True, width="stretch")
        else:
            st.info("Run `python -m src.train_intent` to generate intent metrics.")

    chart_files = [
        ("tabular_model_comparison.png", "Tabular Model Comparison (Test Set)"),
        ("tabular_roc.png", "ROC Curves (Held-Out Test)"),
        ("tabular_feature_importance.png", "Permutation Feature Importance"),
        ("intent_confusion_matrix.png", "Intent Classifier Confusion Matrix"),
    ]
    existing_charts = [(C.REPORT_DIR / fname, cap) for fname, cap in chart_files if (C.REPORT_DIR / fname).exists()]
    if existing_charts:
        with st.expander("📈 View Training Plots from `reports/`", expanded=False):
            pcols = st.columns(2, gap="medium")
            for idx, (fpath, cap) in enumerate(existing_charts):
                pcols[idx % 2].image(str(fpath), caption=cap, width="stretch")

    st.divider()
    ab1, ab2 = st.columns(2, gap="large")
    with ab1:
        st.markdown(ui.helpline_card(helplines, title="Crisis & Support Helplines"), unsafe_allow_html=True)
    with ab2:
        st.markdown(
            """
<div class="mm-card">
<h4>Ethics &amp; Limitations</h4>
<p>• Survey-based screening is <strong>not</strong> a clinical diagnosis.<br>
• Rule-based crisis detection favours false alarms and may miss indirect phrasing.<br>
• Personality adaptation is a design heuristic, not clinically validated.<br>
• Journal entries are stored unencrypted locally in <code>data/journal.csv</code>.</p>
</div>
""",
            unsafe_allow_html=True,
        )
