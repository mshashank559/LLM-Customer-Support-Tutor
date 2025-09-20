"""
Streamlit app for interactive customer support chat simulation with AI coaching suggestions.
"""

import streamlit as st
from engine.suggestion_engine import ConversationState, SuggestionEngine

st.set_page_config(page_title="Customer Support Chat Tutor", layout="wide")

# Initialize conversation state and suggestion engine in session state
if "conv_state" not in st.session_state:
    st.session_state.conv_state = ConversationState("demo_conversation")

if "engine" not in st.session_state:
    st.session_state.engine = SuggestionEngine()

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

def add_message(sender, text):
    st.session_state.conv_state.add_message(sender, text)

def get_suggestions():
    categories = ["tone_adjustment", "empathy", "technical_accuracy", "policy_reminder"]
    suggestions = []
    for cat in categories:
        suggestion = st.session_state.engine.generate_suggestion(
            st.session_state.conv_state.get_context(), cat
        )
        suggestions.append((cat, suggestion))
    return suggestions

def main():
    st.title("LLM-powered Customer Support Chat Tutor")

    # Display conversation history
    st.subheader("Conversation")
    for msg in st.session_state.conv_state.get_context(50):
        if msg["sender"] == "customer":
            st.markdown(f"**Customer:** {msg['text']}")
        else:
            st.markdown(f"**Agent:** {msg['text']}")

    # Input box for agent message
    agent_input = st.text_input("Agent: Type your message here", key="agent_input")

    if st.button("Send"):
        if agent_input.strip():
            add_message("agent", agent_input.strip())
            st.session_state.agent_input = ""
            # For demo, simulate a customer reply
            add_message("customer", "Thank you for your help!")

            # Generate AI coaching suggestions
            st.session_state.suggestions = get_suggestions()

    # Show AI coaching suggestions
    if st.session_state.suggestions:
        st.subheader("AI Coaching Suggestions")
        for cat, suggestion in st.session_state.suggestions:
            st.markdown(f"**{cat.replace('_', ' ').title()}:** {suggestion}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"👍 {cat}", key=f"like_{cat}"):
                    st.session_state.feedback.append((cat, "like"))
            with col2:
                if st.button(f"👎 {cat}", key=f"dislike_{cat}"):
                    st.session_state.feedback.append((cat, "dislike"))

    # Display feedback summary
    if st.session_state.feedback:
        st.subheader("Feedback Summary")
        feedback_counts = {}
        for cat, fb in st.session_state.feedback:
            if cat not in feedback_counts:
                feedback_counts[cat] = {"like": 0, "dislike": 0}
            feedback_counts[cat][fb] += 1
        for cat, counts in feedback_counts.items():
            st.markdown(f"**{cat.replace('_', ' ').title()}**: 👍 {counts['like']} | 👎 {counts['dislike']}")

if __name__ == "__main__":
    main()