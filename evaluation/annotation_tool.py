"""
Streamlit app for QA team to annotate and rate AI coaching suggestions.
"""

import streamlit as st
import json
import os

DATA_PATH = "data/annotations/annotations.json"

def load_annotations():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_annotation(annotation):
    annotations = load_annotations()
    annotations.append(annotation)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

def main():
    st.title("AI Coaching Suggestions Annotation Tool")

    conversation_id = st.text_input("Conversation ID")
    suggestion_category = st.selectbox(
        "Suggestion Category",
        ["tone_adjustment", "empathy", "technical_accuracy", "policy_reminder"]
    )
    suggestion_text = st.text_area("Suggestion Text")
    rating = st.slider("Quality Rating (1=Poor, 5=Excellent)", 1, 5, 3)
    comments = st.text_area("Comments (optional)")

    if st.button("Submit Annotation"):
        if not conversation_id or not suggestion_text:
            st.error("Please provide Conversation ID and Suggestion Text.")
        else:
            annotation = {
                "conversation_id": conversation_id,
                "category": suggestion_category,
                "suggestion": suggestion_text,
                "rating": rating,
                "comments": comments
            }
            save_annotation(annotation)
            st.success("Annotation saved!")

    st.subheader("Previous Annotations")
    annotations = load_annotations()
    if annotations:
        for ann in annotations[-10:][::-1]:  # show last 10 annotations
            st.markdown(f"**Conversation:** {ann['conversation_id']}")
            st.markdown(f"**Category:** {ann['category']}")
            st.markdown(f"**Suggestion:** {ann['suggestion']}")
            st.markdown(f"**Rating:** {ann['rating']}")
            if ann['comments']:
                st.markdown(f"**Comments:** {ann['comments']}")
            st.markdown("---")
    else:
        st.info("No annotations yet.")

if __name__ == "__main__":
    main()