"""
ROI analysis and agent performance improvement statistics.
"""

import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ANNOTATIONS_PATH = "data/annotations/annotations.json"
PERFORMANCE_PATH = "data/performance/agent_performance.json"

def load_annotations():
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_performance():
    if os.path.exists(PERFORMANCE_PATH):
        with open(PERFORMANCE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def calculate_average_rating(annotations):
    if not annotations:
        return 0
    ratings = [ann['rating'] for ann in annotations]
    return np.mean(ratings)

def calculate_roi(improvement_percent, cost, revenue_per_agent=100000):
    """
    Simple ROI calculation:
    improvement_percent: % improvement in agent performance (e.g., 0.1 for 10%)
    cost: cost of AI coaching system
    revenue_per_agent: average revenue generated per agent per year
    """
    gain = improvement_percent * revenue_per_agent
    roi = (gain - cost) / cost if cost > 0 else 0
    return roi

def main():
    st.title("Agent Performance and ROI Analysis")

    annotations = load_annotations()
    performance = load_performance()

    avg_rating = calculate_average_rating(annotations)
    st.metric("Average Suggestion Quality Rating", f"{avg_rating:.2f} / 5")

    if performance:
        df = pd.DataFrame(performance)
        st.subheader("Agent Performance Over Time")
        fig = px.line(df, x="date", y="performance_score", title="Agent Performance Score Over Time")
        st.plotly_chart(fig)

        # Calculate improvement percentage (example: last vs first)
        improvement = (df['performance_score'].iloc[-1] - df['performance_score'].iloc[0]) / df['performance_score'].iloc[0]
        st.metric("Performance Improvement", f"{improvement*100:.2f}%")

        # ROI calculation input
        cost = st.number_input("Enter AI Coaching System Cost ($)", min_value=0, value=50000)
        roi = calculate_roi(improvement, cost)
        st.metric("Estimated ROI", f"{roi*100:.2f}%")
    else:
        st.info("No agent performance data available.")

if __name__ == "__main__":
    main()