import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Load Data ----------
aspect_path = "aspect_inconsistency_summary.csv"
review_path = "reviews_with_inconsistency.csv"

@st.cache_data
def load_data():
    aspect_df = pd.read_csv(aspect_path)
    review_df = pd.read_csv(review_path)
    return aspect_df, review_df

aspect_df, review_df = load_data()

# ---------- Page Config ----------
st.set_page_config(page_title="Restaurant Review Insights", layout="wide")
st.title("🍽️ Restaurant Review – Sentiment vs. Rating Analysis")
st.markdown("Explore inconsistencies between customer review text and star ratings using fine-tuned BERT.")

# ---------- KPI Section ----------
total_reviews = len(review_df)
inconsistent = review_df['is_inconsistent'].sum()
rate = round(inconsistent / total_reviews * 100, 2)

col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", f"{total_reviews:,}")
col2.metric("Inconsistent Reviews", f"{inconsistent:,}")
col3.metric("Inconsistency Rate", f"{rate}%")

st.divider()

# ---------- Aspect-Level Chart ----------
st.subheader("Aspect-Level Inconsistency")
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x='aspects', y='inconsistency_rate', data=aspect_df, ax=ax)
ax.set_ylabel("Inconsistency Rate (%)")
ax.set_xlabel("Aspect")
st.pyplot(fig)

st.divider()

# ---------- Review Explorer ----------
st.subheader("🔍 Explore Inconsistent Reviews")

# Filters
aspect_choice = st.selectbox("Select Aspect", options=["All"] + aspect_df['aspects'].tolist())
rating_choice = st.selectbox("Filter by Star Rating", options=["All"] + sorted(review_df['stars'].unique()))

filtered = review_df.copy()
if aspect_choice != "All":
    filtered = filtered[filtered['aspects'].apply(lambda a: aspect_choice in a if isinstance(a, list) else False)]
if rating_choice != "All":
    filtered = filtered[filtered['stars'] == rating_choice]
filtered = filtered[filtered['is_inconsistent'] == 1]

st.write(f"Showing {len(filtered)} inconsistent reviews")

# Display reviews
for _, row in filtered.head(10).iterrows():
    st.markdown(f"**⭐ {row['stars']} stars** | Predicted: {row['predicted_sentiment']}")
    st.write(row['clean_text'])
    st.divider()


# ---------- Insights Section ----------
st.divider()
st.subheader("📈 Key Insights Summary")

st.markdown("""
- **Overall inconsistency rate:** About **10–12%** of reviews had mismatched sentiment vs. ratings.
- **Highest mismatch aspects:** *Service (11.5%)* and *Price (12.1%)* — customers often praise food but complain about service or cost.
- **More consistent aspects:** *Food (11.1%)* and *Ambience (10.7%)* — text tone closely matches rating.
- **Behavioral takeaway:** Many customers still give high ratings despite negative text sentiment, showing bias toward overall experience rather than single issues.
- **Business implication:** Relying solely on star ratings can hide dissatisfaction trends, especially for service and pricing experiences.
""")


# ---------- Aspect-Level Chart ----------
st.subheader("Aspect-Level Inconsistency")
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x='aspects', y='inconsistency_rate', data=aspect_df, ax=ax)
ax.set_ylabel("Inconsistency Rate (%)")
ax.set_xlabel("Aspect")
st.pyplot(fig)

# ---------- Rating-Level Chart ----------
st.subheader("Rating-Level Inconsistency")

# Compute inconsistency by star rating
star_inconsistency = (
    review_df.groupby('stars')['is_inconsistent']
    .agg(['mean', 'count'])
    .reset_index()
    .rename(columns={'mean': 'inconsistency_rate', 'count': 'review_count'})
)
star_inconsistency['inconsistency_rate'] = (star_inconsistency['inconsistency_rate'] * 100).round(2)

# Display data table
st.dataframe(star_inconsistency.style.format({'inconsistency_rate': '{:.2f}'}))

# Plot the bar chart
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.barplot(x='stars', y='inconsistency_rate', data=star_inconsistency, palette='viridis', ax=ax2)
ax2.set_title("Inconsistency Rate by Star Rating")
ax2.set_xlabel("Star Rating")
ax2.set_ylabel("Inconsistency Rate (%)")
st.pyplot(fig2)
