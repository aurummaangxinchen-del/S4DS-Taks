import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Netflix EDA",
    layout="wide"
)

sns.set_theme(style="whitegrid")


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title(" Netflix Dataset — Exploratory Data Analysis")

st.markdown("""
This dashboard presents the exploratory data analysis performed
on the Netflix titles dataset using Pandas, Matplotlib and Seaborn.
""")


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("netflix_titles.csv")

    return df


df = load_data()


# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

df_clean = df.copy()

# Number of duplicates before cleaning
duplicates_before = df_clean.duplicated().sum()

# Remove duplicates
df_clean = df_clean.drop_duplicates()

# Fill missing values
columns_to_fill = [
    "director",
    "cast",
    "country",
    "rating",
    "duration"
]

for col in columns_to_fill:
    df_clean[col] = df_clean[col].fillna("Unknown")

# Convert date_added
df_clean["date_added"] = pd.to_datetime(
    df_clean["date_added"],
    errors="coerce"
)

# Extract year
df_clean["date_added_year"] = (
    df_clean["date_added"].dt.year
)


# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Original Rows",
        len(df)
    )

with col2:
    st.metric(
        "Rows After Cleaning",
        len(df_clean)
    )

with col3:
    st.metric(
        "Duplicates Removed",
        duplicates_before
    )

with col4:
    st.metric(
        "Columns",
        df_clean.shape[1]
    )


with st.expander("View Dataset"):

    st.dataframe(
        df_clean,
        use_container_width=True
    )


# ---------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------

st.header("2. Missing Value Analysis")

missing = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Percentage": (
        df.isnull().sum() / len(df)
    ) * 100
})

missing = missing[
    missing["Missing Values"] > 0
].sort_values(
    by="Missing Values",
    ascending=False
)

st.dataframe(
    missing,
    use_container_width=True
)


# ---------------------------------------------------
# FINDING 1
# ---------------------------------------------------

st.header("3. Movies vs TV Shows")

type_counts = df_clean["type"].value_counts()

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.barplot(
        x=type_counts.index,
        y=type_counts.values,
        ax=ax
    )

    ax.set_title("Movies vs TV Shows")
    ax.set_xlabel("Type")
    ax.set_ylabel("Number of Titles")

    st.pyplot(fig)


with col2:

    st.subheader("Finding")

    movies = type_counts.get("Movie", 0)
    tv_shows = type_counts.get("TV Show", 0)

    st.write(
        f"The dataset contains **{movies:,} movies** "
        f"and **{tv_shows:,} TV shows**."
    )

    if movies > tv_shows:
        st.success(
            "Movies make up the larger portion of Netflix content."
        )
    else:
        st.success(
            "TV Shows make up the larger portion of Netflix content."
        )


# ---------------------------------------------------
# FINDING 2 — RATINGS
# ---------------------------------------------------

st.header("4. Ratings Distribution")

rating_counts = df_clean["rating"].value_counts()

fig, ax = plt.subplots(figsize=(11, 6))

sns.barplot(
    x=rating_counts.index,
    y=rating_counts.values,
    ax=ax
)

ax.set_title("Netflix Content by Rating")
ax.set_xlabel("Rating")
ax.set_ylabel("Number of Titles")

plt.xticks(rotation=45)

st.pyplot(fig)

most_common_rating = rating_counts.idxmax()
most_common_rating_count = rating_counts.max()

st.info(
    f"**Finding:** {most_common_rating} is the most common rating, "
    f"with **{most_common_rating_count:,} titles**."
)


# ---------------------------------------------------
# FINDING 3 — RELEASE YEAR
# ---------------------------------------------------

st.header("5. Release Year Distribution")

fig, ax = plt.subplots(figsize=(12, 6))

sns.histplot(
    data=df_clean,
    x="release_year",
    bins=30,
    ax=ax
)

ax.set_title(
    "Distribution of Netflix Titles by Release Year"
)

ax.set_xlabel("Release Year")
ax.set_ylabel("Number of Titles")

st.pyplot(fig)


# ---------------------------------------------------
# FINDING 4 — CONTENT OVER TIME
# ---------------------------------------------------

st.header("6. Content Added Over Time")

yearly_content = (
    df_clean
    .dropna(subset=["date_added_year"])
    .groupby("date_added_year")
    .size()
)

fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(
    x=yearly_content.index,
    y=yearly_content.values,
    marker="o",
    ax=ax
)

ax.set_title(
    "Netflix Content Added Over Time"
)

ax.set_xlabel("Year")
ax.set_ylabel("Number of Titles")

st.pyplot(fig)


if len(yearly_content) > 0:

    highest_year = yearly_content.idxmax()
    highest_year_count = yearly_content.max()

    st.info(
        f"**Finding:** The highest number of titles were added in "
        f"**{highest_year}**, with **{highest_year_count:,} titles**."
    )


# ---------------------------------------------------
# MOVIES VS TV SHOWS OVER TIME
# ---------------------------------------------------

st.header("7. Movies vs TV Shows Added Over Time")

year_type = (
    df_clean
    .dropna(subset=["date_added_year"])
    .groupby(
        ["date_added_year", "type"]
    )
    .size()
    .unstack(fill_value=0)
)

fig, ax = plt.subplots(figsize=(12, 6))

year_type.plot(
    marker="o",
    ax=ax
)

ax.set_title(
    "Movies vs TV Shows Added Over Time"
)

ax.set_xlabel("Year")
ax.set_ylabel("Number of Titles")

st.pyplot(fig)


# ---------------------------------------------------
# MULTI-VALUED GENRE COLUMN
# ---------------------------------------------------

st.header("8. Multi-Valued Column Analysis — Genres")

st.write(
    "The `listed_in` column contains multiple genres for many titles. "
    "Instead of treating the entire string as one category, "
    "the genres are split into separate rows."
)

genre_df = df_clean[
    ["show_id", "title", "type", "listed_in"]
].copy()

# Split genres
genre_df["genre"] = genre_df[
    "listed_in"
].str.split(", ")

# Create separate rows
genre_df = genre_df.explode("genre")


with st.expander("View Split Genre Data"):

    st.dataframe(
        genre_df.head(50),
        use_container_width=True
    )


# ---------------------------------------------------
# TOP GENRES
# ---------------------------------------------------

genre_counts = genre_df["genre"].value_counts()

top_genres = genre_counts.head(15)

fig, ax = plt.subplots(figsize=(10, 7))

sns.barplot(
    x=top_genres.values,
    y=top_genres.index,
    ax=ax
)

ax.set_title(
    "Top 15 Netflix Genres"
)

ax.set_xlabel("Number of Titles")
ax.set_ylabel("Genre")

st.pyplot(fig)

top_genre = genre_counts.idxmax()
top_genre_count = genre_counts.max()

st.info(
    f"**Finding:** **{top_genre}** is the most common genre, "
    f"appearing in **{top_genre_count:,} titles**."
)


# ---------------------------------------------------
# MOVIES VS TV SHOWS BY GENRE
# ---------------------------------------------------

st.header("9. Movies vs TV Shows Across Top Genres")

top_10_genres = genre_counts.head(10).index

genre_type = genre_df[
    genre_df["genre"].isin(top_10_genres)
]

fig, ax = plt.subplots(figsize=(12, 7))

sns.countplot(
    data=genre_type,
    y="genre",
    hue="type",
    order=top_10_genres,
    ax=ax
)

ax.set_title(
    "Movies vs TV Shows Across Top Genres"
)

ax.set_xlabel("Number of Titles")
ax.set_ylabel("Genre")

st.pyplot(fig)


# ---------------------------------------------------
# COUNTRY ANALYSIS
# ---------------------------------------------------

st.header("10. Top Countries Producing Netflix Titles")

country_df = df_clean[
    ["show_id", "title", "country"]
].copy()

country_df["country"] = country_df[
    "country"
].replace("Unknown", np.nan)

# Split countries
country_df["country"] = (
    country_df["country"].str.split(", ")
)

# Separate rows
country_df = country_df.explode("country")

country_counts = country_df[
    "country"
].value_counts()

top_countries = country_counts.head(10)

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index,
    ax=ax
)

ax.set_title(
    "Top 10 Countries Producing Netflix Titles"
)

ax.set_xlabel("Number of Titles")
ax.set_ylabel("Country")

st.pyplot(fig)

top_country = country_counts.idxmax()
top_country_count = country_counts.max()

st.info(
    f"**Finding:** **{top_country}** has the highest number "
    f"of Netflix titles, with **{top_country_count:,} titles**."
)


# ---------------------------------------------------
# PIE CHART
# ---------------------------------------------------

st.header("11. Movies vs TV Shows — Percentage")

fig, ax = plt.subplots(figsize=(7, 7))

ax.pie(
    type_counts.values,
    labels=type_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title(
    "Movies vs TV Shows on Netflix"
)

st.pyplot(fig)


# ---------------------------------------------------
# FINAL FINDINGS
# ---------------------------------------------------

st.header("12. Key Findings")

st.markdown(f"""
###  Finding 1 — Movies vs TV Shows

The dataset contains **{movies:,} movies** and **{tv_shows:,} TV shows**.
Therefore, movies represent the larger portion of Netflix titles.

###  Finding 2 — Ratings

**{most_common_rating}** is the most common rating in the dataset,
with **{most_common_rating_count:,} titles**.

### Finding 3 — Countries

**{top_country}** is the country associated with the largest number
of Netflix titles after splitting the multi-valued country column.

###  Finding 4 — Genres

**{top_genre}** is the most frequently occurring genre after
splitting the multi-valued `listed_in` column.

###  Finding 5 — Time

The number of titles added to Netflix changes considerably from
year to year. The highest number of titles in the dataset was added
in **{highest_year}**.
""")


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Netflix EDA Dashboard | Pandas • Matplotlib • Seaborn • Streamlit"
)