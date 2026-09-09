# YouTube Channel Performance Segmentation

An end-to-end data analysis and unsupervised machine learning project using a dataset of trending YouTube videos (`CAvideos.csv`). This project explores key video performance metrics (views, likes, dislikes, and comment counts) and applies data preprocessing, visual exploration, and statistical clustering algorithms.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features & Workflow](#-key-features--workflow)
- [Tech Stack & Libraries](#-tech-stack--libraries)
- [Dataset Summary](#-dataset-summary)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Getting Started](#-getting-started)
- [Usage](#-usage)

---

## 🎯 Project Overview
Understanding video performance and audience engagement is crucial for content creators and marketers on YouTube. This notebook cleans raw trending video data, transforms skewed performance metrics using logarithmic scale mappings, investigates distribution patterns via visual plots, and establishes the foundation for channel and video segmentation using clustering algorithms such as **K-Means**, **Agglomerative Clustering**, and **DBSCAN**.

---

## 🛠️ Key Features & Workflow
1. **Data Ingestion & Inspection:** 
   - Loaded and examined tabular structure consisting of $23,326$ rows and $16$ columns.
   - Assessed data types, distinct value counts, null patterns, and row-level duplicates.
2. **Data Preprocessing & Cleaning:**
   - Parsed ISO datetime fields (`publish_time`) and formatted date strings (`trending_date`) into clean Pandas `datetime64` types.
   - Dropped non-numerical metadata columns (`video_id`, `title`, `thumbnail_link`, `description`, `tags`) to prepare data for feature scaling and mathematical modeling.
3. **Exploratory Data Analysis (EDA):**
   - **Univariate Analysis:** Generated log-transformed ($\log(x + 1)$) feature distribution histograms for highly skewed metrics:
     - View counts
     - Like counts
     - Comment counts
   - **Categorical Breakdown:** Analyzed distribution frequency across various YouTube `category_id` classifications.
4. **Unsupervised Machine Learning Setup:**
   - Imported scikit-learn and SciPy clustering utilities (`StandardScaler`, `PCA`, `KMeans`, `AgglomerativeClustering`, `DBSCAN`, `silhouette_score`, and hierarchical `dendrogram` plotting).

---

## 🧰 Tech Stack & Libraries
* **Language:** Python 3.x
* **Environment:** Jupyter Notebook / Interactive Computing
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Analytics:** `scikit-learn`, `scipy`

---

## 📊 Dataset Summary

| Feature | Description | Data Type |
| :--- | :--- | :--- |
| `trending_date` | Date the video was trending (`YY.DD.MM`) | `datetime64[ns]` |
| `channel_title` | Title of the YouTube channel | `object` |
| `category_id` | Category classification ID | `int64` |
| `publish_time` | Video publication timestamp | `datetime64[ns, UTC]` |
| `views` | Total video views | `int64` |
| `likes` | Total video likes | `int64` |
| `dislikes` | Total video dislikes | `int64` |
| `comment_count` | Total video comments | `int64` |

---

## 📈 Exploratory Data Analysis (EDA)

The notebook highlights key distribution traits using logarithmic transformations ($\log(x+1)$) to normalize heavily right-skewed metrics:

* **Views Distribution:** Transformed view counts reveal a continuous distribution centered between $12$ and $15$ on the log scale.
* **Engagement Signals:** Likes and comment distributions exhibit long-tail characteristics, where top-performing videos account for a disproportionate amount of total user engagement.
* **Category Frequencies:** Displays distinct concentration spikes across specific video categories (such as Music, Entertainment, and News).

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with the required libraries.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
