# YouTube Channel Performance Segmentation

## 📊 Overview
This project analyzes YouTube trending video data to segment channels based on performance metrics using clustering techniques. The analysis explores patterns in views, likes, dislikes, comment counts, and category distributions to identify distinct channel archetypes.

## 🎯 Objectives
- Exploratory data analysis on YouTube trending videos
- Identify key performance metrics and their distributions
- Segment YouTube channels based on engagement patterns
- Discover relationships between engagement metrics
- Provide data-driven insights for content optimization

## 📁 Dataset
`CAvideos.csv` containing trending video data from Canada with:
- **Video metadata**: video_id, title, channel_title, category_id
- **Engagement metrics**: views, likes, dislikes, comment_count
- **Temporal data**: trending_date, publish_time
- **Content attributes**: tags, description, thumbnail_link

## 🔍 Key Analyses
- **Data Preprocessing**: Missing value analysis, datetime conversion, log transformation
- **Univariate Analysis**: Distribution analysis of engagement metrics
- **Bivariate Analysis**: Correlation between views, likes, and comments
- **Categorical Analysis**: Top channels by trending count and average views
- **Segmentation**: K-means, Agglomerative, and DBSCAN clustering

## 🛠️ Tech Stack
- **Python 3.8+**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn (StandardScaler, KMeans, AgglomerativeClustering, DBSCAN, PCA)

## 📈 Key Findings
- Engagement metrics follow right-skewed distributions (log transformation applied)
- Strong positive correlations: views↔likes (r≈0.85), likes↔comments (r≈0.80)
- Entertainment and Music categories dominate trending lists
- Gaming channels show high engagement rates relative to view counts

## 🚀 Getting Started
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
jupyter notebook 3-youtube-channel-performance-segmentation.ipynb
```

## 🤝 Contributing
Contributions welcome! Follow PEP 8 standards and include appropriate documentation.

---
