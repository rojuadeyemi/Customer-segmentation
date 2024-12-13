## Project Title  
**Data-Driven Customer Segmentation for Strategic Marketing**  

## Objective  
The primary goal of this project is to segment customers into distinct groups based on their purchasing behaviors, preferences, and demographics. This segmentation enables businesses to better understand their customer base, personalize marketing strategies, and enhance customer retention and satisfaction.  
## Scope  
This project applies clustering techniques on a dataset containing customer demographics, transaction history, and behavioral data. Key deliverables include identified customer segments, actionable insights, and recommendations for targeted marketing strategies.  

## Steps Undertaken  

### 1. Data Collection and Preprocessing  
- Gathered data from a retail or e-commerce database, including customer demographics, purchase history, and transactional behaviors.  
- Cleaned and processed the dataset by handling missing values, standardizing data formats, and normalizing numeric features for clustering.  

### 2. Exploratory Data Analysis (EDA)  
- Analyzed data distributions and relationships using visualizations like histograms, scatter plots, and heatmaps.  
- Identified key features influencing customer behavior, such as total purchase value, frequency of transactions, and recency of last purchase.  

### 3. Feature Engineering  
- Derived meaningful metrics like RFM (Recency, Frequency, Monetary value) scores to quantify customer behavior.  
- Standardized features to ensure comparability during clustering.  

### 4. Clustering and Segmentation  
- Applied the **K-means algorithm** to cluster customers into distinct segments based on their behaviors.  
- Used techniques like the elbow method and silhouette scores to determine the optimal number of clusters.  

### 5. Validation and Profiling  
- Created detailed profiles for each segment, describing characteristics like average spend, purchase frequency.  

### 6. Insights and Recommendations  
See [Results](Customer Segmentation.pptx)
 

## Outcome  
This project delivered a detailed segmentation of the customer base, empowering the business to make data-driven decisions. It helped identify opportunities for improving customer engagement, optimizing marketing campaigns, and increasing revenue.  

## Tools and Technologies Used  
- **Programming Languages:** Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)  
- **Algorithms:** K-means clustering  
- **Visualization:** Power BI, and Matplotlib for reporting  
- **Additional Tools:** Jupyter Notebook for analysis and presentations  

## Business Impact  
- Enhanced understanding of customer needs and preferences  
- Increased ROI through targeted marketing and personalized customer experiences  
- Improved customer satisfaction and loyalty by tailoring services to specific segments  



## Model Deployment

### Prerequisites

- Python ~3.12
- Pip (Python package installer)
- Jupyter Notebook

### Setup


1. **Create a virtual environment**

Use the provided `Makefile` to create a virtual environment, and install dependencies by running `make` or `make all`.

You can also create a virtual environment manually using the following approaches.

For *Linux/Mac*:

```sh
python -m venv .venv
source .venv/bin/activate 
```

For *Windows*:
    
```sh
python -m venv .venv
.venv\Scripts\activate
```

2. **Install the required dependencies**

    ```sh
    pip install -U pip
    pip install -r requirements.txt
    ```

3. **Launch Jupyter Notebook**

You need to first activate the created environment, by running

For *Linux/Mac*:

```sh
source .venv/bin/activate
```

For *Windows*:

```sh
.venv\Scripts\activate
```

Then launch the `Jupyter Notebook` using
```sh
jupyter notebook
```


### Note
**Following the above steps will ensure reproducibility of the results**

