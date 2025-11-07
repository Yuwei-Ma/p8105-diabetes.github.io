Final Project Proposal
================
Yuwei Ma, Rachel Lu, Chenxi Liu, Xiuhong Fan, Daisy Gui
2025-11-06

# Team Members

- Yuwei Ma (ym3138)
- Rachel Lu (cl4775)
- Chenxi Liu (cl4776)
- Xiuhong Fan (xf2302)
- Daisy Gui (yg3099)

# Tentative Project Title

Diabetes Risk Prediction Using Behavioral and Demographic Factors

# Motivation

Diabetes is one of the most prevalent chronic diseases in the United
States, affecting over 34 million Americans and placing an estimated
\$327 billion annual burden on the healthcare system. The disease
reduces life expectancy and quality of life while contributing to severe
complications including heart disease, kidney failure, vision loss, and
amputations. A significant challenge in diabetes management is
underdiagnosis - 1 in 5 diabetics and 8 in 10 prediabetics are unaware
of their condition. Therefore, early identification of at-risk
individuals enables lifestyle modification and preventive care,
potentially mitigating the burden of diabetes. We aim to discover in
this study the relevant risk factors of diabetes to better understand
predictive factors of diabetes. The CDC’s Behavioral Risk Factor
Surveillance System (BRFSS) provides extensive data on health-related
risk behaviors, chronic health conditions, and preventive care practices
across the United States. Leveraging the cleaned 2015 BRFSS dataset,
this project will: (1) identify key demographic, behavioral, and
clinical factors associated with diabetes, (2) build predictive models
to identify at-risk individuals, and (3) evaluate whether a reduced
subset of survey questions can maintain predictive accuracy. By
integrating statistical and machine learning approaches,we aim to
support the development of accessible, cost-effective screening tools
for early diabetes detection. This work has the potential to support the
development of accessible, cost-effective tools for early diabetes
detection and prevention in large populations, enabling population-level
screening and earlier intervention among high-risk individuals.

# Intended Final Products

**Introduction page**: Overview of diabetes and study motivation,
summary of the BRFSS dataset, and key variables used in the analysis.
This page will introduce the research goals and data structure.

**Statistical analysis page**: Description of the statistical methods,
including logistic regression, subgroup and interaction analyses, and
predictive modeling. This section will summarize key results such as
adjusted odds ratios, ROC curves, and model diagnostics.

**Visualizations page**: Interactive plots and charts showing the
distribution of metabolic, behavioral, and demographic factors across
diabetes groups. This will include histograms, boxplots, correlation
heatmaps, and subgroup comparisons. Final report: A concise written
summary of findings, interpretation of results, sensitivity checks, and
public health implications related to metabolic and lifestyle risk
factors for diabetes.

# Data Description and Sources

The
[dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
is from BRFSS, an annual health-related telephone survey collected by
CDC. This data is from the survey in 2015 that was collected from
441,455 Americans. The original dataset has 330 variables covering
medical, behavioral, and demographic indicators. Variables that are
either directly from the responses of participants or calculated from
their responses.

**Target Variable**: Diabetes_binary: indicates diabetes presence (1 =
prediabetes or diabetes, 0 = no diabetes).

**Feature Variables**: Medical: HighBP (High Blood Pressure), HighChol
(High Cholesterol), BMI Behavioral: Smoker (at least 100 cigarettes in
entire life), Fruits (Consume Fruit \>=1 times per day), Veggies
(Consume Vegetables \>=1 times per day), HvyAlcoholConsump (male \>=14
drinks/wk, female \>=7 drinks/wk) Demographic/Perception: Age, Income,
Sex

# Planned Analyses/Visualizations

**I. Exploring the Relationship Between Health Indicators and Diabetes
Status**

1.1 Distribution of Diabetes Status Across the Population A bar chart
will display the proportion of diabetic vs. non-diabetic individuals.

1.2 Comparison of Key Health Indicators by Diabetes Status Boxplots and
violin plots will compare variables such as BMI and Age between diabetic
and non-diabetic groups.

1.3 Correlation Among Predictors Heatmap of the Pearson correlation
coefficients among numeric predictors to identify multicollinearity.

**II. Logistic Regression and Cross-sectional Analysis**

2.1 Baseline Logistic Regression Model Binary logistic regression with
diabetes status as outcome. Forest plots display odds ratio and 95% CIs
for selected predictors.

2.2 Subgroup and Interaction Analyses Stratified regression by
demographic subgroups visualized with comparative bar charts.
Interactive terms such as Age ✕ BMI explore effect modification.

**III. Prediction and Model Evaluation**

3.1 Predictive Model Evaluation ROC curves, AUC scores, confusion
matrices, and precision-recall curves.

3.2 Feature Importance Ranking Bar charts of standardized coefficients
or SHAP values ranking predictor contributions.

# Coding Challenges

During the course of this project, we expect to encounter several
technical and analytical challenges, such as efficiently processing the
large BRFSS dataset, addressing missing values and class imbalance, and
implementing robust predictive and feature selection models in R.
Particular attention will be given to optimizing computational
performance, validating model accuracy, and ensuring transparency and
reproducibility through well-documented and reproducible workflows.

# Planned Timeline

- November 1-8: Data collection, cleaning, and preliminary exploration.
- November 9-15: Exploratory analysis, feature engineering and
  visualization
- November 16-22: Model building and evaluation
- November 23-30: Finalize data analysis and start on drafting the
  report.
- December 1-6: Complete draft report, with visualizations and
  interpretations.
- December 7-11: Review and finalize project report for submission.

# Project Structure

1.  Introduction:
    - Background information on diabetes
    - Source of the dataset, what it contains, and why it is suitable
      for this analysis
2.  Data Cleaning and Exploration:
    - Explain how missing values or outliers are handled and summarize
      key descriptive statistics and variable distributions.
3.  Association and Survival Analysis:
    - statistical tests to examine relationships between risk factors
      and diabetes outcomes.
4.  Results and Discussion:
    - Present the main findings
    - Figures/tables visualizations and interpret their implications in
      context.
5.  Recommendations:
    - Provide practical insights or public health suggestions based on
      the analysis results.
