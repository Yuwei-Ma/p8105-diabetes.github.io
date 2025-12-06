Final Project
================
Yuwei Ma, Rachel Lu, Chenxi Liu, Xiuhong Fan, Daisy Gui
December 2025

# P8105 Final Project: Diabetes Health Indicators

**Task 1: Data Cleaning, Variable Processing, and Train/Test Dataset
Creation**

Dataset: CDC BRFSS 2015 Diabetes Health Indicators  
Source:
<https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset>

## Load Required Libraries

``` r
library(tidyverse)
library(caret)
library(skimr)
library(janitor)
library(corrplot)
library(DataExplorer)
library(knitr)

# Set seed for reproducibility
set.seed(123)
```

## 1. Data Import

The dataset has three versions: -
`diabetes_012_health_indicators_BRFSS2015.csv` (3 classes: 0, 1, 2) -
`diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (balanced
binary) - `diabetes_binary_health_indicators_BRFSS2015.csv` (full
binary)

For this project, we’ll use the full binary version for more data.

``` r
diabetes_raw <- read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")

glimpse(diabetes_raw)
```

    ## Rows: 253,680
    ## Columns: 22
    ## $ Diabetes_binary      <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0…
    ## $ HighBP               <dbl> 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1…
    ## $ HighChol             <dbl> 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1…
    ## $ CholCheck            <dbl> 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1…
    ## $ BMI                  <dbl> 40, 25, 28, 27, 24, 25, 30, 25, 30, 24, 25, 34, 2…
    ## $ Smoker               <dbl> 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0…
    ## $ Stroke               <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0…
    ## $ HeartDiseaseorAttack <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0…
    ## $ PhysActivity         <dbl> 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1…
    ## $ Fruits               <dbl> 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1…
    ## $ Veggies              <dbl> 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1…
    ## $ HvyAlcoholConsump    <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
    ## $ AnyHealthcare        <dbl> 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1…
    ## $ NoDocbcCost          <dbl> 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0…
    ## $ GenHlth              <dbl> 5, 3, 5, 2, 2, 2, 3, 3, 5, 2, 3, 3, 3, 4, 4, 2, 3…
    ## $ MentHlth             <dbl> 18, 0, 30, 0, 3, 0, 0, 0, 30, 0, 0, 0, 0, 0, 30, …
    ## $ PhysHlth             <dbl> 15, 0, 30, 0, 0, 2, 14, 0, 30, 0, 0, 30, 15, 0, 2…
    ## $ DiffWalk             <dbl> 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0…
    ## $ Sex                  <dbl> 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0…
    ## $ Age                  <dbl> 9, 7, 9, 11, 11, 10, 9, 11, 9, 8, 13, 10, 7, 11, …
    ## $ Education            <dbl> 4, 6, 4, 3, 5, 6, 6, 4, 5, 4, 6, 5, 5, 4, 6, 6, 4…
    ## $ Income               <dbl> 3, 1, 8, 6, 4, 8, 7, 4, 1, 3, 8, 1, 7, 6, 2, 8, 3…

## 2. Initial Data Exploration

``` r
# Check for missing values
colSums(is.na(diabetes_raw))
```

    ##      Diabetes_binary               HighBP             HighChol 
    ##                    0                    0                    0 
    ##            CholCheck                  BMI               Smoker 
    ##                    0                    0                    0 
    ##               Stroke HeartDiseaseorAttack         PhysActivity 
    ##                    0                    0                    0 
    ##               Fruits              Veggies    HvyAlcoholConsump 
    ##                    0                    0                    0 
    ##        AnyHealthcare          NoDocbcCost              GenHlth 
    ##                    0                    0                    0 
    ##             MentHlth             PhysHlth             DiffWalk 
    ##                    0                    0                    0 
    ##                  Sex                  Age            Education 
    ##                    0                    0                    0 
    ##               Income 
    ##                    0

``` r
# Summary statistics
summary(diabetes_raw)
```

    ##  Diabetes_binary      HighBP         HighChol        CholCheck     
    ##  Min.   :0.0000   Min.   :0.000   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.:0.0000   1st Qu.:0.000   1st Qu.:0.0000   1st Qu.:1.0000  
    ##  Median :0.0000   Median :0.000   Median :0.0000   Median :1.0000  
    ##  Mean   :0.1393   Mean   :0.429   Mean   :0.4241   Mean   :0.9627  
    ##  3rd Qu.:0.0000   3rd Qu.:1.000   3rd Qu.:1.0000   3rd Qu.:1.0000  
    ##  Max.   :1.0000   Max.   :1.000   Max.   :1.0000   Max.   :1.0000  
    ##       BMI            Smoker           Stroke        HeartDiseaseorAttack
    ##  Min.   :12.00   Min.   :0.0000   Min.   :0.00000   Min.   :0.00000     
    ##  1st Qu.:24.00   1st Qu.:0.0000   1st Qu.:0.00000   1st Qu.:0.00000     
    ##  Median :27.00   Median :0.0000   Median :0.00000   Median :0.00000     
    ##  Mean   :28.38   Mean   :0.4432   Mean   :0.04057   Mean   :0.09419     
    ##  3rd Qu.:31.00   3rd Qu.:1.0000   3rd Qu.:0.00000   3rd Qu.:0.00000     
    ##  Max.   :98.00   Max.   :1.0000   Max.   :1.00000   Max.   :1.00000     
    ##   PhysActivity        Fruits          Veggies       HvyAlcoholConsump
    ##  Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   
    ##  1st Qu.:1.0000   1st Qu.:0.0000   1st Qu.:1.0000   1st Qu.:0.0000   
    ##  Median :1.0000   Median :1.0000   Median :1.0000   Median :0.0000   
    ##  Mean   :0.7565   Mean   :0.6343   Mean   :0.8114   Mean   :0.0562   
    ##  3rd Qu.:1.0000   3rd Qu.:1.0000   3rd Qu.:1.0000   3rd Qu.:0.0000   
    ##  Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   
    ##  AnyHealthcare     NoDocbcCost         GenHlth         MentHlth     
    ##  Min.   :0.0000   Min.   :0.00000   Min.   :1.000   Min.   : 0.000  
    ##  1st Qu.:1.0000   1st Qu.:0.00000   1st Qu.:2.000   1st Qu.: 0.000  
    ##  Median :1.0000   Median :0.00000   Median :2.000   Median : 0.000  
    ##  Mean   :0.9511   Mean   :0.08418   Mean   :2.511   Mean   : 3.185  
    ##  3rd Qu.:1.0000   3rd Qu.:0.00000   3rd Qu.:3.000   3rd Qu.: 2.000  
    ##  Max.   :1.0000   Max.   :1.00000   Max.   :5.000   Max.   :30.000  
    ##     PhysHlth         DiffWalk           Sex              Age        
    ##  Min.   : 0.000   Min.   :0.0000   Min.   :0.0000   Min.   : 1.000  
    ##  1st Qu.: 0.000   1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.: 6.000  
    ##  Median : 0.000   Median :0.0000   Median :0.0000   Median : 8.000  
    ##  Mean   : 4.242   Mean   :0.1682   Mean   :0.4403   Mean   : 8.032  
    ##  3rd Qu.: 3.000   3rd Qu.:0.0000   3rd Qu.:1.0000   3rd Qu.:10.000  
    ##  Max.   :30.000   Max.   :1.0000   Max.   :1.0000   Max.   :13.000  
    ##    Education        Income     
    ##  Min.   :1.00   Min.   :1.000  
    ##  1st Qu.:4.00   1st Qu.:5.000  
    ##  Median :5.00   Median :7.000  
    ##  Mean   :5.05   Mean   :6.054  
    ##  3rd Qu.:6.00   3rd Qu.:8.000  
    ##  Max.   :6.00   Max.   :8.000

``` r
# Check for duplicates
sum(duplicated(diabetes_raw))
```

    ## [1] 24206

``` r
# Examine outcome variable distribution
table(diabetes_raw$Diabetes_binary)
```

    ## 
    ##      0      1 
    ## 218334  35346

``` r
prop.table(table(diabetes_raw$Diabetes_binary))
```

    ## 
    ##        0        1 
    ## 0.860667 0.139333

## 3. Data Cleaning

``` r
diabetes_clean <- diabetes_raw %>%
  # Clean column names
  clean_names() %>%
  
  # Fix the heart disease variable name
  rename(heart_disease_or_attack = heart_diseaseor_attack) %>%
  
  # Remove duplicates if any
  distinct() %>%
  
  # Remove rows with missing values (if any)
  drop_na()

glimpse(diabetes_clean)
```

    ## Rows: 229,474
    ## Columns: 22
    ## $ diabetes_binary         <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0…
    ## $ high_bp                 <dbl> 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1…
    ## $ high_chol               <dbl> 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0…
    ## $ chol_check              <dbl> 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1…
    ## $ bmi                     <dbl> 40, 25, 28, 27, 24, 25, 30, 25, 30, 24, 25, 34…
    ## $ smoker                  <dbl> 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0…
    ## $ stroke                  <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0…
    ## $ heart_disease_or_attack <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0…
    ## $ phys_activity           <dbl> 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1…
    ## $ fruits                  <dbl> 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0…
    ## $ veggies                 <dbl> 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0…
    ## $ hvy_alcohol_consump     <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
    ## $ any_healthcare          <dbl> 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1…
    ## $ no_docbc_cost           <dbl> 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0…
    ## $ gen_hlth                <dbl> 5, 3, 5, 2, 2, 2, 3, 3, 5, 2, 3, 3, 3, 4, 4, 2…
    ## $ ment_hlth               <dbl> 18, 0, 30, 0, 3, 0, 0, 0, 30, 0, 0, 0, 0, 0, 3…
    ## $ phys_hlth               <dbl> 15, 0, 30, 0, 0, 2, 14, 0, 30, 0, 0, 30, 15, 0…
    ## $ diff_walk               <dbl> 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0…
    ## $ sex                     <dbl> 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0…
    ## $ age                     <dbl> 9, 7, 9, 11, 11, 10, 9, 11, 9, 8, 13, 10, 7, 1…
    ## $ education               <dbl> 4, 6, 4, 3, 5, 6, 6, 4, 5, 4, 6, 5, 5, 4, 6, 6…
    ## $ income                  <dbl> 3, 1, 8, 6, 4, 8, 7, 4, 1, 3, 8, 1, 7, 6, 2, 8…

## 4. Variable Processing and Feature Engineering

### Variable Descriptions

According to the dataset documentation:

1.  **Diabetes_binary**: 0 = no diabetes, 1 = prediabetes or diabetes
2.  **HighBP**: 0 = no, 1 = yes
3.  **HighChol**: 0 = no, 1 = yes
4.  **CholCheck**: 0 = no, 1 = yes (cholesterol check in 5 years)
5.  **BMI**: Body Mass Index
6.  **Smoker**: 0 = no, 1 = yes
7.  **Stroke**: 0 = no, 1 = yes
8.  **HeartDiseaseorAttack**: 0 = no, 1 = yes
9.  **PhysActivity**: 0 = no, 1 = yes
10. **Fruits**: 0 = no, 1 = yes (consume fruit 1+ times per day)
11. **Veggies**: 0 = no, 1 = yes (consume vegetables 1+ times per day)
12. **HvyAlcoholConsump**: 0 = no, 1 = yes
13. **AnyHealthcare**: 0 = no, 1 = yes
14. **NoDocbcCost**: 0 = no, 1 = yes (couldn’t see doctor due to cost)
15. **GenHlth**: 1-5 scale (1 = excellent, 5 = poor)
16. **MentHlth**: days of poor mental health in past 30 days (0-30)
17. **PhysHlth**: days of poor physical health in past 30 days (0-30)
18. **DiffWalk**: 0 = no, 1 = yes (difficulty walking or climbing
    stairs)
19. **Sex**: 0 = female, 1 = male
20. **Age**: 1-13 (age categories)
21. **Education**: 1-6 (education level)
22. **Income**: 1-8 (income categories)

### Processing

``` r
diabetes_processed <- diabetes_clean %>%
  mutate(
    # Convert outcome to factor
    diabetes_binary = factor(diabetes_binary, 
                             levels = c(0, 1),
                             labels = c("No_Diabetes", "Diabetes")),
    
    # Convert binary variables to factors
    high_bp = factor(high_bp, levels = c(0, 1), labels = c("No", "Yes")),
    high_chol = factor(high_chol, levels = c(0, 1), labels = c("No", "Yes")),
    chol_check = factor(chol_check, levels = c(0, 1), labels = c("No", "Yes")),
    smoker = factor(smoker, levels = c(0, 1), labels = c("No", "Yes")),
    stroke = factor(stroke, levels = c(0, 1), labels = c("No", "Yes")),
    heart_disease_or_attack = factor(heart_disease_or_attack, 
                                     levels = c(0, 1), 
                                     labels = c("No", "Yes")),
    phys_activity = factor(phys_activity, levels = c(0, 1), labels = c("No", "Yes")),
    fruits = factor(fruits, levels = c(0, 1), labels = c("No", "Yes")),
    veggies = factor(veggies, levels = c(0, 1), labels = c("No", "Yes")),
    hvy_alcohol_consump = factor(hvy_alcohol_consump, 
                                 levels = c(0, 1), 
                                 labels = c("No", "Yes")),
    any_healthcare = factor(any_healthcare, levels = c(0, 1), labels = c("No", "Yes")),
    no_docbc_cost = factor(no_docbc_cost, levels = c(0, 1), labels = c("No", "Yes")),
    diff_walk = factor(diff_walk, levels = c(0, 1), labels = c("No", "Yes")),
    sex = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
    
    # Convert ordinal variables to factors
    gen_hlth_cat = factor(gen_hlth, 
                          levels = 1:5,
                          labels = c("Excellent", "Very Good", "Good", "Fair", "Poor"),
                          ordered = TRUE),
    
    # Create age groups
    age_cat = factor(age,
                     levels = 1:13,
                     labels = c("18-24", "25-29", "30-34", "35-39", "40-44",
                                "45-49", "50-54", "55-59", "60-64", "65-69",
                                "70-74", "75-79", "80+"),
                     ordered = TRUE),
    
    # Create education categories
    education_cat = factor(education,
                           levels = 1:6,
                           labels = c("Never/K-8", "Grades 9-11", "Grade 12/GED",
                                      "College 1-3 yrs", "College 4+ yrs", "Unknown"),
                           ordered = TRUE),
    
    # Create income categories
    income_cat = factor(income,
                        levels = 1:8,
                        labels = c("<$10k", "$10k-$15k", "$15k-$20k", "$20k-$25k",
                                   "$25k-$35k", "$35k-$50k", "$50k-$75k", "$75k+"),
                        ordered = TRUE),
    
    # Create BMI categories
    bmi_category = case_when(
      bmi < 18.5 ~ "Underweight",
      bmi >= 18.5 & bmi < 25 ~ "Normal",
      bmi >= 25 & bmi < 30 ~ "Overweight",
      bmi >= 30 ~ "Obese"
    ) %>% factor(levels = c("Underweight", "Normal", "Overweight", "Obese"),
                 ordered = TRUE),
    
    # Create mental health categories
    ment_hlth_cat = case_when(
      ment_hlth == 0 ~ "None",
      ment_hlth <= 7 ~ "1-7 days",
      ment_hlth <= 14 ~ "8-14 days",
      ment_hlth <= 30 ~ "15-30 days"
    ) %>% factor(levels = c("None", "1-7 days", "8-14 days", "15-30 days"),
                 ordered = TRUE),
    
    # Create physical health categories
    phys_hlth_cat = case_when(
      phys_hlth == 0 ~ "None",
      phys_hlth <= 7 ~ "1-7 days",
      phys_hlth <= 14 ~ "8-14 days",
      phys_hlth <= 30 ~ "15-30 days"
    ) %>% factor(levels = c("None", "1-7 days", "8-14 days", "15-30 days"),
                 ordered = TRUE),
    
    # Create composite health score
    health_risk_score = as.numeric(high_bp == "Yes") +
      as.numeric(high_chol == "Yes") +
      as.numeric(stroke == "Yes") +
      as.numeric(heart_disease_or_attack == "Yes") +
      as.numeric(smoker == "Yes") +
      as.numeric(hvy_alcohol_consump == "Yes") +
      (gen_hlth - 1) / 4,
    
    # Create healthy lifestyle score
    lifestyle_score = as.numeric(phys_activity == "Yes") +
      as.numeric(fruits == "Yes") +
      as.numeric(veggies == "Yes")
  )

glimpse(diabetes_processed)
```

    ## Rows: 229,474
    ## Columns: 31
    ## $ diabetes_binary         <fct> No_Diabetes, No_Diabetes, No_Diabetes, No_Diab…
    ## $ high_bp                 <fct> Yes, No, Yes, Yes, Yes, Yes, Yes, Yes, Yes, No…
    ## $ high_chol               <fct> Yes, No, Yes, No, Yes, Yes, No, Yes, Yes, No, …
    ## $ chol_check              <fct> Yes, No, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Ye…
    ## $ bmi                     <dbl> 40, 25, 28, 27, 24, 25, 30, 25, 30, 24, 25, 34…
    ## $ smoker                  <fct> Yes, Yes, No, No, No, Yes, Yes, Yes, Yes, No, …
    ## $ stroke                  <fct> No, No, No, No, No, No, No, No, No, No, No, No…
    ## $ heart_disease_or_attack <fct> No, No, No, No, No, No, No, No, Yes, No, No, N…
    ## $ phys_activity           <fct> No, Yes, No, Yes, Yes, Yes, No, Yes, No, No, Y…
    ## $ fruits                  <fct> No, No, Yes, Yes, Yes, Yes, No, No, Yes, No, Y…
    ## $ veggies                 <fct> Yes, No, No, Yes, Yes, Yes, No, Yes, Yes, Yes,…
    ## $ hvy_alcohol_consump     <fct> No, No, No, No, No, No, No, No, No, No, No, No…
    ## $ any_healthcare          <fct> Yes, No, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Ye…
    ## $ no_docbc_cost           <fct> No, Yes, Yes, No, No, No, No, No, No, No, No, …
    ## $ gen_hlth                <dbl> 5, 3, 5, 2, 2, 2, 3, 3, 5, 2, 3, 3, 3, 4, 4, 2…
    ## $ ment_hlth               <dbl> 18, 0, 30, 0, 3, 0, 0, 0, 30, 0, 0, 0, 0, 0, 3…
    ## $ phys_hlth               <dbl> 15, 0, 30, 0, 0, 2, 14, 0, 30, 0, 0, 30, 15, 0…
    ## $ diff_walk               <fct> Yes, No, Yes, No, No, No, No, Yes, Yes, No, No…
    ## $ sex                     <fct> Female, Female, Female, Female, Female, Male, …
    ## $ age                     <dbl> 9, 7, 9, 11, 11, 10, 9, 11, 9, 8, 13, 10, 7, 1…
    ## $ education               <dbl> 4, 6, 4, 3, 5, 6, 6, 4, 5, 4, 6, 5, 5, 4, 6, 6…
    ## $ income                  <dbl> 3, 1, 8, 6, 4, 8, 7, 4, 1, 3, 8, 1, 7, 6, 2, 8…
    ## $ gen_hlth_cat            <ord> Poor, Good, Poor, Very Good, Very Good, Very G…
    ## $ age_cat                 <ord> 60-64, 50-54, 60-64, 70-74, 70-74, 65-69, 60-6…
    ## $ education_cat           <ord> College 1-3 yrs, Unknown, College 1-3 yrs, Gra…
    ## $ income_cat              <ord> $15k-$20k, <$10k, $75k+, $35k-$50k, $20k-$25k,…
    ## $ bmi_category            <ord> Obese, Overweight, Overweight, Overweight, Nor…
    ## $ ment_hlth_cat           <ord> 15-30 days, None, 15-30 days, None, 1-7 days, …
    ## $ phys_hlth_cat           <ord> 15-30 days, None, 15-30 days, None, None, 1-7 …
    ## $ health_risk_score       <dbl> 4.00, 1.50, 3.00, 1.25, 2.25, 3.25, 2.50, 3.50…
    ## $ lifestyle_score         <dbl> 1, 1, 1, 3, 3, 3, 0, 2, 2, 1, 3, 2, 1, 1, 2, 1…

## 5. Data Quality Checks

``` r
# Check for any issues in processed data
sum(is.na(diabetes_processed))
```

    ## [1] 0

``` r
# Verify factor levels
table(diabetes_processed$diabetes_binary)
```

    ## 
    ## No_Diabetes    Diabetes 
    ##      194377       35097

``` r
# Check continuous variables for outliers
diabetes_processed %>%
  select(bmi, ment_hlth, phys_hlth, health_risk_score, lifestyle_score) %>%
  summary()
```

    ##       bmi          ment_hlth       phys_hlth      health_risk_score
    ##  Min.   :12.00   Min.   : 0.00   Min.   : 0.000   Min.   :0.000    
    ##  1st Qu.:24.00   1st Qu.: 0.00   1st Qu.: 0.000   1st Qu.:1.000    
    ##  Median :27.00   Median : 0.00   Median : 0.000   Median :1.750    
    ##  Mean   :28.69   Mean   : 3.51   Mean   : 4.681   Mean   :1.971    
    ##  3rd Qu.:32.00   3rd Qu.: 2.00   3rd Qu.: 4.000   3rd Qu.:2.750    
    ##  Max.   :98.00   Max.   :30.00   Max.   :30.000   Max.   :7.000    
    ##  lifestyle_score
    ##  Min.   :0.00   
    ##  1st Qu.:2.00   
    ##  Median :2.00   
    ##  Mean   :2.14   
    ##  3rd Qu.:3.00   
    ##  Max.   :3.00

## 6. Train/Test Split

Create stratified train/test split (80/20) using stratified sampling to
maintain outcome distribution.

``` r
# Set seed again for reproducibility
set.seed(123)

# Create partition index
train_index <- createDataPartition(
  diabetes_processed$diabetes_binary,
  p = 0.80,
  list = FALSE,
  times = 1
)

# Create training and testing datasets
diabetes_train <- diabetes_processed[train_index, ]
diabetes_test <- diabetes_processed[-train_index, ]

# Verify split proportions
nrow(diabetes_train)
```

    ## [1] 183580

``` r
nrow(diabetes_test)
```

    ## [1] 45894

``` r
# Check outcome distribution in train and test sets
prop.table(table(diabetes_train$diabetes_binary))
```

    ## 
    ## No_Diabetes    Diabetes 
    ##   0.8470531   0.1529469

``` r
prop.table(table(diabetes_test$diabetes_binary))
```

    ## 
    ## No_Diabetes    Diabetes 
    ##   0.8470606   0.1529394

## 7. Export Cleaned Datasets

``` r
# Create output directory if it doesn't exist
if (!dir.exists("data")) {
  dir.create("data")
}

# Save processed datasets
write_csv(diabetes_processed, "data/diabetes_processed.csv")
write_csv(diabetes_train, "data/diabetes_train.csv")
write_csv(diabetes_test, "data/diabetes_test.csv")

# Also save as RDS for faster loading in R
saveRDS(diabetes_processed, "data/diabetes_processed.rds")
saveRDS(diabetes_train, "data/diabetes_train.rds")
saveRDS(diabetes_test, "data/diabetes_test.rds")
```

## 8. Data Dictionary and Variable Summary

``` r
# Create and save data dictionary
data_dictionary <- tribble(
  ~Variable, ~Type, ~Description, ~Values,
  "diabetes_binary", "Factor", "Diabetes diagnosis", "No_Diabetes, Diabetes",
  "high_bp", "Factor", "High blood pressure", "No, Yes",
  "high_chol", "Factor", "High cholesterol", "No, Yes",
  "chol_check", "Factor", "Cholesterol check in past 5 years", "No, Yes",
  "bmi", "Numeric", "Body Mass Index", "Continuous",
  "smoker", "Factor", "Smoked at least 100 cigarettes in life", "No, Yes",
  "stroke", "Factor", "History of stroke", "No, Yes",
  "heart_disease_or_attack", "Factor", "Coronary heart disease or MI", "No, Yes",
  "phys_activity", "Factor", "Physical activity in past 30 days", "No, Yes",
  "fruits", "Factor", "Consume fruit 1+ times per day", "No, Yes",
  "veggies", "Factor", "Consume vegetables 1+ times per day", "No, Yes",
  "hvy_alcohol_consump", "Factor", "Heavy alcohol consumption", "No, Yes",
  "any_healthcare", "Factor", "Have any healthcare coverage", "No, Yes",
  "no_docbc_cost", "Factor", "Could not see doctor due to cost", "No, Yes",
  "gen_hlth", "Numeric", "General health rating", "1-5 (1=excellent, 5=poor)",
  "ment_hlth", "Numeric", "Days of poor mental health (past 30)", "0-30",
  "phys_hlth", "Numeric", "Days of poor physical health (past 30)", "0-30",
  "diff_walk", "Factor", "Difficulty walking or climbing stairs", "No, Yes",
  "sex", "Factor", "Biological sex", "Female, Male",
  "age", "Numeric", "Age category", "1-13 (see age_cat for labels)",
  "education", "Numeric", "Education level", "1-6",
  "income", "Numeric", "Income level", "1-8",
  "bmi_category", "Factor", "BMI category", "Underweight, Normal, Overweight, Obese",
  "health_risk_score", "Numeric", "Composite health risk score", "Continuous (0-7)",
  "lifestyle_score", "Numeric", "Healthy lifestyle score", "0-3"
)

write_csv(data_dictionary, "data/data_dictionary.csv")
```

``` r
# Create comprehensive variable summary table organized by category
variables_summary <- tibble(
  Category = c(
    "Outcome",
    "Health Conditions", "Health Conditions", "Health Conditions", "Health Conditions", "Health Conditions",
    "Physical Metrics", "Physical Metrics", "Physical Metrics",
    "Lifestyle", "Lifestyle", "Lifestyle", "Lifestyle", "Lifestyle",
    "Health Status", "Health Status", "Health Status", "Health Status", "Health Status",
    "Healthcare Access", "Healthcare Access",
    "Demographics", "Demographics", "Demographics", "Demographics", "Demographics", "Demographics", "Demographics",
    "Engineered Features", "Engineered Features"
  ),
  Variable = c(
    "diabetes_binary",
    "high_bp", "high_chol", "stroke", "heart_disease_or_attack", "chol_check",
    "bmi", "bmi_category", "diff_walk",
    "smoker", "phys_activity", "fruits", "veggies", "hvy_alcohol_consump",
    "gen_hlth", "ment_hlth", "phys_hlth", "ment_hlth_cat", "phys_hlth_cat",
    "any_healthcare", "no_docbc_cost",
    "age", "age_cat", "sex", "education", "education_cat", "income", "income_cat",
    "health_risk_score", "lifestyle_score"
  ),
  Description = c(
    "No_Diabetes, Diabetes",
    "High blood pressure (No, Yes)", "High cholesterol (No, Yes)", "History of stroke (No, Yes)", 
    "History of heart disease or attack (No, Yes)", "Cholesterol check in past 5 years (No, Yes)",
    "Body Mass Index (continuous)", "Underweight, Normal, Overweight, Obese", "Difficulty walking/climbing stairs (No, Yes)",
    "Smoked 100+ cigarettes lifetime (No, Yes)", "Physical activity past 30 days (No, Yes)", 
    "Consume fruit 1+ times/day (No, Yes)", "Consume vegetables 1+ times/day (No, Yes)", "Heavy alcohol consumption (No, Yes)",
    "General health (1-5: Excellent to Poor)", "Days poor mental health past 30 days (0-30)", 
    "Days poor physical health past 30 days (0-30)", "Mental health categories (None, 1-7, 8-14, 15-30 days)", 
    "Physical health categories (None, 1-7, 8-14, 15-30 days)",
    "Has healthcare coverage (No, Yes)", "Could not see doctor due to cost (No, Yes)",
    "Age category (1-13)", "Age groups (18-24, 25-29, ..., 80+)", "Biological sex (Female, Male)", 
    "Education level (1-6)", "Education categories", "Income level (1-8)", "Income categories (<$10k, ..., $75k+)",
    "Composite health risk score (0-7)", "Healthy lifestyle score (0-3)"
  )
)

kable(variables_summary, caption = "Dataset Variables by Category")
```

| Category | Variable | Description |
|:---|:---|:---|
| Outcome | diabetes_binary | No_Diabetes, Diabetes |
| Health Conditions | high_bp | High blood pressure (No, Yes) |
| Health Conditions | high_chol | High cholesterol (No, Yes) |
| Health Conditions | stroke | History of stroke (No, Yes) |
| Health Conditions | heart_disease_or_attack | History of heart disease or attack (No, Yes) |
| Health Conditions | chol_check | Cholesterol check in past 5 years (No, Yes) |
| Physical Metrics | bmi | Body Mass Index (continuous) |
| Physical Metrics | bmi_category | Underweight, Normal, Overweight, Obese |
| Physical Metrics | diff_walk | Difficulty walking/climbing stairs (No, Yes) |
| Lifestyle | smoker | Smoked 100+ cigarettes lifetime (No, Yes) |
| Lifestyle | phys_activity | Physical activity past 30 days (No, Yes) |
| Lifestyle | fruits | Consume fruit 1+ times/day (No, Yes) |
| Lifestyle | veggies | Consume vegetables 1+ times/day (No, Yes) |
| Lifestyle | hvy_alcohol_consump | Heavy alcohol consumption (No, Yes) |
| Health Status | gen_hlth | General health (1-5: Excellent to Poor) |
| Health Status | ment_hlth | Days poor mental health past 30 days (0-30) |
| Health Status | phys_hlth | Days poor physical health past 30 days (0-30) |
| Health Status | ment_hlth_cat | Mental health categories (None, 1-7, 8-14, 15-30 days) |
| Health Status | phys_hlth_cat | Physical health categories (None, 1-7, 8-14, 15-30 days) |
| Healthcare Access | any_healthcare | Has healthcare coverage (No, Yes) |
| Healthcare Access | no_docbc_cost | Could not see doctor due to cost (No, Yes) |
| Demographics | age | Age category (1-13) |
| Demographics | age_cat | Age groups (18-24, 25-29, …, 80+) |
| Demographics | sex | Biological sex (Female, Male) |
| Demographics | education | Education level (1-6) |
| Demographics | education_cat | Education categories |
| Demographics | income | Income level (1-8) |
| Demographics | income_cat | Income categories (\<\$10k, …, \$75k+) |
| Engineered Features | health_risk_score | Composite health risk score (0-7) |
| Engineered Features | lifestyle_score | Healthy lifestyle score (0-3) |

Dataset Variables by Category

## Exploratory analysis

``` r
# Load processed data
diabetes <- read_csv("data/diabetes_processed.csv") %>%
  mutate(
    diabetes_binary = factor(diabetes_binary,
                             levels = c("No_Diabetes", "Diabetes"))
  )
```

# Research Questions

In this project we would like to focus on these three questions:

1.  Which demographic, lifestyle, and clinical characteristics are most
    strongly associated with diabetes?

2.  How effectively can logistic regression models predict diabetes
    status using only self-reported survey information?

3.  Does model complexity like interaction terms improve predictive
    performance?

# Data

We use the cleaned [2015 BRFSS
dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)
(253,680 adults, 22 variables), focusing on variables related to
demographics, lifestyle, and self-reported health.

- **Population:** Adults participating in the 2015 BRFSS survey.
- **Outcome:** Self-reported diabetes status (binary outcome used for
  prediction).
- **Main predictors:** BMI, age category, general health rating, blood
  pressure, cholesterol, smoking status, physical activity, sex, and
  socioeconomic indicators

## Key preprocessing steps:

- No missing data were present (already cleaned by BRFSS).

- Created interpretable new variables according to [dataset
  notebook](https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook)
  and
  [codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf).

  - BMI categories

  - Mental & physical health categories

  - Composite lifestyle and health-risk scores

  - Converted categorical variables into meaningful factor labels

- Train/test Split

  - 80/20 stratified split is used to preserve diabetes prevalence.

  - Training set: approximately 202,944 observations (for EDA and model
    fitting)

  - Test set: approximately 50,736 observations (for final evaluation
    only)

- Fixed random seed (123) for reproducibility

## Exploratory analysis revealed:

- **Strong class imbalance (far more non-diabetic respondents).**

``` r
# Distribution of Diabetes Status
diabetes %>%
  ggplot(aes(x = diabetes_binary, fill = diabetes_binary)) +
  geom_bar(width = 0.55, alpha = 0.9, color = "white") +
  scale_fill_manual(values = c("#5DA5DA", "#F17CB0")) + 
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    title = "Distribution of Diabetes Status",
    x = "Diabetes Status",
    y = "Count"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 14, margin = margin(t = 10)),
    axis.title.y = element_text(size = 14, margin = margin(r = 10)),
    axis.text = element_text(size = 12),
    legend.position = "none",
    plot.margin = margin(15, 15, 15, 15)
  )
```

![](p8105_final_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

According to this plot, diabetes individual included is much less than
non-diabetes. This may lead to strong class imbalance.

``` r
# BMI Histogram
diabetes %>%
  ggplot(aes(x = bmi)) +
  geom_histogram(bins = 40, fill = "#6495ED", alpha = 0.8) +
  labs(
    title = "Distribution of BMI",
    x = "BMI",
    y = "Count"
  ) +
  theme_minimal()
```

![](p8105_final_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

The BMI in this sample is right-skewed, with most values falling between
20 and 40.

The concentration of observations in the overweight and obese range
reflects the population distribution in this dataset. A smaller number
of individuals have very high BMI values, forming a long right tail.

``` r
# Age Category Distribution
diabetes %>%
  ggplot(aes(x = age_cat)) +
  geom_bar(fill = "#8B5FBF", alpha = 0.8) +
  labs(
    title = "Age Category Distribution",
    x = "Age Category",
    y = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

![](p8105_final_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Left-skewness in this plot shows that the dataset contains far more
middle-aged and older adults than younger respondents. Counts increase
steadily from the 18–24 group and peak between ages 55–69, before
gradually declining in the 70+ categories. This uneven age distribution
is important because diabetes risk rises with age. This means that the
sample naturally contains more individuals from higher-risk age groups.

- **Diabetic individuals tend to have higher BMI, poor general health,
  and older age.**

``` r
# BMI by Diabetes Status
diabetes %>%
  ggplot(aes(x = diabetes_binary, y = bmi, fill = diabetes_binary)) +
  geom_boxplot(alpha = 0.85) +
  scale_fill_manual(values = c("lightblue", "lightpink")) +
  labs(
    title = "BMI by Diabetes Status",
    x = "Diabetes Status",
    y = "BMI"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
```

![](p8105_final_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

The plot shows that individuals included with diabetes tend to have
higher BMI values than those without diabetes. Both the median and
overall distribution of diabetes group shifted upward.

``` r
# Age Category (Numeric) by Diabetes Status
diabetes %>%
  ggplot(aes(x = diabetes_binary, y = age, fill = diabetes_binary)) +
  geom_boxplot(alpha = 0.85) +
  labs(
    title = "Age Category by Diabetes Status",
    x = "Diabetes Status",
    y = "Age Category (1 = 18–24 … 13 = 80+)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
```

![](p8105_final_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

The plot shows that the median age category is higher for the diabetes
group, and their overall distribution is shifted toward older age
ranges.

``` r
# General Health Score by Diabetes Status
diabetes %>%
  ggplot(aes(x = diabetes_binary, y = gen_hlth, fill = diabetes_binary)) +
  geom_boxplot(alpha = 0.85) +
  scale_fill_manual(values = c("#4C9F70", "#C94C4C")) +
  labs(
    title = "General Health Score by Diabetes Status",
    x = "Diabetes Status",
    y = "General Health (1 = Excellent, 5 = Poor)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
```

![](p8105_final_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

The plot shows that individuals with diabetes generally report worse
overall health than those without diabetes. The diabetes group has a
higher median health score, which indicates poorer health and a
distribution shifted toward less favorable health ratings.

- **Income distribution is skewed toward higher brackets, which may
  influence access to care.**

``` r
# Income Category Distribution
diabetes %>%
  ggplot(aes(x = income_cat)) +
  geom_bar(fill = "#FFA500", alpha = 0.8) +
  labs(
    title = "Income Level Distribution",
    x = "Income Category",
    y = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

![](p8105_final_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
# Correlation heat map
# Select numeric variables
numeric_vars <- diabetes %>%
  select(bmi, ment_hlth, phys_hlth, gen_hlth, income, age)

# Compute and tidy correlation matrix
cor_df <- numeric_vars %>%
  cor(use = "complete.obs") %>%
  as.data.frame() %>%
  rownames_to_column("var1") %>%
  pivot_longer(-var1, names_to = "var2", values_to = "cor")

# Heatmap with correlation values
cor_df %>%
  ggplot(aes(var1, var2, fill = cor)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "#4575b4",
    high = "#d73027",
    mid = "white",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  geom_text(aes(label = round(cor, 2)), size = 3) +
  labs(
    title = "Correlation Heatmap of Numeric Predictors",
    x = "",
    y = "",
    fill = "Correlation"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

![](p8105_final_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The heatmap shows that correlations among the numeric predictors are
generally weak to moderate, indicating low multicollinearity and
supporting their joint use in logistic regression models.

Variables related to overall health, such as general health and physical
or mental health days, are moderately correlated as expected,

While BMI and age show only weak relationships with other predictors.
Overall, the plot confirms that each variable contributes distinct
information for diabetes risk prediction.
