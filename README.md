# Campaign Recommendation Engine

## Multi-Objective Optimization with Explainable ML

### Overview

This project develops an end-to-end campaign recommendation engine for digital advertising, leveraging machine learning, optimization, explainability, and multi-objective decision-making.

Instead of predicting a *single* outcome, the system:  
- Learns relationships between campaign configurations and multiple performance KPIs  
- Explores hypothetical campaign setups under business constraints
- Scores and ranks campaigns using learned models  
- Filters results using Pareto optimality to expose tradeoffs  
- Explains recommendations using SHAP (feature attribution)  

The engine's **purpose** answers the real-world question:  
***“Given my campaign goal and operational constraints, what campaign configurations should I consider — and why?”***

### What This System Does

#### Inputs 

These are parameters a stakeholder can control:  
- Campaign Goal (e.g., Increase Sales, Product Launch)  
- Channel Used (Instagram, Twitter, Facebook, etc.)  
- Target Audience  
- Language  
- Location  
- Customer Segment  
- Season  
- Campaign Duration (days)  
- Acquisition Cost (budget)

#### Outputs (Predicted KPIs)

For each campaign campaign, the engine predicts:  
- ROI  
- Conversion Rate  
- CTR  
- Engagement Score

It then:  
- Combines them into a weighted score (configurable)  
- Identifies Pareto (multi-objective) tradeoffs  
- Produces human-readable explanations for each recommendation

### Why is there a need for Machine Learning

Simple averages or dashboards can describe past performance, but they **cannot**:  
- Evaluate **unseen combinations** of campaign settings  
- Answer **“what-if”** questions  
- Balance multiple **competing KPIs**  
- **Adapt** recommendations to different goals and constraints  

This engine uses ML models to generalize from historical data and support decision-making instead of being limited to reporting.

### Project Structure
project/  
│  
├── campaign_engine.py        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Core engine (models, optimization, Pareto, SHAP)  
├── train_engine.py           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Train models ONCE and save them  
├── run_engine.py             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Run recommendations with goals + constraints  
├── campaigns_clean.csv       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Cleaned dataset  
├── campaign_engine.joblib    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Saved trained models  
└── README.md                 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# This file  

### How to Run the Project

1. **Train the Models** (run once)

    This step trains ML models and saves them to disk.

    `python train_engine.py`

    This creates:

    `campaign_engine.joblib`

    You only need to **re-run** this if:
    - The dataset changes
    - The model architecture changes

2. **Generate Recommendations** (run anytime)  
    `python run_engine.py`
   
    This:

    - loads the saved models  
    - generates campaign campaigns  
    - applies constraints  
    - ranks results  
    - filters via Pareto  
    - prints and saves recommendations
   
### Editing the Campaign Goal (Very Important)

In run_engine.py, you will see:  
`goal = "Increase Sales"`

You can change this to any valid campaign goal, for example:  

`goal = "Product Launch"`

`goal = "Brand Awareness"`

`goal = "Market Expansion"`

***No retraining is required when changing the goal.***

### Editing Constraints (Core Feature)

Constraints define what campaigns are allowed during optimization.

*Example: Default constraints*    
`constraints = {  "Channel_Used": ["Instagram", "Twitter"],  "Season": ["Winter"],  "Acquisition_Cost_Num": (2000, 9000), "Duration_Days": (15, 45),} `   

*How to edit constraints* (examples)

- Restrict to one channel   
`constraints = {"Channel_Used": ["Instagram"]}`  

- Restrict geography   
`constraints = {"Location": ["New York", "Los Angeles"]}.`  

- Restrict audience + language  
`constraints = {"Target_Audience": ["Women 25-34"], "Language": ["English"]}`

- Budget-only scenario  
`constraints = {"Acquisition_Cost_Num": (3000, 6000)}`

- No constraints  
`constraints = {}`  

***Constraints are runtime inputs — they do NOT require retraining.***

### How Optimization Works (High-Level)

1) *campaign Generation*  - Thousands of hypothetical campaigns are randomly sampled within constraints.  

2) *ML Scoring*  
Each campaign is scored using trained ML models for:
    - ROI  
    - Conversion Rate  
    - CTR  
    - Engagement

3) *Weighted Ranking* - Predictions are normalized and combined into a single score.

4) *Pareto Filtering* (Multi-Objective) - Campaigns that are strictly worse in all KPIs are removed.

5) *Explainability* - SHAP explains why top campaigns perform well or poorly.

### Why does Pareto Optimality matter  

A campaign is *Pareto-optimal* if no other campaign is better in all KPIs.

This:
- Eliminates objectively inferior options
- Preserves tradeoffs (e.g., higher ROI vs higher engagement)
- Prevents over-reliance on arbitrary weights

Pareto is applied *after scoring*, as a validation and decision-support layer.

### Example Output (What You’ll See)

Each recommendation includes:
- Campaign configuration  
- Predicted KPIs  
- Final score  
- Explanation text (from SHAP)
  
Example explanation:  
- Higher budget strongly improves ROI  
- Avoiding Pinterest increases ROI   
- Instagram boosts expected ROI   
- Winter slightly reduces performance  
