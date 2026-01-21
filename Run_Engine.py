from Campaign_Engine import CampaignEngine

# Create engine
engine = CampaignEngine()

# Load trained models (NO retraining)
engine.load_models("campaign_engine.joblib")

# Business constraints
constraints = {
    "Channel_Used": ["Instagram", "Twitter"],
    "Acquisition_Cost_Num": (2000, 9000),
    "Duration_Days": (15, 45),
    "Season": ["Winter"],
}

# Optimize campaigns
recs = engine.optimize_campaign(
    goal="Increase Sales",
    n_candidates=8000,
    top_k=10,
    constraints=constraints,
    seed=2
)

# Add explanations
recs = recs.copy()
recs["why"] = recs.apply(engine.explain_row, axis=1)

print(recs)

# Optional: save results
recs.to_csv("recommendations.csv", index=False)
