from Campaign_Engine import CampaignEngine

engine = CampaignEngine()

# Load cleaned dataset
engine.load_data("campaigns_clean.csv")

# Train all models
engine.train_models()

# Save trained models + metadata
engine.save_models("campaign_engine.joblib")

print("Training complete. Models saved.")
