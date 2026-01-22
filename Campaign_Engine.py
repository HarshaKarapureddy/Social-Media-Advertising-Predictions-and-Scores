# Project Name: Campaign Optimization Engine
# Author: Harsha Karapureddy

# IMPORTS
import pandas as pd
import numpy as np
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import shap


# CAMPAIGN ENGINE CLASS
class CampaignEngine:

    def __init__(self):
        # Data + schema
        self.clean_df = None
        self.feature_cols = None
        self.targets = ["ROI", "Conversion_Rate", "CTR", "Engagement_Score"]

        # Encoded training matrix + columns
        self.X_encoded = None
        self.X_encoded_columns = None

        # Models + stats
        self.models = {}
        self.target_means = None
        self.target_stds = None

        # Domain values for candidate generation
        self.domain = None

        # SHAP explainer for ROI
        self.explainer_roi = None

    # Load data
    def load_data(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df["CTR"] = df["Clicks"] / df["Impressions"]
        self.clean_df = df
        return df

    # Train models
    def train_models(self, test_size=0.2, random_state=42):
        if self.clean_df is None:
            raise ValueError("Run load_data() first.")

        self.feature_cols = [
            "Campaign_Goal",
            "Channel_Used",
            "Target_Audience",
            "Language",
            "Location",
            "Customer_Segment",
            "Season",
            "Duration_Days",
            "Acquisition_Cost_Num"
        ]

        X = self.clean_df[self.feature_cols]
        Y = self.clean_df[self.targets]

        # one-hot encode 
        self.X_encoded = pd.get_dummies(
            X,
            columns=[
                "Campaign_Goal",
                "Channel_Used",
                "Target_Audience",
                "Language",
                "Location",
                "Customer_Segment",
                "Season"
            ],
            drop_first=True
        )

        self.X_encoded_columns = self.X_encoded.columns.tolist()

        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X_encoded, Y, test_size=test_size, random_state=random_state
        )

        # train one model per target
        self.models = {}
        for t in self.targets:
            model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state
            )
            model.fit(X_train, Y_train[t])
            self.models[t] = model

        # report MAE
        for t in self.targets:
            preds = self.models[t].predict(X_test)
            mae = mean_absolute_error(Y_test[t], preds)
            print(f"{t} MAE:", round(mae, 4))

        # store train stats for z-scoring
        self.target_means = Y_train.mean()
        self.target_stds = Y_train.std().replace(0, 1e-9)

        # domain values for sampling candidates
        self.domain = {
            "Campaign_Goal": sorted(self.clean_df["Campaign_Goal"].unique()),
            "Channel_Used": sorted(self.clean_df["Channel_Used"].unique()),
            "Target_Audience": sorted(self.clean_df["Target_Audience"].unique()),
            "Language": sorted(self.clean_df["Language"].unique()),
            "Location": sorted(self.clean_df["Location"].unique()),
            "Customer_Segment": sorted(self.clean_df["Customer_Segment"].unique()),
            "Season": sorted(self.clean_df["Season"].unique()),
        }

        # SHAP explainer for ROI
        self.explainer_roi = shap.TreeExplainer(self.models["ROI"])

    # Save / Load 
    def save_models(self, path: str = "campaign_engine.joblib"):
        payload = {
            "feature_cols": self.feature_cols,
            "targets": self.targets,
            "X_encoded_columns": self.X_encoded_columns,
            "models": self.models,
            "target_means": self.target_means,
            "target_stds": self.target_stds,
            "domain": self.domain,
        }
        joblib.dump(payload, path)

    def load_models(self, path: str = "campaign_engine.joblib"):
        payload = joblib.load(path)

        self.feature_cols = payload["feature_cols"]
        self.targets = payload["targets"]
        self.X_encoded_columns = payload["X_encoded_columns"]
        self.models = payload["models"]
        self.target_means = payload["target_means"]
        self.target_stds = payload["target_stds"]
        self.domain = payload["domain"]

        # rebuild SHAP explainer after load
        self.explainer_roi = shap.TreeExplainer(self.models["ROI"])

    # Converts candidate dict -> one-hot encoded row
    def encode_candidate(self, candidate: dict) -> pd.DataFrame:

        if self.X_encoded_columns is None:
            raise ValueError("Train the models first (train_models()).")

        df_row = pd.DataFrame([candidate])
        df_row_encoded = pd.get_dummies(df_row)
        df_row_encoded = df_row_encoded.reindex(columns=self.X_encoded_columns, fill_value=0)
        return df_row_encoded

    # Predict targets, z-score them, compute weighted score
    def score_candidate(self, candidate: dict, weights: dict) -> dict:

        if not self.models:
            raise ValueError("Train the models first (train_models())")

        X_row = self.encode_candidate(candidate)

        preds = {t: float(self.models[t].predict(X_row)[0]) for t in self.models}
        z = {t: (preds[t] - float(self.target_means[t])) / float(self.target_stds[t]) for t in preds}

        total_score = sum(weights[t] * z[t] for t in weights)

        out = candidate.copy()
        out.update({f"pred_{t}": preds[t] for t in preds})
        out["score"] = float(total_score)
        return out

    # Randomly sample candidates, respecting constraints
    def generate_candidates(self, n: int, goal: str, constraints: dict | None = None, seed: int = 42) -> list[dict]:

        if self.domain is None:
            raise ValueError("Train the models first (train_models()) so domain is set.")

        rng = random.Random(seed)
        constraints = constraints or {}

        def pick(cat_name):
            options = constraints.get(cat_name, self.domain[cat_name])
            return rng.choice(list(options))

        dur_min, dur_max = constraints.get("Duration_Days", (15, 60))
        cost_min, cost_max = constraints.get("Acquisition_Cost_Num", (500, 15000))

        candidates = []
        for _ in range(n):
            candidates.append({
                "Campaign_Goal": goal,
                "Channel_Used": pick("Channel_Used"),
                "Target_Audience": pick("Target_Audience"),
                "Language": pick("Language"),
                "Location": pick("Location"),
                "Customer_Segment": pick("Customer_Segment"),
                "Season": pick("Season"),
                "Duration_Days": rng.randint(int(dur_min), int(dur_max)),
                "Acquisition_Cost_Num": rng.uniform(float(cost_min), float(cost_max)),
            })

        return candidates

    # Optmizing sampled candidates
    def optimize_campaign(self,
                          goal: str,
                          n_candidates: int = 5000,
                          top_k: int = 10,
                          constraints: dict | None = None,
                          weights: dict | None = None,
                          seed: int = 42) -> pd.DataFrame:

        if weights is None:
            weights = {
                "ROI": 0.45,
                "Conversion_Rate": 0.25,
                "CTR": 0.15,
                "Engagement_Score": 0.15
            }

        cands = self.generate_candidates(n_candidates, goal, constraints=constraints, seed=seed)

        scored = [self.score_candidate(c, weights=weights) for c in cands]

        df_scored = pd.DataFrame(scored).sort_values("score", ascending=False).head(top_k)
        return df_scored

    # SHAP explanation for ROI.
    def explain_roi(self, candidate: dict, top_n: int = 5) -> dict:

        if self.explainer_roi is None:
            raise ValueError("Train the models first (train_models()) so SHAP explainer exists.")

        X_row = self.encode_candidate(candidate)
        exp = self.explainer_roi(X_row)
        e0 = exp[0]

        values = np.array(e0.values)
        names = np.array(e0.feature_names)

        order = np.argsort(np.abs(values))[::-1]

        top_features = []
        for idx in order[:top_n]:
            top_features.append((names[idx], float(values[idx])))

        pos = [(f, v) for f, v in top_features if v > 0]
        neg = [(f, v) for f, v in top_features if v < 0]

        return {
            "pred_roi": float(e0.base_values + values.sum()),
            "top_pos": pos,
            "top_neg": neg
        }

    def pretty_reason(self, feature, value):

        feature = str(feature)

        if feature.startswith("Channel_Used_Pinterest") and value > 0:
            return "Avoiding Pinterest increases ROI"
        if feature.startswith("Channel_Used_Instagram") and value > 0:
            return "Instagram boosts expected ROI"
        if feature == "Acquisition_Cost_Num":
            return "Higher budget strongly improves ROI"
        if feature.startswith("Season_Winter") and value < 0:
            return "Winter slightly reduces performance"
        if feature.startswith("Language_Spanish") and value < 0:
            return "Not using Spanish slightly lowers ROI"
        if feature.startswith("Language_French") and value < 0:
            return "Not using French slightly lowers ROI"
        if feature.startswith("Customer_Segment"):
            return "Customer segment has a small positive effect"

        return f"{feature} contributes {value:+.2f} to ROI"

    # build candidate dict from a recommendation row, then return reasons.
    def explain_row(self, row: pd.Series) -> str:

        cand = {
            "Campaign_Goal": row["Campaign_Goal"],
            "Channel_Used": row["Channel_Used"],
            "Target_Audience": row["Target_Audience"],
            "Language": row["Language"],
            "Location": row["Location"],
            "Customer_Segment": row["Customer_Segment"],
            "Season": row["Season"],
            "Duration_Days": row["Duration_Days"],
            "Acquisition_Cost_Num": row["Acquisition_Cost_Num"]
        }

        exp = self.explain_roi(cand, top_n=5)

        reasons = []
        for f, v in exp["top_pos"][:3]:
            reasons.append(self.pretty_reason(f, v))
        for f, v in exp["top_neg"][:2]:
            reasons.append(self.pretty_reason(f, v))

        return "; ".join(reasons)

    # returns Pareto non-dominated rows (higher is better for all).
    def pareto_front(self, df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        vals = df[metrics].to_numpy()
        n = vals.shape[0]
        is_efficient = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_efficient[i]:
                continue

            dominates = np.all(vals >= vals[i], axis=1) & np.any(vals > vals[i], axis=1)
            if np.any(dominates):
                is_efficient[i] = False

        return df[is_efficient].copy()
