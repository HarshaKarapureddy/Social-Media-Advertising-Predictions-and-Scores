from Campaign_Engine import CampaignEngine

def main():

    # 1) Load trained engine 
    engine = CampaignEngine()
    engine.load_models("campaign_engine.joblib")


    # 2) Constraints (edit these any time)
    constraints = {
        "Channel_Used": ["Instagram", "Twitter"],
        "Acquisition_Cost_Num": (2000, 9000),
        "Duration_Days": (15, 45),
        "Season": ["Winter"],
        # You can also constrain these if you want:
        # "Customer_Segment": ["Health", "Technology"],
        # "Language": ["English"],
        # "Location": ["Los Angeles", "New York"],
        # "Target_Audience": ["Women 25-34"]
        # Look at read me for more options
    }

    # 3) Score weights 
    weights = {
        "ROI": 0.45,
        "Conversion_Rate": 0.25,
        "CTR": 0.15,
        "Engagement_Score": 0.15
    }

    # goal of campaign, check readme for options for changing
    goal = "Increase Sales" 

    # 4) Generate a LARGE pool so Pareto is meaningful
    n_candidates = 20000
    seed = 42

    df_all = engine.optimize_campaign(
        goal=goal,
        n_candidates=n_candidates,
        top_k=n_candidates,          # keep all scored candidates
        constraints=constraints,
        weights=weights,
        seed=seed
    )

    # Save full scored pool
    df_all.to_csv("all_candidates_scored.csv", index=False)

    # 5) Pareto front (multi-objective filter)
    pareto_metrics = [
        "pred_ROI",
        "pred_Conversion_Rate",
        "pred_CTR",
        "pred_Engagement_Score"
    ]

    pareto_df = engine.pareto_front(df_all, pareto_metrics)

    # 6) Choose the final "top" Pareto items
    #    Pareto gives tradeoffs; we still need to rank for a final list.
    #    Script will rank by your score for consistency.
    top_k = 10
    pareto_top = pareto_df.sort_values("score", ascending=False).head(top_k).copy()

    # Add SHAP explanation strings
    pareto_top["why"] = pareto_top.apply(engine.explain_row, axis=1)

    # Save Pareto results
    pareto_df.to_csv("pareto_front.csv", index=False)
    pareto_top.to_csv("pareto_top10_with_explanations.csv", index=False)

    # 7) Print summary
    print("\n==============================")
    print("Campaign Recommendation Engine")
    print("==============================")
    print(f"Goal: {goal}")
    print(f"Candidates generated: {n_candidates}")
    print(f"Pareto front size: {len(pareto_df)}")
    print(f"Top Pareto results (ranked by score): {top_k}")
    print("==============================\n")

    cols_to_show = [
        "Campaign_Goal",
        "Channel_Used",
        "Target_Audience",
        "Language",
        "Location",
        "Customer_Segment",
        "Season",
        "Duration_Days",
        "Acquisition_Cost_Num",
        "pred_ROI",
        "pred_Conversion_Rate",
        "pred_CTR",
        "pred_Engagement_Score",
        "score",
        "why"
    ]

    print("\nSaved files:")
    print("- all_candidates_scored.csv")
    print("- pareto_front.csv")
    print("- pareto_top10_with_explanations.csv")


if __name__ == "__main__":
    main()
