# 🏆 Enhanced Sports Recommendation API

A sophisticated intelligent recommendation system for sports academies that provides **personalized recommendations** based on user history, preferences, location, and budget analysis.

---

## 🌟 Features

- **Comprehensive User Analysis** — detailed analysis of user spending patterns, sports preferences, and historical behavior  
- **Smart Budget Inference** — automatically calculates an intelligent budget from purchase history with clear reasoning  
- **Dynamic Scoring System** — multi-factor scoring with adaptive thresholds and explainable outputs  
- **Location Intelligence** — distance-based recommendations with travel-pattern insights  
- **Sports Canonicalization** — maps sport name variants (e.g., "football" ↔ "soccer") for robust matching  
- **Skill Level Matching** — recommendations based on user's skill progression and academy fit  
- **Amenities Filtering** — preference-based amenity matching and parsing

---

## 📊 Scoring System Architecture

### Default Weights
```python
weights = {
    'amenities': 0.10,    # 10% - user preferred amenities match
    'distance': 0.50,     # 50% - geographic proximity
    'skill_level': 0.15,  # 15% - skill level compatibility
    'budget': 0.25        # 25% - price vs smart budget fit
}
Total Score = (Amenities_Score × 0.10 + Distance_Score × 0.50 + Skill_Score × 0.15 + Budget_Score × 0.25) × 100
```
Individual Scoring Components
1. Distance Scoring (50% weight)

Dynamic thresholds (based on population distribution of candidate distances):

Excellent (90–100%): ≤ 25th percentile OR ≤ 8 km

Good (70–89%): ≤ median OR ≤ 20 km

Acceptable (40–69%): ≤ 75th percentile OR ≤ 40 km

Poor (15–39%): ≤ 85th percentile OR ≤ 60 km

Very Poor (1–14%): exponential decay beyond poor threshold

Note: Percentile-based thresholds adapt to number and spread of available academies.

2. Budget Scoring (25% weight)

Based on the smart budget inferred from purchase history:

Perfect (100%): price ≤ 40% of smart budget

Excellent (90–99%): 40% < price ≤ 70% of smart budget

Good (80–89%): 70% < price ≤ 100% of smart budget

Acceptable (60%): 100% < price ≤ 110% of smart budget

Moderate (40%): 110% < price ≤ 125% of smart budget

Poor (20%): 125% < price ≤ 150% of smart budget

Very Poor (1–19%): exponential decay beyond 150%

3. Skill Level Scoring (15% weight)

Skill levels: Beginner (1) → Intermediate (2) → Advanced (3) → Professional (4)

Perfect Match (100%): exact skill-level match

Good Match (30%): adjacent level (±1)

Poor Match (10%): non-adjacent levels

4. Amenities Scoring (10% weight)

Perfect (100%): all requested amenities available

Partial: proportional to matched percentage

None (0%): no requested amenities found

No Preference (100%): when ["none"] is provided

Default (50%): when amenities parameter is omitted




