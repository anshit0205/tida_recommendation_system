#  Sports Recommendation API

A sophisticated intelligent recommendation system for sports academies that provides **personalized recommendations** based on user history, preferences, location, and budget analysis.

---

##  Features

- **Comprehensive User Analysis** — detailed analysis of user spending patterns, sports preferences, and historical behavior  
- **Smart Budget Inference** — automatically calculates an intelligent budget from purchase history with clear reasoning  
- **Dynamic Scoring System** — multi-factor scoring with adaptive thresholds and explainable outputs  
- **Location Intelligence** — distance-based recommendations with travel-pattern insights  
- **Sports Canonicalization** — maps sport name variants (e.g., "football" ↔ "soccer") for robust matching  
- **Skill Level Matching** — recommendations based on user's skill progression and academy fit  
- **Amenities Filtering** — preference-based amenity matching and parsing

---

##  Scoring System Architecture

### Default Weights

```python
weights = {
    'amenities': 0.10,    # 10% - user preferred amenities match
    'distance': 0.50,     # 50% - geographic proximity
    'skill_level': 0.15,  # 15% - skill level compatibility
    'budget': 0.25        # 25% - price vs smart budget fit
}
```

### Total Score

```
Total Score = (Amenities_Score × 0.10 + Distance_Score × 0.50 + Skill_Score × 0.15 + Budget_Score × 0.25) × 100
```

### Individual Scoring Components

#### 1. Distance Scoring (50% weight)

Dynamic thresholds (based on population distribution of candidate distances):

- **Excellent (90–100%)**: ≤ 25th percentile OR ≤ 8 km
- **Good (70–89%)**: ≤ median OR ≤ 20 km  
- **Acceptable (40–69%)**: ≤ 75th percentile OR ≤ 40 km
- **Poor (15–39%)**: ≤ 85th percentile OR ≤ 60 km
- **Very Poor (1–14%)**: exponential decay beyond poor threshold

> **Note**: Percentile-based thresholds adapt to number and spread of available academies.

#### 2. Budget Scoring (25% weight)

Based on the smart budget inferred from purchase history:

- **Perfect (100%)**: price ≤ 40% of smart budget
- **Excellent (90–99%)**: 40% < price ≤ 70% of smart budget
- **Good (80–89%)**: 70% < price ≤ 100% of smart budget
- **Acceptable (60%)**: 100% < price ≤ 110% of smart budget
- **Moderate (40%)**: 110% < price ≤ 125% of smart budget
- **Poor (20%)**: 125% < price ≤ 150% of smart budget
- **Very Poor (1–19%)**: exponential decay beyond 150%

#### 3. Skill Level Scoring (15% weight)

Skill levels: **Beginner (1)** → **Intermediate (2)** → **Advanced (3)** → **Professional (4)**

- **Perfect Match (100%)**: exact skill-level match
- **Good Match (30%)**: adjacent level (±1)
- **Poor Match (10%)**: non-adjacent levels

#### 4. Amenities Scoring (10% weight)

- **Perfect (100%)**: all requested amenities available
- **Partial**: proportional to matched percentage
- **None (0%)**: no requested amenities found
- **No Preference (100%)**: when `["none"]` is provided
- **Default (50%)**: when amenities parameter is omitted

---

##  Smart Budget System

### Budget Inference Logic

1. **Base Calculation**: `calculated_budget = max_historical_price × 1.25`
2. **Variance Analysis**: adjust according to spending coefficient of variation (CV)
    CV= mean / std dev 
4. **Safety Bounds**:
   ```
   smart_budget = min(max_price × 2.0, max(avg_price × 1.1, calculated_budget))
   ```

### Budget Growth by CV

- **Consistent Spender (CV < 0.3)**: growth factor = 1.15 → `max_price × 1.15`
- **Moderate Variance (0.3 ≤ CV ≤ 0.7)**: growth factor = 1.25 → `max_price × 1.25`  
- **Experimental Spender (CV > 0.7)**: growth factor = 1.40 → `max_price × 1.40`

**Example Reasoning Text** (for responses):
> "Moderate variance (CV: 0.45) — standard 25% increase applied to maximum historical price."

---

##  API Endpoints

### `POST /recommendations`

**Request parameters** (JSON):

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | string |  Username or customer ID |
| `user_lat` | float | User latitude (-90 to 90) |
| `user_lon` | float |  User longitude (-180 to 180) |
| `preferred_amenities` | array | List of preferred amenities (e.g., `["parking","wifi"]`) |
| `skill_level` | string |  One of: `"Beginner"`,`"Intermediate"`,`"Advanced"`,`"Professional"` |
| `target_sports` | array | Filter for specific sports (e.g., `["football","tennis"]`) |

**Additional Notes**:
- **Amenities Parsing**: accepts separators `;` `,` `/` `|` `&` and (case-insensitive).
- **Sports Canonicalization**: maps common variants to canonical sport names (case-insensitive).

### `GET /health`
Returns simple health & data-load status.

### `GET /stats` 
Returns API usage statistics and available option lists.

### `GET /`
API overview & documentation pointer.

---

## 📋 Example Response

```json
{
  "user_analysis": {
    "customer_name": "John Doe",
    "sports_history": ["football", "cricket"],
    "purchase_count": 5,
    "budget_insights": {
      "smart_budget": 7500.0,
      "reasoning": "Moderate variance (CV: 0.45) - standard 25% increase",
      "historical_range": {
        "min": 3000.0,
        "max": 6000.0,
        "avg": 4500.0,
        "median": 4000.0
      },
      "spending_consistency": "Moderate",
      "budget_growth_factor": 1.25
    },
    "location_analysis": {
      "avg_distance": 15.2,
      "travel_pattern": "Regional",
      "preferred_areas": ["Academy A", "Academy B"]
    },
    "spending_pattern": "Consistent Spender"
  },
  "recommendations": [
    {
      "ID": 123,
      "academy_name": "Elite Sports Academy",
      "sports": ["football", "cricket"],
      "skill_level": "Intermediate",
      "price": 5000.0,
      "distance_km": 8.5,
      "total_score": 87.5,
      "amenities_score": 80.0,
      "distance_score": 92.0,
      "skill_score": 100.0,
      "budget_score": 90.0,
      "recommendation_reason": "Great location & skill_match",
      "budget_indicator": "Within Budget"
    }
  ],
  "summary_stats": {
    "total_recommendations": 10,
    "within_budget_count": 7,
    "excellent_matches": 3,
    "average_distance": 12.4
  }
}
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8+**
- Required data files in a `data/` directory:
  - `post.json`
  - `post_meta.json`
  - `order_addresses.json`
  - `order_product_lookup.json`
  - `customer_lookup.json`

### Quickstart

```bash
# Clone repo
git clone <repository-url>
cd sports-recommendation-api

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pandas numpy phpserialize pydantic

# Run local dev server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Access**:
- API root → http://localhost:8000/
- Swagger docs → http://localhost:8000/docs  
- Health check → http://localhost:8000/health

---

## 🔧 Customization

### Modify Scoring Weights

Edit the `weights` dict in `calculate_enhanced_scores`:

```python
weights = {
    'amenities': 0.10,
    'distance': 0.50,
    'skill_level': 0.15,
    'budget': 0.25
}
```

**Guidelines**: weights must sum to 1.0. Typical ranges:
- **Distance**: 0.4–0.6
- **Budget**: 0.2–0.3  
- **Amenities**: 0.05–0.20
- **Skill**: 0.1–0.25

### Modify Distance Thresholds

Update `get_dynamic_distance_scores` thresholds if you prefer fixed km breakpoints:

```python
 excellent_threshold = min(q25_dist, 8.0)      # Change 8.0 to your desired max km for excellent
 good_threshold = min(median_dist, 20.0)       # Change 20.0 to your desired max km for good
 acceptable_threshold = min(q75_dist, 40.0)    # Change 40.0 to your desired max km for acceptable
 poor_threshold = min(max_dist * 0.85, 60.0)   # Change 60.0 to your desired max km for poor
```

---

## 🔍 API Usage Examples

### open terminal and run 

```bash
python main.py
```
After that go to http://localhost:8000/docs

### Advanced Request

```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user123",
    "user_lat": 29.8543,
    "user_lon": 77.8880,
    "preferred_amenities": ["parking", "wifi", "swimming pool"],
    "skill_level": "Advanced",
    "target_sports": ["football", "tennis"]
  }'
```

---

## 📊 Data Requirements (Input Schema)

- **`post.json`** — academy / package information (IDs, titles, descriptions)
- **`post_meta.json`** — metadata (prices, coordinates, amenities, skill-level tags)  
- **`order_addresses.json`** — customer addresses and coordinates
- **`order_product_lookup.json`** — purchase history mapping (orders → products)
- **`customer_lookup.json`** — customer ID ↔ username mapping

> Ensure coordinates use decimal degrees and prices use a consistent currency/units.

---

## 🎮 Canonical Sports & Variants (case-insensitive)

The API recognizes these canonical sports and common variations (used for matching and filtering):

- **Football** — soccer, football
- **Cricket** — cricket, cricketing  
- **Basketball** — basketball
- **Swimming** — swimming, swim, water
- **Horse Riding** — horse riding, equestrian, horse-riding
- **Tennis** — lawn tennis, tennis, YMCA
- **Badminton** — badminton
- **Skating** — skating, ice skating, roller skating
- **Dance** — dance, dancing
- **Music** — music, vocal, instrument
- **Taekwondo** — taekwondo, tae kwon do, tkd
- **Acting** — acting, drama, theatre, theater
- **Languages** — language, spoken English, language classes
- **Public Speaking** — public speaking, communication, debate
- **Martial Arts** — martial arts, MMA, karate, judo, kung fu, aikido, kickboxing
- **Yoga** — yoga, meditation, pranayama
- **Padel** — padel, paddle tennis
- **Pickleball** — pickleball, pickle ball
- **Roller Hockey** — roller hockey, roller-hockey
- **Boxing** — boxing, kick boxing, kickboxing

---


[Add contact information or support channels here]
