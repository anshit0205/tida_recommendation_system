from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any
from enum import Enum
import pandas as pd
import numpy as np
import phpserialize
import re
from math import radians, sin, cos, sqrt, atan2
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class SkillLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    PROFESSIONAL = "Professional"

class UserPreferencesRequest(BaseModel):
    username: str = Field(..., min_length=1, description="Username or customer ID")
    user_lat: float = Field(..., ge=-90, le=90, description="User's latitude")
    user_lon: float = Field(..., ge=-180, le=180, description="User's longitude")
    preferred_amenities: Optional[List[str]] = Field(default_factory=list, description="Preferred amenities (use ['none'] for no preferences)")
    skill_level: SkillLevel = Field(..., description="User's skill level")
    target_sports: Optional[List[str]] = Field(default=None, description="Specific sports to filter by (optional)")
    
    @validator('preferred_amenities', pre=True)
    def validate_amenities(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [s.strip() for s in re.split(r'[;,\/\|\&]|\band\b', v, flags=re.IGNORECASE) if s.strip()]
            return v
        return v

    @validator('target_sports', pre=True)
    def validate_sports(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = [s.strip() for s in re.split(r'[;,\/\|\&]|\band\b', v, flags=re.IGNORECASE) if s.strip()]
            return v if v else None
        return v

    @validator('skill_level', pre=True)
    def _skill_level_case_insensitive(cls, v):
        if isinstance(v, str):
            s = v.strip().lower()
            mapping = {
                'beginner':'Beginner',
                'intermediate':'Intermediate',
                'advanced':'Advanced',
                'professional':'Professional'
            }
            if s in mapping:
                return mapping[s]
        return v

class RecommendationResponse(BaseModel):
    ID: int
    academy_name: str
    sports: Union[str, List[str]]
    skill_level: str
    price: float
    address: str
    amenities: Union[str, List[str], None]
    latitude: Optional[float]
    longitude: Optional[float]
    distance_km: Optional[float]
    smart_budget: float
    total_score: float
    amenities_score: float
    distance_score: float
    skill_score: float
    budget_score: float
    recommendation_reason: str
    budget_indicator: str

class UserAnalysisResponse(BaseModel):
    customer_name: Optional[str]
    sports_history: List[str]
    purchase_count: int
    budget_insights: Dict[str, Any]
    skill_progression: List[str]
    location_analysis: Dict[str, Any]
    spending_pattern: str

class RecommendationResult(BaseModel):
    user_analysis: UserAnalysisResponse
    recommendations: List[RecommendationResponse]
    summary_stats: Dict[str, Any]
    message: str

# FastAPI App
app = FastAPI(
    title="Enhanced Sports Recommendation API",
    version="2.0.0",
    description="Intelligent Sports Academy Recommendation System with Comprehensive User Analysis"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data
df = None
lookup_with_name = None

# Sport canonicalization (same as original)
SPORT_CANON = {
    "football": ["football", "soccer"],
    "cricket": ["cricket", "cricketing"],
    "basketball": ["basketball"],
    "swimming": ["swimming", "swim", "Water"],
    "horse riding": ["horse riding", "horseriding", "equestrian", "horse-riding"],
    "lawn tennis": ["lawn tennis", "tennis", "YMCA"],
    "badminton": ["badminton"],
    "skating": ["skating", "ice skating", "roller skating"],
    "dance": ["dance", "dancing"],
    "music": ["music", "vocal", "instrument"],
    "taekwondo": ["taekwondo", "tae kwon do", "tkd"],
    "acting": ["acting", "drama", "theatre", "theater"],
    "languages": ["language", "Languages &amp", "languages", "spoken english", "english speaking", "language classes"],
    "public speaking": ["public speaking", "communication", "communication skills", "debate", "elocution"],
    "martial arts": ["Martial Arts","MartialArts", "martial arts", "mma", "mixed martial arts", "karate", "judo",
        "kung fu", "kung-fu", "aikido", "kickboxing", "muay thai", "muay-thai", "self defense", "self-defence", "self defence", "kungfu"],
    "yoga": ["yoga", "meditation", "pranayama"],
    "padel": ["padel", "paddle tennis", "padel tennis"],
    "pickleball": ["pickleball","pickle ball","pickle-ball"],
    "roller hockey": ["roller hockey","RollerHoCkey" ,"roller-hockey"],
    "boxing": ["boxing", "Boxing","box", "kick boxing", "kick-boxing", "kickboxing"]
}

# Create reverse mapping
syn_to_canon = {}
for canon, syns in SPORT_CANON.items():
    for s in syns:
        syn_to_canon[s.lower()] = canon

# Helper Functions (same as original)
def split_raw_parts(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    parts = re.split(r'[;,\/\|\&]|\band\b', text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts

def split_and_clean(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    parts = re.split(r'[;,\/\|\&]|\band\b', text, flags=re.IGNORECASE)
    parts = [p.strip().lower() for p in parts if p and p.strip()]
    return parts

def map_to_canonical(token: str) -> Optional[str]:
    if not isinstance(token, str):
        return None
    t = token.lower().strip()
    if t in syn_to_canon:
        return syn_to_canon[t]
    for syn, canon in syn_to_canon.items():
        if re.search(r'\b' + re.escape(syn) + r'\b', t):
            return canon
    for canon in SPORT_CANON.keys():
        if re.search(r'\b' + re.escape(canon) + r'\b', t):
            return canon
    return None

def parse_available_field_keep_unknowns(val) -> List[str]:
    result = []
    seen_lower = set()

    if pd.isna(val):
        return result

    if isinstance(val, (list, tuple, set)):
        raw_tokens = [str(x).strip() for x in val if x is not None and str(x).strip()]
        for tok in raw_tokens:
            canon = map_to_canonical(tok)
            if canon:
                kl = canon.lower()
                if kl not in seen_lower:
                    result.append(canon)
                    seen_lower.add(kl)
            else:
                norm = tok.strip()
                if norm and norm.lower() not in seen_lower:
                    result.append(norm)
                    seen_lower.add(norm.lower())
        return result

    raw_parts = split_raw_parts(str(val))
    clean_parts = split_and_clean(str(val))
    for raw, clean in zip(raw_parts, clean_parts):
        if not clean:
            continue
        canon = map_to_canonical(clean)
        if canon:
            kl = canon.lower()
            if kl not in seen_lower:
                result.append(canon)
                seen_lower.add(kl)
        else:
            norm = raw.strip()
            if norm and norm.lower() not in seen_lower:
                result.append(norm)
                seen_lower.add(norm.lower())

    return result

def search_post_title_for_sports(title: str) -> List[str]:
    found = []
    if not isinstance(title, str):
        return found
    txt = title.lower()
    for syn, canon in syn_to_canon.items():
        if re.search(r'\b' + re.escape(syn) + r'\b', txt) and canon not in found:
            found.append(canon)
    for canon in SPORT_CANON.keys():
        if re.search(r'\b' + re.escape(canon) + r'\b', txt) and canon not in found:
            found.append(canon)
    return found

def unserialize_php(value):
    if pd.isna(value):
        return None
    try:
        parsed = phpserialize.loads(value.encode("utf-8"), decode_strings=True)
        return list(parsed.values())
    except Exception:
        return None

def create_sports_column(merged_df: pd.DataFrame) -> pd.DataFrame:
    df_local = merged_df.copy()
    
    # Drop trash and ignore posts
    mask_trash = df_local['post_status'].astype(str).str.lower().eq('trash')
    if mask_trash.any():
        df_local = df_local.loc[~mask_trash].reset_index(drop=True)
        
    mask_ignore = df_local['post_title'].astype(str).str.contains(r'please\s+ignore', case=False, na=False)
    if mask_ignore.any():
        df_local = df_local.loc[~mask_ignore].reset_index(drop=True)

    df_local['sports'] = pd.NA

    # Parse available_sports_in_this_academy
    for i in df_local.index:
        val = df_local.at[i, 'available_sports_in_this_academy']
        parsed = parse_available_field_keep_unknowns(val)
        if parsed:
            df_local.at[i, 'sports'] = parsed[0] if len(parsed) == 1 else parsed

    # Search post_title for sports
    missing_idx = df_local[df_local['sports'].isna()].index
    for i in missing_idx:
        title = df_local.at[i, 'post_title']
        found = search_post_title_for_sports(title)
        if found:
            df_local.at[i, 'sports'] = found[0] if len(found) == 1 else found

    return df_local

def clean_data(df_in: pd.DataFrame) -> pd.DataFrame:
    df_local = df_in.copy()
    # Remove drafts, testing posts, etc.
    df_local = df_local[df_local['post_status'] != 'draft'].copy()
    df_local = df_local[df_local['post_status'] != 'private'].copy()
    
    df_local = df_local[
        ~(
            (df_local['post_title'] == "AUTO-DRAFT") |
            (df_local['post_title'].str.contains("testing", case=False, na=False)) |
            (df_local['post_excerpt'].str.contains("testing", case=False, na=False))
        )
    ]
    
    # Handle specific cases
    df_local.loc[
        df_local['post_title'].str.contains("Bala Ji Skaters", case=False, na=False) & df_local['sports'].isna(),
        'sports'
    ] = "skating"
    
    mask = df_local['post_name'].str.contains('skate', case=False, na=False)
    empty_sports = df_local['sports'].isna() | (df_local['sports'].astype(str).str.strip() == '')
    df_local.loc[mask & empty_sports, 'sports'] = 'skating'
    
    # YMCA special handling
    mask_ymca = df_local['post_title'].str.strip().eq("YMCA")
    if mask_ymca.any():
        vals = pd.Series([['football', 'cricket']] * mask_ymca.sum(), index=df_local.loc[mask_ymca].index)
        df_local.loc[mask_ymca, 'sports'] = vals
    
    # Drop rows without sports
    df_local = df_local.dropna(subset=['sports'])
    
    # Clean price and coordinates
    df_local['_price'] = pd.to_numeric(df_local['_price'], errors='coerce')
    df_local['latitude'] = pd.to_numeric(df_local['latitude'], errors='coerce')
    df_local['longitude'] = pd.to_numeric(df_local['longitude'], errors='coerce')
    
    # Process amenities
    df_local['amenities'] = df_local['amenities'].apply(unserialize_php)
    
    return df_local

def calculate_distance(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float('inf')
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def get_customer_history(username: str):
    global lookup_with_name, df
    
    if lookup_with_name is None or lookup_with_name.empty:
        return pd.DataFrame(), None
    
    try:
        customer_orders = lookup_with_name[
            (lookup_with_name['customer_id'].astype(str) == str(username)) |
            (lookup_with_name['name'].str.contains(str(username), case=False, na=False))
        ].copy()
    except Exception:
        return pd.DataFrame(), None
    
    if customer_orders.empty:
        return pd.DataFrame(), None
    
    customer_name = customer_orders['name'].iloc[0]
    
    history = []
    for _, order in customer_orders.iterrows():
        try:
            if order['variation_id'] == 0:
                package = df[df['ID'] == order['product_id']]
            else:
                package = df[df['post_parent'] == order['variation_id']]
            
            if not package.empty:
                pkg_data = package.iloc[0]
                revenue = order.get('product_net_revenue', order.get('price', pkg_data.get('_price', 0)))
                
                history.append({
                    'product_id': order['product_id'],
                    'variation_id': order['variation_id'],
                    'academy_name': pkg_data['post_title'],
                    'sports': pkg_data['sports'],
                    'latitude': pkg_data['latitude'],
                    'longitude': pkg_data['longitude'],
                    'price': revenue,
                    'skill_level': pkg_data['skill_level'],
                    'amenities': pkg_data.get('amenities', ''),
                    'address': pkg_data.get('address', '')
                })
        except Exception:
            continue
    
    return pd.DataFrame(history), customer_name

def infer_smart_budget(customer_history: pd.DataFrame):
    if customer_history.empty:
        return 5000, "Default (no history)"
    
    prices = customer_history['price'].dropna()
    if prices.empty:
        return 5000, "Default (no price data)"
    
    max_price = prices.max()
    avg_price = prices.mean()
    std_price = prices.std() if len(prices) > 1 else 0
    
    base_budget = max_price * 1.25
    
    if std_price > 0:
        cv = std_price / avg_price if avg_price != 0 else 0
        if cv < 0.3:
            smart_budget = max_price * 1.15
            reasoning = f"Consistent spender (CV: {cv:.2f}) - modest 15% increase"
        elif cv > 0.7:
            smart_budget = max_price * 1.4
            reasoning = f"Experimental spender (CV: {cv:.2f}) - generous 40% increase"
        else:
            smart_budget = base_budget
            reasoning = f"Moderate variance (CV: {cv:.2f}) - standard 25% increase"
    else:
        smart_budget = base_budget
        reasoning = "Single purchase history - standard 25% increase"
    
    smart_budget = max(smart_budget, avg_price * 1.1)
    smart_budget = min(smart_budget, max_price * 2.0)
    
    return smart_budget, reasoning

def analyze_user_comprehensive(customer_history: pd.DataFrame, user_prefs: dict, customer_name: str):
    """Enhanced user analysis with comprehensive insights"""
    
    if customer_history.empty:
        return {
            'customer_name': customer_name,
            'sports_history': [],
            'purchase_count': 0,
            'budget_insights': {'smart_budget': 5000, 'reasoning': 'No history available'},
            'skill_progression': [],
            'location_analysis': {'avg_distance': None, 'preferred_areas': []},
            'spending_pattern': 'Unknown'
        }
    
    # Sports analysis
    sports_list = []
    for sport in customer_history['sports'].dropna():
        if isinstance(sport, list):
            sports_list.extend([s.lower() for s in sport])
        else:
            sports_list.append(str(sport).lower())
    
    unique_sports = list(set(sports_list))
    sport_frequency = {}
    for sport in sports_list:
        sport_frequency[sport] = sport_frequency.get(sport, 0) + 1
    
    # Budget analysis with detailed insights
    prices = customer_history['price'].dropna()
    smart_budget, budget_reasoning = infer_smart_budget(customer_history)
    
    if not prices.empty:
        budget_insights = {
            'smart_budget': float(smart_budget),
            'reasoning': budget_reasoning,
            'historical_range': {
                'min': float(prices.min()),
                'max': float(prices.max()),
                'avg': float(prices.mean()),
                'median': float(prices.median())
            },
            'total_spent': float(prices.sum()),
            'spending_consistency': 'High' if prices.std() / prices.mean() < 0.3 else 'Low' if prices.std() / prices.mean() > 0.7 else 'Moderate',
            'price_trend': 'Increasing' if len(prices) > 1 and prices.iloc[-1] > prices.iloc[0] else 'Stable',
            'budget_growth_factor': float(smart_budget / prices.max() if prices.max() > 0 else 1.25)
        }
    else:
        budget_insights = {
            'smart_budget': 5000,
            'reasoning': 'No price data available',
            'historical_range': {'min': 0, 'max': 0, 'avg': 0, 'median': 0},
            'total_spent': 0,
            'spending_consistency': 'Unknown',
            'price_trend': 'Unknown',
            'budget_growth_factor': 1.25
        }
    
    # Location analysis
    user_lat, user_lon = user_prefs['user_lat'], user_prefs['user_lon']
    distances = []
    academy_locations = []
    
    for _, purchase in customer_history.iterrows():
        if not pd.isna(purchase['latitude']) and not pd.isna(purchase['longitude']):
            distance = calculate_distance(user_lat, user_lon, purchase['latitude'], purchase['longitude'])
            if distance != float('inf'):
                distances.append(distance)
                academy_locations.append({
                    'academy': purchase['academy_name'],
                    'distance': distance,
                    'address': purchase.get('address', 'N/A')
                })
    
    location_analysis = {
        'avg_distance': float(np.mean(distances)) if distances else None,
        'max_distance': float(max(distances)) if distances else None,
        'min_distance': float(min(distances)) if distances else None,
        'preferred_areas': [loc['academy'] for loc in sorted(academy_locations, key=lambda x: x['distance'])[:3]],
        'travel_pattern': 'Local' if distances and max(distances) < 10 else 'Regional' if distances and max(distances) < 30 else 'Wide-ranging'
    }
    
    # Skill progression
    skill_levels = customer_history['skill_level'].dropna().tolist()
    
    # Spending pattern classification
    if len(prices) >= 3:
        recent_avg = prices.tail(3).mean()
        overall_avg = prices.mean()
        if recent_avg > overall_avg * 1.2:
            spending_pattern = "Increasing Spender"
        elif recent_avg < overall_avg * 0.8:
            spending_pattern = "Decreasing Spender"
        else:
            spending_pattern = "Consistent Spender"
    else:
        spending_pattern = "New Customer" if len(prices) <= 1 else "Limited History"
    
    return {
        'customer_name': customer_name,
        'sports_history': unique_sports,
        'purchase_count': len(customer_history),
        'budget_insights': budget_insights,
        'skill_progression': skill_levels,
        'location_analysis': location_analysis,
        'spending_pattern': spending_pattern,
        'sport_preferences': dict(sorted(sport_frequency.items(), key=lambda x: x[1], reverse=True))
    }

def get_dynamic_distance_scores(distances: List[float]) -> List[float]:
    valid_distances = [d for d in distances if d != float('inf')]
    
    if not valid_distances:
        return [0.0] * len(distances)
    
    distances_array = np.array(valid_distances)
    
    # Calculate thresholds
    min_dist = np.min(distances_array)
    q25_dist = np.percentile(distances_array, 25)
    median_dist = np.median(distances_array)
    q75_dist = np.percentile(distances_array, 75)
    max_dist = np.max(distances_array)
    
    # Define thresholds
    excellent_threshold = min(q25_dist, 8.0)
    good_threshold = min(median_dist, 20.0)
    acceptable_threshold = min(q75_dist, 40.0)
    poor_threshold = min(max_dist * 0.85, 60.0)
    
    scores = []
    for distance in distances:
        if distance == float('inf'):
            scores.append(0.0)
        elif distance <= excellent_threshold:
            if excellent_threshold > 0:
                score = 1.0 - (distance / excellent_threshold) * 0.1
                scores.append(max(score, 0.90))
            else:
                scores.append(1.0)
        elif distance <= good_threshold:
            if good_threshold > excellent_threshold:
                progress = (distance - excellent_threshold) / (good_threshold - excellent_threshold)
                score = 0.90 - (progress * 0.19)
                scores.append(max(score, 0.70))
            else:
                scores.append(0.70)
        elif distance <= acceptable_threshold:
            if acceptable_threshold > good_threshold:
                progress = (distance - good_threshold) / (acceptable_threshold - good_threshold)
                score = 0.70 - (progress * 0.29)
                scores.append(max(score, 0.40))
            else:
                scores.append(0.40)
        elif distance <= poor_threshold:
            if poor_threshold > acceptable_threshold:
                progress = (distance - acceptable_threshold) / (poor_threshold - acceptable_threshold)
                score = 0.40 - (progress * 0.24)
                scores.append(max(score, 0.15))
            else:
                scores.append(0.15)
        else:
            excess_distance = distance - poor_threshold
            decay_factor = excess_distance / (poor_threshold * 0.5) if poor_threshold > 0 else excess_distance
            score = 0.15 * np.exp(-decay_factor * 0.8)
            scores.append(max(score, 0.01))
    
    return scores

def get_dynamic_budget_scores(prices: List[float], smart_budget: float) -> List[float]:
    scores = []
    for price in prices:
        if pd.isna(price):
            scores.append(0.0)
            continue
        
        if price <= smart_budget * 0.4:
            scores.append(1.0)
        elif price <= smart_budget * 0.7:
            ratio = (price - smart_budget * 0.4) / (smart_budget * 0.3)
            scores.append(1.0 - ratio * 0.1)
        elif price <= smart_budget:
            ratio = (price - smart_budget * 0.7) / (smart_budget * 0.3)
            scores.append(0.9 - ratio * 0.1)
        elif price <= smart_budget * 1.1:
            scores.append(0.6)
        elif price <= smart_budget * 1.25:
            scores.append(0.4)
        elif price <= smart_budget * 1.5:
            scores.append(0.2)
        else:
            over_ratio = (price - smart_budget * 1.5) / smart_budget
            score = 0.2 * np.exp(-over_ratio)
            scores.append(max(score, 0.01))
    
    return scores

def calculate_amenities_score(package, user_prefs: dict) -> float:
    package_amenities = package.get('amenities', []) or []
    if isinstance(package_amenities, str):
        package_amenities = [package_amenities]
    elif package_amenities is None:
        package_amenities = []
    
    user_amenities = user_prefs.get('preferred_amenities', [])
    
    if user_amenities == ['none']:
        return 1.0
    
    if not user_amenities:
        return 0.5
    
    package_amenities_lower = [str(a).lower() for a in package_amenities]
    user_amenities_lower = [str(a).lower() for a in user_amenities]
    
    matches = len(set(package_amenities_lower).intersection(set(user_amenities_lower)))
    total_requested = len(user_amenities)
    
    if matches == 0:
        return 0.0
    elif matches == total_requested:
        return 1.0
    else:
        return matches / total_requested

def calculate_skill_score(package, user_prefs: dict, user_profile: dict) -> float:
    skill_levels = {
        'Beginner': 1,
        'Intermediate': 2, 
        'Advanced': 3,
        'Professional': 4
    }
    
    package_skill = skill_levels.get(package['skill_level'], 2)
    user_skill = skill_levels.get(user_prefs['skill_level'], 2)
    
    if package_skill == user_skill:
        return 1.0
    elif abs(package_skill - user_skill) == 1:
        return 0.3
    else:
        return 0.1

def get_budget_indicator(price: float, smart_budget: float) -> str:
    """Generate budget indicator for recommendations"""
    if price <= smart_budget:
        return "Within Budget"
    elif price <= smart_budget * 1.1:
        return "Slightly Over"
    elif price <= smart_budget * 1.25:
        return "Moderately Over"
    else:
        over_percent = ((price - smart_budget) / smart_budget) * 100
        return f"Over Budget (+{over_percent:.0f}%)"

def get_recommendation_reason(amenities_score: float, distance_score: float, skill_score: float, budget_score: float) -> str:
    scores = {
        'amenities': amenities_score,
        'location': distance_score,
        'skill_match': skill_score,
        'budget_fit': budget_score
    }
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_strengths = [item[0] for item in sorted_scores[:2] if item[1] > 0.6]
    
    if len(top_strengths) >= 2:
        return f"Great {top_strengths[0]} & {top_strengths[1]}"
    elif len(top_strengths) == 1:
        return f"Excellent {top_strengths[0]}"
    else:
        return "Moderate overall match"

def calculate_enhanced_scores(user_prefs: dict, customer_history: pd.DataFrame, user_analysis: dict):
    global df
    
    if customer_history.empty or not user_analysis:
        return pd.DataFrame()
    
    # Determine sports filter
    sports_filter = set()
    if user_prefs.get('target_sports'):
        sports_filter = set([s.lower() for s in user_prefs['target_sports']])
    else:
        sports_filter = set(user_analysis['sports_history'])
    
    # Filter packages by sports
    filtered_packages = []
    for idx, package in df.iterrows():
        package_sports = package['sports']
        if isinstance(package_sports, list):
            package_sports_set = set([s.lower() for s in package_sports])
        else:
            package_sports_set = {str(package_sports).lower()}
        
        if package_sports_set.intersection(sports_filter):
            filtered_packages.append(package)
    
    if not filtered_packages:
        return pd.DataFrame()
    
    filtered_df = pd.DataFrame(filtered_packages)
    
    # Calculate distances
    distances = []
    for _, package in filtered_df.iterrows():
        distance = calculate_distance(
            package['latitude'], package['longitude'],
            user_prefs['user_lat'], user_prefs['user_lon']
        )
        distances.append(distance)
    
    # Get dynamic scores
    distance_scores = get_dynamic_distance_scores(distances)
    prices = filtered_df['_price'].tolist()
    smart_budget = user_analysis['budget_insights']['smart_budget']
    budget_scores = get_dynamic_budget_scores(prices, smart_budget)
    
    # Scoring weights
    weights = {
        'amenities': 0.10,
        'distance': 0.50,
        'skill_level': 0.15,
        'budget': 0.25
    }
    
    # Calculate final scores
    scores_data = []
    for idx, ((_, package), distance, distance_score, budget_score) in enumerate(
        zip(filtered_df.iterrows(), distances, distance_scores, budget_scores)
    ):
        amenities_score = calculate_amenities_score(package, user_prefs)
        skill_score = calculate_skill_score(package, user_prefs, user_analysis)
        
        total_score = (
            amenities_score * weights['amenities'] +
            distance_score * weights['distance'] +
            skill_score * weights['skill_level'] +
            budget_score * weights['budget']
        ) * 100
        
        scores_data.append({
            'ID': package['ID'],
            'academy_name': package['post_title'],
            'sports': package['sports'],
            'skill_level': package['skill_level'],
            'price': package['_price'],
            'address': package.get('address', 'N/A'),
            'amenities': package.get('amenities', 'N/A'),
            'latitude': package['latitude'],
            'longitude': package['longitude'],
            'distance_km': distance if distance != float('inf') else None,
            'total_score': round(total_score, 2),
            'amenities_score': round(amenities_score * 100, 2),
            'distance_score': round(distance_score * 100, 2),
            'skill_score': round(skill_score * 100, 2),
            'budget_score': round(budget_score * 100, 2),
            'recommendation_reason': get_recommendation_reason(
                amenities_score, distance_score, skill_score, budget_score
            ),
            'budget_indicator': get_budget_indicator(package['_price'], smart_budget)
        })
    
    return pd.DataFrame(scores_data).sort_values('total_score', ascending=False)

# Data loading function
def load_data():
    global df, lookup_with_name
    
    try:
        data_dir = Path("data")
        
        # Check if data files exist
        required_files = ["post.json", "post_meta.json", "order_addresses.json", 
                         "order_product_lookup.json", "customer_lookup.json"]
        
        for file in required_files:
            if not (data_dir / file).exists():
                logger.error(f"Required data file not found: {file}")
                return False
        
        # Load main data
        df_posts = pd.read_json(data_dir / "post.json")
        df_meta = pd.read_json(data_dir / "post_meta.json")
        
        # Filter and prepare posts data
        dff = df_posts[(df_posts['post_type'] == 'product') | (df_posts['post_type'] == 'product_variation')]
        dff = dff[['ID','post_parent','post_title','post_excerpt','post_status','post_name','post_type']]
        
        # Pivot meta data
        dfm_pivot = df_meta.pivot_table(
            index='post_id', 
            columns='meta_key', 
            values='meta_value', 
            aggfunc='first'
        ).reset_index()
        
        # Merge data
        merged_df = dff.merge(dfm_pivot, how='left', left_on='ID', right_on='post_id')
        merged_df.drop(columns=['post_id'], inplace=True)
        
        # Process meta keys from parent posts
        meta_keys = ['amenities', 'address', 'latitude', 'longitude', 'skill_level', 'available_sports_in_this_academy']
        
        for key in meta_keys:
            temp_map = (
                df_meta[df_meta['meta_key'] == key][['post_id', 'meta_value']]
                .rename(columns={'post_id': 'post_parent', 'meta_value': f'{key}_value'})
            )
            
            merged_df = merged_df.merge(temp_map, on='post_parent', how='left')
            
            merged_df[key] = merged_df.apply(
                lambda row: row[f'{key}_value'] if row['post_parent'] > 0 and pd.notna(row[f'{key}_value']) else row.get(key, np.nan),
                axis=1
            )
            
            merged_df.drop(columns=[f'{key}_value'], inplace=True)
        
        # Create sports column and clean data
        merged_df = create_sports_column(merged_df)
        df = clean_data(merged_df)
        
        # Load and prepare lookup data
        address = pd.read_json(data_dir / "order_addresses.json")
        lookup = pd.read_json(data_dir / "order_product_lookup.json")
        customer_df = pd.read_json(data_dir / "customer_lookup.json")
        
        # Process lookup data
        address["name"] = address["first_name"] + " " + address["last_name"]
        lookup_with_name = lookup.merge(address[["order_id", "name"]], on="order_id", how="left")
        # if username exists in customer_df, add it
        if 'customer_id' in customer_df.columns and 'username' in customer_df.columns:
            lookup_with_name = lookup_with_name.merge(
                customer_df[['customer_id', 'username']],
                on='customer_id',
                how='left'
            )
        
        logger.info(f"Data loaded successfully: {len(df)} packages, {len(lookup_with_name)} orders")
        return True
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Loading data...")
    success = load_data()
    if not success:
        logger.error("Failed to load data - API will continue but may not work properly")
    else:
        logger.info("API startup complete")

# Root & Health
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Enhanced Sports Recommendation API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "features": [
            "Comprehensive user analysis with spending patterns",
            "Enhanced budget insights and smart budget inference", 
            "Dynamic distance and budget scoring",
            "Detailed location and travel pattern analysis",
            "Sport preference frequency tracking",
            "Skill progression monitoring"
        ],
        "endpoints": {
            "recommendations": "/recommendations",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Enhanced Sports Recommendation API is running",
        "data_loaded": df is not None and lookup_with_name is not None,
        "packages_count": len(df) if df is not None else 0,
        "orders_count": len(lookup_with_name) if lookup_with_name is not None else 0
    }

@app.get("/stats")
async def get_stats():
    try:
        if df is None or lookup_with_name is None:
            return {
                "error": "Data not loaded",
                "packages": 0,
                "orders": 0
            }
        
        # Get unique sports
        all_sports = []
        for sports in df['sports'].dropna():
            if isinstance(sports, list):
                all_sports.extend(sports)
            else:
                all_sports.append(sports)
        
        unique_sports = list(set([str(sport).lower() for sport in all_sports]))
        
        stats = {
            "total_packages": len(df),
            "total_orders": len(lookup_with_name),
            "unique_sports": unique_sports,
            "available_skill_levels": df['skill_level'].dropna().unique().tolist(),
            "price_range": {
                "min": float(df['_price'].min()) if not df['_price'].isna().all() else 0,
                "max": float(df['_price'].max()) if not df['_price'].isna().all() else 0,
                "avg": float(df['_price'].mean()) if not df['_price'].isna().all() else 0
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@app.post("/recommendations", response_model=RecommendationResult)
async def get_recommendations(
    user_prefs: UserPreferencesRequest,
    top_n: int = 10
):
    try:
        # Check if data is loaded
        if df is None or lookup_with_name is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Data not loaded. Please check server logs."
            )
        
        # Validate top_n
        if top_n > 50:
            top_n = 50
        elif top_n < 1:
            top_n = 10
        
        logger.info(f"Getting recommendations for user: {user_prefs.username}")
        
        # Convert pydantic model to dict
        user_prefs_dict = {
            'username': user_prefs.username,
            'user_lat': user_prefs.user_lat,
            'user_lon': user_prefs.user_lon,
            'preferred_amenities': user_prefs.preferred_amenities or [],
            'skill_level': user_prefs.skill_level.value,
            'target_sports': user_prefs.target_sports
        }
        
        # Get customer history
        customer_history, customer_name = get_customer_history(user_prefs.username)
        
        if customer_history.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No purchase history found for user: {user_prefs.username}"
            )
        
        # Enhanced user analysis
        user_analysis = analyze_user_comprehensive(customer_history, user_prefs_dict, customer_name)
        
        # Calculate recommendations
        recommendations_df = calculate_enhanced_scores(user_prefs_dict, customer_history, user_analysis)
        
        if recommendations_df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No packages match your criteria"
            )
        
        # Convert to response format
        recommendations = recommendations_df.head(top_n)
        smart_budget = user_analysis['budget_insights']['smart_budget']
        
        recommendation_list = []
        for _, rec in recommendations.iterrows():
            # Safely handle None values for latitude/longitude
            lat = float(rec['latitude']) if pd.notna(rec['latitude']) else None
            lon = float(rec['longitude']) if pd.notna(rec['longitude']) else None
            distance = float(rec['distance_km']) if pd.notna(rec['distance_km']) else None
            
            recommendation_list.append(RecommendationResponse(
                ID=int(rec['ID']),
                academy_name=str(rec['academy_name']),
                sports=rec['sports'],
                skill_level=str(rec['skill_level']),
                price=float(rec['price']),
                address=str(rec['address']),
                amenities=rec['amenities'],
                latitude=lat,
                longitude=lon,
                distance_km=distance,
                smart_budget=float(smart_budget),
                total_score=float(rec['total_score']),
                amenities_score=float(rec['amenities_score']),
                distance_score=float(rec['distance_score']),
                skill_score=float(rec['skill_score']),
                budget_score=float(rec['budget_score']),
                recommendation_reason=str(rec['recommendation_reason']),
                budget_indicator=str(rec['budget_indicator'])
            ))
        
        # Calculate summary statistics
        within_budget_count = len(recommendations[recommendations['price'] <= smart_budget])
        excellent_matches = len(recommendations[recommendations['total_score'] >= 80])
        good_matches = len(recommendations[recommendations['total_score'] >= 60])
        
        avg_distance = recommendations['distance_km'].mean() if not recommendations['distance_km'].isna().all() else None
        
        summary_stats = {
            "total_recommendations": len(recommendation_list),
            "within_budget_count": within_budget_count,
            "excellent_matches": excellent_matches,
            "good_matches": good_matches,
            "average_distance": float(avg_distance) if avg_distance else None,
            "average_price": float(recommendations['price'].mean()),
            "price_range": {
                "min": float(recommendations['price'].min()),
                "max": float(recommendations['price'].max())
            }
        }
        
        result = RecommendationResult(
            user_analysis=UserAnalysisResponse(
                customer_name=user_analysis['customer_name'],
                sports_history=user_analysis['sports_history'],
                purchase_count=user_analysis['purchase_count'],
                budget_insights=user_analysis['budget_insights'],
                skill_progression=user_analysis['skill_progression'],
                location_analysis=user_analysis['location_analysis'],
                spending_pattern=user_analysis['spending_pattern']
            ),
            recommendations=recommendation_list,
            summary_stats=summary_stats,
            message=f'Enhanced analysis complete for {customer_name or user_prefs.username}: {len(recommendation_list)} intelligent recommendations generated'
        )
        
        logger.info(f"Successfully generated {len(recommendation_list)} recommendations with comprehensive analysis")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Enhanced validation handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        message = error["msg"]
        errors.append(f"{field}: {message}")

    return JSONResponse(
        status_code=422,
        content={
            "status_code": 422,
            "message": "Validation Error",
            "details": errors,
            "example_request": {
                "username": "user123",
                "user_lat": 29.8543,
                "user_lon": 77.8880,
                "preferred_amenities": ["parking", "wifi"],
                "skill_level": "Intermediate",
                "target_sports": ["football", "cricket"]
            },
            "api_features": [
                "Comprehensive user analysis with spending patterns",
                "Smart budget inference with detailed reasoning",
                "Dynamic scoring based on available options",
                "Location and travel pattern analysis"
            ]
        }
    )

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
