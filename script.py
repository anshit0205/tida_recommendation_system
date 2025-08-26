import fastapi
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Set
import pandas as pd
import numpy as np
import requests
import re
import json
from pathlib import Path
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Tida Sports Recommendation API",
    description="API for recommending sports academy packages based on customer preferences and purchase history",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store DataFrames (in production, use a proper database)
df = None
lookup_with_name = None

# ================================
# SPORT CANONICALIZATION CONFIG
# ================================
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
    "martial arts": ["Martial Arts","MartialArts",
        "martial arts", "mma", "mixed martial arts", "karate", "judo",
        "kung fu", "kung-fu", "aikido", "kickboxing", "muay thai",
        "muay-thai", "self defense", "self-defence", "self defence",
        "kungfu"],
    "yoga": ["yoga", "meditation", "pranayama"],
    "padel": ["padel", "paddle tennis", "padel tennis"],
    "pickleball": ["pickleball","pickle ball","pickle-ball"],
    "roller hockey": ["roller hockey","RollerHoCkey" ,"roller-hockey"],
    "boxing": ["boxing", "Boxing","box", "kick boxing", "kick-boxing", "kickboxing"]
}

# Reverse map: synonym -> canonical
syn_to_canon = {}
for canon, syns in SPORT_CANON.items():
    for s in syns:
        syn_to_canon[s.lower()] = canon

# ================================
# PYDANTIC MODELS
# ================================
class ExistingCustomerRequest(BaseModel):
    username: str = Field(..., description="Customer username (case-insensitive)")
    max_distance_km: float = Field(10.0, description="Maximum distance in kilometers", gt=0)
    use_ors: bool = Field(False, description="Use OpenRouteService for routing distances")
    ors_api_key: Optional[str] = Field(None, description="ORS API key (required if use_ors=True)")
    ors_profile: str = Field("driving-car", description="ORS routing profile")

class NewCustomerRequest(BaseModel):
    user_lat: Optional[float] = Field(None, description="User latitude (decimal degrees)")
    user_lon: Optional[float] = Field(None, description="User longitude (decimal degrees)")
    preferred_sports: Optional[Union[str, List[str]]] = Field(None, description="Preferred sports (comma-separated string or list)")
    use_ors: bool = Field(False, description="Use OpenRouteService for routing distances")
    ors_api_key: Optional[str] = Field(None, description="ORS API key (required if use_ors=True)")
    ors_profile: str = Field("driving-car", description="ORS routing profile")
    
    class Config:
        schema_extra = {
            "example": {
                "user_lat": 28.6139,
                "user_lon": 77.2090,
                "preferred_sports": ["football", "cricket"],
                "use_ors": False,
                "ors_api_key": None,
                "ors_profile": "driving-car"
            }
        }

class RecommendationResponse(BaseModel):
    status: str
    message: str
    count: int
    data: List[Dict[str, Any]]
    distance_range: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    data_loaded: bool

# ================================
# UTILITY FUNCTIONS (EXACT SAME LOGIC)
# ================================
def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees)"""
    R = 6371.0
    lon1 = np.array(lon1, dtype=float)
    lat1 = np.array(lat1, dtype=float)
    lon2 = np.array(lon2, dtype=float)
    lat2 = np.array(lat2, dtype=float)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def split_raw_parts(text: str) -> List[str]:
    """Split raw text by common delimiters and return raw trimmed parts (preserve case)."""
    if not isinstance(text, str):
        return []
    parts = re.split(r'[;,\/\|\&]|\band\b', text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts

def split_and_clean(text: str) -> List[str]:
    """Lowercased cleaned tokens (used for matching)."""
    if not isinstance(text, str):
        return []
    parts = re.split(r'[;,\/\|\&]|\band\b', text, flags=re.IGNORECASE)
    parts = [p.strip().lower() for p in parts if p and p.strip()]
    return parts

def map_to_canonical(token: str) -> Optional[str]:
    """Map a token (any case) to canonical item using exact synonyms or word-boundary matching."""
    if not isinstance(token, str):
        return None
    t = token.lower().strip()
    # direct synonym
    if t in syn_to_canon:
        return syn_to_canon[t]
    # try word-boundary partial matches for synonyms
    for syn, canon in syn_to_canon.items():
        if re.search(r'\b' + re.escape(syn) + r'\b', t):
            return canon
    # fallback: check if canonical name itself appears as word
    for canon in SPORT_CANON.keys():
        if re.search(r'\b' + re.escape(canon) + r'\b', t):
            return canon
    return None

def normalize_list_value(val) -> Set[str]:
    """Normalize various input types to a set of lowercase strings."""
    if val is None:
        return set()
    try:
        if not isinstance(val, (list, tuple, set, dict, np.ndarray, pd.Series, bytes, bytearray, str)) and pd.isna(val):
            return set()
    except Exception:
        pass
    
    if isinstance(val, dict):
        items = list(val.values())
    elif isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
        items = list(val)
    elif isinstance(val, (bytes, bytearray)):
        try:
            s = val.decode('utf-8', errors='ignore')
        except Exception:
            s = str(val)
        parts = re.split(r'[;,|/]| and | & ', s)
        items = [p.strip() for p in parts if p.strip()]
    elif isinstance(val, str):
        s = val.strip()
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1]
            parts = re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', inner)
            items = [re.sub(r'^[\"\']|[\"\']$', '', p).strip() for p in parts if p.strip()]
        else:
            parts = re.split(r'[;,|/]| and | & ', s)
            items = [p.strip() for p in parts if p.strip()]
    else:
        s = str(val)
        parts = re.split(r'[;,|/]| and | & ', s)
        items = [p.strip() for p in parts if p.strip()]
    
    cleaned = []
    for p in items:
        if isinstance(p, (list, tuple, np.ndarray, pd.Series)):
            for sub in list(p):
                sub_s = str(sub).strip()
                if sub_s:
                    cleaned.append(sub_s.lower())
        else:
            p_s = str(p).strip()
            if p_s:
                cleaned.append(p_s.lower())
    return set(cleaned)

def safe_notna(val):
    """Check if value is not null/empty in a safe way."""
    try:
        if val is None:
            return False
        if isinstance(val, (list, tuple, set)):
            return len(val) > 0
        if isinstance(val, (np.ndarray, pd.Series)):
            return len(val) > 0 and not pd.isna(val).all()
        if isinstance(val, str):
            return len(val.strip()) > 0
        if isinstance(val, (int, float)):
            return not pd.isna(val)
        return pd.notna(val)
    except Exception:
        return False

def parse_available_field_keep_unknowns(val) -> List[str]:
    """Parse available_sports_in_this_academy and return list of items."""
    result = []
    seen_lower = set()

    if pd.isna(val):
        return result

    # If it's already a list-like, coerce to strings
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

    # else: it's a string
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
    """Search post_title for known canonical items only."""
    found = []
    if not isinstance(title, str):
        return found
    txt = title.lower()
    # check synonyms first
    for syn, canon in syn_to_canon.items():
        if re.search(r'\b' + re.escape(syn) + r'\b', txt) and canon not in found:
            found.append(canon)
    # also check canonical names
    for canon in SPORT_CANON.keys():
        if re.search(r'\b' + re.escape(canon) + r'\b', txt) and canon not in found:
            found.append(canon)
    return found

def ors_matrix_distances_from_origin(user_lon: float, user_lat: float, dest_coords: List[tuple],
                                     ors_api_key: str, profile: str = 'driving-car',
                                     batch_size: int = 50,
                                     ors_base: str = 'https://api.openrouteservice.org') -> List[Optional[float]]:
    """Get driving distances from ORS API."""
    if not ors_api_key:
        raise ValueError('ORS API key required')
    if len(dest_coords) == 0:
        return []
    
    distances_km_all = []
    for i in range(0, len(dest_coords), batch_size):
        chunk = dest_coords[i:i+batch_size]
        locations = [[float(user_lon), float(user_lat)]] + [[float(lon), float(lat)] for (lat, lon) in chunk]
        body = {"locations": locations, "sources": [0], "destinations": list(range(1, len(locations))), "metrics": ["distance"]}
        headers = {"Authorization": ors_api_key, "Content-Type": "application/json"}
        
        try:
            resp = requests.post(f"{ors_base}/v2/matrix/{profile}", json=body, headers=headers, timeout=20)
            resp.raise_for_status()
            j = resp.json()
            distances_matrix = j.get('distances', [])
            if not distances_matrix:
                distances_km_all.extend([None]*len(chunk))
            else:
                row0 = distances_matrix[0]
                for val in row0:
                    if val is None:
                        distances_km_all.append(None)
                    else:
                        try:
                            distances_km_all.append(float(val)/1000.0)
                        except Exception:
                            distances_km_all.append(None)
        except Exception:
            distances_km_all.extend([None]*len(chunk))
    
    if len(distances_km_all) < len(dest_coords):
        distances_km_all.extend([None]*(len(dest_coords)-len(distances_km_all)))
    return distances_km_all

# ================================
# CORE RECOMMENDATION FUNCTIONS (MODIFIED TO SUPPORT ORS)
# ================================
def recommend_nearby_packages_from_purchased(
    res_df: pd.DataFrame,
    main_df: pd.DataFrame,
    max_distance_km: float = 10.0,
    max_recommendations: int = 20,
    sports_cols_res: List[str] = None,
    sports_cols_df: List[str] = None,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col_df: str = "ID",
    minimal_logs: bool = True,
    use_ors: bool = False,
    ors_api_key: Optional[str] = None,
    ors_profile: str = "driving-car",
    ors_batch_size: int = 50
) -> pd.DataFrame:
    """Recommend packages based on purchased history - now with ORS support."""
    if sports_cols_res is None:
        sports_cols_res = ["sports", "sports2"]
    if sports_cols_df is None:
        sports_cols_df = ["sports", "sports2"]

    if res_df is None or res_df.empty:
        return pd.DataFrame()
    if main_df is None or main_df.empty:
        return pd.DataFrame()

    res_work = res_df.copy()
    df_work = main_df.copy()

    # ensure coords numeric
    for col in [lat_col, lon_col]:
        if col in res_work.columns:
            res_work[col] = pd.to_numeric(res_work[col], errors="coerce").astype(float)
        if col in df_work.columns:
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce").astype(float)

    res_work = res_work.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    df_work = df_work.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

    if res_work.empty or df_work.empty:
        return pd.DataFrame()

    def extract_sports_set(row, sports_cols):
        sports_set = set()
        for col in sports_cols:
            if col in row.index:
                val = row[col]
                if safe_notna(val):
                    sports_set |= normalize_list_value(val)
        return sports_set

    res_work['_sports_set'] = res_work.apply(lambda row: extract_sports_set(row, sports_cols_res), axis=1)
    df_work['_sports_set'] = df_work.apply(lambda row: extract_sports_set(row, sports_cols_df), axis=1)

    # Get purchased IDs
    purchased_ids = set()
    if 'product_id' in res_work.columns:
        purchased_ids.update(res_work['product_id'].dropna().astype(str))
    if 'variation_id' in res_work.columns:
        purchased_ids.update(res_work['variation_id'].dropna().astype(str))
    if id_col_df in res_work.columns:
        purchased_ids.update(res_work[id_col_df].dropna().astype(str))

    all_recommendations = []

    for idx, res_row in res_work.iterrows():
        res_lat = res_row[lat_col]
        res_lon = res_row[lon_col]
        res_sports = res_row['_sports_set']
        res_title = res_row.get('post_title', f'Purchased Package {idx+1}')

        if len(res_sports) == 0:
            continue

        # Calculate distances - now with ORS support
        dest_coords = list(zip(df_work[lat_col].astype(float).values, df_work[lon_col].astype(float).values))
        
        distances_km = None
        if use_ors and ors_api_key:
            try:
                distances_km = ors_matrix_distances_from_origin(
                    user_lon=res_lon,
                    user_lat=res_lat,
                    dest_coords=dest_coords,
                    ors_api_key=ors_api_key,
                    profile=ors_profile,
                    batch_size=ors_batch_size
                )
            except Exception:
                distances_km = None

        if distances_km is None:
            # Use haversine distance as fallback
            df_work['_temp_distance'] = haversine(res_lon, res_lat, df_work[lon_col], df_work[lat_col])
        else:
            # Use ORS distances, fallback to haversine for None values
            final_distances = []
            for i, d in enumerate(distances_km):
                if d is None:
                    lat, lon = dest_coords[i]
                    final_distances.append(haversine(res_lon, res_lat, lon, lat))
                else:
                    final_distances.append(d)
            df_work['_temp_distance'] = final_distances

        nearby_packages = df_work[df_work['_temp_distance'] <= max_distance_km].copy()
        
        if nearby_packages.empty:
            continue

        nearby_packages['_sports_match'] = nearby_packages['_sports_set'].apply(lambda df_sports: len(df_sports & res_sports) > 0)
        nearby_packages = nearby_packages[nearby_packages['_sports_match']].copy()

        if nearby_packages.empty:
            continue

        if id_col_df in nearby_packages.columns:
            nearby_packages = nearby_packages[~nearby_packages[id_col_df].astype(str).isin(purchased_ids)].copy()
        
        if nearby_packages.empty:
            continue

        nearby_packages['reference_package'] = res_title
        nearby_packages['reference_lat'] = res_lat
        nearby_packages['reference_lon'] = res_lon
        nearby_packages['distance_km'] = nearby_packages['_temp_distance']
        nearby_packages['matched_sports'] = nearby_packages['_sports_set'].apply(lambda df_sports: ', '.join(sorted(df_sports & res_sports)))

        for _, pkg_row in nearby_packages.iterrows():
            all_recommendations.append(pkg_row.to_dict())

    if not all_recommendations:
        return pd.DataFrame()

    recommendations_df = pd.DataFrame(all_recommendations)

    # Deduplication logic
    if id_col_df in recommendations_df.columns:
        idx_min = recommendations_df.groupby(id_col_df)['distance_km'].idxmin()
        recommendations_df = recommendations_df.loc[idx_min].copy()
    else:
        dedup_cols = []
        if 'address' in recommendations_df.columns:
            dedup_cols.append('address')
        if 'post_title' in recommendations_df.columns:
            dedup_cols.append('post_title')
        if dedup_cols:
            idx_min = recommendations_df.groupby(dedup_cols)['distance_km'].idxmin()
            recommendations_df = recommendations_df.loc[idx_min].copy()

    # Sort by distance
    recommendations_df = recommendations_df.sort_values('distance_km', ascending=True).reset_index(drop=True)

    # Limit recommendations
    try:
        recommendations_df = recommendations_df.head(int(max_recommendations))
    except Exception:
        pass

    # Cleanup columns
    cols_to_drop = ['_temp_distance', '_sports_set', '_sports_match']
    recommendations_df = recommendations_df.drop(columns=[c for c in cols_to_drop if c in recommendations_df.columns])

    return recommendations_df

def get_all_packages_sorted_by_distance_new_customer(
    df: pd.DataFrame,
    user_lat: Optional[float],
    user_lon: Optional[float],
    preferred_sports: Union[str, List[str], None],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    sports_cols: List[str] = None,
    use_ors: bool = False,
    ors_api_key: Optional[str] = None,
    ors_profile: str = "driving-car",
    ors_batch_size: int = 50
) -> pd.DataFrame:
    """Get ALL packages with matching sports sorted by distance for new customers - no filtering except renewals."""
    if sports_cols is None:
        sports_cols = ["sports", "sports2"]

    # normalize preferred sports set
    if preferred_sports is None:
        pref_set = set()
    elif isinstance(preferred_sports, str):
        pref_set = normalize_list_value(preferred_sports)
    else:
        pref_set = set()
        for s in preferred_sports:
            pref_set |= normalize_list_value(s)

    # copy and ensure numeric coords
    df_work = df.copy()
    df_work[lat_col] = pd.to_numeric(df_work[lat_col], errors="coerce")
    df_work[lon_col] = pd.to_numeric(df_work[lon_col], errors="coerce")
    df_work = df_work.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

    # compute normalized sports sets
    def row_sports(r):
        s = set()
        for c in sports_cols:
            if c in r:
                s |= normalize_list_value(r[c])
        return s

    df_work["_sports_set"] = df_work.apply(row_sports, axis=1)

    # apply sports filter ONLY if preferred_sports are specified
    if len(pref_set) > 0:
        df_filtered = df_work[df_work["_sports_set"].apply(lambda s: len(s & pref_set) > 0)].copy()
        # Add matched sports column for filtered results
        df_filtered["matched_sports"] = df_filtered["_sports_set"].apply(lambda s: ', '.join(sorted(s & pref_set)))
    else:
        # If no preferences specified, show ALL packages
        df_filtered = df_work.copy()
        # Add all sports as matched when no preference specified
        df_filtered["matched_sports"] = df_filtered["_sports_set"].apply(lambda s: ', '.join(sorted(s)))

    if df_filtered.shape[0] == 0:
        return pd.DataFrame()

    # ONLY exclude renewal packages - remove all other filtering
    renewal_pattern = re.compile(r"\brenew\w*\b", flags=re.IGNORECASE)
    mask_title = ~df_filtered.get("post_title", pd.Series([""]*len(df_filtered))).astype(str).str.contains(renewal_pattern)
    mask_excerpt = ~df_filtered.get("post_excerpt", pd.Series([""]*len(df_filtered))).astype(str).str.contains(renewal_pattern)
    df_filtered = df_filtered[mask_title & mask_excerpt].copy().reset_index(drop=True)

    if df_filtered.shape[0] == 0:
        return pd.DataFrame()

    # Calculate distances
    user_coords_missing = (user_lat is None) or (user_lon is None) or pd.isna(user_lat) or pd.isna(user_lon)
    
    if user_coords_missing:
        df_filtered["distance_km"] = np.nan
    else:
        # Get coordinates for distance calculation
        dest_coords = list(zip(df_filtered[lat_col].astype(float).values, df_filtered[lon_col].astype(float).values))
        
        distances_km = None
        if use_ors and ors_api_key:
            try:
                distances_km = ors_matrix_distances_from_origin(
                    user_lon=user_lon,
                    user_lat=user_lat,
                    dest_coords=dest_coords,
                    ors_api_key=ors_api_key,
                    profile=ors_profile,
                    batch_size=ors_batch_size
                )
            except Exception:
                distances_km = [None] * len(dest_coords)

        final_distances = []
        if distances_km is None:
            # Use haversine distance
            for (lat, lon) in dest_coords:
                final_distances.append(float(haversine(float(user_lon), float(user_lat), float(lon), float(lat))))
        else:
            # Use ORS distances, fallback to haversine if None
            for idx, d in enumerate(distances_km):
                if d is None:
                    lat, lon = dest_coords[idx]
                    final_distances.append(float(haversine(float(user_lon), float(user_lat), float(lon), float(lat))))
                else:
                    final_distances.append(float(d))

        df_filtered["distance_km"] = final_distances

    # Sort by distance (NaN values will go to end)
    if not user_coords_missing:
        df_filtered = df_filtered.sort_values("distance_km", ascending=True).reset_index(drop=True)
    
    # Cleanup internal columns
    if "_sports_set" in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=["_sports_set"])

    # Replace NaN distances with message
    if df_filtered.shape[0] > 0 and df_filtered["distance_km"].isna().any():
        df_filtered["distance_km"] = df_filtered["distance_km"].apply(lambda x: "lat, long was null" if pd.isna(x) else float(x))

    return df_filtered

# ================================
# DATA LOADING FUNCTIONS (UNCHANGED)
# ================================
def load_data_from_files(data_dir: str = "data"):
    """Load and process data from JSON files - exact same logic as original."""
    global df, lookup_with_name
    
    try:
        data_path = Path(data_dir)
        
        # Load post.json
        post_df = pd.read_json(data_path / "post.json")
        dff = post_df[(post_df['post_type'] == 'product') | (post_df['post_type'] == 'product_variation')]
        dff = dff[['ID','post_parent','post_title','post_excerpt','post_status','post_name','post_type']]
        
        # Load post_meta.json and pivot
        meta_df = pd.read_json(data_path / "post_meta.json")
        dfm_pivot = meta_df.pivot_table(
            index='post_id', 
            columns='meta_key', 
            values='meta_value', 
            aggfunc='first'
        ).reset_index()
        
        # Merge with dff
        merged_df = dff.merge(dfm_pivot, how='left', left_on='ID', right_on='post_id')
        merged_df.drop(columns=['post_id'], inplace=True)
        
        # Pull meta_keys from parent posts
        meta_keys = ['amenities', 'address', 'latitude', 'longitude', 'skill_level', 'available_sports_in_this_academy']
        for key in meta_keys:
            temp_map = (
                meta_df[meta_df['meta_key'] == key][['post_id', 'meta_value']]
                .rename(columns={'post_id': 'post_parent', 'meta_value': f'{key}_value'})
            )
            merged_df = merged_df.merge(temp_map, on='post_parent', how='left')
            merged_df[key] = merged_df.apply(
                lambda row: row[f'{key}_value'] if row['post_parent'] > 0 and pd.notna(row[f'{key}_value']) else row[key],
                axis=1
            )
            merged_df.drop(columns=[f'{key}_value'], inplace=True)
        
        # Create sports column using exact same logic
        df_processed = create_sports_column(merged_df)
        
        # Handle YMCA special case
        mask_ymca = df_processed['post_title'].str.strip().eq("YMCA")
        vals = pd.Series([['football', 'cricket']] * mask_ymca.sum(), index=df_processed.loc[mask_ymca].index)
        df_processed.loc[mask_ymca, 'sports'] = vals
        
        # Remove drafts and testing entries
        df_processed = df_processed[df_processed['post_status'] != 'draft'].copy()
        df_processed = df_processed[
            ~(
                (df_processed['post_title'] == "AUTO-DRAFT") |
                (df_processed['post_title'].str.contains("testing", case=False, na=False)) |
                (df_processed['post_excerpt'].str.contains("testing", case=False, na=False))
            )
        ]
        
        # Handle Bala Ji Skaters
        df_processed.loc[
            df_processed['post_title'].str.contains("Bala Ji Skaters", case=False, na=False) & df_processed['sports'].isna(),
            'sports'
        ] = "skating"
        
        # Drop rows without sports
        df_processed = df_processed.dropna(subset=['sports'])
        
        # Convert sports to string
        def convert_to_string(x):
            if isinstance(x, list):
                return ", ".join(x)
            elif pd.isna(x):
                return np.nan
            else:
                return x
        
        df_processed['sports2'] = df_processed['sports'].apply(convert_to_string)
        
        # Select final columns
        df_processed = df_processed[['ID','post_parent','post_title','post_excerpt','post_status',
                                   'post_name','post_type','_price','amenities', 'address', 'latitude',
                                   'longitude', 'skill_level', 'available_sports_in_this_academy',
                                   'sports', 'sports2']]
        
        # Process amenities (PHP unserialize equivalent)
        import phpserialize
        def unserialize_php(value):
            if pd.isna(value):
                return None
            try:
                parsed = phpserialize.loads(value.encode("utf-8"), decode_strings=True)
                return list(parsed.values())
            except Exception:
                return None
        
        df_processed["amenities"] = df_processed["amenities"].apply(unserialize_php)
        df_processed["amenities2"] = df_processed["amenities"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else None
        )
        
        # Remove private posts
        df_processed = df_processed[df_processed['post_status'] != 'private']
        
        # Load order data
        address_df = pd.read_json(data_path / "order_addresses.json")
        lookup_df = pd.read_json(data_path / "order_product_lookup.json")
        customer_df = pd.read_json(data_path / "customer_lookup.json")
        
        # Create lookup_with_name
        address_df["name"] = address_df["first_name"] + " " + address_df["last_name"]
        lookup_with_name_temp = lookup_df.merge(address_df[["order_id", "name"]], on="order_id", how="left")
        lookup_with_name_temp = lookup_with_name_temp.merge(
            customer_df[['customer_id', 'username']],
            on='customer_id',
            how='left'
        )
        
        # Set global variables
        df = df_processed
        lookup_with_name = lookup_with_name_temp
        
        return True, f"Loaded {len(df)} packages and {len(lookup_with_name)} purchase records"
        
    except Exception as e:
        return False, f"Error loading data: {str(e)}"

def create_sports_column(merged_df: pd.DataFrame,
                        avail_col: str = 'available_sports_in_this_academy',
                        title_col: str = 'post_title',
                        status_col: str = 'post_status',
                        drop_ignore: bool = True,
                        drop_trash: bool = True) -> pd.DataFrame:
    """Create sports column using exact same logic as original."""
    df_work = merged_df.copy()

    # Drop trash posts
    if drop_trash and status_col in df_work.columns:
        mask_trash = df_work[status_col].astype(str).str.lower().eq('trash')
        if mask_trash.any():
            df_work = df_work.loc[~mask_trash].reset_index(drop=True)

    # Drop "please ignore" posts
    if drop_ignore and title_col in df_work.columns:
        mask_ignore = df_work[title_col].astype(str).str.contains(r'please\s+ignore', case=False, na=False)
        if mask_ignore.any():
            df_work = df_work.loc[~mask_ignore].reset_index(drop=True)

    # Initialize sports column
    df_work['sports'] = pd.NA

    # First pass: parse available_sports_in_this_academy
    if avail_col in df_work.columns:
        for i in df_work.index:
            val = df_work.at[i, avail_col]
            parsed = parse_available_field_keep_unknowns(val)
            if parsed:
                df_work.at[i, 'sports'] = parsed[0] if len(parsed) == 1 else parsed

    # Second pass: search post_title for canonical matches
    if title_col in df_work.columns:
        missing_idx = df_work[df_work['sports'].isna()].index
        for i in missing_idx:
            title = df_work.at[i, title_col]
            found = search_post_title_for_sports(title)
            if found:
                df_work.at[i, 'sports'] = found[0] if len(found) == 1 else found

    return df_work

# ================================
# API ENDPOINTS
# ================================
@app.on_event("startup")
async def startup_event():
    """Load data on startup."""
    success, message = load_data_from_files()
    if not success:
        print(f"Warning: {message}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    data_loaded = df is not None and lookup_with_name is not None
    return HealthResponse(
        status="success" if data_loaded else "warning",
        message="API is running" + (" with data loaded" if data_loaded else " but data not loaded"),
        data_loaded=data_loaded
    )

@app.post("/load-data")
async def load_data_endpoint(data_dir: str = "data"):
    """Manually load data from specified directory."""
    success, message = load_data_from_files(data_dir)
    if success:
        return {"status": "success", "message": message}
    else:
        raise HTTPException(status_code=500, detail=message)

@app.post("/recommend/existing-customer", response_model=RecommendationResponse)
async def recommend_existing_customer(request: ExistingCustomerRequest):
    """Recommend packages for existing customer based on purchase history - now with ORS support."""
    if df is None or lookup_with_name is None:
        raise HTTPException(status_code=500, detail="Data not loaded. Please call /load-data first.")
    
    # Validate ORS requirements
    if request.use_ors and not request.ors_api_key:
        raise HTTPException(status_code=400, detail="ORS API key required when use_ors=True")
    
    # Find customer's purchase history
    username_input = request.username.strip().lower()
    res_df = None
    
    # Search in lookup_with_name
    if 'username' in lookup_with_name.columns:
        try:
            mask = lookup_with_name['username'].astype(str).str.lower().str.contains(username_input, na=False)
            candidate = lookup_with_name[mask].copy()
            if not candidate.empty:
                ids = set()
                for col in ('product_id','variation_id','ID','id'):
                    if col in candidate.columns:
                        ids.update(candidate[col].dropna().astype(str).tolist())
                if ids and 'ID' in df.columns:
                    res_df = df[df['ID'].astype(str).isin(ids)].copy()
        except Exception:
            pass
    
    # Fallback search in all text columns
    if (res_df is None or res_df.empty):
        lw = lookup_with_name
        cols = [c for c in lw.columns if lw[c].dtype == object or lw[c].dtype == 'string']
        mask = pd.Series(False, index=lw.index)
        for c in cols:
            try:
                mask = mask | lw[c].astype(str).str.lower().str.contains(username_input, na=False)
            except Exception:
                pass
        candidate = lw[mask].copy()
        if not candidate.empty:
            ids = set()
            for col in ('product_id','variation_id','ID','id'):
                if col in candidate.columns:
                    ids.update(candidate[col].dropna().astype(str).tolist())
            if ids and 'ID' in df.columns:
                res_df = df[df['ID'].astype(str).isin(ids)].copy()
    
    if res_df is None or res_df.empty:
        return RecommendationResponse(
            status="error",
            message=f"No purchase history found for username: {request.username}",
            count=0,
            data=[]
        )
    
    # Get recommendations using modified function with ORS support
    recommendations = recommend_nearby_packages_from_purchased(
        res_df=res_df,
        main_df=df,
        max_distance_km=request.max_distance_km,
        max_recommendations=len(df),  # Show maximum as per original logic
        minimal_logs=True,
        use_ors=request.use_ors,
        ors_api_key=request.ors_api_key,
        ors_profile=request.ors_profile
    )
    
    if recommendations.empty:
        return RecommendationResponse(
            status="success",
            message="No recommendations found within specified criteria",
            count=0,
            data=[]
        )
    
    # Sort by distance ascending
    try:
        recommendations_sorted = recommendations.sort_values('distance_km', ascending=True).reset_index(drop=True)
    except Exception:
        recommendations_sorted = recommendations.copy()
    
    # Calculate distance range
    distance_range = None
    if 'distance_km' in recommendations_sorted.columns and len(recommendations_sorted) > 0:
        try:
            min_d = float(recommendations_sorted['distance_km'].min())
            max_d = float(recommendations_sorted['distance_km'].max())
            distance_range = {"min_km": min_d, "max_km": max_d}
        except Exception:
            pass
    
    # Convert to dict records
    data = recommendations_sorted.to_dict('records')
    
    return RecommendationResponse(
        status="success",
        message=f"Found {len(data)} recommendations" + (" using ORS routing" if request.use_ors else " using direct distance"),
        count=len(data),
        data=data,
        distance_range=distance_range
    )

@app.post("/recommend/new-customer", response_model=RecommendationResponse)
async def recommend_new_customer(request: NewCustomerRequest):
    """Recommend ALL packages for new customer - shows all matching sports sorted by distance."""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded. Please call /load-data first.")
    
    # Validate ORS requirements
    if request.use_ors and not request.ors_api_key:
        raise HTTPException(status_code=400, detail="ORS API key required when use_ors=True")
    
    # Get ALL recommendations sorted by distance using new function
    recommendations = get_all_packages_sorted_by_distance_new_customer(
        df=df,
        user_lat=request.user_lat,
        user_lon=request.user_lon,
        preferred_sports=request.preferred_sports,
        use_ors=request.use_ors,
        ors_api_key=request.ors_api_key,
        ors_profile=request.ors_profile
    )
    
    if recommendations.empty:
        return RecommendationResponse(
            status="success",
            message="No recommendations found matching your criteria",
            count=0,
            data=[]
        )
    
    # Calculate distance range
    distance_range = None
    if 'distance_km' in recommendations.columns and len(recommendations) > 0:
        try:
            numeric_distances = pd.to_numeric(recommendations['distance_km'], errors='coerce')
            numeric_distances = numeric_distances.dropna()
            if len(numeric_distances) > 0:
                min_d = float(numeric_distances.min())
                max_d = float(numeric_distances.max())
                distance_range = {"min_km": min_d, "max_km": max_d}
        except Exception:
            pass
    
    # Convert to dict records
    data = recommendations.to_dict('records')
    
    return RecommendationResponse(
        status="success",
        message=f"Found {len(data)} packages sorted by distance" + (" using ORS routing" if request.use_ors else " using direct distance"),
        count=len(data),
        data=data,
        distance_range=distance_range
    )

@app.get("/sports/canonical")
async def get_canonical_sports():
    """Get list of all canonical sports with their synonyms."""
    return {
        "status": "success",
        "data": SPORT_CANON
    }

@app.get("/customers/search")
async def search_customers(username: str = Query(..., description="Username to search for")):
    """Search for customers by username."""
    if lookup_with_name is None:
        raise HTTPException(status_code=500, detail="Customer data not loaded.")
    
    username_lower = username.lower()
    
    # Search in username column
    matches = []
    if 'username' in lookup_with_name.columns:
        mask = lookup_with_name['username'].astype(str).str.lower().str.contains(username_lower, na=False)
        matches = lookup_with_name[mask][['username', 'name', 'customer_id']].drop_duplicates().to_dict('records')
    
    return {
        "status": "success",
        "message": f"Found {len(matches)} matching customers",
        "data": matches
    }

@app.get("/packages/count")
async def get_package_count():
    """Get total count of packages."""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
    
    return {
        "status": "success",
        "total_packages": len(df),
        "by_sport": df['sports'].value_counts().to_dict() if 'sports' in df.columns else {}
    }

# ================================
# MAIN ENTRY POINT
# ================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
