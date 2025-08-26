# Tida Sports Recommendation API

> FastAPI-based recommendation system for sports academy packages. Suggests relevant academy packages based on customer purchase history or new customer preferences and sorts them by distance.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Quick Start](#quick-start)

   * [Installation](#installation)
   * [Data Setup](#data-setup)
   * [Run the API](#run-the-api)
4. [API Endpoints](#api-endpoints)

   * [Health Check](#health-check)
   * [Load Data](#load-data)
   * [Recommend - Existing Customer](#recommend-for-existing-customer)
   * [Recommend - New Customer](#recommend-for-new-customer)
   * [Other Endpoints](#other-endpoints)
5. [OpenRouteService (ORS) Integration](#openrouteservice-ors-integration)

   * [ORS Routing Profiles](#ors-routing-profiles)
   * [Using ORS in Requests](#using-ors-in-requests)
6. [Supported Sports](#supported-sports)
7. [Response Format](#response-format)
8. [Configuration & Filtering Logic](#configuration--filtering-logic)
9. [Technical Notes](#technical-notes)
10. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
11. [Production Considerations](#production-considerations)

---

## Overview

This project provides a lightweight REST API (FastAPI) that recommends sports academy packages to users. It supports two main recommendation flows:

* **Existing customers** — find packages similar to what they already purchased and nearby academies; excludes packages already purchased by that customer.
* **New customers** — find packages that match user preferred sports, sorted by distance to the user's supplied coordinates.

Recommendations are location-aware and support either Haversine (direct) distance or real-world routing distances via OpenRouteService (ORS).

---

## Features

* Smart distance calculation: Haversine or ORS routing (real-world travel distance/time).
* Sport matching: recognizes 20+ sports and many common variations (case-insensitive).
* Location-based sorting and filtering.
* Purchase-history-aware recommendations (no duplicates).
* Flexible input formats for `preferred_sports`.
* Batch ORS requests and automatic ORS fallback to Haversine when ORS fails.
* Excludes renewal/testing/draft packages by default.

---

## Quick Start

### Installation

```bash
pip install fastapi uvicorn pandas numpy requests phpserialize
```

> Python 3.8+ recommended.

### Data Setup

Create a `data/` folder at the project root and add the following JSON files. These files are used by the API to serve recommendations and must be loaded via `/load-data` (or automatically at startup if the project is configured that way).

* `post.json` — Academy package information (post metadata, title, content, status, price, etc.)
* `post_meta.json` — Package metadata including latitude/longitude, sports tags, canonical sport names, and other custom fields
* `order_addresses.json` — Customer addresses (latitude/longitude per order/customer)
* `order_product_lookup.json` — Purchase history linking customers (or orders) to package IDs
* `customer_lookup.json` — Customer information and username lookup

**Recommended placement:** `./data/post.json`, `./data/post_meta.json`, `./data/order_addresses.json`, `./data/order_product_lookup.json`, `./data/customer_lookup.json`.

**Note:** The `POST /load-data` endpoint accepts a `data_dir` query parameter (default: `data`) to point to a different folder.

### Run the API

Start the API locally with:

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open interactive docs at: `http://localhost:8000/docs`

---

## API Endpoints

> All recommendation endpoints return a JSON response with the following top-level keys: `status`, `message`, `count`, `data`, `distance_range`.

### Health Check

```
GET /health
```

Returns a small JSON indicating the API is running and whether data is loaded.

**Example response:**

```json
{ "status": "success", "message": "OK", "data_loaded": true }
```

---

### Load Data

```
POST /load-data?data_dir=data
```

Loads JSON files from the given directory into memory. Use this if you updated files on disk and want the API to reload them without restarting.

---

### Recommend for Existing Customer

```
POST /recommend/existing-customer
```

**Request Body:**

```json
{
  "username": "john_doe",
  "max_distance_km": 10.0,
  "use_ors": false,
  "ors_api_key": null,
  "ors_profile": "driving-car"
}
```

**Behavior:**

* Looks up the customer by `username` (case-insensitive) in `customer_lookup.json`.
* Finds package(s) the customer previously purchased using `order_product_lookup.json` and `order_addresses.json` to obtain customer location(s).
* Finds nearby packages (within `max_distance_km`) that **match the same sport(s)** as previously purchased packages.
* Excludes packages the customer already purchased and excludes draft/test posts.
* Sorts results by distance (Haversine or ORS, depending on `use_ors`).

**Example (Python):**

```python
import requests
resp = requests.post("http://localhost:8000/recommend/existing-customer", json={
    "username": "jane_smith",
    "max_distance_km": 15.0,
    "use_ors": True,
    "ors_api_key": "YOUR_ORS_API_KEY",
    "ors_profile": "driving-car"
})
print(resp.json())
```

---

### Recommend for New Customer

```
POST /recommend/new-customer
```

**Request Body (examples accepted):**

```json
{
  "user_lat": 28.6139,
  "user_lon": 77.2090,
  "preferred_sports": ["football", "cricket"],
  "use_ors": false,
  "ors_api_key": null,
  "ors_profile": "driving-car"
}
```

or

```json
{
  "user_lat": 28.6139,
  "user_lon": 77.2090,
  "preferred_sports": "football, swimming",
  "use_ors": false
}
```

**Behavior:**

* Accepts `preferred_sports` as a list or comma-separated string. Matching is case-insensitive and supports synonyms (e.g., `soccer` → `football`).
* Finds **all** packages that match any preferred sport, sorts them by distance from (`user_lat`, `user_lon`) using Haversine or ORS routing.

**Example (Python):**

```python
import requests
resp = requests.post("http://localhost:8000/recommend/new-customer", json={
    "user_lat": 28.6139,
    "user_lon": 77.2090,
    "preferred_sports": "football, swimming",
    "use_ors": False
})
for rec in resp.json()["data"][:5]:
    print(rec["post_title"], rec["distance_km"])
```

---

### Other Endpoints

* `GET /sports/canonical` — List all supported canonical sport names and common variations.
* `GET /customers/search?username=john` — Search customers (fuzzy/case-insensitive match).
* `GET /packages/count` — Get a total count of packages currently loaded.

---

## OpenRouteService (ORS) Integration

### What is ORS?

OpenRouteService (ORS) provides real-world routing distances and travel times using road networks. It is more accurate than straight-line (Haversine) distance because it accounts for roads, transport mode, and routing constraints.

### ORS Routing Profiles

Common profiles supported by ORS (use `ors_profile` in requests):

**Driving:**

* `driving-car` (default)
* `driving-hgv`

**Walking / Foot:**

* `foot-walking`
* `foot-hiking`

**Cycling:**

* `cycling-regular`
* `cycling-road`
* `cycling-mountain`
* `cycling-electric`

**Public Transport:**

* `public-transport`

> Choose the profile that best matches how your customers will travel to the academy.

### Getting an ORS API Key

1. Sign up at [https://openrouteservice.org/](https://openrouteservice.org/)
2. Generate an API key in your dashboard
3. Free tier: \~200 requests/day (check ORS for exact limits)

**Security note:** Never hard-code API keys in public repositories. Provide keys with environment variables or secure vaults.

### Using ORS in Requests

Add `use_ors: true` and `ors_api_key: "YOUR_KEY"` to your request body to use ORS routing. Set `ors_profile` to change transport mode. The API batches ORS lookups and falls back to Haversine if ORS fails.

---

## Supported Sports

The API recognizes the following canonical sports and common variations (matching is case-insensitive):

* **Football** — soccer, football
* **Cricket** — cricket, cricketing
* **Basketball** — basketball
* **Swimming** — swimming, swim, water
* **Horse Riding** — horse riding, equestrian, horse-riding
* **Tennis** — lawn tennis, tennis, YMCA
* **Badminton** — badminton
* **Skating** — skating, ice skating, roller skating
* **Dance** — dance, dancing
* **Music** — music, vocal, instrument
* **Taekwondo** — taekwondo, tae kwon do, tkd
* **Acting** — acting, drama, theatre, theater
* **Languages** — language, spoken English, language classes
* **Public Speaking** — public speaking, communication, debate
* **Martial Arts** — martial arts, MMA, karate, judo, kung fu, aikido, kickboxing
* **Yoga** — yoga, meditation, pranayama
* **Padel** — padel, paddle tennis
* **Pickleball** — pickleball, pickle ball
* **Roller Hockey** — roller hockey, roller-hockey
* **Boxing** — boxing, kick boxing, kickboxing

> The canonical list is accessible via `GET /sports/canonical`.

---

## Response Format

All recommendation endpoints return a structure similar to this:

```json
{
  "status": "success",
  "message": "Found 15 recommendations",
  "count": 15,
  "data": [
    {
      "ID": "123",
      "post_title": "Football Academy Delhi",
      "address": "Connaught Place, New Delhi",
      "distance_km": 2.5,
      "matched_sports": "football",
      "latitude": 28.6139,
      "longitude": 77.2090,
      "_price": "5000"
    }
  ],
  "distance_range": { "min_km": 1.2, "max_km": 9.8 }
}
```

---

## Configuration & Filtering Logic

**Distance Calculation Options:**

* `haversine` (default): fast, straight-line distance.
* `ors` (when `use_ors=true`): routing distance/time via ORS.

**Filtering rules:**

* **Existing customers:** Search near previously purchased academy location(s), filter packages matching previously purchased sports, limit by `max_distance_km`, exclude previously purchased packages.
* **New customers:** Find all packages that match preferred sports and sort by distance to the supplied coordinates.

---

## Technical Notes

* Direct (Haversine) distance is calculated using the standard Haversine formula.
* ORS requests are batched (up to 50 locations per batch) for efficiency.
* Case-insensitive matching is used for usernames and sports.
* The system handles common input formats for `preferred_sports` (list or CSV string).
* The API gracefully falls back to Haversine if ORS calls fail or the API key is invalid.

---

## Common Issues & Troubleshooting

**"Data not loaded"**

* Call `POST /load-data?data_dir=data` or restart the API after placing JSON files in the `data/` folder.

**Empty recommendations**

* Confirm the `username` exists in `customer_lookup.json` (for existing-customer requests).
* Ensure valid lat/lon coordinates in `post_meta.json` and `order_addresses.json`.
* Verify `preferred_sports` values are among the supported sports or their variations.

**ORS API errors**

* Confirm your ORS API key and quotas.
* Validate coordinate pairs are within valid ranges (lat: -90..90, lon: -180..180).
* The API will automatically fall back to Haversine on ORS failure.

---

## Production Considerations

When moving to production, consider the following improvements:

* Replace in-memory JSON loading with a proper database (Postgres, ElasticSearch, etc.).
* Add authentication and API keys for protected endpoints.
* Add rate limiting and request throttling to protect ORS usage.
* Implement caching of ORS responses (use a TTL) to reduce external calls and cost.
* Add detailed request/response logging and structured logs for observability.
* Add input validation (pydantic models) and sanitize incoming data.
* Add CI/CD, containerization (Docker), and a deployment strategy (K8s, ECS, etc.).

---
