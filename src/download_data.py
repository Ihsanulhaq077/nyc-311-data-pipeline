# Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-17
"""Download a sample of NYC 311 service requests from the public Socrata API."""
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nyc_311.csv"
URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"


def download_data(limit: int = 20000) -> Path:
    """Download live raw records from NYC Open Data."""
    RAW.parent.mkdir(parents=True, exist_ok=True)
    params = {"$limit": limit, "$order": "created_date DESC"}
    print(f"Fetching {limit} records from {URL}...")
    response = requests.get(URL, params=params, timeout=120)
    response.raise_for_status()
    if not response.text.strip():
        raise RuntimeError("The data API returned an empty response.")
    RAW.write_text(response.text, encoding="utf-8")
    print(f"Downloaded {len(response.text.splitlines()) - 1} rows to {RAW}")
    return RAW


if __name__ == "__main__":
    download_data()
