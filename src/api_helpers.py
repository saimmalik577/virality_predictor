import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SOCIALINSIDER_API_KEY")
BASE_URL = "https://app.socialinsider.io/api"
PROJECT_NAME = "virality_project"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_profiles(project_name=PROJECT_NAME):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "socialinsider_api.get_profiles",
        "params": {
            "projectname": project_name
        }
    }
    res = requests.post(BASE_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
    data = res.json()
    for project in data["result"]:
        if project["projectname"] == project_name:
            return project["profiles"]
    raise ValueError("❌ Project not found or has no profiles")

def get_profile_id_and_type_by_name(name, profiles):
    for profile in profiles:
        if profile["id"].lower() == name.lower():
            return profile["id"], profile["profile_type"]
    print(f"❌ Profile not found in project '{PROJECT_NAME}': {name}")
    return None, None

def get_posts(profile_id, profile_type, start_ts, end_ts):
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "socialinsider_api.get_posts",
        "params": {
            "id": profile_id,
            "profile_type": profile_type,
            "projectname": PROJECT_NAME,
            "date": {
                "start": start_ts,
                "end": end_ts,
                "timezone": "Europe/Berlin"
            },
            "from": 0,
            "size": 1000
        }
    }

    response = requests.post(BASE_URL, headers=HEADERS, json=payload)
    try:
        data = response.json()
        posts = data.get("resp", {}).get("posts", [])
        print(f"✅ {len(posts)} posts fetched.")
        return posts
    except Exception as e:
        print("❌ Error while parsing posts:", e)
        print("Raw response:", response.text)
        return []
