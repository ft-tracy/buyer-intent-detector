"""Step 1: collect public comments from a YouTube channel using the YouTube Data API v3.

Usage:
    export YOUTUBE_API_KEY="your-key"        (Windows PowerShell: $env:YOUTUBE_API_KEY="your-key")
    python collect_comments.py --max-videos 60 --max-per-video 200

Output: data/raw_comments.csv
Privacy: author names and IDs are NOT saved. Only the comment text, date, likes and video title.
"""
import argparse
import os
import sys
import time

import pandas as pd
import requests

API_KEY = os.environ.get("YOUTUBE_API_KEY")
CHANNEL_ID = "UCBs79ESUMkJAtJYqznPL3KA"  # AMG Realtors channel (from the website footer link)
BASE = "https://www.googleapis.com/youtube/v3"


def api_get(endpoint, **params):
    params["key"] = API_KEY
    resp = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    return resp.json()


def list_videos(max_videos):
    """Every channel has an 'uploads' playlist whose ID is the channel ID with UC -> UU."""
    uploads_playlist = "UU" + CHANNEL_ID[2:]
    videos, token = [], None
    while len(videos) < max_videos:
        data = api_get("playlistItems", part="snippet", playlistId=uploads_playlist,
                       maxResults=50, pageToken=token)
        for item in data.get("items", []):
            snip = item["snippet"]
            videos.append({"video_id": snip["resourceId"]["videoId"], "video_title": snip["title"]})
        token = data.get("nextPageToken")
        if not token:
            break
    return videos[:max_videos]


def list_comments(video, max_per_video):
    rows, token = [], None
    while len(rows) < max_per_video:
        try:
            data = api_get("commentThreads", part="snippet", videoId=video["video_id"],
                           maxResults=100, order="time", textFormat="plainText", pageToken=token)
        except RuntimeError as err:
            if "commentsDisabled" in str(err) or "403" in str(err):
                print(f"  comments disabled or blocked: {video['video_title'][:50]}")
                return rows
            raise
        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "comment_id": item["id"],
                "video_id": video["video_id"],
                "video_title": video["video_title"],
                "text": top["textDisplay"],
                "published_at": top["publishedAt"],
                "like_count": top.get("likeCount", 0),
            })
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)
    return rows[:max_per_video]


def main():
    if not API_KEY:
        sys.exit("Set the YOUTUBE_API_KEY environment variable first.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=60)
    parser.add_argument("--max-per-video", type=int, default=200)
    args = parser.parse_args()

    videos = list_videos(args.max_videos)
    print(f"Found {len(videos)} videos")
    all_rows = []
    for i, video in enumerate(videos, 1):
        rows = list_comments(video, args.max_per_video)
        print(f"[{i}/{len(videos)}] {len(rows):4d} comments  {video['video_title'][:60]}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).drop_duplicates("comment_id")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_comments.csv", index=False)
    print(f"\nSaved {len(df)} comments to data/raw_comments.csv")


if __name__ == "__main__":
    main()
