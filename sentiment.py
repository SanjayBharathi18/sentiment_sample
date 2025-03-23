import os
import csv
import re
import time
import requests
from collections import Counter
from googleapiclient.discovery import build
from transformers import pipeline
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# Set your YouTube API key
YOUTUBE_API_KEY = "AIzaSyBhVLN2fbJUTVwEbO3vTMPNPGotaczPKlw"

# Initialize YouTube API
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Initialize RoBERTa sentiment analysis pipeline
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment", truncation=True)

# Function to get Channel ID from Channel Name
def get_channel_id(channel_name):
    try:
        search_response = youtube.search().list(
            q=channel_name,
            type="channel",
            part="id,snippet",
            maxResults=1
        ).execute()
        return search_response["items"][0]["id"]["channelId"]
    except:
        print("⚠️ Error: Invalid Channel Name. Please enter a valid YouTube channel.")
        return None

# Function to get the last 5 videos from a channel
def get_last_videos(channel_id):
    search_response = youtube.search().list(
        channelId=channel_id,
        part="id,snippet",
        order="date",
        maxResults=5
    ).execute()

    videos = []
    for item in search_response["items"]:
        if item["id"]["kind"] == "youtube#video":
            videos.append({"title": item["snippet"]["title"], "videoId": item["id"]["videoId"]})
    return videos

# Function to get comments from a video
def get_video_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(comment["textDisplay"])
        request = youtube.commentThreads().list_next(request, response)

    return comments

# Function to analyze sentiment of comments
def analyze_sentiments(comments):
    results = []
    for comment in comments:
        truncated_comment = comment[:512]  # Ensure within max length
        result = sentiment_pipeline(truncated_comment)[0]
        sentiment_score = result["score"] if result["label"] == "LABEL_2" else -result["score"] if result["label"] == "LABEL_0" else 0
        label = "Positive" if result["label"] == "LABEL_2" else "Negative" if result["label"] == "LABEL_0" else "Neutral"
        results.append([comment, label, round(sentiment_score, 3)])
    return results

# Function to process all 5 videos and save results
def process_videos(channel_name, save_csv=True):
    # Get channel ID
    channel_id = get_channel_id(channel_name)
    if not channel_id:
        return {"error":"Invalid Channel Name"}

    # Get last 5 videos
    videos = get_last_videos(channel_id)
    all_comments = []
    sentiment_summary = {}
    negative_comments = []

    for video in videos:
        print(f"\n📥 Fetching comments from: {video['title']}")

        # Get comments
        comments = get_video_comments(video["videoId"])
        if not comments:
            print("⚠️ No comments found for this video.")
            continue

        print(f"✅ Retrieved {len(comments)} comments. Performing sentiment analysis...")

        # Analyze sentiment
        analyzed_comments = analyze_sentiments(comments)

        # Store results
        for comment, sentiment, score in analyzed_comments:
            all_comments.append({
                "video_title": video["title"],
                "comment": comment,
                "sentiment": sentiment,
                "score": score
            })
            if sentiment == "Negative":
                negative_comments.append(comment)

        # Calculate sentiment percentages
        total_comments = len(analyzed_comments)
        positive_count = sum(1 for _, label, _ in analyzed_comments if label == "Positive")
        negative_count = sum(1 for _, label, _ in analyzed_comments if label == "Negative")
        neutral_count = total_comments - (positive_count + negative_count)

        sentiment_summary[video["title"]] = {
            "positive": round((positive_count / total_comments) * 100, 2),
            "negative": round((negative_count / total_comments) * 100, 2),
            "neutral": round((neutral_count / total_comments) * 100, 2),
        }

    # Save to CSV if needed
    if save_csv:
        csv_filename = "youtube_comments_sentiment.csv"
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Video Title", "Comment", "Sentiment", "Score"])
            for row in all_comments:
                writer.writerow([row["video_title"], row["comment"], row["sentiment"], row["score"]])

        print(f"\n✅ Data saved to {csv_filename}")

        summary_filename = "sentiment_summary.csv"
        with open(summary_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Video Title", "Positive (%)", "Negative (%)", "Neutral (%)"])
            for video, sentiments in sentiment_summary.items():
                writer.writerow([video, sentiments["positive"], sentiments["negative"], sentiments["neutral"]])

        print(f"\n✅ Sentiment summary saved to {summary_filename}")

    # Extract common negative words
    most_common_negative_words = extract_most_common_words(negative_comments)

    # Return JSON structure
    result_json = {
        "channel_name": channel_name,
        "comments_analysis": all_comments,
        "sentiment_summary": sentiment_summary,
        "negative_comment_keywords": most_common_negative_words
    }

    return result_json


# Function to extract most common words from negative comments (without stopwords)
def extract_most_common_words(comments):
    all_words = []
    for comment in comments:
        words = re.findall(r'\b\w+\b', comment.lower())  # Extract words
        filtered_words = [word for word in words if word not in STOPWORDS]  # Remove stopwords
        all_words.extend(filtered_words)

    word_counts = Counter(all_words)
    return dict(word_counts.most_common(10))

# Main function
if __name__ == "__main__":
    channel_name = input("Enter YouTube Channel Name: ")
    process_videos(channel_name)

