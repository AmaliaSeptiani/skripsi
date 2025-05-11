import os
from googleapiclient.discovery import build
import openpyxl
from langdetect import detect, LangDetectException
import re  # Untuk regex mendeteksi link

# Set up API key and service
API_KEY = ''
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def contains_link(text):
    """Cek apakah teks mengandung link."""
    link_pattern = r'(https?://\S+|www\.\S+|\S+\.(com|id|net|org|co|info))'
    return bool(re.search(link_pattern, text))

def fetch_comments(video_id, min_likes=0, max_results=100):
    """
    Fetch comments from a YouTube video filtered by minimum likes, language detection, and no links.

    :param video_id: The YouTube video ID.
    :param min_likes: Minimum number of likes a comment must have to be included.
    :param max_results: Maximum number of comments to fetch per page.
    :return: A list of comment texts and their like counts.
    """
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )

    while request:
        response = request.execute()
        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment = snippet["textDisplay"]
            like_count = snippet.get("likeCount", 0)

            try:
                if (like_count >= min_likes 
                    and detect(comment) == "id" 
                    and not contains_link(comment)):  # Tambahkan pengecekan tidak mengandung link
                    comments.append((comment, like_count))
            except LangDetectException:
                continue  # Skip jika gagal deteksi bahasa

        # Get the next page of results
        request = youtube.commentThreads().list_next(request, response)

    return comments

def save_to_excel(data, filename):
    """
    Save a list of comments and their like counts to an Excel file.

    :param data: List of tuples containing comments and like counts.
    :param filename: Excel file name.
    """
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "YouTube Comments"
    sheet.append(["Video ID", "Comments", "Likes"])  # Header row

    for video_id, comments in data.items():
        for comment, like_count in comments:
            sheet.append([video_id, comment, like_count])

    workbook.save(filename)
    print(f"✅ Data saved to {filename}")

if __name__ == "__main__":
    video_ids = ['QydJ4CjJsOk', 'gM8PlLp4xXQ', 'yztD26Z7FzY', 'MSmv4ImbZjk', 'If90WVTuh8Y', 'lvp8rc4YWiM', 'FiBXRmiMgX8', '3OY0CD49KzI']
    min_likes = 2

    all_comments = {}

    for video_id in video_ids:
        comments = fetch_comments(video_id, min_likes=min_likes)
        all_comments[video_id] = comments

    if all_comments:
        save_to_excel(all_comments, "dataset.xlsx")
    else:
        print("⚠️ No comments found or an error occurred.")
