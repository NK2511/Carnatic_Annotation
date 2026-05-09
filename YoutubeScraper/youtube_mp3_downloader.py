import yt_dlp
import os

def download_audio(video_urls, output_folder="downloaded_mp3s"):
    """
    Downloads audio from YouTube videos and converts them to MP3.

    Args:
        video_urls (list): List of YouTube video URLs.
        output_folder (str): Directory to save the MP3 files.
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
        'noplaylist': True, 
        
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    }

    import shutil
    if not shutil.which("ffmpeg") and not shutil.which("ffmpeg.exe"):
        # Check if ffmpeg is in the sibling demucs_env/Scripts folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ffmpeg_dir = os.path.join(script_dir, "..", "demucs_env", "Scripts")
        local_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        
        if os.path.exists(local_ffmpeg):
            # Add to PATH so yt-dlp can find it
            os.environ["PATH"] += os.pathsep + os.path.abspath(ffmpeg_dir)
        else:
            print("Error: FFmpeg not found. Please install FFmpeg to convert audio to MP3.")
            print("You can download it from: https://gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip")
            print(f"Extract it and place ffmpeg.exe and ffprobe.exe in {ffmpeg_dir} or add them to your system PATH.")
            return

    # Sequential download to avoid HTTP 403 Forbidden
    import time
    import random
    import copy

    # Remove empty/invalid items
    valid_items = [x for x in video_urls if x]
    
    print(f"Starting sequential download of {len(valid_items)} videos...")
    
    for i, item in enumerate(valid_items):
        url = item
        custom_name = None
        
        # Check if item is a tuple: (url, custom_name)
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            url = item[0]
            custom_name = item[1]
        
        if not url or not isinstance(url, str) or url.strip() == "":
            continue

        print(f"\n[{i+1}/{len(valid_items)}] Processing: {url}")
        
        try:
            current_opts = copy.deepcopy(ydl_opts)
            
            # Identify source file as specific as possible to avoid caching issues
            current_opts.update({
                'sleep_interval': 3,
                'max_sleep_interval': 10,
            })
            
            if custom_name:
                # yt-dlp automatically adds the extension based on the conversion
                # We just need to ensure we don't double it up if the user PROVIDED it
                if custom_name.lower().endswith('.mp3'):
                    base_name = custom_name[:-4]
                else:
                    base_name = custom_name
                    
                # Use specific filename template. %(ext)s will be filled by the converter (mp3)
                current_opts['outtmpl'] = os.path.join(output_folder, f"{base_name}.%(ext)s")
                print(f"Target Filename: {base_name}.mp3")
            
            with yt_dlp.YoutubeDL(current_opts) as ydl:
                ydl.download([url])
                print(f"✅ Successfully downloaded.")
                
            # Sleep between requests
            wait_time = random.uniform(5, 12)
            print(f"Sleeping for {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")

if __name__ == "__main__":
    # Example usage:
    # Add your YouTube links here
    links = [
        ("https://www.youtube.com/watch?v=JHBh-AwIiOY", "Keeravani Kaligiyunte_01"),
        ("https://www.youtube.com/watch?v=L441ShYg14g", "Keeravani Kaligiyunte_02"),
        ("https://www.youtube.com/watch?v=aG8HX4RP2gs", "Keeravani Kaligiyunte_03"),
        ("https://www.youtube.com/watch?v=_Frdne-nsJ0", "Keeravani Kaligiyunte_04"),
        ("https://www.youtube.com/watch?v=rL93plL5lgU", "Keeravani Varumu_Losagi_01"),
        ("https://www.youtube.com/watch?v=MUyUUJw4_XA", "Keeravani Amma_Vani_01"),
        ("https://www.youtube.com/watch?v=DSXB_KgZm2c", "Keeravani Amma_Vani_02"),
        ("https://www.youtube.com/watch?v=WV9RyqlUfds", "Keeravani Alogaye_Rukmini_01"),
        ("https://www.youtube.com/watch?v=OVkg5eLDFrY", "Keeravani Jagadambha_01"),
        ("https://www.youtube.com/watch?v=4kcdmRB6pjc", "Keeravani Velava_01"),
        ("https://www.youtube.com/watch?v=_1A3XzU8uFM", "Keeravani Velava_02"),
        ("https://www.youtube.com/watch?v=k_Fdp5RcENI", "Keeravani Velava_03"),
        ("https://www.youtube.com/watch?v=dxNvjkvdPMw", "Keeravani Vetri_Pera_01"),
        ("https://www.youtube.com/watch?v=3zzTYMMBrZg", "Keeravani Maya_Vithai_01"),
        ("https://www.youtube.com/watch?v=G3UAX455idI", "Keeravani Hayi_Hayi_01"),
        ("https://www.youtube.com/watch?v=l7MQ1BGLsQo", "Keeravani Innamum_sandehapadalaamo_01"),
        ("https://www.youtube.com/watch?v=Il_rLfQjBcY", "Keeravani Devi_Neeye_Thunai_01 "),
        ("https://www.youtube.com/watch?v=8VHJYvl61Qo", "Keeravani Innamum_sandehapadalaamo_02"),

    ]
    
    if links:
        download_audio(links)
        print("All downloads complete.")
    else:
        print("No links provided. Please add YouTube links to the 'links' list.")
