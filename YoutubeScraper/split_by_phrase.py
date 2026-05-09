import os
import whisper
import warnings
import subprocess
import json
from fuzzywuzzy import process, fuzz # You might need to install 'fuzzywuzzy' and 'python-Levenshtein'

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Ensure ffmpeg from sibling demucs_env folder is in PATH for Whisper
script_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_dir = os.path.join(script_dir, "..", "demucs_env", "Scripts")
if os.path.exists(os.path.join(ffmpeg_dir, "ffmpeg.exe")):
    os.environ["PATH"] += os.pathsep + os.path.abspath(ffmpeg_dir)

def find_time_for_phrase(segments, phrase, search_type="start"):
    """
    Searches for a phrase in the Whisper segments and returns the timestamp.
    search_type: "start" returns the start time of the matching segment.
                 "end" returns the end time of the matching segment.
    """
    best_ratio = 0
    best_time = None
    best_text = ""
    
    # Simple linear search for best match
    # Concatenate all text to find the general area? 
    # Or strict segment matching?
    # Let's try segment-by-segment fuzzy match first. 
    # Note: Phrases might span segments. This is tricky. 
    # For now, we assume the phrase is distinctive enough to match a single segment or we search sliding window?
    # Let's clean the phrase
    
    target_clean = phrase.lower().strip().replace("...", "")
    
    for seg in segments:
        seg_text = seg['text'].lower()
        ratio = fuzz.partial_ratio(target_clean, seg_text)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_text = seg['text']
            if search_type == "start":
                best_time = seg['start']
            else:
                best_time = seg['end']
    
    # Heuristic: If match is too weak, warn user
    if best_ratio < 80:
        print(f"Warning: Low match confidence ({best_ratio}%) for phrase: '{phrase}'")
        print(f"Best match found: '{best_text}'")
        
    return best_time

def get_audio_duration(file_path):
    """Gets audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        file_path
    ]
    try:
        # Check sibling demucs_env folder for ffprobe
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ffmpeg_dir = os.path.join(script_dir, "..", "demucs_env", "Scripts")
        local_ffprobe = os.path.join(ffmpeg_dir, "ffprobe.exe")
        if os.path.exists(local_ffprobe):
            cmd[0] = local_ffprobe
    
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    except Exception as e:
        print(f"Error getting duration: {e}")
        return None

def split_audio_by_text(audio_path, parts_config, model=None):
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    # Check for cached transcript with timestamps (JSON)
    json_path = os.path.splitext(audio_path)[0] + "_whisper_data.json"
    segments = None
    
    if os.path.exists(json_path):
        print(f"Loading cached transcript from {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                segments = data['segments']
        except Exception as e:
            print(f"Error loading cache: {e}")
            
    if segments is None:
        # 1. Transcribe with timestamps
        print(f"Transcribing {audio_path} to find timestamps... (this takes time)")
        
        if model is None:
            print("Loading Whisper model...")
            model = whisper.load_model("base")
            
        result = model.transcribe(audio_path, verbose=False)
        segments = result['segments']
        
        # Save cache
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Saved transcript cache to {json_path}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    # Get total duration for the last segment
    total_duration = get_audio_duration(audio_path)
    
    output_dir = os.path.join(os.path.dirname(audio_path), "text_split_chapters")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Check for ffmpeg
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_dir = os.path.join(script_dir, "..", "demucs_env", "Scripts")
    ffmpeg_cmd = "ffmpeg"
    local_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if os.path.exists(local_ffmpeg):
        ffmpeg_cmd = local_ffmpeg

    print("\nCalculating split points...")
    for i, part in enumerate(parts_config):
        title = part['title']
        start_phrase = part['start_text']
        end_phrase = part['end_text']
        
        print(f"\nPart: {title}")
        
        # Find start
        start_time = find_time_for_phrase(segments, start_phrase, "start")
        if start_time is None:
            print("  Could not find start phrase! Skipping.")
            continue
            
        # Find end
        # Special handling for the LAST segment: End of segment MUST be end of episode
        is_last_segment = (i == len(parts_config) - 1)
        
        if is_last_segment and total_duration:
            print(f"  Last segment detected. Overriding end time to file duration: {total_duration}s")
            end_time = total_duration
        else:
            end_time = find_time_for_phrase(segments, end_phrase, "end")
            
        if end_time is None:
            print("  Could not find end phrase! Skipping.")
            continue
            
        print(f"  Time Range: {start_time:.2f}s to {end_time:.2f}s")
        
        # Validation: Check if start < end
        if start_time >= end_time:
            print(f"  Error: Start time ({start_time:.2f}s) is >= End time ({end_time:.2f}s). Skipping segment.")
            print(f"  Likely cause: Phrase matching failed or incorrect order.")
            continue

        # Output filename
        clean_title = title.replace(" ", "_").replace(":", "").replace("?", "").replace(",", "")
        output_file = os.path.join(output_dir, f"{clean_title}.mp3")
        
        # Split
        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i", audio_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c", "copy",
            output_file
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  Created: {output_file}")


if __name__ == "__main__":
    
    episode1_config = [
        {
            "title": "01_Cosmic_Mechanics_and_the_Mathematics_of_Yugas",
            "start_text": "It is in time that we exist",
            "end_text": "upward moment of human consciousness"
        },
        {
            "title": "02_The_Etheric_Atmosphere_and_the_Nature_of_the_Story",
            "start_text": "So, if you go by this, here starts this Satt Yuga",
            "end_text": "stairway to the divine"
        },
        {
            "title": "03_Internal_Fires_Agni_and_The_Limits_of_the_Brain",
            "start_text": "The time and the impact of time upon the system is very, very important",
            "end_text": "It cannot go further"
        },
        {
            "title": "04_Energy_Kalki_and_The_Evolution_of_Perception",
            "start_text": "but at the same time yogic sciences once again say",
            "end_text": "still have allowed it to live within you"
        }
    ]

    episode2_config = [
        {
            "title": "05_The_Origins_of_Mahabharata_and_the_Birth_of_Mercury",
            "start_text": "The great sage, who was also known as Vyasa",
            "end_text": "life will come your way"
        },
        {
            "title": "06_The_Lunar_Dynasty_Nahushas_Fall_and_Kachas_Sacrifice",
            "start_text": "Such Buddha grew up",
            "end_text": "He is not good"
        },
        {
            "title": "07_The_Secret_of_Sanjivini_and_the_Rise_of_the_Kurus",
            "start_text": "But Deviani is heartbroken",
            "end_text": "from which both the Pandavas and the Kauravas come"
        },
        {
            "title": "08_The_Science_of_Sages_Blessings_and_Curses",
            "start_text": "The whole story is about this clan of people",
            "end_text": "Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes."
        }
    ]

    episode3_config = [
        {
            "title": "09_The_Legend_of_Shakuntala_and_Bharata",
            "start_text": "Puru became the king",
            "end_text": "wisdom and balance"
        },
        {
            "title": "10_Shantanu_Ganga_and_the_Birth_of_Bhishma",
            "start_text": "So Bharata was celebrated",
            "end_text": "coronated him as the Yuvaraj or as the next king"
        },
        {
            "title": "11_The_Terrible_Vow_and_Ambas_Revenge",
            "start_text": "Now that Devavrata was there",
            "end_text": "you will have it"
        },
        {
            "title": "12_Bhishmas_Dharma_and_the_Identity_of_Bharat",
            "start_text": "So she sat there and left her body",
            "end_text": "mispronounced word"
        }
    ]

    episode4_config = [
        {
            "title": "13_The_Yadava_Council_and_the_Prophecy_of_Kamsa",
            "start_text": "There's another important event taking shape elsewhere",
            "end_text": "kill all of them"
        },
        {
            "title": "14_The_Birth_of_Krishna_and_the_Joy_of_Life",
            "start_text": "Soldiers went out slaughtering every infant",
            "end_text": "never low battery"
        },
        {
            "title": "15_Vyasa_the_Blind_Kings_Lineage_and_the_Pandavas_Birth",
            "start_text": "Ambika and Ambaliqa",
            "end_text": "we are not using the mantra anymore"
        },
        {
            "title": "16_The_Dark_Omen_of_the_Kauravas_Birth",
            "start_text": "So these five boys grew up as Pancha Pandavas",
            "end_text": "in Hastinapur"
        },
        {
            "title": "17_Krishnas_Sadhana_and_the_Nature_of_Nara_Narayana",
            "start_text": "Sadguru, I want to know a little bit about Gandhari",
            "end_text": "glowed with wisdom for all ages"
        }
    ]

    episode5_config = [
        {
            "title": "18_The_Death_of_Pandu_and_Return_to_Hastinapur",
            "start_text": "the five brothers, have grown up well in the forest",
            "end_text": "official beginning of their hatred happened"
        },
        {
            "title": "19_Shakuni_The_Poisoning_of_Bhima_and_the_Nagapashana",
            "start_text": "when both of them got into the wrestling ring",
            "end_text": "one day a roxessor, a wild man"
        },
        {
            "title": "20_The_Wax_Palace_Escape_and_Bakasoora",
            "start_text": "who is almost on the edge of being a beast", # Context: Start of Hidimba/Bakasura arc
            "end_text": "Bakaashura"
        }
    ]
    
    episode6_config = [
        {
            "title": "21_Revenge_The_Rise_of_Draupadi_and_Drishtadyumna",
            "start_text": "So almost a year after the burning down",
            "end_text": "a very formidable enemy is growing in the neighborhood"
        },
        {
            "title": "22_The_Swayamvara_Karna_Shamed_and_Arjunas_Victory",
            "start_text": "Draupadi wants his daughter to be married",
            "end_text": "Arjuna picked up the bow and created howak around him"
        },
        {
            "title": "23_The_Five_Husbands_and_Return_to_Hastinapur",
            "start_text": "then the yadva was moored in to control",
            "end_text": "where nothing will happen"
        }
    ]
    
    episode7_config = [
        {
            "title": "24_The_Burning_of_Khandava_Forest_and_Indraprastha",
            "start_text": "But out of his goodness",
            "end_text": "Narada came"
        },
        {
            "title": "25_The_Slaying_of_Jarasandha_and_Shishupala",
            "start_text": "When Narada comes",
            "end_text": "mass killing would have happened"
        },
        {
            "title": "26_The_Dice_Game_Conspiracy_and_Draupadis_Laughter",
            "start_text": "So after the Raja Svereya Agna Shukra",
            "end_text": "they went back"
        }
    ]

    episode8_config = [
        {
            "title": "27_The_Garish_Hall_and_the_Game_of_Dice",
            "start_text": "Duryodhana wanted to build an equally good assembly hall",
            "end_text": "Duriyodhana left the sabbhah"
        },
        {
            "title": "28_Exile_The_Forest_Life_and_Akshaya_Patra",
            "start_text": "So Pandavas got back everything",
            "end_text": "Next day it'll come again"
        },
        {
            "title": "29_Durvasas_Visit_and_Krishnas_Miracle",
            "start_text": "So blessed with this, guests and the Brahmins",
            "end_text": "I don't want you to be a compassionate case"
        }
    ]

    episode9_config = [
        {
            "title": "30_The_Cow_Counting_Picnic_and_Arjunas_Magnanimity",
            "start_text": "So life goes on in the forest",
            "end_text": "Narakasura spirit has entered Karna"
        },
        {
            "title": "31_Arjunas_Exile_Chitrangada_and_Subhadra",
            "start_text": "Once he heard this, suddenly enthusiasm",
            "end_text": "A man, a man, a man"
        },
        {
            "title": "32_Lessons_in_the_Wild_Hanuman_and_the_Yaksha_Prashna",
            "start_text": "They see that they're beginning to have too many guests",
            "end_text": "making sure you get the point"
        }
    ]

    episode10_config = [
        {
            "title": "33_The_Incognito_Exile_and_the_Slaying_of_Kichaka",
            "start_text": "Now that the twelve years are over",
            "end_text": "So he said it is Bima. That's gone."
        },
        {
            "title": "34_The_Battle_of_Virata_and_Krishnas_Choice",
            "start_text": "So the cow of our army decided to ride",
            "end_text": "That one wrong choice."
        },  
        {
            "title": "35_The_Cosmic_Form_and_Mystics_of_Kurukshetra",
            "start_text": "When war becomes inevitable, when Krishna comes to so peace",
            "end_text": "It is an angled bob."
        }
    ]

    episode11_config = [
        {
            "title": "36_The_Kurukshetra_War_Begins_and_Bhishmas_Fall",
            "start_text": "Bhisma is the commander of the Kaurvas",
            "end_text": "and war continued like this"
        },
        {
            "title": "37_Abhimanyus_Death_and_the_Killing_of_Jayadratha",
            "start_text": "It is the day 13 which kind of turned things over",
            "end_text": "So it continues like this on 15th day I think"
        },
        {
            "title": "38_Karnas_Fall_Duryodhanas_Death_and_the_Aftermath",
            "start_text": "On 15th day or so they even break the night rules",
            "end_text": "turmoil of the situation."
        }
    ]

    episode12_config = [
        {
            "title": "39_The_Three_Aspects_of_Success_and_the_Logic_of_Destiny",
            "start_text": "Whatever that may be. There are three aspects.",
            "end_text": "What can be changed is your business, isn't it?"
        },
        {
            "title": "40_The_Iron_Statue_The_Forest_Fire_and_The_Fall_of_Dwarka",
            "start_text": "Having finished with the war, now they all move towards Hastinapur",
            "end_text": "Nobody had ever seen him like that"
        },
        {
            "title": "41_Bhishmas_Statecraft_and_the_Nature_of_Duryodhana",
            "start_text": "Here, he was sitting like half a life",
            "end_text": "he cries"
        }
    ]

    episode13_config = [
        {
            "title": "42_The_Destruction_of_the_Yadavas_and_Krishnas_Departure",
            "start_text": "he other was inebrated, not just with drink",
            "end_text": "Arjuna's grandson."
        },
        {
            "title": "43_The_Mahaprasthan_The_Dog_Heaven_and_Hell",
            "start_text": "These six people, the Pandavas and Draupadi",
            "end_text": "Now it is Jaya."
        },
        {
            "title": "44_Parikshit_The_Snake_Sacrifice_and_the_End_of_the_Cycle",
            "start_text": "The idea of external conquest is a silly idea",
            "end_text": "That's Mahabharata's story for you."
        },
        {
            "title": "45_QA_Nahusha_Arjunas_Death_and_the_Art_of_Forgiveness",
            "start_text": "Now you are in Kwaris.",
            "end_text": "You remember every bitter moment of your life"
        }
    ]

    episode14_config = [
        {
            "title": "46_The_Philosophy_of_Dharma_and_Matsyanyaya",
            "start_text": "Most people tend to wanting to think in terms of right and wrong",
            "end_text": "probe it with your intellect and ask questions"
        },
        {
            "title": "47_Establishing_Your_Dharma_and_Leaving_the_Dead",
            "start_text": "You said that everybody can have their own life",
            "end_text": "This is freedom, isn't it?"
        },
        {
            "title": "48_Krishnas_Dream_The_Many_Faces_of_False_Dharma",
            "start_text": "And Krishna gives you an insight into his own values",
            "end_text": "you cannot pass in the very nature of things"
        },
        {
            "title": "49_The_Reptilian_Brain_Gandharis_Choice_and_Cosmic_Cycles",
            "start_text": "Param Sathguru, as the story goes on",
            "end_text": "I keep a certain fire between me and the world"
        }
    ]

    episode15_config = [
        {
            "title": "50_Living_Through_the_Drama_and_the_Essence_of_Gita",
            "start_text": "There were many moments in the past few days",
            "end_text": "I look at them the way they are."
        },
        {
            "title": "51_Cultural_Context_The_Dice_and_the_Dravidian_Resistance",
            "start_text": "In the early days, they said, innocentious to give their dyes at home",
            "end_text": "It's too long, a period to resist."
        },
        {
            "title": "52_History_as_Living_Truth_and_Krishnas_Political_Spirituality",
            "start_text": "You can keep Mahabharata, sense will come into you",
            "end_text": "We cannot give up the mission."
        },
        {
            "title": "53_Creating_Your_God_and_Riding_the_Cosmic_Wave",
            "start_text": "Namaste, this is Guruji.",
            "end_text": "How to write the way?"
        }
    ]

    episode16_config = [
        {
            "title": "54_The_Birth_of_Karna_and_the_Three_Curses",
            "start_text": "Human beings who get labeled when we look back",
            "end_text": "bound for life"
        },
        {
            "title": "55_Loyalty_Lineage_and_the_Sacrifice_of_the_Armor",
            "start_text": "You will see this loyalty",
            "end_text": "that is all he wanted"
        },
        {
            "title": "56_The_Fall_of_Karna_and_the_Fairness_of_Existence",
            "start_text": "So both of them got what they wanted",
            "end_text": "Life doesn't work like that"
        }
    ]

    audio_file1 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_1_Yugas tides of time.mp3"
    audio_file2 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_2_Of Boons and Curses.mp3"
    audio_file3 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_3_Bhisma.mp3"
    audio_file4 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_4_Entry of Krishna.mp3"
    audio_file5 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_5_The Poison Of Hate.mp3"
    audio_file6 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_6_Five Husbands.mp3"
    audio_file7 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_7_Indraprastha The City of Woe.mp3"
    audio_file8 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_8_The Gamble Where Humanity Was Lost.mp3"
    audio_file9 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_9_Vanvasa Parv.mp3"
    audio_file10 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_10_Agnathavasa Parv.mp3"
    audio_file11 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_11_Kurukshetra Beyond Fair and Unfair.mp3"
    audio_file12 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_12_Dance Of Destiny.mp3"
    audio_file13 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_13_Vanaprastha - End of an Era.mp3"
    audio_file14 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_14_Dharma Adharma.mp3"
    audio_file15 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_15_Living through the Story.mp3"
    audio_file16 = r"C:\Desktop\Python\CarnaticAnnotater\downloaded_mp3s\Mahabharat_ep_16_Karna-The Fate's Child.mp3"

    # List of all (audio_path, config) pairs
    all_tasks = [
        (audio_file1, episode1_config),
        (audio_file2, episode2_config),
        (audio_file3, episode3_config),
        (audio_file4, episode4_config),
        (audio_file5, episode5_config),
        (audio_file6, episode6_config),
        (audio_file7, episode7_config),
        (audio_file8, episode8_config),
        (audio_file9, episode9_config),
        (audio_file10, episode10_config),
        (audio_file11, episode11_config),
        (audio_file12, episode12_config),
        (audio_file13, episode13_config),
        (audio_file14, episode14_config),
        (audio_file15, episode15_config),
        (audio_file16, episode16_config),
    ]

    print(f"Loaded {len(all_tasks)} audio tasks.")
    
    # Load model once
    print("Loading Whisper model (base)...")
    model = whisper.load_model("base")

    for audio_path, config in all_tasks:
        print(f"\nProcessing File: {audio_path}")
        split_audio_by_text(audio_path, config, model=model)

