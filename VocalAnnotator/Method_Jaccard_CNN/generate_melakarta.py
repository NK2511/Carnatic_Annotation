"""
Generate Melakarta Raga Signatures mapped to Physical Note Names used in VocalAnnotator.
Physical Notes: Sa, Ri1, Ri2(Ga1), Ga2(Ri3), Ga3, Ma1, Ma2, Pa, Da1, Da2(Ni1), Ni2(Da3), Ni3.
"""

# Physical Note Mapping
# Key: Logical Melakarta Name
# Value: Physical Note Name (as used in carva_*.csv and extractor)
NOTE_MAP = {
    'S': 'Sa',
    'R1': 'Ri1',
    'R2': 'Ri2',
    'R3': 'Ga2', # Shatsruti Rishabham = Sadharana Gandharam
    'G1': 'Ri2', # Shudda Gandharam = Chatushruti Rishabham
    'G2': 'Ga2',
    'G3': 'Ga3',
    'M1': 'Ma1',
    'M2': 'Ma2',
    'P': 'Pa',
    'D1': 'Da1',
    'D2': 'Da2',
    'D3': 'Ni2', # Shatsruti Dhaivatam = Kaisiki Nishadam
    'N1': 'Da2', # Shudda Nishadam = Chatushruti Dhaivatam
    'N2': 'Ni2',
    'N3': 'Ni3'
}

# The 6 Chakras for R/G
RG_PAIRS = [
    ('R1', 'G1'), # 1. Shuddha R, Shuddha G
    ('R1', 'G2'), # 2. Shuddha R, Sadharana G
    ('R1', 'G3'), # 3. Shuddha R, Antara G
    ('R2', 'G2'), # 4. Chatushruti R, Sadharana G
    ('R2', 'G3'), # 5. Chatushruti R, Antara G
    ('R3', 'G3')  # 6. Shatsruti R, Antara G
]

# The 6 Chakras for D/N
DN_PAIRS = [
    ('D1', 'N1'), # 1. Shuddha D, Shuddha N
    ('D1', 'N2'), # 2. Shuddha D, Kaisiki N
    ('D1', 'N3'), # 3. Shuddha D, Kakali N
    ('D2', 'N2'), # 4. Chatushruti D, Kaisiki N
    ('D2', 'N3'), # 5. Chatushruti D, Kakali N
    ('D3', 'N3')  # 6. Shatsruti D, Kakali N
]

def get_melakarta_name(index):
    # Simplified list of names (or just use Index if names are too many)
    # Ideally should use proper names.
    # List from Wikipedia or standard source.
    melakarta_names = [
        "Kanakangi", "Ratnangi", "Ganamurti", "Vanaspati", "Manavati", "Tanarupi",
        "Senavati", "Hanumatodi", "Dhenuka", "Natakapriya", "Kokilapriya", "Rupavati",
        "Gayakapriya", "Vakulabharanam", "Mayamalavagowla", "Chakravakam", "Suryakantam", "Hatakambari",
        "Jhankaradhwani", "Natabhairavi", "Keeravani", "Kharaharapriya", "Gaurimanohari", "Varunapriya",
        "Mararanjani", "Charukesi", "Sarasangi", "Harikambhoji", "Dhee-Shankarabharanam", "Naganandini",
        "Yagapriya", "Ragavardhini", "Gangeyabhushani", "Vagadheeswari", "Shulini", "Chalanata",
        "Salagam", "Jalarnavam", "Jhalavarali", "Navaneetam", "Pavani", "Raghupriya",
        "Gavambhodi", "Bhavapriya", "Shubhapantuvarali", "Shadvidhamargini", "Suvarnangi", "Divyamani",
        "Dhavalambari", "Namanarayani", "Kamavardhini", "Ramapriya", "Gamanashrama", "Vishwambhari",
        "Shamalangi", "Shanmukhapriya", "Simhendramadhyamam", "Hemavati", "Dharmavati", "Neetimati",
        "Kantamani", "Rishabhapriya", "Latangi", "Vachaspati", "Mechakalyani", "Chitrambari",
        "Sucharitra", "Jyotiswarupini", "Dhatuvardhini", "Nasikabhushani", "Kosalam", "Rasikapriya"
    ]
    if 1 <= index <= 72:
        return melakarta_names[index-1]
    return f"Melakarta-{index}"

RAGA_SIGNATURES = {}

count = 1
for m in ['M1', 'M2']:
    for r, g in RG_PAIRS:
        for d, n in DN_PAIRS:
            logical_notes = ['S', r, g, m, 'P', d, n]
            physical_notes = {NOTE_MAP[note] for note in logical_notes}
            
            # Map canonical names to our dataset usage
            name = get_melakarta_name(count)
            # Aliases for dataset matching
            if name == "Mayamalavagowla": name = "Mayamalavagowlai"
            if name == "Hanumatodi": name = "Thodi" # Assuming Thodi is Hanumatodi
            if name == "Dhee-Shankarabharanam": name = "Shankarabharanam"
            if name == "Mechakalyani": name = "Kalyani"
            
            RAGA_SIGNATURES[name] = physical_notes
            count += 1

# Generate Python file content
content = "MELAKARTA_RAGAS = {\n"
for name, notes in RAGA_SIGNATURES.items():
    sorted_notes = sorted(list(notes))
    content += f"    '{name}': {set(sorted_notes)},\n"
content += "}\n"

with open("melakarta_signatures.py", "w") as f:
    f.write(content)

print(f"Generated melakarta_signatures.py with {len(RAGA_SIGNATURES)} Ragas.")
