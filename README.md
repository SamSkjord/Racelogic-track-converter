# RaceLogic Converter

This script converts **RaceLogic** `.CIR` track files to **Google Earth-compatible** `.KMZ` files using a mix of real and synthetic data based on the track metadata.

It scans the full RaceLogic directory structure, handles different file types and formats, and ensures output even for incomplete tracks using heuristics.

---

## 🔧 Requirements

- Python 3.6+
- No external dependencies

---

## 📥 Setup Instructions

1. **Install the official RaceLogic track database**  
   Download the **Video Setup & Circuit Tools** installer from:  
   👉 https://www.vboxmotorsport.co.uk/index.php/en/customer-ct-track-database

   This will install the tracks into:  
   `C:\ProgramData\Racelogic`

2. **Clone this repo or copy the script**

3. **Run the converter**

   ```bash
   python racelogic_converter.py C:\ProgramData\Racelogic path\to\output_directory

---

## 📝 TODO:

- Start/Finish line orientation is off
- Optimisations

---
