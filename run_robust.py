import subprocess
import time
import sys
import os

# Base command (reduced to 2 workers for stability)
cmd = [
    sys.executable, "feature_engineering.py",
    "--trajectories", "data/clean_flights_train/",
    "--flightlist", "prc_data/flightlist_train.parquet",
    "--fuel", "prc_data/fuel_train.parquet",
    "--output", "data/features_train.parquet",
    "--workers", "8",  # 1 worker = max stability (slower but safer)
    "--resume"
]

MAX_RETRIES = 100
retries = 0

print("🛡️  Launching ROBUST mode (Auto-Restart)")
print(f"Command: {' '.join(cmd)}")

while retries < MAX_RETRIES:
    print(f"\n[Attempt {retries + 1}/{MAX_RETRIES}] Launching...")
    
    try:
        # Build command with current blacklist
        current_cmd = cmd.copy()
        blacklist = []
        if os.path.exists("blacklist.txt"):
            with open("blacklist.txt", "r") as f:
                blacklist = [line.strip() for line in f if line.strip()]
        
        if blacklist:
            current_cmd.extend(["--ignore-flights", ",".join(blacklist)])
            print(f"🚫 Blacklist active ({len(blacklist)} flights)")

        # Run process
        result = subprocess.run(current_cmd)
        
        # If code 0, success
        if result.returncode == 0:
            print("\n✅  Processing finished successfully!")
            break
            
        # Otherwise (crash), identify the culprit
        retries += 1
        print(f"\n⚠️  Process exited with code {result.returncode}.")
        print(f"💥 TOTAL CRASHES: {retries}")
        
        # Read the flight that crashed
        if os.path.exists("current_flight.txt"):
            with open("current_flight.txt", "r") as f:
                bad_flight = f.read().strip()
            
            if bad_flight and bad_flight not in blacklist:
                print(f"💀 CULPRIT FLIGHT IDENTIFIED: {bad_flight}")
                print(f"   -> Adding to blacklist.txt")
                with open("blacklist.txt", "a") as f:
                    f.write(bad_flight + "\n")
            else:
                print("❓ Unable to identify culprit flight (or already blacklisted).")
        
        print("Restarting in 5 seconds...")
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n🛑  Manual stop by user.")
        break
    except Exception as e:
        print(f"\n❌  Unexpected error: {e}")
        break

if retries >= MAX_RETRIES:
    print("\n❌  Max retries reached. Aborting.")
