import os
import time
from data_processor import DataProcessor
from vlm_client import VLMClient

def main():
    # Define file paths
    # The current working directory is expected to be c:/hackerrank or the project root
    input_csv = "claims.csv"
    output_csv = "output.csv"
    
    # Check if we have the Gemini API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Please set it before running the pipeline.")
        return

    print("Initializing components...")
    # Initialize processor and VLM client
    processor = DataProcessor(input_csv=input_csv, output_csv=output_csv)
    vlm_client = VLMClient()
    
    print(f"Loading data from {input_csv}...")
    try:
        df = processor.load_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    total_claims = len(df)
    print(f"Total claims to process: {total_claims}")

    # Process each claim
    user_history_db = {}
    for index, row in df.iterrows():
        user_id = row.get("user_id", f"unknown_user_{index}")
        user_claim = row.get("user_claim", "")
        claim_object = row.get("claim_object", "unknown object")
        
        print(f"\n[{index + 1}/{total_claims}] Processing claim for user_id: {user_id}")
        
        # Extract image paths
        image_paths = processor.get_image_paths(row)
        
        # get history
        history = user_history_db.get(user_id, [])
        
        # Evaluate using VLM
        result_dict = vlm_client.evaluate_claim(
            user_claim=user_claim,
            claim_object=claim_object,
            image_paths=image_paths,
            base_dir=".", # Assuming paths are relative to where main.py is run
            user_history=history
        )
        
        # update history
        user_history_db.setdefault(user_id, []).append({
            "claim": user_claim,
            "status": result_dict.get('claim_status')
        })
        
        print(f"  -> Claim Status: {result_dict.get('claim_status')}")
        print(f"  -> Valid Image: {result_dict.get('valid_image')}")
        
        # Append result to dataframe
        processor.append_result(index, result_dict)
        
        # Continuous saving: save after every 5 rows and at the very end
        if (index + 1) % 5 == 0:
            print(f"Saving intermediate progress...")
            processor.save_output()
            
        time.sleep(13)

    # Final save
    print("\nProcessing complete. Saving final output...")
    processor.save_output()
    print("Done!")

if __name__ == "__main__":
    main()
