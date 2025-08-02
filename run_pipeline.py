#!/usr/bin/env python
# Titanic Survival Prediction Pipeline Runner

import os
import subprocess
import time
import sys

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80)

def run_command(command, description):
    """Run a shell command and print its output."""
    print_header(description)
    print(f"Running: {command}\n")
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            
        # Wait for process to complete
        process.wait()
        
        # Check return code
        if process.returncode != 0:
            print(f"\nCommand failed with return code {process.returncode}")
            return False
        else:
            print(f"\nCommand completed successfully")
            return True
    except Exception as e:
        print(f"\nError executing command: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    print_header("Checking Dependencies")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("Error: requirements.txt not found")
            return False
        
        # Install dependencies
        return run_command("pip install -r requirements.txt", "Installing Dependencies")
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False

def check_data():
    """Check if the required data files exist."""
    print_header("Checking Data Files")
    
    required_files = ["train.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: The following required files are missing: {', '.join(missing_files)}")
        return False
    else:
        print("All required data files are present")
        return True

def run_pipeline():
    """Run the complete Titanic survival prediction pipeline."""
    start_time = time.time()
    
    print_header("Titanic Survival Prediction Pipeline")
    print("This script will run the complete pipeline for the Titanic survival prediction project.")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\nFailed to install dependencies. Exiting.")
        return False
    
    # Step 2: Check data
    if not check_data():
        print("\nRequired data files are missing. Exiting.")
        return False
    
    # Step 3: Train the model
    if not run_command("python model.py", "Training Model"):
        print("\nModel training failed. Exiting.")
        return False
    
    # Step 4: Create visualizations
    if not run_command("python visualize.py", "Creating Visualizations"):
        print("\nVisualization creation failed, but continuing...")
    
    # Step 5: Check if test.csv exists for predictions
    if os.path.exists("test.csv"):
        if not run_command("python predict.py", "Making Predictions"):
            print("\nPrediction failed, but continuing...")
    else:
        print("\ntest.csv not found. Skipping prediction step.")
    
    # Calculate total runtime
    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    
    print_header("Pipeline Complete")
    print(f"Total runtime: {int(minutes)} minutes and {seconds:.2f} seconds")
    
    # Print summary
    print("\nSummary:")
    print("- Model trained and saved as 'titanic_model.joblib'")
    
    if os.path.exists("visualizations"):
        visualization_files = os.listdir("visualizations")
        print(f"- {len(visualization_files)} visualizations created in 'visualizations' folder")
    
    if os.path.exists("predictions.csv"):
        print("- Predictions saved as 'predictions.csv'")
    
    print("\nNext steps:")
    print("1. Review the model performance metrics")
    print("2. Examine the visualizations in the 'visualizations' folder")
    print("3. If predictions were made, check 'predictions.csv'")
    
    return True

if __name__ == "__main__":
    try:
        success = run_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)