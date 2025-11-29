#!/usr/bin/env python3
"""
LULC Training Automation Script
Simplifies common operations with single commands
"""

import argparse
import subprocess
import os
import sys

def run_command(cmd, description):
    """Execute command and handle errors."""
    print(f"\n{'='*60}")
    print(f"➤ {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print(f"\n❌ Error: Command failed with code {result.returncode}")
        return False
    print(f"\n✅ {description} - Complete!")
    return True

def setup():
    """Setup environment."""
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                "Installing dependencies")

def generate_dummy_data():
    """Generate synthetic test data."""
    run_command([sys.executable, "demo_setup.py"],
                "Generating dummy test data")

def train_single_year(year):
    """Train for a single year."""
    cmd = [sys.executable, "model/train.py",
           "--data_dir", "LULC_Continual_Learning_Data",
           "--year", str(year)]
    run_command(cmd, f"Training model for year {year}")

def train_continual(start_year, end_year):
    """Train continuously for multiple years."""
    for year in range(start_year, end_year + 1):
        success = train_single_year(year)
        if not success:
            print(f"❌ Training stopped at year {year}")
            return False
    print(f"✅ Continual training complete for {start_year}-{end_year}")
    return True

def generate_predictions(year, image_path):
    """Generate predictions for an image."""
    checkpoint = f"model/checkpoint_{year}.pth"
    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint not found: {checkpoint}")
        return False
    
    output_path = f"output/classification_{year}.tif"
    os.makedirs("output", exist_ok=True)
    
    cmd = [sys.executable, "model/inference.py",
           "--checkpoint", checkpoint,
           "--image", image_path,
           "--output", output_path,
           "--save_probabilities"]
    
    return run_command(cmd, f"Generating predictions for {image_path}")

def validate_model(year):
    """Validate model performance."""
    checkpoint = f"model/checkpoint_{year}.pth"
    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint not found: {checkpoint}")
        return False
    
    cmd = [sys.executable, "model/validate.py",
           "--checkpoint", checkpoint,
           "--data_dir", "LULC_Continual_Learning_Data",
           "--year", str(year)]
    
    return run_command(cmd, f"Validating model for year {year}")

def generate_dashboard_assets():
    """Generate prediction assets for the dashboard."""
    print("\n" + "="*60)
    print("GENERATING DASHBOARD ASSETS")
    print("="*60)
    
    data_dir = "LULC_Continual_Learning_Data"
    years = range(2018, 2026) # 2018 to 2025
    
    for year in years:
        checkpoint = f"model/checkpoint_{year}.pth"
        image_path = f"{data_dir}/Sentinel2_{year}.tif"
        
        if not os.path.exists(checkpoint):
            print(f"⚠️  Skipping {year}: Checkpoint not found ({checkpoint})")
            continue
            
        if not os.path.exists(image_path):
            # Try to find any image for that year
            potential_images = [f for f in os.listdir(data_dir) if f.startswith(f"Sentinel2_{year}")]
            if potential_images:
                image_path = os.path.join(data_dir, potential_images[0])
            else:
                print(f"⚠️  Skipping {year}: No satellite image found")
                continue
        
        print(f"Processing year {year}...")
        generate_predictions(year, image_path)
    
    print("\n✅ Dashboard assets generation complete!")
    return True

def quick_test():
    """Quick test: setup → dummy data → train → validate."""
    print("\n" + "="*60)
    print("QUICK TEST SEQUENCE")
    print("="*60)
    
    steps = [
        ("Installing dependencies", lambda: setup()),
        ("Generating test data", lambda: generate_dummy_data()),
        ("Training model for 2019", lambda: train_single_year(2019)),
        ("Validating model", lambda: validate_model(2019)),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Test failed at: {step_name}")
            return False
    
    print("\n" + "="*60)
    print("✅ QUICK TEST PASSED!")
    print("="*60)
    print("\nYour implementation is working correctly!")
    print("Next: Use the full training pipeline for real data.")
    return True

def full_workflow():
    """Complete workflow: setup → data → continual training → validation."""
    print("\n" + "="*60)
    print("FULL PRODUCTION WORKFLOW")
    print("="*60)
    
    # Note: This assumes data already exists
    # For real data, run: python gee_scripts/export_data.py first
    
    print("""
This workflow requires:
1. Real satellite data in LULC_Continual_Learning_Data/
2. Download from Google Earth Engine: python gee_scripts/export_data.py

Then run:
    python runbook.py --continual 2018 2020
    
This will:
- Train for 2018
- Train for 2019 (with EWC constraint from 2018)
- Train for 2020 (with EWC constraint from 2019)
    """)

def main():
    parser = argparse.ArgumentParser(
        description='LULC Training Runbook - Automate common tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runbook.py --test              # Quick validation test
  python runbook.py --setup             # Install dependencies
  python runbook.py --dummy             # Generate test data
  python runbook.py --train 2019        # Train single year
  python runbook.py --continual 2018 2020  # Train 2018→2019→2020
  python runbook.py --predict 2019 image.tif  # Generate predictions
  python runbook.py --validate 2019     # Check model accuracy
        """
    )
    
    parser.add_argument('--test', action='store_true',
                        help='Run quick validation test')
    parser.add_argument('--setup', action='store_true',
                        help='Setup environment (install dependencies)')
    parser.add_argument('--dummy', action='store_true',
                        help='Generate synthetic test data')
    parser.add_argument('--train', type=int, metavar='YEAR',
                        help='Train model for specific year')
    parser.add_argument('--continual', type=int, nargs=2, metavar=('START', 'END'),
                        help='Train continuously from START_YEAR to END_YEAR')
    parser.add_argument('--predict', type=str, nargs=2, metavar=('YEAR', 'IMAGE'),
                        help='Generate predictions for given image')
    parser.add_argument('--validate', type=int, metavar='YEAR',
                        help='Validate model for specific year')
    parser.add_argument('--dashboard', action='store_true',
                        help='Generate all assets for the dashboard')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute requested command
    if args.test:
        quick_test()
    elif args.setup:
        setup()
    elif args.dummy:
        generate_dummy_data()
    elif args.train:
        train_single_year(args.train)
    elif args.continual:
        train_continual(args.continual[0], args.continual[1])
    elif args.predict:
        generate_predictions(int(args.predict[0]), args.predict[1])
    elif args.validate:
        validate_model(args.validate)
    elif args.dashboard:
        generate_dashboard_assets()

if __name__ == "__main__":
    main()
