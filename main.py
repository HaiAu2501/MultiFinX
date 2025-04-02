import os
import sys
import locale
import io
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from agents.nodes import AgentState
from agents.groups import main_graph

# Force UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_system.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("investment_analysis")

# Load environment variables
load_dotenv()

def main():
    """Entry point for the investment strategy optimization system."""
    logger.info("=== INVESTMENT STRATEGY ANALYSIS AND OPTIMIZATION SYSTEM ===")
    
    # Get user question via console
    print("\nNhap cau hoi dau tu hoac muc tieu cua ban:")
    user_question = input("> ")
    
    if not user_question.strip():
        logger.error("No question or goal provided.")
        print("Vui long nhap cau hoi hoac muc tieu dau tu va chay lai.")
        sys.exit(1)
    
    # Get output folder name
    print("\nNhap ten thu muc luu tru bao cao (mac dinh: investment_strategies):")
    output_folder_name = input("> ").strip()
    if not output_folder_name:
        output_folder_name = "investment_strategies"
    
    logger.info(f"Received question: {user_question}")
    logger.info(f"Output folder: {output_folder_name}")
    
    # Create output directories
    output_dir = Path(__file__).parent / output_folder_name
    output_dir.mkdir(exist_ok=True)
    
    group_output_dir = output_dir / "group_responses"
    group_output_dir.mkdir(exist_ok=True)
    
    expert_output_dir = output_dir / "expert_responses"
    expert_output_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nAnalyzing question: {user_question}")
    print("=" * 50)
    
    # Initialize state with question and output folder
    initial_state = {
        "question": user_question,
        "output_folder": output_folder_name
    }
    
    # Run analysis with LangGraph
    logger.info("Calling analysis graph for question")
    start_time = datetime.now()
    result = main_graph.invoke(initial_state)
    end_time = datetime.now()
    
    # Log execution time
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Analysis completed in {duration:.2f} seconds")
    
    # Print final investment strategy to console
    print("\n=== INVESTMENT STRATEGY ===\n")
    print(result["final_report"])
    
    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()