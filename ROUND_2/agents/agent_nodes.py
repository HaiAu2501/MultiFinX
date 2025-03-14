from typing import Dict, List, Any, Annotated, TypedDict, cast, Optional
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Now import from the correct paths
from experts.group_1 import (
    MARKET_ANALYST, TECHNICAL_ANALYST, FUNDAMENTAL_ANALYST, 
    SENTIMENT_ANALYST, ECONOMIC_INDICATORS_EXPERT
)
from experts.group_2 import (
    FINANCIAL_STATEMENT_ANALYST, FINANCIAL_RATIO_EXPERT, VALUATION_EXPERT,
    CASH_FLOW_ANALYST, CAPITAL_STRUCTURE_EXPERT
)
from experts.group_3 import (
    BANKING_FINANCE_EXPERT, REAL_ESTATE_EXPERT, CONSUMER_GOODS_EXPERT,
    INDUSTRIAL_EXPERT, TECHNOLOGY_EXPERT
)
from experts.group_4 import (
    GLOBAL_MARKETS_EXPERT, GEOPOLITICAL_RISK_ANALYST, REGULATORY_FRAMEWORK_EXPERT,
    MONETARY_POLICY_EXPERT, DEMOGRAPHIC_TRENDS_EXPERT
)
from experts.group_5 import (
    GAME_THEORY_STRATEGIST, RISK_MANAGEMENT_EXPERT, PORTFOLIO_OPTIMIZATION_EXPERT,
    ASSET_ALLOCATION_STRATEGIST, INVESTMENT_PSYCHOLOGY_EXPERT
)

from search_tools import simple_search

# Load environment variables
load_dotenv()

# Define state structure
# Define state structure
class InputState(TypedDict):
    question: Annotated[str, "merge"]  # User question or goal

class OutputState(TypedDict):
    question_1: Annotated[str, "merge"]  # Copy of question for group 1
    question_2: Annotated[str, "merge"]  # Copy of question for group 2
    question_3: Annotated[str, "merge"]  # Copy of question for group 3
    question_4: Annotated[str, "merge"]  # Copy of question for group 4
    question_5: Annotated[str, "merge"]  # Copy of question for group 5
    analyses: Annotated[Dict[str, str], "merge"]  # Analyses from different agents
    group_1: Annotated[Dict[str, str], "merge"]   # Analyses from group 1 - Market Analysis
    group_2: Annotated[Dict[str, str], "merge"]   # Analyses from group 2 - Financial Analysis
    group_3: Annotated[Dict[str, str], "merge"]   # Analyses from group 3 - Sectoral Analysis
    group_4: Annotated[Dict[str, str], "merge"]   # Analyses from group 4 - External Factors
    group_5: Annotated[Dict[str, str], "merge"]   # Analyses from group 5 - Strategy
    group_summaries: Annotated[Dict[str, str], "merge"]  # Summaries from each group
    final_report: str                # Final synthesis report
    search_results: Annotated[Dict[str, Dict[str, Any]], "merge"]  # Search results

class AgentState(InputState, OutputState):
    """Combined state for the agent system, inheriting from both input and output states."""
    pass

def get_model():
    """
    Get the appropriate LLM based on environment config.
    Returns OpenAI model by default.
    """
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("No API key found for OpenAI")

def generate_search_queries(llm, system_prompt, question, expert_name):
    """Generate search queries based on the expert's domain and the question."""
    query_generation_prompt = f"""
    {system_prompt}
    
    Your task is to generate 3-5 specific search queries that will help gather information to answer the following question from your expert perspective:
    
    QUESTION: {question}
    
    INSTRUCTIONS:
    1. Consider what information you need as a {expert_name} to properly answer this question
    2. Create search queries that will find relevant, current information about the Vietnamese market
    3. Make your queries specific and focused
    4. Format your response as a JSON list of strings containing only the queries
    
    Format example:
    {{
        "queries": [
            "query 1",
            "query 2",
            "query 3"
        ]
    }}
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant that generates search queries."),
        HumanMessage(content=query_generation_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Extract queries from response
    try:
        content = response.content
        # Extract JSON portion
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content
            
        queries_data = json.loads(json_str)
        return queries_data.get("queries", [])
    except Exception as e:
        print(f"Error extracting queries: {e}")
        # Fallback to basic extraction
        queries = []
        for line in response.content.split("\n"):
            if line.strip().startswith('"') or line.strip().startswith("'"):
                queries.append(line.strip().strip('"\'').strip(','))
        return queries[:5]  # Limit to 5 queries

def perform_searches(queries, expert_name):
    """Perform searches with the generated queries."""
    all_results = []
    
    for i, query in enumerate(queries):
        print(f"[{expert_name}] Searching ({i+1}/{len(queries)}): {query}")
        try:
            results = simple_search(query)
            all_results.extend(results)
            print(f"  Found {len(results)} results")
        except Exception as e:
            print(f"  Search error: {e}")
    
    return all_results

def compile_search_results(results):
    """Compile search results into a formatted text."""
    compiled = "SEARCH RESULTS:\n\n"
    
    for i, result in enumerate(results):
        compiled += f"Result {i+1}:\n"
        compiled += f"Title: {result.get('title', 'No title')}\n"
        compiled += f"Link: {result.get('link', 'No link')}\n"
        compiled += f"Snippet: {result.get('snippet', 'No snippet')}\n\n"
    
    return compiled

def analyze_results(llm, system_prompt, question, search_results, expert_name):
    """Generate an expert analysis based on the search results."""
    analysis_prompt = f"""
    {system_prompt}
    
    USER QUESTION:
    {question}
    
    {search_results}
    
    INSTRUCTIONS:
    As a {expert_name}, provide a detailed analysis to answer the question based on:
    1. Your expert knowledge of the Vietnamese market
    2. The information from the search results
    
    Your analysis should:
    - Be thorough and insightful
    - Include specific recommendations where appropriate
    - Cite sources from the search results where possible
    - End with a "References" section listing your sources
    
    Format your response as a professional analysis report.
    """
    
    messages = [
        SystemMessage(content="You are a financial expert specialized in the Vietnamese market."),
        HumanMessage(content=analysis_prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content

def save_expert_analysis(expert_name, analysis, question):
    """Save the expert's analysis to a file."""
    output_dir = Path(__file__).parent / "investment_strategies" / "expert_responses"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / f"{expert_name}.txt", 'w', encoding='utf-8') as f:
        f.write(f"=== ANALYSIS FROM {expert_name.upper()} ===\n\n")
        f.write(f"Question: {question}\n\n")
        f.write(analysis)
    
    print(f"[{expert_name}] Analysis saved to {output_dir / f'{expert_name}.txt'}")

def create_expert_agent(system_prompt: str, agent_name: str, group_key: str):
    """Create an expert agent function to be used as a node in the graph."""
    # Get the corresponding question key for this group
    question_key = f"question_{group_key[-1]}"  # e.g., "question_1" for "group_1"
    
    def expert_analysis(state: AgentState) -> Dict:
        """Run expert analysis and store in state."""
        try:
            # Get values from state using the group-specific question key
            question = state.get(question_key, "")
            
            print(f"\n[DEBUG] Running {agent_name} for question: {question}")
            
            # Get LLM
            llm = get_model()
            
            # Step 1: Generate search queries
            queries = generate_search_queries(llm, system_prompt, question, agent_name)
            
            # Step 2: Perform searches
            search_results = perform_searches(queries, agent_name)
            
            # Step 3: Compile search results
            compiled_results = compile_search_results(search_results)
            
            # Step 4: Analyze results
            analysis = analyze_results(llm, system_prompt, question, compiled_results, agent_name)
            
            # Step 5: Save analysis
            save_expert_analysis(agent_name, analysis, question)
            
            # Return only the group-specific data
            return {
                group_key: {agent_name: analysis},
                "search_results": {
                    f"{agent_name}_search": {
                        "queries": queries,
                        "results": search_results
                    }
                }
            }
            
        except Exception as e:
            print(f"[ERROR] Error in {agent_name}: {str(e)}")
            return {
                group_key: {agent_name: f"Error in analysis: {str(e)}"},
                "search_results": {f"{agent_name}_search": {"error": str(e)}}
            }
    
    return expert_analysis

# Create expert nodes with their group keys
# Group 1: Market Analysis
market_analyst_node = create_expert_agent(MARKET_ANALYST, "market_analyst", "group_1")
technical_analyst_node = create_expert_agent(TECHNICAL_ANALYST, "technical_analyst", "group_1")
fundamental_analyst_node = create_expert_agent(FUNDAMENTAL_ANALYST, "fundamental_analyst", "group_1")
sentiment_analyst_node = create_expert_agent(SENTIMENT_ANALYST, "sentiment_analyst", "group_1")
economic_indicators_node = create_expert_agent(ECONOMIC_INDICATORS_EXPERT, "economic_indicators_expert", "group_1")

# Group 2: Financial Analysis
financial_statement_node = create_expert_agent(FINANCIAL_STATEMENT_ANALYST, "financial_statement_analyst", "group_2")
financial_ratio_node = create_expert_agent(FINANCIAL_RATIO_EXPERT, "financial_ratio_expert", "group_2")
valuation_node = create_expert_agent(VALUATION_EXPERT, "valuation_expert", "group_2")
cash_flow_node = create_expert_agent(CASH_FLOW_ANALYST, "cash_flow_analyst", "group_2")
capital_structure_node = create_expert_agent(CAPITAL_STRUCTURE_EXPERT, "capital_structure_expert", "group_2")

# Group 3: Sectoral Analysis
banking_finance_node = create_expert_agent(BANKING_FINANCE_EXPERT, "banking_finance_expert", "group_3")
real_estate_node = create_expert_agent(REAL_ESTATE_EXPERT, "real_estate_expert", "group_3")
consumer_goods_node = create_expert_agent(CONSUMER_GOODS_EXPERT, "consumer_goods_expert", "group_3")
industrial_node = create_expert_agent(INDUSTRIAL_EXPERT, "industrial_expert", "group_3")
technology_node = create_expert_agent(TECHNOLOGY_EXPERT, "technology_expert", "group_3")

# Group 4: External Factors
global_markets_node = create_expert_agent(GLOBAL_MARKETS_EXPERT, "global_markets_expert", "group_4")
geopolitical_risk_node = create_expert_agent(GEOPOLITICAL_RISK_ANALYST, "geopolitical_risk_analyst", "group_4")
regulatory_framework_node = create_expert_agent(REGULATORY_FRAMEWORK_EXPERT, "regulatory_framework_expert", "group_4")
monetary_policy_node = create_expert_agent(MONETARY_POLICY_EXPERT, "monetary_policy_expert", "group_4")
demographic_trends_node = create_expert_agent(DEMOGRAPHIC_TRENDS_EXPERT, "demographic_trends_expert", "group_4")

# Group 5: Strategy
game_theory_node = create_expert_agent(GAME_THEORY_STRATEGIST, "game_theory_strategist", "group_5")
risk_management_node = create_expert_agent(RISK_MANAGEMENT_EXPERT, "risk_management_expert", "group_5")
portfolio_optimization_node = create_expert_agent(PORTFOLIO_OPTIMIZATION_EXPERT, "portfolio_optimization_expert", "group_5")
asset_allocation_node = create_expert_agent(ASSET_ALLOCATION_STRATEGIST, "asset_allocation_strategist", "group_5")
investment_psychology_node = create_expert_agent(INVESTMENT_PSYCHOLOGY_EXPERT, "investment_psychology_expert", "group_5")

def create_group_summarizer(group_name: str, expert_names: List[str], group_key: str):
    """Create a group summarizer function to be used as a node in the graph."""
    # Get the corresponding question key for this group
    question_key = f"question_{group_key[-1]}"  # e.g., "question_1" for "group_1"
    
    def summarize_group(state: AgentState) -> Dict:
        """Summarize analyses from experts in the group."""
        try:
            # Extract analyses from experts in this group
            expert_analyses = ""
            
            # Get analyses from the group-specific state
            if group_key in state:
                for expert, analysis in state[group_key].items():
                    expert_analyses += f"### Analysis from {expert}:\n{analysis}\n\n"
            
            # Get search information
            search_info = ""
            for expert in expert_names:
                search_key = f"{expert}_search"
                if search_key in state.get("search_results", {}):
                    search_data = state["search_results"][search_key]
                    if "queries" in search_data:
                        search_info += f"\n### Search queries from {expert}:\n"
                        for query in search_data["queries"]:
                            search_info += f"- {query}\n"
            
            # Use group-specific question key
            question = state.get(question_key, "No question provided")
            
            print(f"\n[DEBUG] Running summarizer for {group_name} for question: {question}")
            
            llm = get_model()
            
            summary_prompt = f"""
            You are a group coordinator for the {group_name} expert team.
            
            Your task is to create a comprehensive summary of the following expert analyses to answer this user question:
            
            USER QUESTION:
            {question}
            
            EXPERT ANALYSES:
            {expert_analyses}
            
            SEARCH INFORMATION:
            {search_info}
            
            Create a thorough summary that:
            1. Highlights the key insights from all experts
            2. Identifies areas of consensus and important differences
            3. Directly answers the user's question
            4. Provides actionable investment recommendations
            5. Includes citations to sources where appropriate
            
            Format your response as a professional group analysis report.
            """
            
            messages = [
                SystemMessage(content="You are a financial analysis coordinator."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = llm.invoke(messages)
            summary = response.content
            
            # Save the summary
            output_dir = Path(__file__).parent / "investment_strategies" / "group_responses"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            clean_group_name = group_name.split("(")[0].strip() if "(" in group_name else group_name
            
            with open(output_dir / f"{clean_group_name}.txt", 'w', encoding='utf-8') as f:
                f.write(f"=== GROUP SUMMARY: {clean_group_name.upper()} ===\n\n")
                f.write(f"Question: {question}\n\n")
                f.write(summary)
            
            # Return only the group summaries
            return {
                "group_summaries": {group_name: summary}
            }
            
        except Exception as e:
            print(f"[ERROR] Error in {group_name} summarizer: {str(e)}")
            return {
                "group_summaries": {group_name: f"Error in summary: {str(e)}"}
            }
    
    return summarize_group

# Create group summarizer nodes with their respective group keys
market_analysis_group_summarizer = create_group_summarizer(
    "Phân tích Thị trường (Market Analysis)",
    ["market_analyst", "technical_analyst", "fundamental_analyst", 
     "sentiment_analyst", "economic_indicators_expert"],
    "group_1"
)

financial_analysis_group_summarizer = create_group_summarizer(
    "Phân tích Tài chính (Financial Analysis)",
    ["financial_statement_analyst", "financial_ratio_expert", "valuation_expert", 
     "cash_flow_analyst", "capital_structure_expert"],
    "group_2"
)

sectoral_analysis_group_summarizer = create_group_summarizer(
    "Phân tích Ngành (Sectoral Analysis)",
    ["banking_finance_expert", "real_estate_expert", "consumer_goods_expert", 
     "industrial_expert", "technology_expert"],
    "group_3"
)

external_factors_group_summarizer = create_group_summarizer(
    "Yếu tố Bên ngoài (External Factors)",
    ["global_markets_expert", "geopolitical_risk_analyst", "regulatory_framework_expert", 
     "monetary_policy_expert", "demographic_trends_expert"],
    "group_4"
)

strategy_group_summarizer = create_group_summarizer(
    "Lập chiến lược (Strategy)",
    ["game_theory_strategist", "risk_management_expert", "portfolio_optimization_expert", 
     "asset_allocation_strategist", "investment_psychology_expert"],
    "group_5"
)

def final_synthesizer(state: AgentState) -> Dict:
    """Create the final investment strategy report."""
    try:
        # Format all group summaries
        group_summaries_text = ""
        for group_name, summary in state.get("group_summaries", {}).items():
            group_summaries_text += f"### Summary from {group_name}:\n{summary}\n\n"
        
        # Use the original question, or any of the question copies if available
        question = state.get("question", "")
        if not question:
            # Try any of the question copies
            for i in range(1, 6):
                question_key = f"question_{i}"
                if question_key in state and state[question_key]:
                    question = state[question_key]
                    break
        
        if not question:
            question = "No question provided"
        
        print(f"\n[DEBUG] Running final synthesizer for question: {question}")
        
        llm = get_model()
        
        synthesis_prompt = f"""
        You are a chief investment strategist specialized in the Vietnamese market.
        
        Your task is to create a comprehensive investment strategy based on the following group summaries:
        
        USER QUESTION:
        {question}
        
        GROUP SUMMARIES:
        {group_summaries_text}
        
        Create a detailed investment strategy that:
        1. Directly answers the user's question
        2. Provides a market analysis and current trends
        3. Includes a strategic asset allocation recommendation
        4. Recommends specific sectors and stocks
        5. Advises on market entry timing
        6. Includes a risk management plan
        7. Provides specific actionable steps for investors
        
        Format your response as a professional investment strategy report with clear sections.
        """
        
        messages = [
            SystemMessage(content="You are a chief investment strategist for the Vietnamese market."),
            HumanMessage(content=synthesis_prompt)
        ]
        
        response = llm.invoke(messages)
        final_report = response.content
        
        # Save the final report
        output_dir = Path(__file__).parent / "investment_strategies"
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "final_investment_strategy.txt", 'w', encoding='utf-8') as f:
            f.write("=== OPTIMAL INVESTMENT STRATEGY ===\n\n")
            f.write(f"Question: {question}\n\n")
            f.write(final_report)
        
        print(f"Final investment strategy saved to {output_dir / 'final_investment_strategy.txt'}")
        
        # Return only the final_report key
        return {
            "final_report": final_report
        }
        
    except Exception as e:
        print(f"[ERROR] Error in final synthesizer: {str(e)}")
        return {
            "final_report": f"Error in final synthesis: {str(e)}"
        }