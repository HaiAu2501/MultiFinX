import sys
from pathlib import Path
from typing import Dict, List, Any, Annotated, Callable, cast
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.graph import CompiledGraph
from langgraph.pregel import Pregel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from agent_nodes import (
    # Expert nodes - Group 1: Market Analysis
    market_analyst_node, technical_analyst_node, fundamental_analyst_node,
    sentiment_analyst_node, economic_indicators_node,
    
    # Expert nodes - Group 2: Financial Analysis
    financial_statement_node, financial_ratio_node, valuation_node,
    cash_flow_node, capital_structure_node,
    
    # Expert nodes - Group 3: Sectoral Analysis
    banking_finance_node, real_estate_node, consumer_goods_node,
    industrial_node, technology_node,
    
    # Expert nodes - Group 4: External Factors
    global_markets_node, geopolitical_risk_node, regulatory_framework_node,
    monetary_policy_node, demographic_trends_node,
    
    # Expert nodes - Group 5: Strategy
    game_theory_node, risk_management_node, portfolio_optimization_node,
    asset_allocation_node, investment_psychology_node,
    
    # Group summarizer nodes
    market_analysis_group_summarizer, financial_analysis_group_summarizer,
    sectoral_analysis_group_summarizer, external_factors_group_summarizer,
    strategy_group_summarizer,
    
    # Final synthesizer
    final_synthesizer,
    
    # State types
    InputState, OutputState, AgentState
)

def create_expert_group_graph(group_name: str, expert_nodes: List[Callable], summarizer_node: Callable) -> CompiledGraph:
    """
    Create a graph for a group of experts.
    
    Args:
        group_name: Name of the group
        expert_nodes: List of expert node functions
        summarizer_node: Group summarizer function
        
    Returns:
        Compiled graph for the expert group
    """
    # Create workflow for the group
    workflow = StateGraph(AgentState)
    
    # Add all expert nodes
    expert_names = [f"expert_{i}" for i in range(len(expert_nodes))]
    for i, expert_node in enumerate(expert_nodes):
        workflow.add_node(expert_names[i], expert_node)
    
    # Add summarizer node
    summarizer_name = f"{group_name}_summarizer"
    workflow.add_node(summarizer_name, summarizer_node)
    
    # Connect the nodes
    if expert_names:
        # Connect START to first expert
        workflow.add_edge(START, expert_names[0])
        
        # Connect experts sequentially
        for i in range(len(expert_names) - 1):
            workflow.add_edge(expert_names[i], expert_names[i + 1])
        
        # Connect last expert to summarizer
        workflow.add_edge(expert_names[-1], summarizer_name)
    else:
        # If no experts, connect START directly to summarizer
        workflow.add_edge(START, summarizer_name)
    
    # Connect summarizer to END
    workflow.add_edge(summarizer_name, END)
    
    # Compile the graph
    return workflow.compile()

# Create expert group graphs
market_analysis_group_graph = create_expert_group_graph(
    "market_analysis",
    [
        market_analyst_node, technical_analyst_node, fundamental_analyst_node,
        sentiment_analyst_node, economic_indicators_node
    ],
    market_analysis_group_summarizer
)

financial_analysis_group_graph = create_expert_group_graph(
    "financial_analysis",
    [
        financial_statement_node, financial_ratio_node, valuation_node,
        cash_flow_node, capital_structure_node
    ],
    financial_analysis_group_summarizer
)

sectoral_analysis_group_graph = create_expert_group_graph(
    "sectoral_analysis",
    [
        banking_finance_node, real_estate_node, consumer_goods_node,
        industrial_node, technology_node
    ],
    sectoral_analysis_group_summarizer
)

external_factors_group_graph = create_expert_group_graph(
    "external_factors",
    [
        global_markets_node, geopolitical_risk_node, regulatory_framework_node,
        monetary_policy_node, demographic_trends_node
    ],
    external_factors_group_summarizer
)

strategy_group_graph = create_expert_group_graph(
    "strategy",
    [
        game_theory_node, risk_management_node, portfolio_optimization_node,
        asset_allocation_node, investment_psychology_node
    ],
    strategy_group_summarizer
)

def create_main_graph() -> CompiledGraph:
    """
    Create the main graph that orchestrates all the expert groups.
    
    Returns:
        Compiled main graph
    """
    # Create main workflow with specified input and output
    main_workflow = StateGraph(AgentState, input=InputState, output=OutputState)
    
    # Create initializer node to copy question into separate keys for each group
    def initializer(state):
        # Copy the original question to separate keys for each group
        original_question = state.get("question", "")
        return {
            "question_1": original_question,  # For market analysis group
            "question_2": original_question,  # For financial analysis group
            "question_3": original_question,  # For sectoral analysis group
            "question_4": original_question,  # For external factors group
            "question_5": original_question,  # For strategy group
        }
    
    # Add the initializer node
    main_workflow.add_node("initializer", initializer)
    
    # Add each group graph as a node
    main_workflow.add_node("market_analysis_group", market_analysis_group_graph)
    main_workflow.add_node("financial_analysis_group", financial_analysis_group_graph)
    main_workflow.add_node("sectoral_analysis_group", sectoral_analysis_group_graph)
    main_workflow.add_node("external_factors_group", external_factors_group_graph)
    main_workflow.add_node("strategy_group", strategy_group_graph)
    
    # Add final synthesizer node
    main_workflow.add_node("final_synthesis", final_synthesizer)
    
    # Connect the nodes sequentially but with multiple parallel paths
    # First, connect START to the initializer
    main_workflow.add_edge(START, "initializer")

    # Connect cleaner to all analysis groups
    main_workflow.add_edge("initializer", "market_analysis_group")
    main_workflow.add_edge("initializer", "financial_analysis_group")
    main_workflow.add_edge("initializer", "sectoral_analysis_group")
    main_workflow.add_edge("initializer", "external_factors_group")
    
    # Create a join node to wait for all analysis groups to complete
    def join_node(state):
        # Return an empty dictionary as we don't need to modify state
        return {}
    
    main_workflow.add_node("join", join_node)
    
    # Connect all analysis groups to the join node
    main_workflow.add_edge("market_analysis_group", "join")
    main_workflow.add_edge("financial_analysis_group", "join")
    main_workflow.add_edge("sectoral_analysis_group", "join")
    main_workflow.add_edge("external_factors_group", "join")
    
    # Connect join to strategy group (which runs after all analysis is done)
    main_workflow.add_edge("join", "strategy_group")
    
    # Connect strategy group to final synthesis
    main_workflow.add_edge("strategy_group", "final_synthesis")
    main_workflow.add_edge("final_synthesis", END)
    
    # Compile the main graph
    return main_workflow.compile()

# Create the main graph
main_graph = create_main_graph()