from typing import Dict, List, Any, Annotated, TypedDict, cast, Optional
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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

from agents.tools import simple_search

# Load environment variables
load_dotenv()

# Define state structure with output_folder
class InputState(TypedDict):
    question: Annotated[str, "merge"]  # User question or goal
    output_folder: Annotated[str, "merge"]  # Output folder name

class OutputState(TypedDict):
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
    Lấy mô hình LLM phù hợp dựa trên cấu hình môi trường.
    Trả về mô hình OpenAI theo mặc định.
    """
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("Không tìm thấy khóa API cho OpenAI")

def generate_search_queries(llm, system_prompt, question, expert_name):
    """Tạo các truy vấn tìm kiếm dựa trên lĩnh vực chuyên gia và câu hỏi."""
    query_generation_prompt = f"""
    {system_prompt}
    
    Nhiệm vụ của bạn là tạo 3-5 truy vấn tìm kiếm cụ thể sẽ giúp thu thập thông tin để trả lời câu hỏi sau từ góc độ chuyên gia của bạn:
    
    CÂU HỎI: {question}
    
    HƯỚNG DẪN:
    1. Xem xét những thông tin bạn cần với tư cách là một {expert_name} để trả lời câu hỏi này một cách đúng đắn
    2. Tạo các truy vấn tìm kiếm sẽ tìm thông tin liên quan, hiện tại về thị trường Việt Nam
    3. Làm cho các truy vấn của bạn cụ thể và tập trung
    4. Định dạng phản hồi của bạn dưới dạng danh sách JSON các chuỗi chỉ chứa các truy vấn
    
    Ví dụ định dạng:
    {{
        "queries": [
            "truy vấn 1",
            "truy vấn 2",
            "truy vấn 3"
        ]
    }}
    """
    
    messages = [
        SystemMessage(content="Bạn là trợ lý hữu ích tạo ra các truy vấn tìm kiếm."),
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
        print(f"Lỗi khi trích xuất truy vấn: {e}")
        # Fallback to basic extraction
        queries = []
        for line in response.content.split("\n"):
            if line.strip().startswith('"') or line.strip().startswith("'"):
                queries.append(line.strip().strip('"\'').strip(','))
        return queries[:5]  # Limit to 5 queries

def perform_searches(queries, expert_name):
    """Thực hiện tìm kiếm với các truy vấn được tạo."""
    all_results = []
    
    for i, query in enumerate(queries):
        print(f"[{expert_name}] Đang tìm kiếm ({i+1}/{len(queries)}): {query}")
        try:
            results = simple_search(query)
            all_results.extend(results)
            print(f"  Tìm thấy {len(results)} kết quả")
        except Exception as e:
            print(f"  Lỗi tìm kiếm: {e}")
    
    return all_results

def compile_search_results(results):
    """Biên soạn kết quả tìm kiếm thành văn bản định dạng."""
    compiled = "KẾT QUẢ TÌM KIẾM:\n\n"
    
    for i, result in enumerate(results):
        compiled += f"Kết quả {i+1}:\n"
        compiled += f"Tiêu đề: {result.get('title', 'Không có tiêu đề')}\n"
        compiled += f"Liên kết: {result.get('link', 'Không có liên kết')}\n"
        compiled += f"Đoạn trích: {result.get('snippet', 'Không có đoạn trích')}\n\n"
    
    return compiled

def analyze_results(llm, system_prompt, question, search_results, expert_name):
    """Tạo phân tích chuyên gia dựa trên kết quả tìm kiếm."""
    analysis_prompt = f"""
    {system_prompt}
    
    CÂU HỎI NGƯỜI DÙNG:
    {question}
    
    {search_results}
    
    HƯỚNG DẪN:
    Với tư cách là {expert_name}, hãy cung cấp phân tích chi tiết để trả lời câu hỏi dựa trên:
    1. Kiến thức chuyên môn của bạn về thị trường Việt Nam
    2. Thông tin từ kết quả tìm kiếm
    
    Phân tích của bạn cần:
    - Toàn diện và sâu sắc
    - Bao gồm các khuyến nghị cụ thể khi thích hợp
    - Trích dẫn nguồn từ kết quả tìm kiếm khi có thể
    - Kết thúc bằng phần "Tài liệu tham khảo" liệt kê các nguồn của bạn
    
    Định dạng phản hồi của bạn như một báo cáo phân tích chuyên nghiệp.
    """
    
    messages = [
        SystemMessage(content="Bạn là chuyên gia tài chính chuyên về thị trường Việt Nam."),
        HumanMessage(content=analysis_prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content

def save_expert_analysis(expert_name, analysis, question, output_folder):
    """Lưu phân tích của chuyên gia vào tệp."""
    output_dir = Path(__file__).parent / output_folder / "expert_responses"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / f"{expert_name}.txt", 'w', encoding='utf-8') as f:
        f.write(f"=== PHÂN TÍCH TỪ {expert_name.upper()} ===\n\n")
        f.write(f"Câu hỏi: {question}\n\n")
        f.write(analysis)
    
    print(f"[{expert_name}] Phân tích đã được lưu vào {output_dir / f'{expert_name}.txt'}")

def create_expert_agent(system_prompt: str, agent_name: str, group_key: str):
    """Tạo hàm agent chuyên gia để sử dụng như một node trong đồ thị."""
    def expert_analysis(state: AgentState) -> Dict:
        """Chạy phân tích chuyên gia và lưu trong state."""
        try:
            # Get values from state
            question = state.get("question", "")
            output_folder = state.get("output_folder", "investment_strategies")
            
            print(f"\n[DEBUG] Đang chạy {agent_name} cho câu hỏi: {question}")
            
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
            save_expert_analysis(agent_name, analysis, question, output_folder)
            
            # Return the group-specific data
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
            print(f"[LỖI] Lỗi trong {agent_name}: {str(e)}")
            return {
                group_key: {agent_name: f"Lỗi trong phân tích: {str(e)}"},
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
    """Tạo một hàm tổng hợp nhóm để sử dụng như một node trong đồ thị."""
    def summarize_group(state: AgentState) -> Dict:
        """Tổng hợp phân tích từ các chuyên gia trong nhóm."""
        try:
            # Extract analyses from experts in this group
            expert_analyses = ""
            
            # Get analyses from the group-specific state
            if group_key in state:
                for expert, analysis in state[group_key].items():
                    expert_analyses += f"### Phân tích từ {expert}:\n{analysis}\n\n"
            
            # Get search information
            search_info = ""
            for expert in expert_names:
                search_key = f"{expert}_search"
                if search_key in state.get("search_results", {}):
                    search_data = state["search_results"][search_key]
                    if "queries" in search_data:
                        search_info += f"\n### Truy vấn tìm kiếm từ {expert}:\n"
                        for query in search_data["queries"]:
                            search_info += f"- {query}\n"
            
            # Use original question and output folder
            question = state.get("question", "Không có câu hỏi nào được cung cấp")
            output_folder = state.get("output_folder", "investment_strategies")
            
            print(f"\n[DEBUG] Đang chạy người tổng hợp cho {group_name} cho câu hỏi: {question}")
            
            llm = get_model()
            
            summary_prompt = f"""
            Bạn là điều phối viên nhóm cho đội chuyên gia {group_name}.
            
            Nhiệm vụ của bạn là tạo một bản tóm tắt toàn diện về các phân tích chuyên gia sau đây để trả lời câu hỏi của người dùng:
            
            CÂU HỎI NGƯỜI DÙNG:
            {question}
            
            PHÂN TÍCH CHUYÊN GIA:
            {expert_analyses}
            
            THÔNG TIN TÌM KIẾM:
            {search_info}
            
            Tạo một bản tóm tắt kỹ lưỡng:
            1. Nêu bật những hiểu biết chính từ tất cả các chuyên gia
            2. Xác định các lĩnh vực đồng thuận và sự khác biệt quan trọng
            3. Trực tiếp trả lời câu hỏi của người dùng
            4. Cung cấp khuyến nghị đầu tư có thể thực hiện được
            5. Bao gồm trích dẫn nguồn khi thích hợp
            
            Định dạng phản hồi của bạn như một báo cáo phân tích nhóm chuyên nghiệp.
            """
            
            messages = [
                SystemMessage(content="Bạn là điều phối viên phân tích tài chính."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = llm.invoke(messages)
            summary = response.content
            
            # Save the summary
            output_dir = Path(__file__).parent / output_folder / "group_responses"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            clean_group_name = group_name.split("(")[0].strip() if "(" in group_name else group_name
            
            with open(output_dir / f"{clean_group_name}.txt", 'w', encoding='utf-8') as f:
                f.write(f"=== TÓM TẮT NHÓM: {clean_group_name.upper()} ===\n\n")
                f.write(f"Câu hỏi: {question}\n\n")
                f.write(summary)
            
            # Return only the group summaries
            return {
                "group_summaries": {group_name: summary}
            }
            
        except Exception as e:
            print(f"[LỖI] Lỗi trong người tổng hợp {group_name}: {str(e)}")
            return {
                "group_summaries": {group_name: f"Lỗi trong tóm tắt: {str(e)}"}
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
    """Tạo báo cáo chiến lược đầu tư cuối cùng."""
    try:
        # Format all group summaries
        group_summaries_text = ""
        for group_name, summary in state.get("group_summaries", {}).items():
            group_summaries_text += f"### Tóm tắt từ {group_name}:\n{summary}\n\n"
        
        # Use the original question and output folder
        question = state.get("question", "Không có câu hỏi nào được cung cấp")
        output_folder = state.get("output_folder", "investment_strategies")
        
        print(f"\n[DEBUG] Đang chạy người tổng hợp cuối cùng cho câu hỏi: {question}")
        
        llm = get_model()
        
        synthesis_prompt = f"""
        Bạn là trưởng chiến lược đầu tư chuyên về thị trường Việt Nam.
        
        Nhiệm vụ của bạn là tạo một chiến lược đầu tư toàn diện dựa trên các bản tóm tắt nhóm sau:
        
        CÂU HỎI NGƯỜI DÙNG:
        {question}
        
        TÓM TẮT NHÓM:
        {group_summaries_text}
        
        Tạo một chiến lược đầu tư chi tiết:
        1. Trực tiếp trả lời câu hỏi của người dùng
        2. Cung cấp phân tích thị trường và xu hướng hiện tại
        3. Bao gồm khuyến nghị phân bổ tài sản chiến lược
        4. Khuyến nghị các ngành và cổ phiếu cụ thể
        5. Tư vấn về thời điểm tham gia thị trường
        6. Bao gồm kế hoạch quản lý rủi ro
        7. Cung cấp các bước cụ thể có thể thực hiện cho nhà đầu tư
        
        Định dạng phản hồi của bạn như một báo cáo chiến lược đầu tư chuyên nghiệp với các phần rõ ràng.
        """
        
        messages = [
            SystemMessage(content="Bạn là trưởng chiến lược đầu tư cho thị trường Việt Nam."),
            HumanMessage(content=synthesis_prompt)
        ]
        
        response = llm.invoke(messages)
        final_report = response.content
        
        # Save the final report
        output_dir = Path(__file__).parent / output_folder
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "final_investment_strategy.txt", 'w', encoding='utf-8') as f:
            f.write("=== CHIẾN LƯỢC ĐẦU TƯ TỐI ƯU ===\n\n")
            f.write(f"Câu hỏi: {question}\n\n")
            f.write(final_report)
        
        print(f"Chiến lược đầu tư cuối cùng đã được lưu vào {output_dir / 'final_investment_strategy.txt'}")
        
        # Return only the final_report key
        return {
            "final_report": final_report
        }
        
    except Exception as e:
        print(f"[LỖI] Lỗi trong người tổng hợp cuối cùng: {str(e)}")
        return {
            "final_report": f"Lỗi trong tổng hợp cuối cùng: {str(e)}"
        }