from typing import List, Dict, Optional, Any
import os
import time
import random
from datetime import datetime
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

# Initialize DuckDuckGo search
search_api = DuckDuckGoSearchAPIWrapper()
DEFAULT_MAX_RESULTS = 5
MAX_RETRIES = 3  # Maximum number of retries for search
BASE_DELAY = 1  # Base delay in seconds before retrying
JITTER = 0.5  # Random jitter to add to retry timing

def simple_search(query: str) -> List[Dict[str, str]]:
    """
    Thực hiện tìm kiếm đơn giản sử dụng DuckDuckGo với cơ chế thử lại để tránh RateLimit.
    
    Args:
        query: Truy vấn tìm kiếm
        
    Returns:
        Danh sách kết quả tìm kiếm với title, link, và snippet
    """
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            results = search_api.results(query, max_results=DEFAULT_MAX_RESULTS)
            
            # Format the results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
            
            return formatted_results
        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                print(f"Đã đạt đến số lần thử lại tối đa. Lỗi tìm kiếm cuối cùng: {str(e)}")
                return []
            
            # Calculate delay with linear backoff (not exponential as requested) and jitter
            delay = BASE_DELAY * retries + random.uniform(0, JITTER)
            print(f"Lỗi tìm kiếm: {str(e)}. Thử lại sau {delay:.2f} giây... (lần thử {retries}/{MAX_RETRIES})")
            time.sleep(delay)

def search_with_context(query: str, context: str) -> List[Dict[str, str]]:
    """
    Thực hiện tìm kiếm với ngữ cảnh bổ sung.
    
    Args:
        query: Truy vấn cơ bản
        context: Ngữ cảnh bổ sung để thêm vào truy vấn
        
    Returns:
        Danh sách kết quả tìm kiếm
    """
    enhanced_query = f"{query} {context}"
    return simple_search(enhanced_query)

def search_vietnam_market(query: str) -> List[Dict[str, str]]:
    """
    Thực hiện tìm kiếm cụ thể về thị trường Việt Nam.
    
    Args:
        query: Truy vấn cơ bản
        
    Returns:
        Danh sách kết quả tìm kiếm
    """
    return search_with_context(query, "Vietnam stock market")

def search_financial_data(company_or_ticker: str) -> List[Dict[str, str]]:
    """
    Tìm kiếm dữ liệu tài chính về một công ty hoặc mã cổ phiếu cụ thể.
    
    Args:
        company_or_ticker: Tên công ty hoặc mã cổ phiếu
        
    Returns:
        Danh sách kết quả tìm kiếm
    """
    return search_with_context(company_or_ticker, "financial data Vietnam stock market")

def search_sector_performance(sector: str) -> List[Dict[str, str]]:
    """
    Tìm kiếm dữ liệu hiệu suất về một ngành cụ thể tại Việt Nam.
    
    Args:
        sector: Ngành kinh tế (ví dụ: ngân hàng, bất động sản)
        
    Returns:
        Danh sách kết quả tìm kiếm
    """
    return search_with_context(sector, "sector performance Vietnam stock market")

def search_economic_indicators() -> List[Dict[str, str]]:
    """
    Tìm kiếm các chỉ số kinh tế hiện tại tại Việt Nam.
    
    Returns:
        Danh sách kết quả tìm kiếm
    """
    return simple_search("Vietnam GDP inflation interest rate economic indicators current")

def perform_batch_search(queries: List[str], max_queries_per_batch: int = 3, batch_delay: float = 2.0) -> List[Dict[str, str]]:
    """
    Thực hiện tìm kiếm theo lô để tránh giới hạn tốc độ.
    
    Args:
        queries: Danh sách các truy vấn cần tìm kiếm
        max_queries_per_batch: Số lượng truy vấn tối đa mỗi lô
        batch_delay: Thời gian chờ giữa các lô (giây)
        
    Returns:
        Danh sách kết quả tìm kiếm từ tất cả các truy vấn
    """
    all_results = []
    
    # Split queries into batches
    for i in range(0, len(queries), max_queries_per_batch):
        batch = queries[i:i + max_queries_per_batch]
        
        print(f"Xử lý lô truy vấn {i//max_queries_per_batch + 1}/{(len(queries) + max_queries_per_batch - 1)//max_queries_per_batch}")
        
        # Process each query in the batch
        for query in batch:
            results = simple_search(query)
            all_results.extend(results)
            
            # Small delay between individual queries within a batch
            time.sleep(random.uniform(0.5, 1.0))
        
        # Delay between batches if there are more batches to process
        if i + max_queries_per_batch < len(queries):
            actual_delay = batch_delay + random.uniform(-0.5, 0.5)  # Add some jitter
            print(f"Chờ {actual_delay:.2f} giây trước khi xử lý lô tiếp theo...")
            time.sleep(actual_delay)
    
    return all_results