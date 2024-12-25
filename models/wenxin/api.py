from ..base_api import BaseAPI
from typing import List, Dict, Union, Generator
import requests
import json
from ...logger import logger
from ...utils.error_handler import handle_wenxin_error

class API(BaseAPI):
    BASE_URL = "https://aip.baidubce.com"
    
    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials.get("api_key")
        self.secret_key = credentials.get("secret_key")
        self.access_token = self._get_access_token()
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        logger.info("Wenxin API initialized")

    def _get_access_token(self) -> str:
        """获取access_token"""
        url = f"{self.BASE_URL}/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()["access_token"]

    def _call_api(self, endpoint: str, data: Dict, stream: bool = False) -> Union[Dict, Generator]:
        url = f"{self.BASE_URL}{endpoint}?access_token={self.access_token}"
        try:
            response = self.session.post(url, json=data, stream=stream)
            response.raise_for_status()
            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
        except requests.RequestException as e:
            raise handle_wenxin_error(e)

    def _handle_stream_response(self, response: requests.Response) -> Generator:
        for line in response.iter_lines():
            if line:
                yield json.loads(line.decode('utf-8').split('data: ')[1])

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        endpoint = self._get_endpoint(model)
        data = {
            "messages": messages,
            **kwargs
        }
        logger.info(f"Generating response with model: {model}")
        return self._call_api(endpoint, data)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        endpoint = self._get_endpoint(model)
        data = {
            "messages": messages,
            "stream": True,
            **kwargs
        }
        logger.info(f"Generating streaming response with model: {model}")
        return self._call_api(endpoint, data, stream=True)

    def _get_endpoint(self, model: str) -> str:
        if model == "ERNIE-Bot":
            return "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions"
        elif model == "ERNIE-Bot-turbo":
            return "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"
        elif model == "BLOOMZ-7B":
            return "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/bloomz_7b1"
        else:
            raise ValueError(f"Unsupported model: {model}")

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count tokens in a message."""
        # 文心一言 API 没有提供计算 token 的端点，所以我们使用一个估算方法
        token_count = sum(len(str(message.get('content', '')).split()) for message in messages)
        logger.info(f"Estimated token count for model {model}: {token_count}")
        return token_count

    def set_proxy(self, proxy_url: str):
        """Set a proxy for API calls."""
        self.session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Proxy set to {proxy_url}")