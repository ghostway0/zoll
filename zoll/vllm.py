import time
import json
from typing import List, Dict, Any
import requests

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from zoll.llm import LLMClient, SamplingParams


class VLLMClient(LLMClient):
    def __init__(
        self,
        host: str = "127.0.0.1",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 30.0,
        request_timeout: float = 60.0,
    ):
        self.session: requests.Session = requests.Session()
        self.host = host
        self.base_url = f"http://{self.host}:{server_port}"
        self.group_port = group_port
        self.request_timeout = request_timeout
        self.pynccl_comm: PyNcclCommunicator
        self.rank: int

        self._check_server(connection_timeout)
        self._init_communicator()

    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(
            method, url, timeout=self.request_timeout, **kwargs
        )
        response.raise_for_status()
        return response.json() if response.content else None

    def _check_server(self, total_timeout: float, retry_interval: float = 2.0):
        start_time = time.time()
        while True:
            try:
                self._request("GET", "/health/")
                return
            except (
                requests.RequestException,
                requests.HTTPError,
                json.JSONDecodeError,
                RuntimeError,
            ):
                pass

            elapsed = time.time() - start_time
            if total_timeout > 0 and elapsed >= total_timeout:
                raise TimeoutError(
                    f"vLLM connection timeout after {total_timeout:.1f}s"
                )

            time.sleep(retry_interval)

    def _init_communicator(self):
        tp_data = self._request("GET", "/get_tensor_parallel_size/")
        tensor_parallel_size = tp_data["tensor_parallel_size"]
        world_size = tensor_parallel_size + 1
        self.rank = tensor_parallel_size

        comm_init_payload = {
            "host": "0.0.0.0",
            "port": self.group_port,
            "world_size": world_size,
        }
        self._request("POST", "/init_communicator/", json=comm_init_payload)

        pg = StatelessProcessGroup.create(
            host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device="cuda:0")  # they assume cuda:0

    def generate(self, message: List[Dict[str, str]], params: SamplingParams) -> str:
        payload = {"messages": [message], **params.dict()}
        payload.setdefault("max_tokens", 16)
        response_data = self._request("POST", "/chat/", json=payload)
        try:
            return response_data["responses"][0]["outputs"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Failed to parse generate response structure: {e}") from e

    def load_weight(self, name: str, weight: torch.Tensor):
        if self.pynccl_comm is None or self.rank is None:
            raise RuntimeError("Communicator not initialized.")

        metadata = {
            "name": name,
            "dtype": str(weight.dtype),
            "shape": tuple(weight.shape),
        }
        self._request("POST", "/update_named_param/", json=metadata)

        target_device = torch.device(f"cuda:{self.pynccl_comm.group.device.index}")
        if weight.device != target_device:
            weight = weight.to(target_device)

        stream = torch.cuda.current_stream(device=target_device)
        self.pynccl_comm.broadcast(weight, src=self.rank, stream=stream)
        self.pynccl_comm.group.barrier()

    def close(self):
        try:
            self._request("POST", "/close_communicator/")
        except (requests.RequestException, requests.HTTPError, RuntimeError):
            pass
        finally:
            self.session.close()
