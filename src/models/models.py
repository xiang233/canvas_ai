"""
模型管理器 - 多 provider 版本

支持两个后端：
  - Azure OpenAI  (AZURE_OPENAI_*)   —— model_id 必须填 **deployment name**
  - OpenAI 官方   (OPENAI_API_KEY)

由 LLM_PROVIDER 控制：
  azure   仅 Azure
  openai  仅 OpenAI 官方
  auto    两个都注册，Azure 优先占用“裸别名”（默认；任一缺失就自动降级到另一个）
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv(verbose=True)

from src.logger import logger
from src.models.openaillm import OpenAIServerModel
from src.utils import Singleton

PLACEHOLDER = "PLACEHOLDER"


class ModelManager(metaclass=Singleton):
    """模型管理器 - 负责注册和管理可用的模型"""

    def __init__(self):
        self.registed_models: Dict[str, Any] = {}

    def init_models(self, use_local_proxy: bool = False) -> int:
        """初始化所有模型（Azure OpenAI 和/或 OpenAI 官方）"""

        # 重新初始化时先清空，保证配置变更即时生效
        self.registed_models.clear()

        provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()

        # 注册顺序决定“裸别名”(如 gpt-4.1-mini) 归谁：先注册的先占位
        if provider == "azure":
            order: List[str] = ["azure"]
        elif provider == "openai":
            order = ["openai"]
        else:  # auto / both / 其他值：两个都试，Azure 优先
            order = ["azure", "openai"]

        registered_count = 0
        for backend in order:
            if backend == "azure":
                registered_count += self._register_azure_models()
            else:
                registered_count += self._register_openai_models(use_local_proxy=use_local_proxy)

        if registered_count == 0:
            logger.error("=" * 70)
            logger.error("未检测到可用的大模型配置！")
            logger.error("=" * 70)
            logger.error("请在 .env 中至少配置以下任一组：")
            logger.error("- AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY (+ AZURE_OPENAI_DEPLOYMENTS)")
            logger.error("- OPENAI_API_KEY (+ 可选 OPENAI_BASE_URL / OPENAI_ORGANIZATION / OPENAI_PROJECT)")
            logger.error("=" * 70)
            raise ValueError("模型配置缺失，请检查 .env 文件")

        logger.info(f"模型注册完成，共 {registered_count} 个别名可用: {list(self.registed_models.keys())}")
        return registered_count

    # ------------------------------------------------------------------ Azure

    def _register_azure_models(self) -> int:
        """注册 Azure OpenAI 部署。

        注意：Azure 上传给 API 的 model 字段是 **deployment name**，不是模型名。
        所以这里的 model_id 直接用 deployment name。
        """
        logger.info("注册 Azure OpenAI 模型")

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", PLACEHOLDER)
        api_key = os.getenv("AZURE_OPENAI_API_KEY", PLACEHOLDER)
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        if azure_endpoint == PLACEHOLDER or api_key == PLACEHOLDER:
            logger.warning("未检测到 AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY，跳过 Azure 模型注册")
            return 0

        # 逗号分隔的 deployment 名称列表
        deployments = [
            d.strip()
            for d in os.getenv("AZURE_OPENAI_DEPLOYMENTS", "gpt-4.1-mini").split(",")
            if d.strip()
        ]
        if not deployments:
            logger.warning("AZURE_OPENAI_DEPLOYMENTS 为空，跳过 Azure 模型注册")
            return 0

        try:
            from openai import AsyncAzureOpenAI
        except ModuleNotFoundError as exc:  # noqa: BLE001
            logger.error(f"未安装 openai SDK，无法注册 Azure 模型: {exc}")
            return 0

        client_kwargs: Dict[str, Any] = {}
        max_retries = os.getenv("AZURE_OPENAI_MAX_RETRIES") or os.getenv("OPENAI_MAX_RETRIES")
        if max_retries:
            try:
                client_kwargs["max_retries"] = int(max_retries)
            except ValueError:
                logger.warning("AZURE_OPENAI_MAX_RETRIES 不是有效的整数，已忽略该配置")

        timeout = os.getenv("AZURE_OPENAI_TIMEOUT") or os.getenv("OPENAI_TIMEOUT")
        if timeout:
            try:
                client_kwargs["timeout"] = float(timeout)
            except ValueError:
                logger.warning("AZURE_OPENAI_TIMEOUT 不是有效的数字，已忽略该配置")

        try:
            # 所有 deployment 共用同一个 client，复用底层连接池
            azure_client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                **client_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"创建 Azure OpenAI 客户端失败: {exc}")
            return 0

        registered = 0
        for deployment in deployments:
            try:
                model = OpenAIServerModel(
                    model_id=deployment,
                    http_client=azure_client,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"注册 Azure 部署 {deployment} 失败: {exc}")
                continue

            # 显式别名 azure-xxx 始终注册；裸别名仅在未被占用时注册
            for alias in (f"azure-{deployment}", deployment):
                if alias in self.registed_models:
                    logger.debug(f"模型别名 {alias} 已存在，跳过覆盖")
                    continue
                self.registed_models[alias] = model
                registered += 1
                logger.info(f"成功注册 Azure 模型: {alias} (deployment: {deployment})")

        if registered == 0:
            logger.warning("没有成功注册任何 Azure 模型，请检查 AZURE_OPENAI_DEPLOYMENTS")

        return registered

    # ----------------------------------------------------------------- OpenAI

    def _register_openai_models(self, use_local_proxy: bool = False) -> int:
        """注册 OpenAI 官方模型"""
        logger.info("注册 OpenAI 模型")

        if use_local_proxy:
            api_key = self._check_local_api_key("LOCAL_OPENAI_API_KEY", "OPENAI_API_KEY")
            api_base = self._check_local_api_base("LOCAL_OPENAI_API_BASE", "OPENAI_API_BASE")
        else:
            api_key = os.getenv("OPENAI_API_KEY", PLACEHOLDER)
            api_base = os.getenv("OPENAI_API_BASE", PLACEHOLDER)

        # 兼容其他常见的基础地址环境变量命名
        if api_base == PLACEHOLDER:
            api_base = os.getenv("OPENAI_BASE_URL", PLACEHOLDER)
        if api_base == PLACEHOLDER:
            api_base = os.getenv("OPENAI_API_URL", PLACEHOLDER)

        if api_key == PLACEHOLDER:
            logger.warning("未检测到 OPENAI_API_KEY，跳过 OpenAI 模型注册")
            return 0

        organization = os.getenv("OPENAI_ORGANIZATION") or os.getenv("OPENAI_ORG")
        project = os.getenv("OPENAI_PROJECT")

        client_kwargs: Dict[str, Any] = {}
        max_retries = os.getenv("OPENAI_MAX_RETRIES")
        if max_retries:
            try:
                client_kwargs["max_retries"] = int(max_retries)
            except ValueError:
                logger.warning("OPENAI_MAX_RETRIES 不是有效的整数，已忽略该配置")

        timeout = os.getenv("OPENAI_TIMEOUT")
        if timeout:
            try:
                client_kwargs["timeout"] = float(timeout)
            except ValueError:
                logger.warning("OPENAI_TIMEOUT 不是有效的数字，已忽略该配置")

        if api_base == PLACEHOLDER:
            api_base = None

        openai_models = [
            {"model_id": "gpt-5", "aliases": ["openai-gpt-5", "gpt-5"]},
            {"model_id": "gpt-5-mini", "aliases": ["openai-gpt-5-mini"]},
            {"model_id": "gpt-5-nano", "aliases": ["openai-gpt-5-nano"]},
            {"model_id": "gpt-4o-mini", "aliases": ["openai-gpt-4o-mini", "gpt-4o-mini"]},
            {"model_id": "gpt-4o", "aliases": ["openai-gpt-4o", "gpt-4o"]},
            {"model_id": "gpt-4.1-mini", "aliases": ["openai-gpt-4.1-mini", "gpt-4.1-mini"]},
            {"model_id": "gpt-4.1", "aliases": ["openai-gpt-4.1"]},
            {"model_id": "o1-mini", "aliases": ["openai-o1-mini"]},
            {"model_id": "o1-preview", "aliases": ["openai-o1-preview"]},
        ]

        registered = 0

        for model_config in openai_models:
            model_id = model_config["model_id"]
            aliases = model_config.get("aliases", []) or [model_id]

            try:
                model = OpenAIServerModel(
                    model_id=model_id,
                    api_base=api_base,
                    api_key=api_key,
                    organization=organization,
                    project=project,
                    client_kwargs=client_kwargs or None,
                )
            except Exception as exc:  # noqa: BLE001 - 捕获并记录初始化异常
                logger.error(f"注册 OpenAI 模型 {model_id} 失败: {exc}")
                continue

            for alias in aliases:
                if alias in self.registed_models:
                    logger.debug(f"模型别名 {alias} 已存在，跳过覆盖")
                    continue

                self.registed_models[alias] = model
                registered += 1
                logger.info(f"成功注册 OpenAI 模型: {alias} (ID: {model_id})")

        if registered == 0:
            logger.warning("没有成功注册任何 OpenAI 模型，请检查模型别名或配置")

        return registered

    # ------------------------------------------------------------------ utils

    def _check_local_api_key(self, local_api_key_name: str, remote_api_key_name: str) -> str:
        """检查本地和远程 API Key"""
        api_key = os.getenv(local_api_key_name, PLACEHOLDER)
        if api_key == PLACEHOLDER:
            logger.warning(f"Local API key {local_api_key_name} is not set, using remote API key {remote_api_key_name}")
            api_key = os.getenv(remote_api_key_name, PLACEHOLDER)
        return api_key

    def _check_local_api_base(self, local_api_base_name: str, remote_api_base_name: str) -> str:
        """检查本地和远程 API Base"""
        api_base = os.getenv(local_api_base_name, PLACEHOLDER)
        if api_base == PLACEHOLDER:
            logger.warning(f"Local API base {local_api_base_name} is not set, using remote API base {remote_api_base_name}")
            api_base = os.getenv(remote_api_base_name, PLACEHOLDER)
        return api_base

    def get_model(self, model_name: str):
        """获取已注册的模型实例"""
        if model_name not in self.registed_models:
            raise ValueError(f"模型 {model_name} 未注册。可用模型: {list(self.registed_models.keys())}")
        return self.registed_models[model_name]

    def list_models(self):
        """列出所有已注册的模型"""
        return list(self.registed_models.keys())


# 全局模型管理器实例
model_manager = ModelManager()
