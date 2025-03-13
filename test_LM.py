from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

from llm import HUGGINGFACE_LLM
import logging
logger = logging.getLogger(__name__)
# 首先确保已安装必要依赖：
# pip install torch transformers accelerate bitsandbytes

# 测试代码
if __name__ == "__main__":
    # 初始化模型（首次运行会自动下载约3GB的模型）
    llm = HUGGINGFACE_LLM(
        model_id="agentica-org/DeepScaleR-1.5B-Preview",
        n=2  # 默认生成2个结果
    )

    # 测试单prompt生成
    logger.info("单prompt测试:")
    single_result = llm("The meaning of life is", temperature=0.7, max_tokens=2048)
    logger.info(single_result)

    # 测试批量prompt生成
    # logger.info("\n批量prompt测试:")
    # batch_result = llm([
    #     "In the future, AI will",
    #     "Quantum computing is"
    # ], n=1, temperature=0)
    # logger.info(batch_result)
