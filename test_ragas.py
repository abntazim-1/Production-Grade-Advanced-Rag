import sys
import types

try:
    import langchain_community.chat_models
    if not hasattr(langchain_community.chat_models, "vertexai"):
        dummy = types.ModuleType("langchain_community.chat_models.vertexai")
        dummy.ChatVertexAI = type("ChatVertexAI", (object,), {})
        sys.modules["langchain_community.chat_models.vertexai"] = dummy
        langchain_community.chat_models.vertexai = dummy
except Exception:
    pass

import traceback
try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Success")
except Exception as e:
    traceback.print_exc()
