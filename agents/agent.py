from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/beam:cloud",  # Or try other Ollama-supported models
    api_base="http://127.0.0.1:11434",  # Default Ollama local server
    num_ctx=8192,
)

agent = CodeAgent(tools=[], model=model)
agent.run("Write the optimal algorithm for leetcode two-sum problem")