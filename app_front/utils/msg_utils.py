def get_response(result, llm_provider) -> str:

    if llm_provider == 'anthropic':
        return result["messages"][-1].content[-1]['text']
