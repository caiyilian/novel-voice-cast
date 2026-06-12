import sys
sys.path.insert(0, "..")

from app.core.ollama_client import OllamaClient, OllamaConfig, OllamaError


def test_config_loading():
    config = OllamaConfig.from_ip_config()
    assert config.base_url, "base_url should not be empty"
    assert config.model, "model should not be empty"
    print(f"[PASS] Config loaded: {config.base_url} / {config.model}")
    return config


def test_connection(config):
    client = OllamaClient(config)
    status = client.check_connection()
    if status.ok:
        print(f"[PASS] Connection OK: model={status.model}")
    else:
        print(f"[FAIL] Connection failed: {status.message}")
    return status.ok


def test_chat_simple(config):
    client = OllamaClient(config)
    result = client.chat([{"role": "user", "content": "Say hello in one word."}])
    assert result.content, "Response content should not be empty"
    print(f"[PASS] Simple chat: {result.content[:50]}")
    return True


def test_chat_with_tools(config):
    client = OllamaClient(config)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_novel",
                "description": "Search for a keyword in the novel text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "The keyword to search for"}
                    },
                    "required": ["keyword"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the search_novel tool to find information."},
        {"role": "user", "content": "Search for the word 'chapter' in the text."},
    ]
    result = client.chat(messages, tools=tools)
    if result.tool_calls:
        tc = result.tool_calls[0]
        print(f"[PASS] Tool calling: {tc.name}({tc.arguments})")
        return True
    else:
        print(f"[WARN] No tool calls returned, model responded: {result.content[:100]}")
        return False


def test_tool_result(config):
    client = OllamaClient(config)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_lines",
                "description": "Read lines from a file by line range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer", "description": "Start line number"},
                        "end": {"type": "integer", "description": "End line number"},
                    },
                    "required": ["start", "end"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "You are a novel analyst. Use read_lines to explore the text."},
        {"role": "user", "content": "Read lines 1 to 10."},
    ]
    result = client.chat(messages, tools=tools)
    if result.tool_calls:
        tc = result.tool_calls[0]
        print(f"[PASS] Tool with params: {tc.name}(start={tc.arguments.get('start')}, end={tc.arguments.get('end')})")

        messages.append({"role": "assistant", "content": result.content, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}]})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Line 1: The story begins...\nLine 2: In a far away land..."})

        result2 = client.chat(messages, tools=tools)
        print(f"[PASS] After tool result, model responded: {result2.content[:80]}")
        return True
    else:
        print(f"[WARN] No tool calls: {result.content[:100]}")
        return False


if __name__ == "__main__":
    print("=== Stage 4.5a: Ollama Client Test ===\n")

    config = test_config_loading()
    print()

    if not test_connection(config):
        print("\nCannot connect to Ollama. Skipping remaining tests.")
        sys.exit(1)
    print()

    test_chat_simple(config)
    print()

    test_chat_with_tools(config)
    print()

    test_tool_result(config)
    print()

    print("=== All tests completed ===")
