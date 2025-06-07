import csv
import re
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = "You are a Graph generating expert, helpful assistant. Follow the user instructions exactly."

def create_user_prompt(description):
    """
    In this user prompt, we instruct the LLM to:
    1) Carefully read the graph description and understand its properties (e.g. number of nodes, edges, etc.).
    2) Construct a graph that exactly matches those properties.
    3) Output only the edge list for that graph in the exact format (0, 1), (0, 2), etc.
    4) Provide no explanation, just the edge list.
    """
    instructions = (
        "Read the following graph description carefully. It may mention the number of nodes, edges, "
        "average degree, number of triangles, or other properties. Construct a graph that matches "
        "these properties exactly, then output ONLY that graph's edge list in the format:\n\n"
        "(0, 1), (0, 2), (1, 2), (1, 3)\n\n"
        "No additional text, just the edge list. Do not say 'please generate' or 'here is...'—"
        "just provide the list. For example, if the graph is a star with node 0 in the center "
        "and nodes 1, 2, 3 on the leaves, you would write:\n\n"
        "(2, 1), (5, 2), (5, 3)\n\n"
        "If there's no reason to include an edge (i, i) or duplicates, do not include them.\n\n"
        "Graph description:\n"
        f"{description}"
    )
    return instructions

def read_test_file(filename):
    """
    Reads the test file and extracts (graph_id, description) pairs.
    Each line in test.txt might look like:
        001,This graph has 5 nodes and 4 edges...
    """
    lines = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            graph_id, desc = line.split(",", 1)
            lines.append((graph_id, desc))
    return lines

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", gpu_memory_utilization=0.6)

sampling_params = SamplingParams(
    max_tokens=1024,
    temperature=0.0,
    top_p=1.0,
    stop=["</SYSTEM>", "</USER>"]
)

def main():
    data = read_test_file("data/test/test.txt")
    submission_rows = []

    for graph_id, desc in data:
        user_prompt = create_user_prompt(desc)
        full_prompt = f"<SYSTEM>{SYSTEM_PROMPT}</SYSTEM>\n<USER>{user_prompt}</USER>"

        outputs = llm.generate([full_prompt], sampling_params)
        raw_text = outputs[0].outputs[0].text

        print(f"Graph ID: {graph_id}\nRaw LLM Response:\n{raw_text}")
        print("-" * 50)

        submission_rows.append((graph_id, raw_text))

    with open("output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph_id", "edge_list_raw"])
        writer.writerows(submission_rows)

if __name__ == "__main__":
    main()
