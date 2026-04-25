"""
Smoke test for the NIM-backed expand_query() entry point in query_expander.py.

Runs a single sample query through the standalone function and prints each
returned string with an index. A successful run prints 4 strings:
  [0] original
  [1] conceptual reformulation
  [2] methodology-focused reformulation
  [3] synonym-based reformulation

If NIM is unreachable or the response is malformed, expand_query() returns
[query] — so the smoke test will print exactly 1 string in that case.
"""
from query_expander import expand_query


def main() -> None:
    query = "What is retrieval-augmented generation?"
    results = expand_query(query, n=3)

    print(f"Query: {query}")
    print(f"Returned {len(results)} string(s):\n")
    for i, s in enumerate(results):
        label = "original" if i == 0 else f"reform {i}"
        print(f"[{i}] {label}: {s}")


if __name__ == "__main__":
    main()
