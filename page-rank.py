import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Create a directed graph
G = nx.DiGraph()

# Step 2: Add web pages
pages = ["A", "B", "C", "D"]

G.add_nodes_from(pages)

# Step 3: Add hyperlinks between pages
G.add_edges_from([
    ("B", "A"),
    ("B", "C"),
    ("C", "A"),
    ("D", "A"),
    ("D", "B"),
    ("D", "C")
])

# Step 4: Display graph
print("Web Page Link Structure:")
print(G.edges())

# Step 5: Calculate PageRank
pagerank = nx.pagerank(
    G,
    alpha=0.85,
    max_iter=100,
    tol=1e-6
)

# Step 6: Display PageRank scores
print("\nPageRank Scores:")

for page, score in sorted(
    pagerank.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(page, ":", round(score, 6))

# Step 7: Find highest-ranked page
highest_page = max(
    pagerank,
    key=pagerank.get
)

print("\nHighest Ranked Web Page:", highest_page)

# Step 8: Visualize directed graph
plt.figure(figsize=(8, 6))

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2500,
    node_color="lightblue",
    font_size=14,
    font_weight="bold",
    arrows=True,
    arrowsize=20
)

plt.title("PageRank Web Link Graph")

plt.show()

# Step 9: Visualize PageRank scores
plt.figure(figsize=(8, 5))

plt.bar(
    pagerank.keys(),
    pagerank.values()
)

plt.xlabel("Web Pages")
plt.ylabel("PageRank Score")

plt.title("PageRank Algorithm Results")

plt.show()