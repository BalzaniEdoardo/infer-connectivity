import matplotlib.pyplot as plt
import networkx as nx


def visualize_connectivity_matrix(G, N_E):
    """Visualize the adjacency matrix as a heatmap"""
    # Get adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Full connectivity matrix
    im1 = ax1.imshow(adj_matrix, cmap="Blues", aspect="auto")
    ax1.set_title("Full Connectivity Matrix")
    ax1.set_xlabel("Postsynaptic")
    ax1.set_ylabel("Presynaptic")

    # Add lines to separate E and I populations
    ax1.axhline(y=N_E - 0.5, color="red", linestyle="--", linewidth=2)
    ax1.axvline(x=N_E - 0.5, color="red", linestyle="--", linewidth=2)
    ax1.text(N_E / 2, -0.1 * len(G), "Excitatory", ha="center", transform=ax1.transData)
    ax1.text(
        (N_E + len(G)) / 2,
        -0.1 * len(G),
        "Inhibitory",
        ha="center",
        transform=ax1.transData,
    )

    plt.colorbar(im1, ax=ax1)

    # Connection density by type
    ee_connections = adj_matrix[:N_E, :N_E].sum()  # EE
    ei_connections = adj_matrix[N_E:, :N_E].sum()  # IE
    ie_connections = adj_matrix[:N_E, N_E:].sum()  # EI
    ii_connections = adj_matrix[N_E:, N_E:].sum()  # II

    connection_types = ["EE", "IE", "EI", "II"]
    connection_counts = [ee_connections, ei_connections, ie_connections, ii_connections]

    ax2.bar(
        connection_types,
        connection_counts,
        color=["red", "purple", "orange", "blue"],
        alpha=0.7,
    )
    ax2.set_title("Connections by Type")
    ax2.set_ylabel("Number of Connections")

    plt.tight_layout()
    plt.show()
