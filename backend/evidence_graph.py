import json
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DOCUMENTS_PATH = PROCESSED_DIR / "documents.json"
ENTITIES_PATH = PROCESSED_DIR / "entities.json"
RELATIONS_PATH = PROCESSED_DIR / "relations.json"
FRAMES_PATH = PROCESSED_DIR / "frames.json"

GRAPH_OUTPUT_PATH = PROCESSED_DIR / "evidence_graph.json"


# =========================================================
# HELPERS
# =========================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required processed file missing: {path}\n"
            "Run backend/preprocess.py first."
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def readable_predicate(predicate):
    """
    Keep original DWIE predicate internally,
    but provide a cleaner human-facing label.
    """

    mapping = {
        "in0": "located in",
        "in1": "located in",
        "in2": "located in",
        "in0-x": "located in",
        "in1-x": "located in",
        "in2-x": "located in",

        "citizen_of": "citizen of",
        "member_of": "member of",
        "based_in0": "based in",
        "based_in1": "based in",
        "based_in2": "based in",
        "based_in0-x": "based in",
        "based_in1-x": "based in",
        "based_in2-x": "based in",
    }

    return mapping.get(
        predicate,
        predicate.replace("_", " ")
    )


# =========================================================
# EVIDENCE GRAPH
# =========================================================

class EvidenceGraph:

    def __init__(self):

        self.documents = load_json(
            DOCUMENTS_PATH
        )

        self.entities = load_json(
            ENTITIES_PATH
        )

        self.relations = load_json(
            RELATIONS_PATH
        )

        self.frames = load_json(
            FRAMES_PATH
        )

        # MultiDiGraph:
        # - directed relationships matter
        # - multiple evidence edges may exist
        #   between the same pair of entities
        self.graph = nx.MultiDiGraph()

        self._build()


    # =====================================================
    # BUILD GRAPH
    # =====================================================

    def _build(self):

        self._add_document_nodes()

        self._add_entity_nodes()

        self._add_entity_document_edges()

        self._add_relation_edges()

        self._add_frame_nodes_and_edges()


    # =====================================================
    # DOCUMENT NODES
    # =====================================================

    def _add_document_nodes(self):

        for document in self.documents:

            document_id = document[
                "document_id"
            ]

            node_id = (
                f"document:{document_id}"
            )

            self.graph.add_node(
                node_id,

                node_type="document",

                document_id=document_id,

                label=document.get(
                    "title",
                    document_id
                ),

                title=document.get(
                    "title"
                ),

                iptc=document.get(
                    "iptc",
                    []
                ),

                verified=document.get(
                    "verified",
                    True
                ),

                source=document.get(
                    "source",
                    "DWIE"
                ),
            )


    # =====================================================
    # ENTITY NODES
    # =====================================================

    def _add_entity_nodes(self):

        for entity in self.entities:

            entity_id = entity[
                "entity_id"
            ]

            self.graph.add_node(
                entity_id,

                node_type="entity",

                entity_id=entity_id,

                label=entity.get(
                    "name",
                    entity_id
                ),

                name=entity.get(
                    "name"
                ),

                category=entity.get(
                    "category",
                    "other"
                ),

                link=entity.get(
                    "link"
                ),

                tags=entity.get(
                    "tags",
                    []
                ),

                aliases=entity.get(
                    "aliases",
                    []
                ),
            )


    # =====================================================
    # ENTITY → DOCUMENT
    # =====================================================

    def _add_entity_document_edges(self):

        for entity in self.entities:

            entity_id = entity[
                "entity_id"
            ]

            for document_id in entity.get(
                "document_ids",
                []
            ):

                document_node = (
                    f"document:{document_id}"
                )

                if not self.graph.has_node(
                    document_node
                ):
                    continue

                self.graph.add_edge(
                    entity_id,
                    document_node,

                    edge_type="evidence",

                    relation="mentioned_in",

                    label="mentioned in",

                    document_id=document_id,

                    verified=True,

                    source="DWIE",
                )


    # =====================================================
    # ENTITY → ENTITY RELATIONSHIPS
    # =====================================================

    def _add_relation_edges(self):

        for relation in self.relations:

            subject = relation[
                "subject_entity_id"
            ]

            object_ = relation[
                "object_entity_id"
            ]

            if not self.graph.has_node(
                subject
            ):
                continue

            if not self.graph.has_node(
                object_
            ):
                continue

            predicate = relation[
                "predicate"
            ]

            self.graph.add_edge(
                subject,
                object_,

                key=relation[
                    "relation_id"
                ],

                edge_type="relation",

                relation_id=relation[
                    "relation_id"
                ],

                predicate=predicate,

                relation=predicate,

                label=readable_predicate(
                    predicate
                ),

                document_id=relation[
                    "document_id"
                ],

                verified=relation.get(
                    "verified",
                    True
                ),

                source=relation.get(
                    "source",
                    "DWIE"
                ),
            )


    # =====================================================
    # EVENT / FRAME NODES
    # =====================================================

    def _add_frame_nodes_and_edges(self):

        for frame in self.frames:

            frame_id = frame[
                "frame_id"
            ]

            event_node = (
                f"event:{frame_id}"
            )

            frame_type = frame[
                "frame_type"
            ]

            document_id = frame[
                "document_id"
            ]

            self.graph.add_node(
                event_node,

                node_type="event",

                frame_id=frame_id,

                frame_type=frame_type,

                label=frame_type.replace(
                    "-",
                    " "
                ),

                document_id=document_id,

                verified=frame.get(
                    "verified",
                    True
                ),

                source=frame.get(
                    "source",
                    "DWIE"
                ),
            )

            # -----------------------------------------
            # Event → source document
            # -----------------------------------------

            document_node = (
                f"document:{document_id}"
            )

            if self.graph.has_node(
                document_node
            ):

                self.graph.add_edge(
                    event_node,
                    document_node,

                    edge_type="evidence",

                    relation="supported_by",

                    label="supported by",

                    document_id=document_id,

                    verified=True,

                    source="DWIE",
                )

            # -----------------------------------------
            # Event → participant entities
            # -----------------------------------------

            for slot in frame.get(
                "slots",
                []
            ):

                entity_id = slot.get(
                    "entity_id"
                )

                role = slot.get(
                    "role",
                    "participant"
                )

                if (
                    not entity_id
                    or not self.graph.has_node(
                        entity_id
                    )
                ):
                    continue

                self.graph.add_edge(
                    event_node,
                    entity_id,

                    edge_type="event_role",

                    relation=role,

                    label=role.replace(
                        "_",
                        " "
                    ),

                    document_id=document_id,

                    verified=True,

                    source="DWIE",
                )


    # =====================================================
    # STATS
    # =====================================================

    def stats(self):

        node_counts = {
            "entity": 0,
            "document": 0,
            "event": 0,
        }

        for _, data in (
            self.graph.nodes(data=True)
        ):

            node_type = data.get(
                "node_type"
            )

            if node_type in node_counts:
                node_counts[
                    node_type
                ] += 1

        edge_counts = {}

        for _, _, data in (
            self.graph.edges(data=True)
        ):

            edge_type = data.get(
                "edge_type",
                "unknown"
            )

            edge_counts[
                edge_type
            ] = (
                edge_counts.get(
                    edge_type,
                    0
                )
                + 1
            )

        return {
            "total_nodes":
                self.graph.number_of_nodes(),

            "total_edges":
                self.graph.number_of_edges(),

            "nodes_by_type":
                node_counts,

            "edges_by_type":
                edge_counts,
        }


    # =====================================================
    # SERIALIZATION
    # =====================================================

    def serialize_graph(
        self,
        graph=None,
        max_nodes=None
    ):

        graph = graph or self.graph

        nodes = []

        selected_nodes = list(
            graph.nodes()
        )

        if max_nodes is not None:
            selected_nodes = (
                selected_nodes[
                    :max_nodes
                ]
            )

        selected_set = set(
            selected_nodes
        )

        for node_id in selected_nodes:

            data = dict(
                graph.nodes[
                    node_id
                ]
            )

            nodes.append({
                "id": node_id,
                **data,
            })

        edges = []

        for source, target, key, data in (
            graph.edges(
                keys=True,
                data=True
            )
        ):

            if (
                source not in selected_set
                or target not in selected_set
            ):
                continue

            edges.append({
                "id":
                    f"{source}"
                    f"->{target}"
                    f":{key}",

                "source":
                    source,

                "target":
                    target,

                **dict(data),
            })

        return {
            "nodes": nodes,
            "edges": edges,
        }


    # =====================================================
    # ENTITY NEIGHBORHOOD
    # =====================================================

    def entity_neighborhood(
        self,
        entity_id,
        depth=1,
        max_nodes=100
    ):

        if not self.graph.has_node(
            entity_id
        ):
            raise ValueError(
                f"Unknown entity: "
                f"{entity_id}"
            )

        # Convert to undirected only for
        # neighborhood discovery.
        undirected = (
            self.graph.to_undirected()
        )

        distances = (
            nx.single_source_shortest_path_length(
                undirected,
                entity_id,
                cutoff=depth
            )
        )

        nodes = list(
            distances.keys()
        )[:max_nodes]

        subgraph = self.graph.subgraph(
            nodes
        ).copy()

        return self.serialize_graph(
            subgraph
        )


    # =====================================================
    # DOCUMENT-SCOPED CASE GRAPH
    # =====================================================

    def document_subgraph(
        self,
        document_ids,
        max_nodes=250
    ):

        requested = set(
            str(document_id)
            for document_id
            in document_ids
        )

        selected_nodes = set()

        # -----------------------------------------
        # Documents
        # -----------------------------------------

        for document_id in requested:

            node_id = (
                f"document:{document_id}"
            )

            if self.graph.has_node(
                node_id
            ):
                selected_nodes.add(
                    node_id
                )

        # -----------------------------------------
        # Entities appearing in those docs
        # -----------------------------------------

        for entity in self.entities:

            entity_docs = set(
                entity.get(
                    "document_ids",
                    []
                )
            )

            if entity_docs & requested:

                selected_nodes.add(
                    entity["entity_id"]
                )

        # -----------------------------------------
        # Events from those docs
        # -----------------------------------------

        for frame in self.frames:

            if (
                frame["document_id"]
                in requested
            ):

                selected_nodes.add(
                    f"event:"
                    f"{frame['frame_id']}"
                )

        selected_nodes = list(
            selected_nodes
        )[:max_nodes]

        subgraph = self.graph.subgraph(
            selected_nodes
        ).copy()

        return self.serialize_graph(
            subgraph
        )


    # =====================================================
    # SAVE COMPLETE GRAPH
    # =====================================================

    def save(self):

        data = self.serialize_graph()

        GRAPH_OUTPUT_PATH.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        return GRAPH_OUTPUT_PATH


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

def main():

    evidence_graph = EvidenceGraph()

    stats = evidence_graph.stats()

    output_path = (
        evidence_graph.save()
    )

    print("Evidence graph built.")

    print(
        f"Nodes: "
        f"{stats['total_nodes']}"
    )

    print(
        f"Edges: "
        f"{stats['total_edges']}"
    )

    print(
        "Nodes by type:",
        stats["nodes_by_type"]
    )

    print(
        "Edges by type:",
        stats["edges_by_type"]
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()