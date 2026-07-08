// frontend/components/graph/GraphViewer.tsx
// Read-only knowledge graph visualization using ReactFlow.
"use client";
import ReactFlow, {
  MiniMap, Controls, Background, Node, Edge, useNodesState, useEdgesState,
} from "reactflow";
import "reactflow/dist/style.css";

interface GraphData {
  entities: { entity_id: string; name: string; type: string }[];
  relations: { source_entity_id: string; target_entity_id: string; relation: string }[];
}

export default function GraphViewer({ data }: { data: GraphData }) {
  const initialNodes: Node[] = (data.entities || []).map((e, i) => ({
    id: e.entity_id,
    data: { label: `${e.name} (${e.type})` },
    position: { x: (i % 4) * 220 + 50, y: Math.floor(i / 4) * 120 + 50 },
    style: {
      background: e.type === "Algorithm" ? "#6366f1" : e.type === "Concept" ? "#8b5cf6" : "#a78bfa",
      color: "#fff", borderRadius: 12, padding: 12, fontSize: 13, border: "none",
    },
  }));

  const initialEdges: Edge[] = (data.relations || []).map((r, i) => ({
    id: `edge-${i}`,
    source: r.source_entity_id,
    target: r.target_entity_id,
    label: r.relation.replace(/_/g, " "),
    animated: true,
    style: { stroke: "#94a3b8" },
    labelStyle: { fontSize: 10, fill: "#94a3b8" },
  }));

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  return (
    <div style={{ width: "100%", height: 500 }}>
      <ReactFlow nodes={nodes} edges={edges}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
        nodesDraggable={true} nodesConnectable={false} elementsSelectable={false}
        fitView>
        <Controls />
        <MiniMap />
        <Background />
      </ReactFlow>
    </div>
  );
}
