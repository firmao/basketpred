import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# Initialize Figure (High DPI for publication crispness)
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.8, 3.8)
ax.axis("off")

# ==================== DATA DEFINITIONS ====================

# Nodes: (x, y, label, category)
nodes = {
    # Layer 3: External Ontologies (Top Row)
    "duo": (0.2, 3.2, "duo:0000017", "ext"),
    "foaf": (1.8, 3.2, "foaf:Person\nschema:Athlete", "ext"),
    "sosa": (3.4, 3.2, "sosa:Observation", "ext"),
    
    # Layer 2: Core Domain Classes (Middle Row)
    "motor": (0.2, 2.0, "ciatec:Motor\nClassification", "domain"),
    "athlete": (1.8, 2.0, "ciatec:Athlete", "core"),
    "metric": (3.4, 2.0, "ciatec:Performance\nMetric", "domain"),
    
    # Layer 1: Subsidiary Classes & Events (Lower Middle)
    "device": (0.2, 0.8, "ciatec:Assistive\nDevice", "domain"),
    "zone": (1.8, 0.8, "ciatec:CourtZone", "domain"),
    "event": (3.4, 0.8, "ciatec:ShotEvent", "domain"),
    
    # Layer 0: Subclasses & Standards (Bottom Row)
    "wheelchair": (-0.4, -0.4, "ciatec:\nWheelchair", "domain"),
    "ball": (0.8, -0.4, "ciatec:\nSmartBall", "domain"),
    "geo": (1.8, -0.4, "geo:Feature", "ext"),
    "acc": (3.4, -0.4, "achievedAccuracy\n(xsd:float)", "data")
}

# Style Palette
styles = {
    "core":   {"fc": "#E3F2FD", "ec": "#1565C0", "tc": "#0D47A1", "lw": 2.0, "shape": "round"},
    "domain": {"fc": "#E0F2F1", "ec": "#00897B", "tc": "#004D40", "lw": 1.5, "shape": "round"},
    "ext":    {"fc": "#F3E5F5", "ec": "#8E24AA", "tc": "#4A148C", "lw": 1.5, "shape": "dashed"},
    "data":   {"fc": "#FFF8E1", "ec": "#FFB300", "tc": "#E65100", "lw": 1.5, "shape": "square"}
}

# Edge Relationships: (source, target, label, color_style)
edges = [
    # Top Subclass Alignments
    ("motor", "duo", "rdfs:subClassOf", "#8E24AA"),
    ("athlete", "foaf", "rdfs:subClassOf", "#8E24AA"),
    ("metric", "sosa", "rdfs:subClassOf", "#8E24AA"),
    
    # Core Domain Relationships
    ("athlete", "motor", "ciatec:belongsToClass", "#1565C0"),
    ("metric", "athlete", "sosa:hasFeatureOfInterest", "#1565C0"),
    ("athlete", "device", "ciatec:recommendedDevice", "#1565C0"),
    ("athlete", "zone", "ciatec:predictedHighAccuracy", "#1565C0"),
    ("event", "metric", "ciatec:executedBy", "#1565C0"),
    
    # Equipment Hierarchy & External Mappings
    ("wheelchair", "device", "rdfs:subClassOf", "#8E24AA"),
    ("ball", "device", "rdfs:subClassOf", "#8E24AA"),
    ("zone", "geo", "rdfs:subClassOf", "#8E24AA"),
    ("event", "acc", "datatype", "#FFB300")
]

# ==================== DRAWING LOGIC ====================

# 1. Draw Curved Edges First (Behind Nodes)
for src, dst, label, color in edges:
    x1, y1 = nodes[src][0], nodes[src][1]
    x2, y2 = nodes[dst][0], nodes[dst][1]
    
    # Compute Bezier Curve Control Points
    dx, dy = x2 - x1, y2 - y1
    cx, cy = x1 + dx * 0.5, y1 + dy * 0.5
    
    # Slight curvature based on orientation
    if abs(dx) > abs(dy):
        cy += 0.2
    else:
        cx += 0.1
        
    path = Path([(x1, y1), (cx, cy), (x2, y2)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    patch = patches.PathPatch(path, facecolor="none", edgecolor=color, lw=1.2, alpha=0.7, zorder=1)
    ax.add_patch(patch)
    
    # Draw Edge Label Badge (White mask background prevents text overlap)
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mid_x, mid_y, f" {label} ", fontsize=6.5, fontfamily="sans-serif",
            fontstyle="italic", color=color, ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))

# 2. Draw Nodes (In Front of Edges)
for key, (x, y, text, cat) in nodes.items():
    st = styles[cat]
    
    # Box Style
    if st["shape"] == "dashed":
        bbox_props = dict(boxstyle="round,pad=0.5,rounding_size=0.3", fc=st["fc"], ec=st["ec"], lw=st["lw"], ls="--")
    elif st["shape"] == "square":
        bbox_props = dict(boxstyle="square,pad=0.4", fc=st["fc"], ec=st["ec"], lw=st["lw"])
    else:
        bbox_props = dict(boxstyle="round,pad=0.5,rounding_size=0.4", fc=st["fc"], ec=st["ec"], lw=st["lw"])

    ax.text(x, y, text, fontsize=8, fontfamily="sans-serif", fontweight="bold" if cat == "core" else "normal",
            color=st["tc"], ha="center", va="center", zorder=4, bbox=bbox_props)

plt.tight_layout()
plt.savefig("fig_kg_schema.png", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved elegant pure Python PNG as 'fig_kg_schema.png'!")