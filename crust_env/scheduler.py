"""
CRust Dependency Scheduler — scheduler.py

Builds a Directed Acyclic Graph (DAG) from a legacy C codebase by parsing
local `#include "..."` directives, then produces a bottom-up (leaf-first)
topological migration schedule.

This is the core of the "Divide-Transpile-Reconstruct" paradigm:
  1. Leaf nodes  → no internal dependencies  → translate first
  2. Middle nodes → depend on translated leaves → translate next
  3. Root nodes  → entire DAG already migrated → translate last

This strategy limits the "blast radius" of any single breaking change and
prevents sparse-reward stalls by keeping early tasks tractable.
"""

import os
import re
from collections import deque
from typing import Dict, List, Set, Optional, Tuple


class CDependencyGraph:
    """
    Parses a C project directory tree and constructs a file-level dependency DAG.

    Nodes  = .c and .h files (identified by basename)
    Edges  = directed from includer → included
             (A depends on B means A must be translated AFTER B)

    Topological sort produces a leaf-first ordering suitable for the
    bottom-up migration strategy described in the research paper.
    """

    def __init__(self, c_code_dir: str):
        self.c_code_dir = c_code_dir
        # adjacency: file → list of files it depends on
        self.graph: Dict[str, List[str]] = {}
        # reverse: file → list of files that depend on it
        self.reverse_graph: Dict[str, List[str]] = {}
        self.nodes: Set[str] = set()
        # mapping from basename → absolute path
        self._file_map: Dict[str, str] = {}

    # ── Graph construction ─────────────────────────────────────────────────

    def _find_files(self) -> List[str]:
        """Recursively find all .c and .h files in the project directory."""
        files: List[str] = []
        if not os.path.isdir(self.c_code_dir):
            return files
        for root, _, filenames in os.walk(self.c_code_dir):
            for filename in filenames:
                if filename.endswith((".c", ".h")):
                    files.append(os.path.join(root, filename))
        return files

    def _parse_includes(self, file_path: str) -> List[str]:
        """
        Extract local includes: `#include "filename.h"` or `#include "path/file.h"`
        Ignores system includes: `#include <stdio.h>`.
        Returns basenames of included files (for DAG node matching).
        """
        pattern = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
        includes: List[str] = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            for match in pattern.finditer(content):
                # Normalize to basename for graph matching
                included_name = os.path.basename(match.group(1))
                includes.append(included_name)
        except OSError:
            pass
        return includes

    def build_graph(self) -> None:
        """
        Build the forward and reverse dependency graphs.
        Uses basenames as node IDs for simplicity (MVP-safe; handles flat/nested src/).
        """
        files = self._find_files()

        # Build filename → absolute path mapping
        self._file_map = {}
        for f in files:
            basename = os.path.basename(f)
            # Last-one-wins if duplicates (prefer src/ over include/ for .c files)
            self._file_map[basename] = f

        # Initialize graph nodes
        for basename in self._file_map:
            self.nodes.add(basename)
            self.graph.setdefault(basename, [])
            self.reverse_graph.setdefault(basename, [])

        # Add edges based on #include directives
        for basename, abs_path in self._file_map.items():
            for dep_name in self._parse_includes(abs_path):
                if dep_name not in self.nodes:
                    # Header referenced but not found → add as virtual node
                    self.nodes.add(dep_name)
                    self.graph.setdefault(dep_name, [])
                    self.reverse_graph.setdefault(dep_name, [])

                # Edge: basename depends on dep_name
                if dep_name not in self.graph[basename]:
                    self.graph[basename].append(dep_name)
                if basename not in self.reverse_graph[dep_name]:
                    self.reverse_graph[dep_name].append(basename)

    def get_topological_schedule(self) -> List[str]:
        """
        Kahn's algorithm for topological sort — returns leaf-first order.

        Leaf nodes (in-degree = 0, no dependencies) are served first.
        If a cycle is detected (circular includes), remaining nodes are appended
        as a best-effort fallback so the environment can still make progress.

        Returns:
            List of C/H filenames in the recommended migration order.
        """
        self.build_graph()

        if not self.nodes:
            return []

        # Compute in-degree: number of local dependencies each file has
        in_degree: Dict[str, int] = {node: 0 for node in self.nodes}
        for node, deps in self.graph.items():
            in_degree[node] = len(deps)

        # Start with all nodes that have no dependencies (true leaf nodes)
        queue: deque = deque(
            sorted(n for n in self.nodes if in_degree[n] == 0)
        )
        schedule: List[str] = []

        while queue:
            node = queue.popleft()
            schedule.append(node)

            # For every node that depends on `node`, reduce its in-degree
            for dependent in sorted(self.reverse_graph.get(node, [])):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Cycle detection fallback: add remaining unscheduled nodes
        if len(schedule) != len(self.nodes):
            scheduled_set = set(schedule)
            for node in sorted(self.nodes):
                if node not in scheduled_set:
                    schedule.append(node)

        # Filter to only .c files for the migration schedule
        # (.h headers are context, not translation targets)
        c_schedule = [f for f in schedule if f.endswith(".c")]
        h_schedule = [f for f in schedule if f.endswith(".h")]

        # Headers come first as context, then .c files in dependency order
        return h_schedule + c_schedule

    def get_dependency_info(self) -> Dict[str, Dict]:
        """
        Returns rich dependency metadata for each node.
        Used by the observation to give the agent architectural context.
        """
        self.build_graph()
        info: Dict[str, Dict] = {}
        for node in self.nodes:
            info[node] = {
                "depends_on": self.graph.get(node, []),
                "depended_by": self.reverse_graph.get(node, []),
                "abs_path": self._file_map.get(node),
            }
        return info
